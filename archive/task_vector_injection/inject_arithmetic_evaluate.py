"""
Evaluate task vector injection on zero-shot arithmetic examples.

For each format (Direct / MCQ / Verification), reports accuracy under:

  No injection  — pure zero-shot baseline
  Matching      — inject that format's mean task vector (diagonal)
  Mismatching   — inject another format's vector (off-diagonal control)

All conditions use a single forward pass and read logits directly —
no text generation or parsing required. Each format restricts the model
to a fixed candidate set:

  Direct       — integers 2–98 that tokenise as a single token
  MCQ          — ["A", "B", "C", "D"]
  Verification — ["yes", "no"]

Vectors must be pre-extracted with extract_arithmetic_vectors.py.

Usage:
  python task_vector_injection/inject_arithmetic_evaluate.py
  python task_vector_injection/inject_arithmetic_evaluate.py --layer 8 --alpha 2.0
  python task_vector_injection/inject_arithmetic_evaluate.py --sweep_layers
"""

import sys
import os

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "layer_stratification"))

import argparse
import random
import numpy as np
import torch

from local_model import predict_constrained, predict_constrained_with_injection, tokenizer
import data_arithmetic_formats as data

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--layer",        type=int,   default=8)
_parser.add_argument("--alpha",        type=float, default=2.0)
_parser.add_argument("--n_test",       type=int,   default=20)
_parser.add_argument("--sweep_layers", action="store_true")
_args, _ = _parser.parse_known_args()

VEC_DIR = os.path.join(_dir, "vectors")
LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------
# Format-specific candidate sets
# --------------------------------------------------------------------------

def _single_token_integers(lo: int, hi: int) -> list[str]:
    """All integers in [lo, hi] that the tokeniser encodes as a single token."""
    out = []
    for n in range(lo, hi + 1):
        s = str(n)
        for variant in (s, f" {s}"):
            if len(tokenizer.encode(variant, add_special_tokens=False)) == 1:
                out.append(s)
                break
    return out


# Computed once at import time; covers all possible Direct answers (op1,op2 ∈ [1,49])
_DIRECT_CANDIDATES   = _single_token_integers(2, 98)
_MCQ_CANDIDATES      = ["A", "B", "C", "D"]
_VERIF_CANDIDATES    = ["yes", "no"]

_CANDIDATES: dict[str, list[str]] = {
    "Direct":       _DIRECT_CANDIDATES,
    "MCQ":          _MCQ_CANDIDATES,
    "Verification": _VERIF_CANDIDATES,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_vector(fmt: str, layer: int) -> torch.Tensor:
    path = os.path.join(VEC_DIR, f"arith_contrast_{fmt}_layer{layer}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No vector at {path}. Run extract_arithmetic_vectors.py first."
        )
    return torch.tensor(np.load(path))


def zero_shot_messages(fmt: str, ex: dict) -> list[dict]:
    inp, _ = data.format_icl_example(fmt, ex)
    return [{"role": "user", "content": inp}]


def correct_answer(fmt: str, ex: dict) -> str:
    """Ground-truth answer string for each format."""
    if fmt == "Direct":
        return str(ex["answer"])
    elif fmt == "MCQ":
        return LETTERS[ex["correct_idx"]]
    else:
        return ex["label"]


def evaluate_condition(
    pools: dict,
    target_fmt: str,
    inject_fmt: str | None,
    layer: int,
    alpha: float,
    n_test: int,
) -> float:
    """Accuracy on zero-shot `target_fmt` examples, with optional injection."""
    candidates = _CANDIDATES[target_fmt]
    test_pool  = pools[target_fmt]["test"]
    examples   = random.Random(42).sample(test_pool, min(n_test, len(test_pool)))

    correct = 0
    for ex in examples:
        msgs = zero_shot_messages(target_fmt, ex)

        if inject_fmt is None:
            pred = predict_constrained(msgs, candidates)
        else:
            tv   = load_vector(inject_fmt, layer)
            pred = predict_constrained_with_injection(msgs, candidates, tv, layer, alpha)

        correct += pred == correct_answer(target_fmt, ex)

    return correct / len(examples)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def run_layer(pools: dict, layer: int, alpha: float, n_test: int) -> None:
    fmts  = data.TASKS
    col_w = 14

    print(f"\n--- Layer {layer}, alpha={alpha}, n_test={n_test} ---")
    header = f"{'Target':<{col_w}} {'No inject':>{col_w}}"
    for f in fmts:
        header += f"  {'+ ' + f:>{col_w}}"
    print(header)
    print("-" * len(header))

    for target in fmts:
        baseline = evaluate_condition(pools, target, None, layer, alpha, n_test)
        row = f"{target:<{col_w}} {baseline:>{col_w}.3f}"
        for inject in fmts:
            acc    = evaluate_condition(pools, target, inject, layer, alpha, n_test)
            marker = "*" if inject == target else " "
            row   += f"  {acc:>{col_w - 1}.3f}{marker}"
        print(row)

    print("  * = matching format injection (diagonal)")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating arithmetic pools...")
    pools = data.load_pools()

    if _args.sweep_layers:
        saved = sorted(
            int(f.split("_layer")[1].split(".npy")[0])
            for f in os.listdir(VEC_DIR)
            if f.endswith(".npy") and f"arith_contrast_{data.TASKS[0]}" in f
        )
        print(f"Sweeping layers: {saved}")
        for layer in saved:
            run_layer(pools, layer, _args.alpha, _args.n_test)
    else:
        run_layer(pools, _args.layer, _args.alpha, _args.n_test)
