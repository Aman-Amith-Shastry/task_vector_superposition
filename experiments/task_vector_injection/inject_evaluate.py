"""
Evaluate task vector injection on MMLU zero-shot examples.

For each test question, compares three probability distributions over A/B/C/D:
  p_zs   — zero-shot (no ICL context)
  p_icl  — full 3-shot ICL (upper-bound reference)
  p_inj  — zero-shot + injected contrast vector

Metrics reported per condition:
  P(correct)         — probability assigned to the correct answer letter
  KL(p || p_icl)     — divergence from 3-shot ICL behaviour

Key comparison: KL(p_inj || p_icl) vs KL(p_zs || p_icl)
  If injection works, the injected distribution should be closer to ICL than
  the raw zero-shot distribution — i.e., KL goes down.

Vectors must be pre-extracted with extract_vectors.py.

Usage:
  python task_vector_injection/inject_evaluate.py
  python task_vector_injection/inject_evaluate.py --layer 10 --alpha 2.0
  python task_vector_injection/inject_evaluate.py --sweep_layers
"""

import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "layer_stratification"))

import argparse
import math
import random
import numpy as np
import torch

from local_model import predict_mcq, predict_mcq_with_injection
import data_semantic_domains as data

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--layer",        type=int,   default=10)
_parser.add_argument("--alpha",        type=float, default=2.0)
_parser.add_argument("--n_test",       type=int,   default=20)
_parser.add_argument("--sweep_layers", action="store_true")
_parser.add_argument("--residual",     action="store_true",
                     help="Inject domain-residual vectors (contrast − common) "
                          "instead of full contrast vectors.")
_args, _ = _parser.parse_known_args()

VEC_DIR = os.path.join(_dir, "vectors")
LETTERS = ["A", "B", "C", "D"]
EPS     = 1e-9


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_vector(domain: str, layer: int, residual: bool = False) -> torch.Tensor:
    suffix = "_residual" if residual else ""
    path   = os.path.join(VEC_DIR, f"{domain}{suffix}_layer{layer}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No vector at {path}. Run extract_vectors.py first."
        )
    return torch.tensor(np.load(path))


def zero_shot_messages(ex: dict) -> list[dict]:
    c = ex["choices"]
    return [{"role": "user", "content": (
        f"Question: {ex['question']}\n"
        f"A) {c[0]}  B) {c[1]}  C) {c[2]}  D) {c[3]}\n"
        "Answer:"
    )}]


def icl_messages(pools: dict, target_domain: str, test_ex: dict, sample_idx: int) -> list[dict]:
    """3-shot ICL prompt for target_domain with test_ex as the held-out question."""
    pure_ratio = data.PURE_RATIOS[data.TASKS.index(target_domain)]
    # Offset seed so ICL examples don't overlap with extraction seeds (0..N_EXTRACT-1)
    rng  = random.Random(sample_idx + 10000)
    msgs = data.build_messages(pools, rng, pure_ratio, sample_idx=sample_idx)
    # Swap in the specific test question (build_messages picks one at random)
    msgs[-1] = {"role": "user", "content": data._format_question(test_ex)}
    return msgs


def kl(p: dict, q: dict) -> float:
    """KL(p || q) over LETTERS with epsilon smoothing."""
    return sum(
        (p[l] + EPS) * math.log((p[l] + EPS) / (q[l] + EPS))
        for l in LETTERS
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_target(
    pools: dict,
    target_domain: str,
    layer: int,
    alpha: float,
    n_test: int,
) -> dict:
    """
    Runs all injection conditions for one target domain.
    Computes p_zs and p_icl once per example, then evaluates each condition.

    Returns a dict keyed by condition (None = no inject, or domain str):
      {"p_correct": float, "kl_to_icl": float}
    Also returns "kl_zs_baseline": mean KL(p_zs || p_icl).
    """
    test_pool = pools[target_domain]["test"]
    examples  = random.Random(42).sample(test_pool, min(n_test, len(test_pool)))

    # Load all vectors once
    vectors = {d: load_vector(d, layer, residual=_args.residual) for d in data.TASKS}

    # Accumulators: None = no injection, domain str = that injection
    conditions = [None] + data.TASKS
    p_corrects = {c: [] for c in conditions}
    kl_to_icls = {c: [] for c in conditions}

    for idx, ex in enumerate(examples):
        zs_msgs  = zero_shot_messages(ex)
        icl_msgs = icl_messages(pools, target_domain, ex, idx)
        true_letter = LETTERS[ex["answer"]]

        p_zs  = predict_mcq(zs_msgs)[1]
        p_icl = predict_mcq(icl_msgs)[1]

        # No-injection condition
        p_corrects[None].append(p_zs[true_letter])
        kl_to_icls[None].append(kl(p_zs, p_icl))

        # Each injection condition
        for inj_domain in data.TASKS:
            p_inj = predict_mcq_with_injection(
                zs_msgs, vectors[inj_domain], layer, alpha
            )[1]
            p_corrects[inj_domain].append(p_inj[true_letter])
            kl_to_icls[inj_domain].append(kl(p_inj, p_icl))

    return {
        c: {
            "p_correct": float(np.mean(p_corrects[c])),
            "kl_to_icl": float(np.mean(kl_to_icls[c])),
        }
        for c in conditions
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def run_layer(pools: dict, layer: int, alpha: float, n_test: int) -> None:
    col_w = 14

    print(f"\n{'='*70}")
    print(f"Layer {layer}  |  alpha={alpha}  |  n_test={n_test}")
    print(f"{'='*70}")

    all_results = {t: evaluate_target(pools, t, layer, alpha, n_test) for t in data.TASKS}

    # --- Table 1: P(correct answer) ---
    print(f"\nP(correct answer)  [↑ better]")
    header = f"{'Target':<{col_w}} {'No inject':>{col_w}}"
    for d in data.TASKS:
        header += f"  {'+ ' + d:>{col_w}}"
    print(header)
    print("-" * len(header))
    for target in data.TASKS:
        r   = all_results[target]
        row = f"{target:<{col_w}} {r[None]['p_correct']:>{col_w}.3f}"
        for inj in data.TASKS:
            marker = "*" if inj == target else " "
            row   += f"  {r[inj]['p_correct']:>{col_w - 1}.3f}{marker}"
        print(row)

    # --- Table 2: KL(condition || 3-shot ICL) ---
    print(f"\nKL(condition || 3-shot ICL)  [↓ = closer to ICL behaviour]")
    header2 = f"{'Target':<{col_w}} {'ZS baseline':>{col_w}}"
    for d in data.TASKS:
        header2 += f"  {'+ ' + d:>{col_w}}"
    header2 += f"  {'KL reduction*':>{col_w}}"
    print(header2)
    print("-" * len(header2))
    for target in data.TASKS:
        r        = all_results[target]
        kl_base  = r[None]["kl_to_icl"]
        kl_match = r[target]["kl_to_icl"]
        pct      = 100 * (kl_base - kl_match) / (kl_base + EPS)
        row = f"{target:<{col_w}} {kl_base:>{col_w}.3f}"
        for inj in data.TASKS:
            marker = "*" if inj == target else " "
            row   += f"  {r[inj]['kl_to_icl']:>{col_w - 1}.3f}{marker}"
        row += f"  {pct:>{col_w - 1}.1f}%"
        print(row)
    print("  * = matching domain injection  |  KL reduction = (ZS − match) / ZS")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading MMLU pools...")
    pools = data.load_pools()

    if _args.sweep_layers:
        saved = sorted(
            int(f.split("_layer")[1].split(".npy")[0])
            for f in os.listdir(VEC_DIR)
            if f.endswith(".npy") and data.TASKS[0] in f
        )
        print(f"Sweeping layers: {saved}")
        for layer in saved:
            run_layer(pools, layer, _args.alpha, _args.n_test)
    else:
        run_layer(pools, _args.layer, _args.alpha, _args.n_test)
