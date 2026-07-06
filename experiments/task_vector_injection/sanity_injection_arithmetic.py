"""
Pure-injection sanity gate (go/no-go for the behavioral injection program).

Question: does additively injecting a single stored pure-task contrast vector
into a neutral zero-shot arithmetic prompt steer the model's first-token output
toward that task's format?

  Direct       → a digit   (the numeric sum)
  MCQ          → a letter   A-D
  Verification → yes / no

Default focus is MCQ: weights (alpha, beta, gamma) = (0, 1, 0), i.e. we inject
the pure MCQ contrast centroid v_MCQ at `layer` and ask whether the A/B/C/D mass
rises relative to the no-injection baseline.

Design choices (see discussion):
  * Neutral stem "{op1} + {op2} A:" — the Direct format with NO options shown,
    so A/B/C/D are not structurally available. If MCQ injection still pulls
    toward letters here, that is clean evidence of format steering.
  * Effect is read as the SHIFT from the alpha=0 baseline on the same prompts,
    so the stem's intrinsic Direct lean is differenced out.
  * Category "mass" is reported in logit space (log-sum-exp over the category's
    token ids), where the residual-stream injection is additive, and also as
    probabilities (sum over the category's token ids) for interpretability.
  * alpha is swept: a MONOTONE rise in the injected task's mass is the real pass
    criterion, not a single-alpha bump.

Usage:
  python experiments/task_vector_injection/sanity_injection_arithmetic.py
  python experiments/task_vector_injection/sanity_injection_arithmetic.py --task MCQ --layer 8 --alphas 0 2 4 6 8 10
  python experiments/task_vector_injection/sanity_injection_arithmetic.py --task Verification
"""

import sys, os, argparse, random

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "data"))

os.environ.setdefault("TASK_VECTOR_MODEL",    "meta-llama/Llama-3.2-3B-Instruct")
os.environ.setdefault("TASK_VECTOR_QUANTIZE", "")

_p = argparse.ArgumentParser()
_p.add_argument("--task", choices=["Direct", "MCQ", "Verification"], default="MCQ",
                help="Which pure contrast vector to inject.")
_p.add_argument("--layer", type=int, default=8,
                help="Layer to inject at (default 8 = peak-linearity for Llama-3B arithmetic).")
_p.add_argument("--alphas", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10],
                help="Injection scales to sweep. alpha=0 is the baseline.")
_p.add_argument("--n_prompts", type=int, default=20,
                help="Neutral prompts to average over.")
_args = _p.parse_args()

import numpy as np
import torch
from local_model import model, tokenizer, _device

TASKS   = ["Direct", "MCQ", "Verification"]
VEC_DIR = os.path.join(_dir, "vectors")

# First-token category token sets (filled with actual single-token ids below).
CATEGORY_WORDS = {
    "Direct":       [str(d) for d in range(10)],
    "MCQ":          ["A", "B", "C", "D"],
    "Verification": ["yes", "no", "Yes", "No"],
}


def category_token_ids() -> dict[str, list[int]]:
    """Deduped single-token ids per category, covering bare and space-prefixed forms."""
    cat_ids: dict[str, list[int]] = {}
    for cat, words in CATEGORY_WORDS.items():
        ids: set[int] = set()
        for word in words:
            for variant in (word, f" {word}"):
                enc = tokenizer.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
        cat_ids[cat] = sorted(ids)
    return cat_ids


def load_vec(task: str, layer: int) -> torch.Tensor:
    path = os.path.join(VEC_DIR, f"arith_contrast_{task}_layer{layer}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No stored vector at {path}.\n"
            f"Run extract_arithmetic_vectors.py first."
        )
    return torch.from_numpy(np.load(path)).float()


def neutral_prompts(n: int) -> list[list[dict]]:
    """Neutral Direct-format stems: bare addition + shared 'A:' trigger, no options."""
    rng = random.Random(0)
    return [
        [{"role": "user", "content": f"{rng.randint(1, 49)} + {rng.randint(1, 49)} A:"}]
        for _ in range(n)
    ]


def read_categories(
    messages: list[dict],
    cat_ids: dict[str, list[int]],
    task_vector: torch.Tensor | None = None,
    layer: int | None = None,
    alpha: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-category (logit, probability) at the first answer token, optional injection.

    logit  = log-sum-exp over the category's token logits (additive in injection).
    prob   = summed softmax probability over the category's token ids.
    """
    handle = None
    if task_vector is not None and alpha != 0.0:
        tv = task_vector.to(_device)

        def _hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            if hidden.dim() == 3:
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * tv
            else:
                hidden[-1, :] = hidden[-1, :] + alpha * tv
            return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

        handle = model.model.layers[layer].register_forward_hook(_hook)

    try:
        text   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(_device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
    finally:
        if handle is not None:
            handle.remove()

    probs = torch.softmax(logits, dim=-1)
    out_logit, out_prob = {}, {}
    for cat, ids in cat_ids.items():
        idx = torch.tensor(ids, device=logits.device)
        out_logit[cat] = torch.logsumexp(logits[idx], dim=0).item()
        out_prob[cat]  = probs[idx].sum().item()
    return out_logit, out_prob


def avg_over_prompts(prompts, cat_ids, tv=None, layer=None, alpha=0.0):
    logits, probs = [], []
    for m in prompts:
        lg, pr = read_categories(m, cat_ids, tv, layer, alpha)
        logits.append(lg); probs.append(pr)
    mean_logit = {c: float(np.mean([d[c] for d in logits])) for c in cat_ids}
    mean_prob  = {c: float(np.mean([d[c] for d in probs]))  for c in cat_ids}
    # Distribution normalised over the three categories (relative format preference)
    tot = sum(mean_prob.values()) or 1.0
    norm = {c: mean_prob[c] / tot for c in cat_ids}
    return mean_logit, mean_prob, norm


if __name__ == "__main__":
    print(f"Sanity gate: inject pure v_{_args.task} | layer {_args.layer} | "
          f"alphas {_args.alphas} | {_args.n_prompts} neutral prompts\n")

    cat_ids = category_token_ids()
    for c, ids in cat_ids.items():
        print(f"  {c:<13} {len(ids)} tokens: {ids}")
    print()

    prompts = neutral_prompts(_args.n_prompts)
    tv      = load_vec(_args.task, _args.layer)

    # Baseline (alpha = 0) for shift reference
    base_logit, base_prob, base_norm = avg_over_prompts(prompts, cat_ids)
    print("Baseline (no injection):")
    print("  normalised probs:  " + "  ".join(f"{c}={base_norm[c]:.3f}" for c in TASKS))
    print(f"  {_args.task} logit:    {base_logit[_args.task]:+.3f}")
    print()

    # Header
    cats_hdr = "  ".join(f"{c[:4]:>7}" for c in TASKS)
    print(f"{'alpha':>6}  | norm prob [{cats_hdr}] | "
          f"{_args.task[:4]} logit  Δlogit  Δprob   argmax")
    print("-" * 88)

    for a in _args.alphas:
        if a == 0.0:
            mlogit, mprob, mnorm = base_logit, base_prob, base_norm
        else:
            mlogit, mprob, mnorm = avg_over_prompts(prompts, cat_ids, tv, _args.layer, a)
        argmax   = max(mnorm, key=mnorm.get)
        d_logit  = mlogit[_args.task] - base_logit[_args.task]
        d_prob   = mprob[_args.task]  - base_prob[_args.task]
        flag     = " <-" if argmax == _args.task else ""
        norm_str = "  ".join(f"{mnorm[c]:>7.3f}" for c in TASKS)
        print(f"{a:>6.1f}  | {norm_str} | "
              f"{mlogit[_args.task]:>9.3f}  {d_logit:+6.3f}  {d_prob:+6.3f}   {argmax}{flag}")

    print("\nPASS if the injected task's Δlogit rises monotonically with alpha "
          f"and {_args.task} mass crosses over (argmax '<-').")
