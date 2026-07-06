"""
Behavioral injection test for the entity→attribute group (Capital / Currency /
Language) — does injecting a pure-task contrast vector make that task's gold
answer more likely?

Setup:
  * Prompt is a bare country name (the symmetric-affordance zero-shot baseline):
        [{"role": "user", "content": "Brazil"}]
  * For each test country we know all three gold answers (capital, currency,
    language). We teacher-force-score each gold answer as a continuation and read
    the summed per-token log-probability (NOT raw logits — those are unnormalised;
    NOT raw probs — those underflow for multi-token answers).
  * Baseline = no injection. Injected = add alpha * v_task at the assistant-header
    position of layer `layer`, propagated through the rest of the network.
  * The behavioral signal is ΔlogP = logP(gold | inject) − logP(gold | baseline),
    per (injected task, gold task). The same gold string appears on both sides, so
    its length/frequency bias cancels — we never compare raw logP across different
    gold strings, only each gold answer to its own baseline.

Injection fires ONLY at the assistant-header token (the last prompt token, where
the contrast vector was extracted), not at the appended gold tokens. The first
gold token's log-prob is read at that position; later tokens are steered via
causal attention back to it.

Pass criterion: the 3×3 mean-ΔlogP matrix is DIAGONALLY DOMINANT — injecting v_T
raises logP(gold_T) more than it raises the other two tasks' gold answers.

Usage:
  python experiments/task_vector_injection/inject_entity_behavioral.py
  python experiments/task_vector_injection/inject_entity_behavioral.py --layer 18 --alphas 0 2 4 6 8
"""

import sys, os, argparse

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "data"))

os.environ.setdefault("TASK_VECTOR_MODEL",    "meta-llama/Llama-3.2-3B-Instruct")
os.environ.setdefault("TASK_VECTOR_QUANTIZE", "")

_p = argparse.ArgumentParser()
_p.add_argument("--layer", type=int, default=18,
                help="Layer to inject at (default 18 = peak-linearity for Llama-3B).")
_p.add_argument("--alphas", type=float, nargs="+", default=[0, 2, 4, 6, 8],
                help="Injection scales to sweep. alpha=0 is the baseline.")
_p.add_argument("--n_countries", type=int, default=None,
                help="How many test-pool countries to average over (default: all).")
_p.add_argument("--mode", choices=["raw", "residual"], default="residual",
                help="raw = inject the stored centroid (dominated by the shared ICL "
                     "direction); residual = inject centroid minus the tri-center, "
                     "isolating the task-discriminative direction d_T (recommended).")
_args = _p.parse_args()

import numpy as np
import torch
from local_model import model, tokenizer, _device
import data_entity_attribute as data

TASKS   = data.TASKS
VEC_DIR = os.path.join(_dir, "vectors")


def load_vec(task: str, layer: int) -> torch.Tensor:
    path = os.path.join(VEC_DIR, f"entity_contrast_{task}_layer{layer}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No stored vector at {path}.\n"
            f"Run experiments/task_vector_injection/extract_entity.py first."
        )
    return torch.from_numpy(np.load(path)).float()


def score_gold(
    messages: list[dict],
    continuation: str,
    task_vector: torch.Tensor | None = None,
    layer: int | None = None,
    alpha: float = 0.0,
) -> float:
    """Summed teacher-forced log-prob of `continuation`, optional header injection.

    One forward pass over [prompt + gold]. If a vector is given, alpha*vector is
    added at the assistant-header token (the last prompt token, index n_ctx-1),
    matching the contrast-vector extraction position.
    """
    context_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    context_ids = tokenizer.encode(context_text, add_special_tokens=False, return_tensors="pt")
    # Leading space matters for BPE: " Paris" != "Paris" as a continuation.
    cont_ids = tokenizer.encode(" " + continuation, add_special_tokens=False, return_tensors="pt")
    full_ids = torch.cat([context_ids, cont_ids], dim=1).to(_device)

    n_ctx      = context_ids.shape[1]
    header_pos = n_ctx - 1

    handle = None
    if task_vector is not None and alpha != 0.0:
        tv = task_vector.to(_device)

        def _hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            if hidden.dim() == 3:
                hidden[:, header_pos, :] = hidden[:, header_pos, :] + alpha * tv
            else:
                hidden[header_pos, :] = hidden[header_pos, :] + alpha * tv
            return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

        handle = model.model.layers[layer].register_forward_hook(_hook)

    try:
        with torch.no_grad():
            logits = model(full_ids).logits[0]   # [seq, vocab]
    finally:
        if handle is not None:
            handle.remove()

    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i, tid in enumerate(cont_ids[0]):
        total += log_probs[n_ctx - 1 + i, tid].item()
    # First gold token, read at the header position (where the injection fires):
    # length-invariant and the most direct measure of the injection's effect.
    first = log_probs[n_ctx - 1, cont_ids[0][0]].item()
    return total, first


def compute_baseline(countries):
    """Per-country baseline (no injection) summed & first-token logP for each gold task.

    Computed once and reused across all alphas (the baseline is injection-free).
    Returns (base_sum, base_first, base_sum_mean, base_first_mean).
    """
    base_sum   = {g: [] for g in TASKS}
    base_first = {g: [] for g in TASKS}
    for rec in countries:
        prompt = [{"role": "user", "content": rec["country"]}]
        for g in TASKS:
            tot, fst = score_gold(prompt, data.gold_answer(g, rec))
            base_sum[g].append(tot)
            base_first[g].append(fst)
    sum_mean   = {g: float(np.mean(base_sum[g]))   for g in TASKS}
    first_mean = {g: float(np.mean(base_first[g])) for g in TASKS}
    return base_sum, base_first, sum_mean, first_mean


def delta_matrices(countries, vecs, layer, alpha, base_sum, base_first):
    """Mean ΔlogP[injected_task][gold_task] at one alpha, for both readouts.

    Returns (delta_sum, delta_first): summed-token and first-token ΔlogP matrices.
    """
    d_sum   = {t: {g: [] for g in TASKS} for t in TASKS}
    d_first = {t: {g: [] for g in TASKS} for t in TASKS}
    for ci, rec in enumerate(countries):
        prompt = [{"role": "user", "content": rec["country"]}]
        for t in TASKS:                       # injected task
            for g in TASKS:                   # gold answer scored
                tot, fst = score_gold(prompt, data.gold_answer(g, rec), vecs[t], layer, alpha)
                d_sum[t][g].append(tot - base_sum[g][ci])
                d_first[t][g].append(fst - base_first[g][ci])
    mean = lambda d: {t: {g: float(np.mean(d[t][g])) for g in TASKS} for t in TASKS}
    return mean(d_sum), mean(d_first)


def print_matrix(title: str, delta: dict) -> bool:
    """Print a 3×3 ΔlogP matrix; mark each row's argmax and report diagonal dominance."""
    hdr = "  ".join(f"gold:{g[:4]:>8}" for g in TASKS)
    row_label = "inject \\ gold"
    print(f"  [{title}]")
    print(f"  {row_label:>16}   {hdr}")
    diag_ok = True
    for t in TASKS:
        row_argmax = max(delta[t], key=delta[t].get)
        if row_argmax != t:
            diag_ok = False
        cells = "  ".join(
            f"{delta[t][g]:>+8.2f}{'*' if g == row_argmax else ' '}"
            for g in TASKS
        )
        flag = "" if row_argmax == t else "   <- off-diagonal"
        print(f"  {'inject ' + t:>16}   {cells}{flag}")
    diag_pos = all(delta[t][t] > 0 for t in TASKS)
    print(f"  Diagonal all positive: {'YES' if diag_pos else 'NO'}   "
          f"Diagonally dominant: {'YES' if diag_ok else 'NO'}")
    return diag_ok


def double_center(delta: dict) -> dict:
    """Remove additive row (injection-strength) and column (gold-headroom) effects,
    leaving the task-matching interaction:

        M'[t][g] = M[t][g] − rowmean[t] − colmean[g] + grandmean

    A positive diagonal means injecting d_T helps gold_T more than the row/column
    main effects predict — i.e. genuine task-specific selectivity. (First-order
    correction: assumes additive effects; the log-prob ceiling is only approximately
    additive, so pair with the softmax-preference readout for a stricter check.)
    """
    grand   = float(np.mean([delta[t][g] for t in TASKS for g in TASKS]))
    rowmean = {t: float(np.mean([delta[t][g] for g in TASKS])) for t in TASKS}
    colmean = {g: float(np.mean([delta[t][g] for t in TASKS])) for g in TASKS}
    return {t: {g: delta[t][g] - rowmean[t] - colmean[g] + grand for g in TASKS}
            for t in TASKS}


if __name__ == "__main__":
    print(f"Behavioral injection | layer {_args.layer} | alphas {_args.alphas}")

    pools     = data.load_pools()
    countries = pools[TASKS[0]]["test"]
    if _args.n_countries:
        countries = countries[:_args.n_countries]
    print(f"Test countries ({len(countries)}): "
          f"{', '.join(c['country'] for c in countries)}\n")

    raw_vecs = {t: load_vec(t, _args.layer) for t in TASKS}
    # tri_center = mean of the three pure centroids = the shared ICL/"answer" direction.
    # Residual d_T = v_T - tri_center is the task-discriminative part (the displacement
    # vector whose linearity κ measures in Section 3).
    tri_center = sum(raw_vecs.values()) / len(TASKS)
    residual   = {t: raw_vecs[t] - tri_center for t in TASKS}

    print(f"\nVector norms at layer {_args.layer}:")
    print(f"  ||tri-center (common)|| = {tri_center.norm().item():.2f}")
    for t in TASKS:
        print(f"  {t:>9}:  ||raw|| = {raw_vecs[t].norm().item():6.2f}   "
              f"||residual d_T|| = {residual[t].norm().item():6.2f}")

    vecs = residual if _args.mode == "residual" else raw_vecs
    print(f"\nInjecting '{_args.mode}' vectors "
          f"({'task-discriminative d_T' if _args.mode == 'residual' else 'full centroid'}).\n")

    # Baseline (no injection), computed once and reused across alphas.
    base_sum, base_first, sum_mean, first_mean = compute_baseline(countries)
    print("Baseline mean logP(gold) per task (no injection):")
    print("  summed:      " + "  ".join(f"{g}={sum_mean[g]:+.2f}"   for g in TASKS))
    print("  first-token: " + "  ".join(f"{g}={first_mean[g]:+.2f}" for g in TASKS))

    for alpha in _args.alphas:
        if alpha == 0.0:
            continue
        print(f"\n=== alpha = {alpha} ===")
        d_sum, d_first = delta_matrices(countries, vecs, _args.layer, alpha,
                                        base_sum, base_first)
        # Double-centered: strip injection-strength (row) and gold-headroom (column)
        # main effects, leaving the task-matching interaction. First-token is the
        # primary, length-invariant readout.
        print_matrix("first-token ΔlogP, double-centered  (interaction; primary)",
                     double_center(d_first))
        print()
        print_matrix("summed ΔlogP, double-centered  (interaction; holistic)",
                     double_center(d_sum))

    print("\nPASS if, at some alpha, the double-centered diagonal is positive and "
          "row-dominant (* marks each row's argmax): injecting d_T helps gold_T more "
          "than the injection-strength and gold-headroom main effects predict.")
