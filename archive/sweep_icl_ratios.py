"""
ICL ratio sweep — tests the linear superposition hypothesis by varying the
number of ICL examples drawn from each task/specialty.

If task vectors superpose linearly, the contrast vector centroid for ratio
(n_0, n_1, n_2) should fall at the weighted average of the pure centroids:

    (n_0·V_0 + n_1·V_1 + n_2·V_2) / (n_0 + n_1 + n_2)

Supports three experiments via --experiment:
  specialization  Medicine / Surgery / Pharmacology MCQ (default)
  format_tasks    MCQ / PubMedQA / Symptom2Disease
  arithmetic      Direct / MCQ / Verification (programmatic arithmetic)

Usage:
  python sweep_icl_ratios.py
  python sweep_icl_ratios.py --experiment format_tasks
  python sweep_icl_ratios.py --experiment arithmetic
"""

import argparse
import os
import re
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --------------------------------------------------------------------------
# Experiment + model selection — parsed before importing local_model
# --------------------------------------------------------------------------

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--experiment",
    choices=["specialization", "format_tasks", "arithmetic", "semantic_domains"],
    default="specialization",
)
_parser.add_argument("--layer", type=int, default=None,
                     help="Override the default layer for this experiment.")
_parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                     help="HuggingFace model ID to use.")
_parser.add_argument("--quantize", default="", choices=["", "int8", "int4"],
                     help="Quantize weights via quanto (recommended for 8B on MPS).")
_args, _ = _parser.parse_known_args()

os.environ["TASK_VECTOR_MODEL"]    = _args.model
os.environ["TASK_VECTOR_QUANTIZE"] = _args.quantize
_size_match = re.search(r'(\d+\.?\d*[Bb])', _args.model)
MODEL_TAG = _size_match.group(1).upper() if _size_match else _args.model.split("/")[-1]

from local_model import get_activation, n_layers
from log_utils import log_run_header, log_icl_sample

if _args.experiment == "specialization":
    from data_specialization import (
        TASKS, RATIOS, PURE_RATIOS, N_VECTOR_SAMPLES, load_pools, build_messages,
    )
    LAYER      = 14
    OUT_PREFIX = "specialization"
elif _args.experiment == "format_tasks":
    from data_format_tasks import (
        TASKS, RATIOS, PURE_RATIOS, N_VECTOR_SAMPLES, load_pools, build_messages,
    )
    LAYER      = 3
    OUT_PREFIX = "format_tasks"
elif _args.experiment == "arithmetic":
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer_stratification"))
    from data_arithmetic_formats import (
        TASKS, RATIOS, PURE_RATIOS, N_VECTOR_SAMPLES, load_pools, build_messages,
    )
    LAYER      = 8
    OUT_PREFIX = "arithmetic"
else:  # semantic_domains
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer_stratification"))
    from data_semantic_domains import (
        TASKS, RATIOS, PURE_RATIOS, N_VECTOR_SAMPLES, load_pools, build_messages,
    )
    LAYER      = 14
    OUT_PREFIX = "semantic_domains"

if _args.layer is not None:
    LAYER = _args.layer


# --------------------------------------------------------------------------
# Contrast vector
# --------------------------------------------------------------------------

def get_contrast_vector(messages: list[dict], layer: int) -> torch.Tensor:
    """activation(ICL + test_q) − activation(test_q alone) at `layer`."""
    icl_act  = get_activation(messages,       layer=layer, add_generation_prompt=True)
    zero_act = get_activation(messages[-1:],  layer=layer, add_generation_prompt=True)
    return icl_act - zero_act


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def collect_all_ratios(
    pools,
    ratios: list[tuple[int, int, int]] = RATIOS,
    n_samples: int = N_VECTOR_SAMPLES,
    layer: int = LAYER,
) -> dict[tuple[int, int, int], torch.Tensor]:
    """Collect contrast vectors for every ratio.

    pools is (icl_pool, test_pool) for specialization or the pools dict for
    format_tasks — build_messages handles both via the data module interface.
    """
    results: dict[tuple[int, int, int], torch.Tensor] = {}
    log_run_header(f"sweep_icl_ratios ({_args.experiment}) — layer {layer}")
    for counts in ratios:
        label = "-".join(map(str, counts))
        print(f"  Ratio {label} ({n_samples} samples)...")
        vecs = []
        for i in range(n_samples):
            rng  = random.Random(i)
            msgs = build_messages(pools, rng, counts, sample_idx=i)
            log_icl_sample(label, i, msgs, layer)
            vecs.append(get_contrast_vector(msgs, layer))
        results[counts] = torch.stack(vecs)
    return results


# --------------------------------------------------------------------------
# LDA
# --------------------------------------------------------------------------

def fit_and_project(
    vectors: dict[tuple[int, int, int], torch.Tensor],
    n_samples: int = N_VECTOR_SAMPLES,
) -> tuple[LinearDiscriminantAnalysis, dict[tuple[int, int, int], np.ndarray]]:
    """Fit LDA on the three pure single-task conditions, project all ratios."""
    X = np.vstack([vectors[p].float().numpy() for p in PURE_RATIOS])
    y = np.repeat(np.arange(len(PURE_RATIOS)), n_samples)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)

    projected = {ratio: lda.transform(v.float().numpy()) for ratio, v in vectors.items()}
    return lda, projected


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

_TASK_COLORS = ["steelblue", "seagreen", "darkorange"]


def plot_ratio_sweep(
    projected: dict[tuple[int, int, int], np.ndarray],
    layer: int = LAYER,
) -> None:
    _, ax = plt.subplots(figsize=(8, 7))

    pure_centroids: dict[tuple[int, int, int], np.ndarray] = {}
    for ratio, task, color in zip(PURE_RATIOS, TASKS, _TASK_COLORS):
        c = projected[ratio].mean(axis=0)
        pure_centroids[ratio] = c
        ax.scatter(*c, color=color, s=200, zorder=5)
        ax.annotate(
            f"{task}\n({'-'.join(map(str, ratio))})", c,
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )

    tri_pts = np.array([pure_centroids[p] for p in PURE_RATIOS])
    ax.add_patch(mpatches.Polygon(
        tri_pts, fill=False, edgecolor="gray", linestyle="--", linewidth=1
    ))

    for ratio in projected:
        if ratio in PURE_RATIOS:
            continue
        actual    = projected[ratio].mean(axis=0)
        total     = sum(ratio)
        predicted = sum(
            (n / total) * pure_centroids[p]
            for n, p in zip(ratio, PURE_RATIOS)
        )
        label = "-".join(map(str, ratio))
        dist  = np.linalg.norm(actual - predicted)
        ax.scatter(*actual,    color="crimson", s=80, zorder=4)
        ax.scatter(*predicted, color="gray",    s=80, marker="x", linewidths=1.5, zorder=4)
        ax.plot([predicted[0], actual[0]], [predicted[1], actual[1]],
                color="gray", linewidth=0.8, linestyle=":")
        ax.annotate(f"{label}\n({dist:.3f})", actual,
                    textcoords="offset points", xytext=(5, -14), fontsize=7, color="crimson")

    handles = [
        plt.scatter([], [], color="crimson", s=80, label="Actual centroid"),
        plt.scatter([], [], color="gray",    s=80, marker="x", label="Predicted (linear combo)"),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("LDA component 1")
    ax.set_ylabel("LDA component 2")
    ax.set_title(
        f"ICL ratio sweep ({_args.experiment}) — contrast vectors, layer {layer}\n"
        "Dotted line = gap between predicted and actual centroid"
    )
    plt.tight_layout()
    out = f"{OUT_PREFIX}_ratio_sweep_{MODEL_TAG}_layer{layer}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved → {out}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Experiment: {_args.experiment}")
    print("Loading data...")
    pools = load_pools()

    print(f"\nCollecting contrast vectors at layer {LAYER}...")
    vectors        = collect_all_ratios(pools)
    _, projected   = fit_and_project(vectors)
    pure_centroids = {p: projected[p].mean(axis=0) for p in PURE_RATIOS}

    print("\n=== Ratio sweep results ===")
    for ratio in RATIOS:
        actual    = projected[ratio].mean(axis=0)
        total     = sum(ratio)
        predicted = sum(
            (n / total) * pure_centroids[p]
            for n, p in zip(ratio, PURE_RATIOS)
        )
        dist  = np.linalg.norm(actual - predicted)
        label = "-".join(map(str, ratio))
        print(f"  {label:>8}: |actual − predicted| = {dist:.4f}")

    plot_ratio_sweep(projected)
