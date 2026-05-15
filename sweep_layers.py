"""
Layer sweep — tests task-vector discriminability across transformer layers.

For each layer, collects contrast vectors for the three pure single-task
conditions and the equal mixed condition, projects with LDA, and reports
|mixed − predicted| as the superposition quality metric.

Supports two experiments via --experiment:
  specialization  Medicine / Surgery / Pharmacology MCQ (default)
  format_tasks    MCQ / PubMedQA / Symptom2Disease

Usage:
  python sweep_layers.py
  python sweep_layers.py --experiment format_tasks
"""

import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from local_model import get_activation
from log_utils import log_run_header, log_icl_sample

# --------------------------------------------------------------------------
# Experiment selection
# --------------------------------------------------------------------------

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--experiment",
    choices=["specialization", "format_tasks"],
    default="specialization",
)
_args, _ = _parser.parse_known_args()

if _args.experiment == "specialization":
    from data_specialization import (
        TASKS, PURE_RATIOS, N_VECTOR_SAMPLES, load_pools, build_messages,
    )
    LAYER      = 14
    OUT_PREFIX = "specialization"
else:
    from data_format_tasks import (
        TASKS, PURE_RATIOS, N_VECTOR_SAMPLES, load_pools, build_messages,
    )
    LAYER      = 3
    OUT_PREFIX = "format_tasks"

SWEEP_LAYERS = [3, 8, 14, 20, 26]
MIXED_RATIO  = (1, 1, 1)
SWEEP_RATIOS = PURE_RATIOS + [MIXED_RATIO]


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

def collect_contrast_vectors(
    pools,
    n_samples: int = N_VECTOR_SAMPLES,
    layer: int = LAYER,
) -> dict[tuple[int, int, int], torch.Tensor]:
    conditions: dict[tuple[int, int, int], torch.Tensor] = {}
    log_run_header(f"sweep_layers ({_args.experiment}) — layer {layer}")

    for counts in SWEEP_RATIOS:
        label      = "-".join(map(str, counts))
        task_label = TASKS[counts.index(max(counts))] if counts in PURE_RATIOS else "Mixed"
        print(f"  {task_label} ({label}, {n_samples} samples)...")
        vecs = []
        for i in range(n_samples):
            rng  = random.Random(i)
            msgs = build_messages(pools, rng, counts, sample_idx=i)
            log_icl_sample(label, i, msgs, layer)
            vecs.append(get_contrast_vector(msgs, layer))
        conditions[counts] = torch.stack(vecs)
    return conditions


# --------------------------------------------------------------------------
# LDA and separation metric
# --------------------------------------------------------------------------

def fit_lda(
    vectors: dict[tuple[int, int, int], torch.Tensor],
    n_samples: int = N_VECTOR_SAMPLES,
) -> dict[tuple[int, int, int], np.ndarray]:
    X = np.vstack([vectors[p].float().numpy() for p in PURE_RATIOS])
    y = np.repeat(np.arange(len(PURE_RATIOS)), n_samples)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    return {ratio: lda.transform(v.float().numpy()) for ratio, v in vectors.items()}


def mixed_centroid_distance(projected: dict[tuple[int, int, int], np.ndarray]) -> float:
    """Distance from mixed centroid to the ⅓+⅓+⅓ predicted position in LDA space."""
    pure_centroids = np.array([projected[p].mean(axis=0) for p in PURE_RATIOS])
    predicted      = pure_centroids.mean(axis=0)
    mixed_c        = projected[MIXED_RATIO].mean(axis=0)
    return float(np.linalg.norm(mixed_c - predicted))


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

_COLORS = ["steelblue", "seagreen", "darkorange"]


def _plot_layer(
    projected: dict[tuple[int, int, int], np.ndarray],
    layer: int,
) -> None:
    _, ax = plt.subplots(figsize=(7, 6))

    centroids: dict[tuple[int, int, int], np.ndarray] = {}
    for ratio, task, color in zip(PURE_RATIOS, TASKS, _COLORS):
        pts = projected[ratio]
        c   = pts.mean(axis=0)
        centroids[ratio] = c
        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.2, color=color, s=20)
        ax.scatter(*c, color=color, s=150, zorder=5)
        ax.annotate(task, c, textcoords="offset points", xytext=(6, 4), fontsize=9)

    tri_pts = np.array(list(centroids.values()))
    ax.add_patch(mpatches.Polygon(
        tri_pts, fill=False, edgecolor="gray", linestyle="--", linewidth=1
    ))

    predicted = tri_pts.mean(axis=0)
    ax.scatter(*predicted, color="gray", s=150, marker="x", linewidths=2, zorder=4)
    ax.annotate("Predicted\n(⅓+⅓+⅓)", predicted,
                textcoords="offset points", xytext=(6, -16), fontsize=8, color="gray")

    mixed_pts = projected[MIXED_RATIO]
    mixed_c   = mixed_pts.mean(axis=0)
    ax.scatter(mixed_pts[:, 0], mixed_pts[:, 1], alpha=0.2, color="crimson", s=20, marker="^")
    ax.scatter(*mixed_c, color="crimson", s=150, zorder=5, marker="^")
    ax.annotate("Mixed", mixed_c, textcoords="offset points", xytext=(6, 4), fontsize=9)

    dist = np.linalg.norm(mixed_c - predicted)
    ax.set_xlabel("LDA component 1")
    ax.set_ylabel("LDA component 2")
    ax.set_title(
        f"Contrast vectors (LDA) — {_args.experiment}, layer {layer}\n"
        f"|mixed − predicted| = {dist:.4f}"
    )
    plt.tight_layout()
    fname = f"{OUT_PREFIX}_layer{layer}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Plot saved → {fname}")


def plot_summary(scores: dict[int, float]) -> None:
    layers = sorted(scores)
    vals   = [scores[l] for l in layers]
    best   = min(scores, key=scores.get)

    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, vals, marker="o", linewidth=2)
    ax.axvline(best, color="crimson", linestyle="--", linewidth=1.5,
               label=f"best: layer {best}  ({scores[best]:.4f})")
    for layer, val in zip(layers, vals):
        ax.annotate(f"{val:.3f}", (layer, val),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|mixed − predicted| in LDA space")
    ax.set_title(f"Task-vector superposition by layer — {_args.experiment}")
    ax.legend()
    plt.tight_layout()
    out = f"{OUT_PREFIX}_layer_sweep.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Summary plot saved → {out}")


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def run_sweep(
    pools,
    layers: list[int] = SWEEP_LAYERS,
    n_samples: int = N_VECTOR_SAMPLES,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        vectors   = collect_contrast_vectors(pools, n_samples=n_samples, layer=layer)
        projected = fit_lda(vectors)
        _plot_layer(projected, layer)
        scores[layer] = mixed_centroid_distance(projected)
        print(f"  |mixed − predicted| = {scores[layer]:.4f}")
    return scores


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Experiment: {_args.experiment}")
    print("Loading data...")
    pools  = load_pools()
    scores = run_sweep(pools)
    best   = min(scores, key=scores.get)

    print("\n=== Layer sweep results ===")
    for layer in sorted(scores):
        marker = "  <- best" if layer == best else ""
        print(f"  Layer {layer:2d}: {scores[layer]:.4f}{marker}")

    plot_summary(scores)
