"""
Layer stratification sweep — tests whether format-distinct task identity
encodes in early layers and semantic-domain identity encodes in middle layers.

Hypothesis: LDA separability (Fisher criterion) for format-varying tasks peaks
at early layers (~3); separability for semantic-domain tasks peaks at middle
layers (~14).

Three task groups:
  Format-varying Arithmetic   Direct / MCQ / Verification           (format varies)
  Semantic Medical Domains    Medicine / Surgery / Pharmacology      (semantic, existing exp)
  Semantic MMLU Domains       Mathematics / History / Law            (semantic, new)

LDA separability is used as the metric: fit LDA on the pure-task vectors,
project to 2D, then compute tr(S_B) / tr(S_W) in that 2D space.
Task vectors are raw residual-stream activations at the assistant-header
token (no zero-shot subtraction). Higher = tasks more linearly decodable.

Usage:
  python layer_stratification/sweep_stratification.py
  python sweep_stratification.py        # if run from layer_stratification/
"""

import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_dir))  # repo root  (local_model, data_specialization)
sys.path.insert(0, _dir)                    # layer_stratification/

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from local_model import get_activations_all_layers

import data_arithmetic_formats
import data_semantic_domains
import data_specialization  # root-level module

SWEEP_LAYERS = [1, 3, 5, 8, 10, 14, 18, 20, 24, 26]
OUT_DIR      = os.path.join(_dir, "results")
N_SAMPLES    = data_arithmetic_formats.N_VECTOR_SAMPLES   # 20


# --------------------------------------------------------------------------
# Task vectors at all sweep layers — one forward pass per prompt
# --------------------------------------------------------------------------

def get_task_vectors(
    messages: list[dict],
    layers: list[int] = SWEEP_LAYERS,
) -> dict:
    """Raw residual-stream activation at the last token for every layer in
    `layers`. One forward pass; values are torch.Tensor."""
    return get_activations_all_layers(messages, layers)


# --------------------------------------------------------------------------
# Separability metrics
# --------------------------------------------------------------------------

def _fisher_criterion_2d(group_vecs: list[np.ndarray]) -> float:
    """tr(S_B) / tr(S_W) in whatever space group_vecs live in."""
    all_vecs     = np.vstack(group_vecs)
    overall_mean = all_vecs.mean(axis=0)
    S_B = float(sum(
        v.shape[0] * np.sum((v.mean(axis=0) - overall_mean) ** 2)
        for v in group_vecs
    ))
    S_W = float(sum(
        np.sum((v - v.mean(axis=0, keepdims=True)) ** 2)
        for v in group_vecs
    ))
    return S_B / (S_W + 1e-8)


def lda_separability(group_vecs: list[np.ndarray]) -> float:
    """Fit LDA on raw contrast vectors, return Fisher criterion in the 2D projection.

    Steps:
      1. Stack all pure-task vectors and fit LDA — finds the 2 directions in
         3072-D space that maximise between-class / within-class scatter.
      2. Project every vector down to those 2 directions.
      3. Compute Fisher criterion in the resulting 2-D space.

    The projection removes ~3070 noise dimensions first, so the criterion is
    well-estimated from 20 samples per class and is sensitive to subtle
    semantic differences that are linearly decodable but don't dominate
    the raw vector magnitude.
    """
    X = np.vstack(group_vecs)
    y = np.repeat(np.arange(len(group_vecs)), [v.shape[0] for v in group_vecs])

    lda     = LinearDiscriminantAnalysis(n_components=2)
    X_proj  = lda.fit_transform(X, y)

    proj_groups = [X_proj[y == c] for c in range(len(group_vecs))]
    return _fisher_criterion_2d(proj_groups)


# --------------------------------------------------------------------------
# Layer sweep for one task group
# --------------------------------------------------------------------------

def sweep_group(
    pools,
    build_fn,
    pure_ratios: list[tuple],
    label: str,
    layers: list[int] = SWEEP_LAYERS,
    n_samples: int = N_SAMPLES,
) -> dict[int, float]:
    # Accumulate task vectors for all layers in one pass per sample.
    # Layout: layer_vecs[layer][ratio_idx] = list of 1-D numpy vectors
    layer_vecs: dict[int, list[list]] = {l: [[] for _ in pure_ratios] for l in layers}

    for r_idx, ratio in enumerate(pure_ratios):
        print(f"    {label} ratio {'-'.join(map(str, ratio))} ({n_samples} samples)...",
              flush=True)
        for i in range(n_samples):
            msgs = build_fn(pools, random.Random(i), ratio, sample_idx=i)
            vecs = get_task_vectors(msgs, layers)
            for l, vec in vecs.items():
                layer_vecs[l][r_idx].append(vec.float().numpy())

    scores: dict[int, float] = {}
    for l in layers:
        group_vecs = [np.vstack(layer_vecs[l][r]) for r in range(len(pure_ratios))]
        scores[l]  = lda_separability(group_vecs)
        print(f"    Layer {l:2d}: LDA separability = {scores[l]:.4f}")
    return scores


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

_GROUP_META = [
    ("Format-varying Arithmetic", "steelblue",  "o"),
    ("Semantic Medical Domains",  "seagreen",    "s"),
    ("Semantic MMLU Domains",     "darkorange",  "^"),
]


def plot_stratification(
    all_scores: dict[str, dict[int, float]],
    out_path: str,
) -> None:
    _, ax = plt.subplots(figsize=(9, 5))
    layers = sorted(next(iter(all_scores.values())))

    for (name, scores), (_, color, marker) in zip(all_scores.items(), _GROUP_META):
        vals = [scores[l] for l in layers]
        ax.plot(layers, vals, marker=marker, color=color, linewidth=2, label=name)
        for l, v in zip(layers, vals):
            ax.annotate(f"{v:.3g}", (l, v),
                        textcoords="offset points", xytext=(0, 7),
                        fontsize=7, ha="center", color=color)

    ax.set_yscale("log")
    ax.set_xlabel("Layer")
    ax.set_ylabel("LDA separability  tr(S_B) / tr(S_W) in 2D LDA space  [log scale]")
    ax.set_title(
        "Layer stratification: where does task identity encode?\n"
        "Format-varying tasks (early) vs. semantic-domain tasks (middle)"
    )
    ax.legend(fontsize=9)
    ax.set_xticks(layers)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved → {out_path}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    task_groups = [
        (
            "Format-varying Arithmetic",
            data_arithmetic_formats.load_pools,
            data_arithmetic_formats.build_messages,
            data_arithmetic_formats.PURE_RATIOS,
        ),
        (
            "Semantic Medical Domains",
            data_specialization.load_pools,
            data_specialization.build_messages,
            data_specialization.PURE_RATIOS,
        ),
        (
            "Semantic MMLU Domains",
            data_semantic_domains.load_pools,
            data_semantic_domains.build_messages,
            data_semantic_domains.PURE_RATIOS,
        ),
    ]

    all_scores: dict[str, dict[int, float]] = {}
    for name, load_fn, build_fn, pure_ratios in task_groups:
        print(f"\n=== {name} ===")
        print("  Loading data...")
        pools = load_fn()
        print("  Sweeping layers...")
        all_scores[name] = sweep_group(pools, build_fn, pure_ratios, name)

    print("\n=== Summary ===")
    for name, scores in all_scores.items():
        best = max(scores, key=lambda l: scores[l])
        print(f"  {name}:")
        for l in sorted(scores):
            marker = "  <- best" if l == best else ""
            print(f"    Layer {l:2d}: {scores[l]:.4f}{marker}")

    plot_stratification(all_scores, os.path.join(OUT_DIR, "layer_stratification.png"))
