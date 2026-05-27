"""
Task vector analysis: do ICL examples cause the residual stream to encode
task identity, and do mixed-task prompts superpose single-task vectors?

Single neutral LLM (no system prompt). The hook fires at the
add_generation_prompt=True position — after a held-out test question is
appended — so the activation reflects the full ICL context rather than
being dominated by the last ICL answer token.

Uses raw activations (not contrast vectors) — valid here because all three
tasks share the same A/B/C/D output format, so the test question format
does not bias activations toward any particular task.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from local_model import get_activation
from log_utils import log_run_header, log_icl_sample
from data_specialization import (
    TASKS, N_ICL_POOL, N_VECTOR_SAMPLES, ICL_SEED, PURE_RATIOS,
    load_icl_pool, load_test_pool, load_pools, _format_question, build_messages,
)

# Alias for backward compatibility with compare_accuracies and sweep_system_prompt
SUBJECTS = TASKS

TASK_VECTOR_LAYER = 13


# --------------------------------------------------------------------------
# Task vector collection (raw activations)
# --------------------------------------------------------------------------

def collect_task_vectors(
    pools: tuple[dict[str, list], list],
    n_samples: int = N_VECTOR_SAMPLES,
    layer: int = TASK_VECTOR_LAYER,
) -> dict[str, torch.Tensor]:
    """Collect n_samples raw activations per condition (pure tasks + mixed).

    Returns {condition_name: tensor[n_samples, hidden_size]}.
    Uses raw activations rather than contrast vectors — the shared MCQ format
    means the test question does not bias activations across tasks.
    """
    conditions: dict[str, torch.Tensor] = {}
    log_run_header(f"analyze_specializations — collect_task_vectors — layer {layer}")

    for ratio, subject in zip(PURE_RATIOS, SUBJECTS):
        print(f"  {subject} ({n_samples} samples)...")
        vecs = []
        for i in range(n_samples):
            msgs = build_messages(pools, random.Random(i), ratio)
            log_icl_sample(subject, i, msgs, layer)
            vecs.append(get_activation(msgs, layer=layer, add_generation_prompt=True))
        conditions[subject] = torch.stack(vecs)

    print(f"  Mixed ({n_samples} samples)...")
    mixed = []
    for i in range(n_samples):
        msgs = build_messages(pools, random.Random(i), (1, 1, 1))
        log_icl_sample("Mixed", i, msgs, layer)
        mixed.append(get_activation(msgs, layer=layer, add_generation_prompt=True))
    conditions["Mixed"] = torch.stack(mixed)

    return conditions


# --------------------------------------------------------------------------
# LDA projection and plotting
# --------------------------------------------------------------------------

def project_with_lda(
    vectors: dict[str, torch.Tensor],
    n_vector_samples: int = N_VECTOR_SAMPLES,
) -> dict[str, np.ndarray]:
    """Fit LDA on the three single-task classes, project all conditions to 2D.

    Mixed is projected as a test point — never used to fit the LDA.
    """
    X = np.vstack([vectors[s].float().numpy() for s in SUBJECTS])
    y = np.repeat(np.arange(len(SUBJECTS)), n_vector_samples)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)

    return {key: lda.transform(vecs.float().numpy()) for key, vecs in vectors.items()}


def plot_task_vectors(
    projected: dict[str, np.ndarray],
    layer: int = TASK_VECTOR_LAYER,
    fname: str | None = None,
) -> None:
    palette = dict(zip(SUBJECTS, ["steelblue", "seagreen", "darkorange"]))
    palette["Mixed"] = "crimson"

    _, ax = plt.subplots(figsize=(7, 6))

    centroids: dict[str, np.ndarray] = {}
    for subject in SUBJECTS:
        pts = projected[subject]
        c   = pts.mean(axis=0)
        centroids[subject] = c
        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.2, color=palette[subject], s=20)
        ax.scatter(*c, color=palette[subject], s=150, zorder=5)
        ax.annotate(subject, c, textcoords="offset points", xytext=(6, 4), fontsize=9)

    tri_pts = np.array([centroids[s] for s in SUBJECTS])
    ax.add_patch(mpatches.Polygon(
        tri_pts, fill=False, edgecolor="gray", linestyle="--", linewidth=1
    ))

    predicted = tri_pts.mean(axis=0)
    ax.scatter(*predicted, color="gray", s=150, marker="x", linewidths=2, zorder=4)
    ax.annotate("Predicted\n(⅓+⅓+⅓)", predicted,
                textcoords="offset points", xytext=(6, -16), fontsize=8, color="gray")

    mixed_pts = projected["Mixed"]
    mixed_c   = mixed_pts.mean(axis=0)
    ax.scatter(mixed_pts[:, 0], mixed_pts[:, 1], alpha=0.2,
               color=palette["Mixed"], s=20, marker="^")
    ax.scatter(*mixed_c, color=palette["Mixed"], s=150, zorder=5, marker="^")
    ax.annotate("Mixed", mixed_c, textcoords="offset points", xytext=(6, 4), fontsize=9)

    dist = np.linalg.norm(mixed_c - predicted)
    ax.set_xlabel("LDA component 1")
    ax.set_ylabel("LDA component 2")
    ax.set_title(f"Task vectors (LDA) — neutral LLM, layer {layer}\n"
                 f"|mixed − predicted| = {dist:.4f}")
    plt.tight_layout()
    out = fname or f"task_vectors_layer{layer}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved → {out}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_task_vector_analysis(layer: int = TASK_VECTOR_LAYER) -> None:
    pools = load_pools()

    print(f"Collecting task vectors (layer {layer})...")
    vectors   = collect_task_vectors(pools, layer=layer)
    projected = project_with_lda(vectors)
    plot_task_vectors(projected, layer=layer, fname="task_vectors_neutral.png")

    predicted        = np.mean([projected[s].mean(axis=0) for s in SUBJECTS], axis=0)
    actual           = projected["Mixed"].mean(axis=0)
    dist             = np.linalg.norm(actual - predicted)
    single_centroids = np.array([projected[s].mean(axis=0) for s in SUBJECTS])
    coeffs, _, _, _  = np.linalg.lstsq(single_centroids.T, actual, rcond=None)

    print(f"  Centroid distance (mixed vs ⅓+⅓+⅓): {dist:.4f}")
    print("  Least-squares mixture weights: " +
          ", ".join(f"{s}={c:.3f}" for s, c in zip(SUBJECTS, coeffs)))


if __name__ == "__main__":
    run_task_vector_analysis()
