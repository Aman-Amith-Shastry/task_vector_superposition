"""
ICL ratio sweep at layer 14.

Tests the linear superposition hypothesis by varying the number of ICL
examples drawn from each specialty. If task vectors superpose linearly,
the contrast vector centroid for ratio (n_med, n_sur, n_phar) should fall at:

    (n_med·V_med + n_sur·V_sur + n_phar·V_phar) / (n_med + n_sur + n_phar)

where V_x is the centroid of the pure single-task condition at that layer.
The three pure conditions (3-0-0, 0-3-0, 0-0-3) are used to fit the LDA
and serve as the triangle vertices. All mixed ratios are projected into that
same space and compared against their predicted positions.

Usage: python sweep_icl_ratios.py
"""

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from analyze_specialization import (
    SUBJECTS,
    N_VECTOR_SAMPLES,
    load_icl_pool,
    load_test_pool,
    _format_question,
)
from local_model import get_activation

LAYER = 14

# (n_medicine, n_surgery, n_pharmacology) — must sum to 3 for comparable context length
RATIOS: list[tuple[int, int, int]] = [
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (2, 1, 0),
    (1, 2, 0),
    (0, 2, 1),
    (0, 1, 2),
    (1, 1, 1),
]

PURE_RATIOS = [(3, 0, 0), (0, 3, 0), (0, 0, 3)]


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def build_ratio_messages(
    pool: dict,
    test_pool: list,
    rng: random.Random,
    counts: tuple[int, int, int],
) -> list[dict]:
    """ICL prompt with the given per-subject example counts + a held-out test question.

    Examples are shuffled so no subject is systematically last, avoiding the
    recency bias that caused the original Pharmacology-dominance artifact.
    """
    all_examples = []
    for subject, n in zip(SUBJECTS, counts):
        if n > 0:
            all_examples.extend(rng.sample(pool[subject], n))
    rng.shuffle(all_examples)

    messages: list[dict] = []
    for ex in all_examples:
        correct = ["A", "B", "C", "D"][ex["cop"]]
        messages.append({"role": "user",      "content": _format_question(ex)})
        messages.append({"role": "assistant",  "content": correct})
    messages.append({"role": "user", "content": _format_question(rng.choice(test_pool))})
    return messages


# --------------------------------------------------------------------------
# Contrast vector
# --------------------------------------------------------------------------

def get_contrast_vector(messages: list[dict], layer: int) -> torch.Tensor:
    """activation(ICL + test_q) − activation(test_q only) at `layer`."""
    icl_act  = get_activation(messages,      layer=layer, add_generation_prompt=True)
    zero_act = get_activation(messages[-1:], layer=layer, add_generation_prompt=True)
    return icl_act - zero_act


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def collect_all_ratios(
    pool: dict,
    test_pool: list,
    ratios: list[tuple[int, int, int]] = RATIOS,
    n_samples: int = N_VECTOR_SAMPLES,
    layer: int = LAYER,
) -> dict[tuple[int, int, int], torch.Tensor]:
    """Collect contrast vectors for every ratio. Returns {ratio: [n_samples, hidden]}."""
    results: dict[tuple[int, int, int], torch.Tensor] = {}
    for counts in ratios:
        label = "-".join(map(str, counts))
        print(f"  Ratio {label} ({n_samples} samples)...")
        vecs = [
            get_contrast_vector(
                build_ratio_messages(pool, test_pool, random.Random(i), counts),
                layer,
            )
            for i in range(n_samples)
        ]
        results[counts] = torch.stack(vecs)
    return results


# --------------------------------------------------------------------------
# LDA: fit on pure conditions, project everything
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

    projected = {ratio: lda.transform(vecs.float().numpy()) for ratio, vecs in vectors.items()}
    return lda, projected


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_ratio_sweep(
    projected: dict[tuple[int, int, int], np.ndarray],
) -> None:
    palette = dict(zip(SUBJECTS, ["steelblue", "seagreen", "darkorange"]))

    _, ax = plt.subplots(figsize=(8, 7))

    # Pure single-task anchors and triangle
    pure_centroids: dict[tuple[int, int, int], np.ndarray] = {}
    for ratio, subject in zip(PURE_RATIOS, SUBJECTS):
        c = projected[ratio].mean(axis=0)
        pure_centroids[ratio] = c
        ax.scatter(*c, color=palette[subject], s=200, zorder=5)
        ax.annotate(
            f"{subject}\n({'-'.join(map(str, ratio))})", c,
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )

    tri_pts = np.array([pure_centroids[p] for p in PURE_RATIOS])
    ax.add_patch(mpatches.Polygon(
        tri_pts, fill=False, edgecolor="gray", linestyle="--", linewidth=1
    ))

    # Mixed ratios: actual vs predicted
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

    # Legend
    handles = [
        plt.scatter([], [], color="crimson", s=80, label="Actual centroid"),
        plt.scatter([], [], color="gray",    s=80, marker="x", label="Predicted (linear combo)"),
    ]
    ax.legend(handles=handles, fontsize=8)

    ax.set_xlabel("LDA component 1")
    ax.set_ylabel("LDA component 2")
    ax.set_title(
        f"ICL ratio sweep — contrast vectors, layer {LAYER}\n"
        "Dotted line = gap between predicted and actual centroid"
    )
    plt.tight_layout()
    plt.savefig("icl_ratio_sweep.png", dpi=150)
    plt.close()
    print("Plot saved → icl_ratio_sweep.png")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    pool      = load_icl_pool()
    test_pool = load_test_pool()

    print(f"Collecting contrast vectors at layer {LAYER}...")
    vectors         = collect_all_ratios(pool, test_pool)
    _, projected    = fit_and_project(vectors)
    pure_centroids  = {p: projected[p].mean(axis=0) for p in PURE_RATIOS}

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
