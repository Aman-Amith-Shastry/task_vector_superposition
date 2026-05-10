"""
Layer sweep using contrast vectors (ICL activation − zero-shot activation).

Subtracting the zero-shot baseline removes the structural signal that comes
from the token position itself, leaving only the shift caused by the ICL
examples. Both passes use the same test question so the only variable is
whether the ICL context is present.

Usage: python sweep_layers.py
"""

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from analyze_specialization import (
    SUBJECTS,
    N_VECTOR_SAMPLES,
    load_icl_pool,
    load_test_pool,
    build_single_task_messages,
    build_mixed_task_messages,
    project_with_lda,
    plot_task_vectors,
)
from local_model import get_activation

SWEEP_LAYERS = [3, 8, 14, 20, 26]


# --------------------------------------------------------------------------
# Contrast vector
# --------------------------------------------------------------------------

def get_contrast_vector(messages: list[dict], layer: int) -> torch.Tensor:
    """activation(ICL + test_q) − activation(test_q only) at `layer`.

    `messages` ends with the test question as a user turn (messages[-1]).
    The zero-shot baseline uses that same message in isolation, so the
    subtraction isolates the ICL-induced shift and nothing else.
    """
    icl_act  = get_activation(messages,       layer=layer, add_generation_prompt=True)
    zero_act = get_activation(messages[-1:],  layer=layer, add_generation_prompt=True)
    return icl_act - zero_act


def collect_contrast_vectors(
    pool: dict,
    test_pool: list,
    n_samples: int = N_VECTOR_SAMPLES,
    layer: int = 13,
) -> dict[str, torch.Tensor]:
    """Collect contrast vectors for each condition at a single layer.

    Returns {condition: tensor[n_samples, hidden_size]}.
    The same random seed i is used for both build functions and both forward
    passes within get_contrast_vector, so ICL and zero-shot always share
    the same test question.
    """
    conditions: dict[str, torch.Tensor] = {}

    for subject in SUBJECTS:
        print(f"  {subject} ({n_samples} samples)...")
        vecs = [
            get_contrast_vector(
                build_single_task_messages(subject, pool, test_pool, random.Random(i)),
                layer,
            )
            for i in range(n_samples)
        ]
        conditions[subject] = torch.stack(vecs)

    print(f"  Mixed ({n_samples} samples)...")
    mixed = [
        get_contrast_vector(
            build_mixed_task_messages(pool, test_pool, random.Random(i)),
            layer,
        )
        for i in range(n_samples)
    ]
    conditions["Mixed"] = torch.stack(mixed)

    return conditions


# --------------------------------------------------------------------------
# Separation metric
# --------------------------------------------------------------------------

def mixed_centroid_distance(vectors: dict) -> float:
    """Mean distance from each single-task centroid to the mixed centroid in LDA space.

    Measures how centrally the mixed task vector sits among the three single-task
    centroids. Minimized when the mixed centroid is equidistant from all three —
    i.e., at the centroid of the triangle — which is the superposition prediction.
    Lower = better alignment with the convex combination hypothesis.
    """
    projected = project_with_lda(vectors)
    centroids = [projected[s].mean(axis=0) for s in SUBJECTS]
    mixed_c   = projected["Mixed"].mean(axis=0)
    return float(np.mean([np.linalg.norm(c - mixed_c) for c in centroids]))


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def run_sweep(
    pool: dict,
    test_pool: list,
    layers: list[int] = SWEEP_LAYERS,
    n_samples: int = N_VECTOR_SAMPLES,
) -> dict[int, float]:
    """For each layer: collect contrast vectors, save LDA plot, record separation."""
    scores: dict[int, float] = {}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        vectors      = collect_contrast_vectors(pool, test_pool, n_samples=n_samples, layer=layer)
        projected    = project_with_lda(vectors)
        plot_task_vectors(projected, layer=layer)
        scores[layer] = mixed_centroid_distance(vectors)
        print(f"  Mean dist (single → mixed) = {scores[layer]:.4f}")

    return scores


def plot_summary(scores: dict[int, float]) -> None:
    layers = sorted(scores)
    vals   = [scores[l] for l in layers]
    best   = max(scores.items(), key=lambda kv: kv[1])[0]

    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, vals, marker="o", linewidth=2)
    ax.axvline(best, color="crimson", linestyle="--", linewidth=1.5,
               label=f"best: layer {best}  ({scores[best]:.4f})")
    for layer, val in zip(layers, vals):
        ax.annotate(f"{val:.3f}", (layer, val),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean pairwise centroid distance (LDA)")
    ax.set_title("Task-vector discriminability by layer — contrast vectors")
    ax.legend()
    plt.tight_layout()
    plt.savefig("layer_sweep.png", dpi=150)
    plt.close()
    print("\nSummary plot saved → layer_sweep.png")


if __name__ == "__main__":
    pool      = load_icl_pool()
    test_pool = load_test_pool()

    scores = run_sweep(pool, test_pool)
    best   = min(scores.items(), key=lambda kv: kv[1])[0]

    print("\n=== Layer sweep results ===")
    for layer in sorted(scores):
        marker = "  <- best" if layer == best else ""
        print(f"  Layer {layer:2d}: {scores[layer]:.4f}{marker}")

    plot_summary(scores)
