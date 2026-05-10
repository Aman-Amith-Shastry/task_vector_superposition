"""
Task vector analysis: do ICL examples cause the residual stream to encode
task identity, and do mixed-task prompts superpose single-task vectors?

Single neutral LLM (no system prompt). The hook fires at the
add_generation_prompt=True position — after a held-out test question is
appended — so the activation reflects the full ICL context rather than
being dominated by the last ICL answer token.
"""

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset
from local_model import get_activation

SUBJECTS = ["Medicine", "Surgery", "Pharmacology"]

N_ICL_POOL       = 200
N_TEST_POOL      = 100
N_SHOTS          = 3
N_VECTOR_SAMPLES = 20
TASK_VECTOR_LAYER = 13
ICL_SEED         = 99
TEST_SEED        = 77


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_icl_pool() -> dict[str, list]:
    """200 train examples per subject for ICL prompts."""
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    pool = {}
    for subject in SUBJECTS:
        subset = ds.filter(lambda x, s=subject: x["subject_name"] == s)
        pool[subject] = list(subset.shuffle(seed=ICL_SEED).select(range(N_ICL_POOL)))
    return pool


def load_test_pool() -> list:
    """100 validation examples (mixed subjects) as held-out test questions."""
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    return list(ds.shuffle(seed=TEST_SEED).select(range(N_TEST_POOL)))


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def _format_question(example: dict) -> str:
    return (
        f"Question: {example['question']}\n"
        f"A) {example['opa']}\n"
        f"B) {example['opb']}\n"
        f"C) {example['opc']}\n"
        f"D) {example['opd']}\n"
        "Answer:"
    )


def build_single_task_messages(
    subject: str,
    pool: dict[str, list],
    test_pool: list,
    rng: random.Random,
) -> list[dict]:
    """N_SHOTS examples from one subject + held-out test question. No system prompt."""
    messages: list[dict] = []
    for ex in rng.sample(pool[subject], N_SHOTS):
        correct = ["A", "B", "C", "D"][ex["cop"]]
        messages.append({"role": "user",      "content": _format_question(ex)})
        messages.append({"role": "assistant",  "content": correct})
    messages.append({"role": "user", "content": _format_question(rng.choice(test_pool))})
    return messages


def build_mixed_task_messages(
    pool: dict[str, list],
    test_pool: list,
    rng: random.Random,
) -> list[dict]:
    """1 example per subject (order shuffled each call) + held-out test question. No system prompt."""
    subjects = SUBJECTS[:]
    rng.shuffle(subjects)
    messages: list[dict] = []
    for subject in subjects:
        ex = rng.choice(pool[subject])
        correct = ["A", "B", "C", "D"][ex["cop"]]
        messages.append({"role": "user",      "content": _format_question(ex)})
        messages.append({"role": "assistant",  "content": correct})
    messages.append({"role": "user", "content": _format_question(rng.choice(test_pool))})
    return messages


# --------------------------------------------------------------------------
# Task vector collection
# --------------------------------------------------------------------------

def collect_task_vectors(
    pool: dict[str, list],
    test_pool: list,
    n_samples: int = N_VECTOR_SAMPLES,
    layer: int = TASK_VECTOR_LAYER,
) -> dict[str, torch.Tensor]:
    """Collect n_samples individual task vectors per condition.

    Returns {condition: tensor[n_samples, hidden_size]}.
    hook fires at add_generation_prompt=True so the activation encodes
    'which task should I do' rather than 'I just finished a task answer'.
    """
    conditions: dict[str, torch.Tensor] = {}

    for subject in SUBJECTS:
        print(f"  {subject} ({n_samples} samples)...")
        vecs = [
            get_activation(
                build_single_task_messages(subject, pool, test_pool, random.Random(i)),
                layer=layer,
                add_generation_prompt=True,
            )
            for i in range(n_samples)
        ]
        conditions[subject] = torch.stack(vecs)

    print(f"  Mixed ({n_samples} samples)...")
    mixed = [
        get_activation(
            build_mixed_task_messages(pool, test_pool, random.Random(i)),
            layer=layer,
            add_generation_prompt=True,
        )
        for i in range(n_samples)
    ]
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

    Mixed is projected as a test point — it is never used to fit the LDA.
    With 3 classes, LDA yields exactly 2 discriminant components.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
# Main runner
# --------------------------------------------------------------------------

def run_task_vector_analysis(layer: int = TASK_VECTOR_LAYER) -> None:
    pool      = load_icl_pool()
    test_pool = load_test_pool()

    print(f"Collecting task vectors (layer {layer})...")
    vectors   = collect_task_vectors(pool, test_pool, layer=layer)
    projected = project_with_lda(vectors)
    plot_task_vectors(projected, layer=layer, fname="task_vectors_neutral.png")

    predicted = np.mean([projected[s].mean(axis=0) for s in SUBJECTS], axis=0)
    actual    = projected["Mixed"].mean(axis=0)
    dist      = np.linalg.norm(actual - predicted)
    print(f"  Centroid distance (mixed vs ⅓+⅓+⅓): {dist:.4f}")

    single_centroids = np.array([projected[s].mean(axis=0) for s in SUBJECTS])
    coeffs, _, _, _  = np.linalg.lstsq(single_centroids.T, actual, rcond=None)
    print("  Least-squares mixture weights: " +
          ", ".join(f"{s}={c:.3f}" for s, c in zip(SUBJECTS, coeffs)))


if __name__ == "__main__":
    run_task_vector_analysis()
