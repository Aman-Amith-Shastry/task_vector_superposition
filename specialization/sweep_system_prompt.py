"""
System prompt steering experiment at layer 14.

Tests whether a specialist system prompt predictably steers the mixed-task
contrast vector (1-1-1 ICL ratio) toward the corresponding specialist anchor
in LDA space.

Conditions:
  - No system prompt + 1-1-1 ICL   (baseline)
  - Physician system prompt + 1-1-1 ICL
  - Surgeon system prompt + 1-1-1 ICL
  - Pharmacologist system prompt + 1-1-1 ICL

The LDA axes are fit on the three pure single-task conditions (3-0-0, 0-3-0,
0-0-3) without any system prompt — the same anchors used in the ratio sweep.
All steered conditions are projected into that space as test points.

The contrast baseline is always just the test question with no system prompt
and no ICL examples, so each contrast vector captures the total shift caused
by both the ICL examples and the system prompt together.

Usage: python sweep_system_prompt.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data_specialization import (
    TASKS as SUBJECTS,
    N_VECTOR_SAMPLES,
    load_pools,
    _format_question,
    PURE_RATIOS,
)
from sweep_icl_ratios import collect_all_ratios, fit_and_project
from local_model import get_activation
from log_utils import log_run_header, log_icl_sample

LAYER = 14
MIXED_RATIO = (1, 1, 1)

SYSTEM_PROMPTS = {
    "Physician": (
        "You are a General Physician. Internal medicine is the branch of medicine concerned with "
        "the prevention, diagnosis, and non-surgical treatment of diseases affecting the internal "
        "organ systems. It encompasses conditions of the cardiovascular, respiratory, "
        "gastrointestinal, endocrine, renal, and immune systems, among others."
    ),
    "Surgeon": (
        "You are a Surgeon. Surgery is the branch of medicine that treats diseases, injuries, and "
        "deformities through manual and instrumental operations on the body. It covers the "
        "indications, techniques, and outcomes of operative intervention, as well as the "
        "pathological conditions that require surgical rather than medical management."
    ),
    "Pharmacologist": (
        "You are a Pharmacologist. Pharmacology is the science of how chemical substances interact "
        "with living systems, covering the mechanisms by which drugs exert their effects "
        "(pharmacodynamics), how the body absorbs, distributes, metabolises, and eliminates them "
        "(pharmacokinetics), and the relationships between drug structure, activity, toxicity, and "
        "therapeutic use."
    ),
}

# System prompt → which subject it should steer toward
PROMPT_SUBJECT = {
    "Physician":      "Medicine",
    "Surgeon":        "Surgery",
    "Pharmacologist": "Pharmacology",
}


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def build_steered_messages(
    pool: dict,
    test_pool: list,
    rng: random.Random,
    system_prompt: str | None,
    counts: tuple[int, int, int] = MIXED_RATIO,
) -> list[dict]:
    """ICL prompt with an optional system prompt + 1-1-1 examples + test question.

    Examples are shuffled to avoid recency bias. The system prompt (if given)
    is prepended as the first message so the model adopts the specialist role
    before seeing any ICL examples.
    """
    all_examples = []
    for subject, n in zip(SUBJECTS, counts):
        if n > 0:
            all_examples.extend(rng.sample(pool[subject], n))
    rng.shuffle(all_examples)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
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
    """activation(prompt + test_q) − activation(test_q only) at `layer`.

    The zero-shot baseline always uses just the bare test question with no
    system prompt, so the contrast captures the combined effect of the
    system prompt and the ICL examples.
    """
    icl_act  = get_activation(messages,       layer=layer, add_generation_prompt=True)
    zero_act = get_activation(messages[-1:],  layer=layer, add_generation_prompt=True)
    return icl_act - zero_act


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def collect_steered_vectors(
    pool: dict,
    test_pool: list,
    n_samples: int = N_VECTOR_SAMPLES,
    layer: int = LAYER,
) -> dict[str, torch.Tensor]:
    """Collect contrast vectors for the baseline and each steered condition."""
    conditions: dict[str, torch.Tensor] = {}
    log_run_header(f"sweep_system_prompt — collect_steered_vectors — layer {layer}")

    print("  Baseline (no system prompt, 1-1-1)...")
    baseline_vecs = []
    for i in range(n_samples):
        msgs = build_steered_messages(pool, test_pool, random.Random(i), None)
        log_icl_sample("Baseline", i, msgs, layer)
        baseline_vecs.append(get_contrast_vector(msgs, layer))
    conditions["Baseline"] = torch.stack(baseline_vecs)

    for name, prompt in SYSTEM_PROMPTS.items():
        print(f"  {name} system prompt + 1-1-1...")
        vecs = []
        for i in range(n_samples):
            msgs = build_steered_messages(pool, test_pool, random.Random(i), prompt)
            log_icl_sample(name, i, msgs, layer)
            vecs.append(get_contrast_vector(msgs, layer))
        conditions[name] = torch.stack(vecs)

    return conditions


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_steering(
    pure_projected: dict[tuple[int, int, int], np.ndarray],
    steered_projected: dict[str, np.ndarray],
) -> None:
    subject_color = dict(zip(SUBJECTS, ["steelblue", "seagreen", "darkorange"]))
    prompt_color  = {
        "Physician":      "steelblue",
        "Surgeon":        "seagreen",
        "Pharmacologist": "darkorange",
        "Baseline":       "crimson",
    }

    _, ax = plt.subplots(figsize=(8, 7))

    # Pure single-task anchors and triangle
    pure_centroids: dict[tuple[int, int, int], np.ndarray] = {}
    for ratio, subject in zip(PURE_RATIOS, SUBJECTS):
        c = pure_projected[ratio].mean(axis=0)
        pure_centroids[ratio] = c
        ax.scatter(*c, color=subject_color[subject], s=200, zorder=5)
        ax.annotate(subject, c, textcoords="offset points", xytext=(6, 4), fontsize=9)

    tri_pts = np.array([pure_centroids[p] for p in PURE_RATIOS])
    ax.add_patch(mpatches.Polygon(
        tri_pts, fill=False, edgecolor="gray", linestyle="--", linewidth=1
    ))

    # Baseline centroid
    baseline_c = steered_projected["Baseline"].mean(axis=0)
    ax.scatter(*baseline_c, color="crimson", s=120, zorder=5, marker="D")
    ax.annotate("Baseline\n(no sys prompt)", baseline_c,
                textcoords="offset points", xytext=(6, 4), fontsize=8, color="crimson")

    # Steered centroids + arrows from baseline
    for name in SYSTEM_PROMPTS:
        c     = steered_projected[name].mean(axis=0)
        color = prompt_color[name]
        ax.scatter(*c, color=color, s=120, zorder=5, marker="*", linewidths=1)
        ax.annotate(f"{name}", c,
                    textcoords="offset points", xytext=(6, -12), fontsize=8, color=color)
        ax.annotate(
            "", xy=c, xytext=baseline_c,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

    ax.set_xlabel("LDA component 1")
    ax.set_ylabel("LDA component 2")
    ax.set_title(
        f"System prompt steering — 1-1-1 ICL, layer {LAYER}\n"
        "Arrows show shift from unsteered baseline to each specialist prompt"
    )
    plt.tight_layout()
    plt.savefig("system_prompt_steering.png", dpi=150)
    plt.close()
    print("Plot saved → system_prompt_steering.png")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    pools              = load_pools()
    icl_pool, test_pool = pools

    # Fit LDA once on the pure single-task anchors and reuse it for all projections
    print("Collecting pure single-task anchors (no system prompt)...")
    pure_vectors   = collect_all_ratios(pools, ratios=list(PURE_RATIOS))
    lda, pure_proj = fit_and_project(pure_vectors)
    pure_centroids = {p: pure_proj[p].mean(axis=0) for p in PURE_RATIOS}

    # Project steered conditions into the same LDA space
    print("\nCollecting steered contrast vectors...")
    steered_vectors = collect_steered_vectors(icl_pool, test_pool)
    steered_proj = {
        name: lda.transform(vecs.float().numpy())
        for name, vecs in steered_vectors.items()
    }

    # Report displacement toward expected anchor
    print("\n=== Steering results ===")
    baseline_c = steered_proj["Baseline"].mean(axis=0)
    for name, subject in PROMPT_SUBJECT.items():
        anchor_ratio = PURE_RATIOS[SUBJECTS.index(subject)]
        anchor_c     = pure_centroids[anchor_ratio]
        steered_c    = steered_proj[name].mean(axis=0)

        dist_before = np.linalg.norm(baseline_c - anchor_c)
        dist_after  = np.linalg.norm(steered_c  - anchor_c)
        shift       = dist_before - dist_after   # positive = moved closer to anchor

        print(f"  {name:>14} → {subject}: "
              f"dist before={dist_before:.4f}, after={dist_after:.4f}, "
              f"shift={shift:+.4f} ({'closer' if shift > 0 else 'farther'})")

    plot_steering(pure_proj, steered_proj)
