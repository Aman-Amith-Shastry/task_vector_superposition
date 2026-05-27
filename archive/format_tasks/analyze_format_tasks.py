"""
Representational superposition across format-distinct tasks.

Three tasks with the same input style (clinical/biomedical text) but
different output formats:
  - MCQ         (MedMCQA):             question + A/B/C/D options → letter
  - Yes/No      (PubMedQA):            question + Yes/No/Maybe → word
  - Diagnosis   (Symptom2Disease):     symptoms + numbered options → index

Unlike the medical-specialty experiments (Medicine/Surgery/Pharmacology),
these tasks have maximally separated task vectors because their ICL examples
look nothing alike. The hypothesis is that mixed ICL (1 example per task)
produces a contrast vector that sits inside the triangle spanned by the three
pure-task contrast vectors — representational superposition.

Contrast vectors (activation(ICL + test_q) − activation(test_q alone)) are
used throughout so the test question's content domain does not bias the
representation toward any particular task.

Datasets are loaded directly from HuggingFace — no manual downloads needed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from local_model import get_activation
from log_utils import log_run_header, log_icl_sample

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TASKS            = ["MCQ", "PubMedQA", "Symptom2Disease"]
LAYER            = 3
N_SHOTS          = 3
N_VECTOR_SAMPLES = 20
SEED             = 42


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_all_pools() -> dict[str, dict]:
    """Load ICL and test pools for all three tasks from HuggingFace.

    MedMCQA:         train/validation splits (separate by design)
    PubMedQA:        single train split — divided manually at index 700
    Symptom2Disease: train/test splits
    """
    print("  Loading MedMCQA (Medicine)...")
    mcq_train = load_dataset("openlifescienceai/medmcqa", split="train")
    mcq_val   = load_dataset("openlifescienceai/medmcqa", split="validation")
    mcq_icl   = list(mcq_train.filter(lambda x: x["subject_name"] == "Medicine")
                     .shuffle(seed=SEED).select(range(200)))
    mcq_test  = list(mcq_val.filter(lambda x: x["subject_name"] == "Medicine")
                     .shuffle(seed=SEED).select(range(N_VECTOR_SAMPLES)))

    print("  Loading PubMedQA (pqa_labeled)...")
    pubmed_all  = list(load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
                       .shuffle(seed=SEED))
    pubmed_icl  = pubmed_all[:700]
    pubmed_test = pubmed_all[700:700 + N_VECTOR_SAMPLES]

    print("  Loading Symptom2Disease...")
    s2d_icl  = list(load_dataset("gretelai/symptom_to_diagnosis", split="train")
                    .shuffle(seed=SEED))
    s2d_test = list(load_dataset("gretelai/symptom_to_diagnosis", split="test")
                    .shuffle(seed=SEED).select(range(N_VECTOR_SAMPLES)))

    return {
        "MCQ":             {"icl": mcq_icl,    "test": mcq_test},
        "PubMedQA":        {"icl": pubmed_icl,  "test": pubmed_test},
        "Symptom2Disease": {"icl": s2d_icl,     "test": s2d_test},
    }


# Populated once at startup by init_disease_list().
_DISEASE_LIST:    list[str] = []
_DISEASE_OPTIONS: str       = ""


def init_disease_list(pools: dict) -> None:
    """Build sorted disease list and numbered options string for S2D formatting."""
    global _DISEASE_LIST, _DISEASE_OPTIONS
    all_ex = pools["Symptom2Disease"]["icl"] + pools["Symptom2Disease"]["test"]
    _DISEASE_LIST    = sorted({ex["output_text"].lower() for ex in all_ex})
    _DISEASE_OPTIONS = "  ".join(
        f"{i + 1}) {d}" for i, d in enumerate(_DISEASE_LIST)
    )


# --------------------------------------------------------------------------
# Prompt formatting
# --------------------------------------------------------------------------

def format_icl_example(task: str, ex: dict) -> tuple[str, str]:
    """Return (input_text, output_text) for a single ICL example."""
    if task == "MCQ":
        inp = (
            f"Question: {ex['question']}\n"
            f"A) {ex['opa']}  B) {ex['opb']}  C) {ex['opc']}  D) {ex['opd']}\n"
            "Answer:"
        )
        out = ["A", "B", "C", "D"][ex["cop"]]

    elif task == "PubMedQA":
        inp = (
            f"Question: {ex['question']}\n"
            "Options: Yes / No / Maybe\n"
            "Answer:"
        )
        out = ex["final_decision"]

    else:   # Symptom2Disease — numbered options, answer is the 1-based index
        idx = _DISEASE_LIST.index(ex["output_text"].lower()) + 1
        inp = (
            f"Question: {ex['input_text']}\n"
            f"{_DISEASE_OPTIONS}\n"
            "Answer:"
        )
        out = str(idx)

    return inp, out


def format_test_input(task: str, ex: dict) -> str:
    """Format test question with no explicit output format cues.

    Options are stripped so the model must infer the expected output format
    from the ICL examples alone. This is the input to both the ICL pass and
    the zero-shot baseline pass in get_contrast_vector.
    """
    if task == "MCQ":
        return f"Question: {ex['question']}\nAnswer:"
    elif task == "PubMedQA":
        return f"Question: {ex['question']}\nAnswer:"
    else:
        return f"Question: {ex['input_text']}\nAnswer:"


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def build_pure_messages(
    task: str,
    pools: dict,
    test_input: str,
    rng: random.Random,
) -> list[dict]:
    """N_SHOTS ICL examples from one task + pre-formatted test question."""
    messages: list[dict] = []
    for ex in rng.sample(pools[task]["icl"], N_SHOTS):
        inp, out = format_icl_example(task, ex)
        messages.append({"role": "user",      "content": inp})
        messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": test_input})
    return messages


def build_mixed_messages(
    pools: dict,
    test_input: str,
    rng: random.Random,
) -> list[dict]:
    """1 ICL example per task (order shuffled) + pre-formatted test question."""
    task_order = TASKS[:]
    rng.shuffle(task_order)
    messages: list[dict] = []
    for task in task_order:
        ex = rng.choice(pools[task]["icl"])
        inp, out = format_icl_example(task, ex)
        messages.append({"role": "user",      "content": inp})
        messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": test_input})
    return messages


# --------------------------------------------------------------------------
# Contrast vectors
# --------------------------------------------------------------------------

def get_contrast_vector(messages: list[dict], layer: int) -> torch.Tensor:
    """activation(ICL + test_q) − activation(test_q alone) at `layer`.

    The subtraction removes whatever the test question contributes on its own
    — including content-domain bias — leaving only the shift caused by the
    ICL examples. The test question is always messages[-1] (the final user
    turn), so the baseline is always consistent with the full prompt.
    """
    icl_act  = get_activation(messages,       layer=layer, add_generation_prompt=True)
    zero_act = get_activation(messages[-1:],  layer=layer, add_generation_prompt=True)
    return icl_act - zero_act


def collect_contrast_vectors(
    pools: dict,
    n_samples: int = N_VECTOR_SAMPLES,
    layer: int = LAYER,
) -> dict[str, torch.Tensor]:
    """Collect contrast vectors for pure single-task and mixed conditions.

    Each pure condition uses that task's own test pool so the test question
    is contextually appropriate. For the mixed condition the test question
    is drawn from the MCQ pool — the content cancels out via subtraction so
    the choice is arbitrary.
    """
    log_run_header(f"behavioral_superposition — collect_contrast_vectors — layer {layer}")
    vectors: dict[str, torch.Tensor] = {}

    for task in TASKS:
        print(f"  {task} ({n_samples} samples)...")
        vecs = []
        for i in range(n_samples):
            rng        = random.Random(i)
            test_ex    = rng.choice(pools[task]["test"])
            test_input = format_test_input(task, test_ex)
            msgs       = build_pure_messages(task, pools, test_input, rng)
            log_icl_sample(task, i, msgs, layer)
            vecs.append(get_contrast_vector(msgs, layer))
        vectors[task] = torch.stack(vecs)

    print(f"  Mixed ({n_samples} samples)...")
    mixed = []
    for i in range(n_samples):
        rng        = random.Random(i)
        # Rotate test question source across all three tasks so no single task's
        # format biases the attention patterns in the mixed ICL context.
        test_task  = TASKS[i % len(TASKS)]
        test_ex    = rng.choice(pools[test_task]["test"])
        test_input = format_test_input(test_task, test_ex)
        msgs       = build_mixed_messages(pools, test_input, rng)
        log_icl_sample("Mixed", i, msgs, layer)
        mixed.append(get_contrast_vector(msgs, layer))
    vectors["Mixed"] = torch.stack(mixed)

    return vectors


# --------------------------------------------------------------------------
# LDA
# --------------------------------------------------------------------------

def fit_and_project(
    vectors: dict[str, torch.Tensor],
    n_samples: int = N_VECTOR_SAMPLES,
) -> tuple[LinearDiscriminantAnalysis, dict[str, np.ndarray]]:
    """Fit LDA on the three pure-task conditions, project all vectors."""
    X = np.vstack([vectors[t].float().numpy() for t in TASKS])
    y = np.repeat(np.arange(len(TASKS)), n_samples)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)

    projected = {k: lda.transform(v.float().numpy()) for k, v in vectors.items()}
    return lda, projected


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

TASK_COLORS = {
    "MCQ":             "steelblue",
    "PubMedQA":        "seagreen",
    "Symptom2Disease": "darkorange",
    "Mixed":           "crimson",
}


def plot_contrast_lda(projected: dict[str, np.ndarray], layer: int = LAYER) -> None:
    _, ax = plt.subplots(figsize=(7, 6))

    centroids: dict[str, np.ndarray] = {}
    for task in TASKS:
        pts = projected[task]
        c   = pts.mean(axis=0)
        centroids[task] = c
        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.2, color=TASK_COLORS[task], s=20)
        ax.scatter(*c, color=TASK_COLORS[task], s=150, zorder=5)
        ax.annotate(task, c, textcoords="offset points", xytext=(6, 4), fontsize=9)

    tri_pts = np.array([centroids[t] for t in TASKS])
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
               color=TASK_COLORS["Mixed"], s=20, marker="^")
    ax.scatter(*mixed_c, color=TASK_COLORS["Mixed"], s=150, zorder=5, marker="^")
    ax.annotate("Mixed", mixed_c, textcoords="offset points", xytext=(6, 4), fontsize=9)

    dist = np.linalg.norm(mixed_c - predicted)
    ax.set_xlabel("LDA component 1")
    ax.set_ylabel("LDA component 2")
    ax.set_title(
        f"Contrast vectors (LDA) — MCQ / PubMedQA / Symptom2Disease, layer {layer}\n"
        f"|mixed − predicted| = {dist:.4f}"
    )
    plt.tight_layout()
    plt.savefig("task_superposition_lda.png", dpi=150)
    plt.close()
    print("Plot saved → task_superposition_lda.png")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading datasets from HuggingFace...")
    pools = load_all_pools()
    init_disease_list(pools)
    print(f"  Disease list ({len(_DISEASE_LIST)} classes): {', '.join(_DISEASE_LIST)}")

    print(f"\nCollecting contrast vectors (layer {LAYER})...")
    vectors      = collect_contrast_vectors(pools)
    _, projected = fit_and_project(vectors)
    plot_contrast_lda(projected)

    centroids = {t: projected[t].mean(axis=0) for t in TASKS}
    predicted = np.mean(list(centroids.values()), axis=0)
    mixed_c   = projected["Mixed"].mean(axis=0)

    print("\n=== Superposition summary ===")
    print(f"  |mixed − predicted (⅓+⅓+⅓)|: {np.linalg.norm(mixed_c - predicted):.4f}")
    for t in TASKS:
        print(f"  |{t} − mixed|: {np.linalg.norm(centroids[t] - mixed_c):.4f}")

    # Least-squares mixture weights: how much of each pure task is in the mixed vector?
    pure_mat = np.array([centroids[t] for t in TASKS])
    coeffs, _, _, _ = np.linalg.lstsq(pure_mat.T, mixed_c, rcond=None)
    print("  Least-squares weights: " +
          ", ".join(f"{t}={c:.3f}" for t, c in zip(TASKS, coeffs)))
