"""
Behavioral validation: does the ICL condition predict accuracy?

For each ICL condition (Medicine-only, Surgery-only, Pharmacology-only, zero-shot),
evaluate accuracy on 20 held-out validation questions from each specialty.

If the representational finding (task vectors cluster by specialty) has behavioral
consequences, accuracy should be highest when the ICL condition matches the question
specialty.

Prediction uses log-probability scoring over A/B/C/D rather than greedy generation,
giving a deterministic and comparable signal across conditions.

No contamination: ICL examples are drawn from the train split; accuracy test
questions are drawn from a separate specialty-filtered slice of the validation
split using a different random seed from any prior experiment.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from data_specialization import (
    TASKS as SUBJECTS,
    N_ICL_POOL,
    ICL_SEED,
    load_icl_pool,
    _format_question,
)
from local_model import predict_mcq
from log_utils import log_run_header, log_icl_sample

N_TEST_QUESTIONS = 100  # held-out questions per specialty per condition
N_ICL_SHOTS      = 3
ACCURACY_SEED    = 42   # distinct from ICL_SEED=99 and test_pool seed=77


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_accuracy_test_pool() -> dict[str, list]:
    """20 validation examples per specialty for accuracy evaluation.

    Specialty-filtered so accuracy can be measured per domain. Uses ACCURACY_SEED
    (42) to avoid overlap with load_test_pool (seed=77). Cross-split contamination
    is impossible: all ICL examples are from the train split.
    """
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    test_pool: dict[str, list] = {}
    for subject in SUBJECTS:
        subset = ds.filter(lambda x, s=subject: x["subject_name"] == s)
        test_pool[subject] = list(
            subset.shuffle(seed=ACCURACY_SEED).select(range(N_TEST_QUESTIONS))
        )
    return test_pool


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def build_icl_messages(
    icl_subject: str | None,
    pool: dict[str, list],
    test_question: dict,
    rng: random.Random,
) -> list[dict]:
    """Prompt with N_ICL_SHOTS examples from icl_subject (None = zero-shot) + test question."""
    messages: list[dict] = []
    if icl_subject is not None:
        for ex in rng.sample(pool[icl_subject], N_ICL_SHOTS):
            correct = ["A", "B", "C", "D"][ex["cop"]]
            messages.append({"role": "user",      "content": _format_question(ex)})
            messages.append({"role": "assistant", "content": correct})
    messages.append({"role": "user", "content": _format_question(test_question)})
    return messages




# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_condition(
    icl_subject: str | None,
    pool: dict[str, list],
    test_pool: dict[str, list],
) -> dict[str, float]:
    """Accuracy on each specialty's held-out questions under one ICL condition.

    ICL examples for each test question are drawn with seed=question_index so
    the prompt is reproducible but varies across questions to avoid over-fitting
    to a single set of ICL examples.
    """
    label = icl_subject if icl_subject is not None else "Zero-shot"
    log_run_header(f"compare_accuracies — ICL condition={label}")

    accuracies: dict[str, float] = {}
    for test_subject in SUBJECTS:
        correct = 0
        questions = test_pool[test_subject]
        for i, q in enumerate(questions):
            msgs = build_icl_messages(icl_subject, pool, q, random.Random(i))
            log_icl_sample(f"{label}→{test_subject}", i, msgs)
            pred, _ = predict_mcq(msgs)
            gold = ["A", "B", "C", "D"][q["cop"]]
            if pred == gold:
                correct += 1
        acc = correct / len(questions)
        accuracies[test_subject] = acc
        match_marker = " <-- match" if icl_subject == test_subject else ""
        print(f"    {test_subject}: {correct}/{len(questions)} = {acc:.2f}{match_marker}")

    return accuracies


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_accuracy_bars(
    results: dict[str, dict[str, float]],
    condition_labels: list[str],
) -> None:
    n_conditions = len(condition_labels)
    x            = np.arange(len(SUBJECTS))
    width        = 0.75 / n_conditions

    condition_colors = {
        "Zero-shot":    "lightgray",
        "Medicine":     "steelblue",
        "Surgery":      "seagreen",
        "Pharmacology": "darkorange",
    }

    _, ax = plt.subplots(figsize=(11, 6))

    for i, label in enumerate(condition_labels):
        heights = [results[label][s] for s in SUBJECTS]
        offset  = (i - n_conditions / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, heights, width,
            label=f"{label} ICL" if label != "Zero-shot" else label,
            color=condition_colors[label],
            edgecolor="white", linewidth=0.6,
        )
        for bar, h, s in zip(bars, heights, SUBJECTS):
            # Bold annotation when ICL condition matches question specialty
            weight = "bold" if label == s else "normal"
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.012,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight=weight,
            )

    ax.axhline(0.25, color="black", linestyle=":", linewidth=1.2, label="Chance (0.25)")

    # Box the diagonal (matching) bars to make them visually salient
    for i, label in enumerate(condition_labels):
        if label in SUBJECTS:
            j = SUBJECTS.index(label)
            offset = (i - n_conditions / 2 + 0.5) * width
            bar_x  = x[j] + offset - width / 2
            bar_h  = results[label][label]
            ax.add_patch(plt.Rectangle(
                (bar_x, 0), width, bar_h,
                fill=False, edgecolor="black", linewidth=1.5, zorder=5,
            ))

    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f"MCQ accuracy by specialty under each ICL condition  "
        f"({N_ICL_SHOTS}-shot, {N_TEST_QUESTIONS} questions per specialty)\n"
        "Bold/boxed bars = ICL condition matches question specialty"
    )
    ax.legend(title="ICL condition", fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=150)
    plt.close()
    print("Plot saved → accuracy_comparison.png")


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def print_summary(
    results: dict[str, dict[str, float]],
    condition_labels: list[str],
) -> None:
    print("\n=== Accuracy summary (rows = ICL condition, cols = question specialty) ===")
    header = f"{'':>15}" + "".join(f"{s:>16}" for s in SUBJECTS)
    print(header)
    for label in condition_labels:
        row = f"{label:>15}"
        for s in SUBJECTS:
            acc = results[label][s]
            marker = " *" if label == s else "  "
            row += f"{acc:.2f}{marker}{'':>10}"
        print(row)
    print("  * = ICL condition matches question specialty")

    print("\n=== Match vs. mismatch lift ===")
    for s in SUBJECTS:
        match_acc    = results[s][s]
        mismatch_acc = np.mean([results[other][s] for other in SUBJECTS if other != s])
        zeroshot_acc = results["Zero-shot"][s]
        print(f"  {s}: match={match_acc:.2f}  mismatch_avg={mismatch_acc:.2f}  "
              f"zero-shot={zeroshot_acc:.2f}  lift={match_acc - mismatch_acc:+.2f}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    pool      = load_icl_pool()
    test_pool = load_accuracy_test_pool()

    conditions      = [None] + list(SUBJECTS)
    condition_labels = ["Zero-shot"] + list(SUBJECTS)

    results: dict[str, dict[str, float]] = {}
    for icl_subject, label in zip(conditions, condition_labels):
        print(f"\nCondition: {label}")
        results[label] = evaluate_condition(icl_subject, pool, test_pool)

    print_summary(results, condition_labels)
    plot_accuracy_bars(results, condition_labels)
