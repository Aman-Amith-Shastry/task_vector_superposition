"""
Data loading and prompt construction for the semantic-domain (MMLU) experiment.

Three MMLU subjects with the same A/B/C/D output format but clearly distinct
semantic domains:

  History       high_school_world_history  (events, dates, political figures)
  Law           professional_law           (cases, statutes, procedural reasoning)
  ML            machine_learning           (technical CS/AI concepts)

Same output format as the specialisation experiment; only domain semantics vary.
This lets the layer stratification sweep compare format-driven vs. semantic-driven
task-vector separability across layers.
"""

import random
from datasets import load_dataset

TASKS            = ["History", "Law", "ML"]
_MMLU_SUBJECTS   = {
    "History": "high_school_world_history",
    "Law":     "professional_law",
    "ML":      "machine_learning",
}
N_ICL_POOL       = 150
N_VECTOR_SAMPLES = 20
SEED             = 42

PURE_RATIOS: list[tuple[int, int, int]] = [(3, 0, 0), (0, 3, 0), (0, 0, 3)]
RATIOS: list[tuple[int, int, int]] = [
    (3, 0, 0),
    (0, 3, 0),
    (0, 0, 3),
    (2, 1, 0),
    (1, 2, 0),
    (0, 2, 1),
    (0, 1, 2),
    (2, 0, 1),
    (1, 0, 2),
    (1, 1, 1),
]

_LETTERS = ["A", "B", "C", "D"]


def load_pools() -> dict[str, dict]:
    """Load ICL and test pools from MMLU test split for all three subjects."""
    pools: dict[str, dict] = {}
    for task in TASKS:
        subject = _MMLU_SUBJECTS[task]
        print(f"  Loading MMLU {subject}...")
        all_ex = list(
            load_dataset("cais/mmlu", subject, split="test").shuffle(seed=SEED)
        )
        # ICL and test draw from non-overlapping slices of the same shuffled list,
        # so the two pools are disjoint by construction — a test question is never
        # shown as an in-context example.
        n_icl   = min(N_ICL_POOL, len(all_ex) - N_VECTOR_SAMPLES)
        icl_ex  = all_ex[:n_icl]
        test_ex = all_ex[n_icl: n_icl + N_VECTOR_SAMPLES]
        pools[task] = {"icl": icl_ex, "test": test_ex}
    return pools


def _format_question(ex: dict) -> str:
    c = ex["choices"]
    return (
        f"Question: {ex['question']}\n"
        f"A) {c[0]}  B) {c[1]}  C) {c[2]}  D) {c[3]}\n"
        "Answer:"
    )


def build_messages(
    pools: dict,
    rng: random.Random,
    counts: tuple[int, int, int],
    sample_idx: int = 0,
    rotate_test: bool = False,
) -> list[dict]:
    """ICL messages for the given per-task counts + a held-out test question.

    All tasks share A/B/C/D format so test question rotation has no effect on
    output format — but setting rotate_test=True ensures all conditions draw
    test questions with the same domain distribution, removing any semantic
    bias from the test question itself.
    """
    is_pure   = counts in PURE_RATIOS
    if rotate_test:
        test_task = TASKS[sample_idx % len(TASKS)]
    else:
        test_task = TASKS[counts.index(max(counts))] if is_pure else TASKS[sample_idx % len(TASKS)]

    test_ex    = rng.choice(pools[test_task]["test"])
    test_input = _format_question(test_ex)

    all_examples: list[dict] = []
    for task, n in zip(TASKS, counts):
        all_examples.extend(rng.sample(pools[task]["icl"], n))
    rng.shuffle(all_examples)

    messages: list[dict] = []
    for ex in all_examples:
        messages.append({"role": "user",      "content": _format_question(ex)})
        messages.append({"role": "assistant", "content": _LETTERS[ex["answer"]]})
    messages.append({"role": "user", "content": test_input})
    return messages
