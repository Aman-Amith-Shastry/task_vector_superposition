"""
Data loading and prompt construction for the medical specialisation experiment.

Three MCQ specialties (Medicine / Surgery / Pharmacology) from MedMCQA.
All tasks share the same A/B/C/D output format, so test question format does
not bias contrast vectors and raw activations are equally valid.
"""

import random
from datasets import load_dataset

TASKS            = ["Medicine", "Surgery", "Pharmacology"]
N_ICL_POOL       = 200
N_TEST_POOL      = 100
N_VECTOR_SAMPLES = 20
ICL_SEED         = 99
TEST_SEED        = 77

# (n_medicine, n_surgery, n_pharmacology) — must sum to 3
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

PURE_RATIOS: list[tuple[int, int, int]] = [(3, 0, 0), (0, 3, 0), (0, 0, 3)]


def load_icl_pool() -> dict[str, list]:
    """200 train examples per specialty."""
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    pool: dict[str, list] = {}
    for task in TASKS:
        subset = ds.filter(lambda x, s=task: x["subject_name"] == s)
        pool[task] = list(subset.shuffle(seed=ICL_SEED).select(range(N_ICL_POOL)))
    return pool


def load_test_pool() -> list:
    """100 mixed-subject validation examples as held-out test questions."""
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    return list(ds.shuffle(seed=TEST_SEED).select(range(N_TEST_POOL)))


def load_pools() -> tuple[dict[str, list], list]:
    """Return (icl_pool, test_pool) — the standard pools for this experiment."""
    return load_icl_pool(), load_test_pool()


def _format_question(example: dict) -> str:
    return (
        f"Question: {example['question']}\n"
        f"A) {example['opa']}\n"
        f"B) {example['opb']}\n"
        f"C) {example['opc']}\n"
        f"D) {example['opd']}\n"
        "Answer:"
    )


def build_messages(
    pools: tuple[dict[str, list], list],
    rng: random.Random,
    counts: tuple[int, int, int],
    sample_idx: int = 0,
) -> list[dict]:
    """ICL messages for the given per-task example counts + a test question.

    pools = (icl_pool, test_pool) from load_pools().
    Examples are shuffled to avoid recency bias.
    sample_idx is unused here (all tasks share the same output format so test
    question rotation is unnecessary) but kept for interface parity with
    data_format_tasks.build_messages.
    """
    icl_pool, test_pool = pools
    all_examples: list[dict] = []
    for task, n in zip(TASKS, counts):
        if n > 0:
            all_examples.extend(rng.sample(icl_pool[task], n))
    rng.shuffle(all_examples)

    messages: list[dict] = []
    for ex in all_examples:
        correct = ["A", "B", "C", "D"][ex["cop"]]
        messages.append({"role": "user",      "content": _format_question(ex)})
        messages.append({"role": "assistant", "content": correct})
    messages.append({"role": "user", "content": _format_question(rng.choice(test_pool))})
    return messages
