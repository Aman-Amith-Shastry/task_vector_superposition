"""
Data loading and prompt construction for the format-distinct tasks experiment.

Three tasks with the same input style but different output formats:
  MCQ            (MedMCQA Medicine):      question + A/B/C/D options → letter
  PubMedQA       (pqa_labeled):           question + Yes/No/Maybe    → word
  Symptom2Disease (gretelai):             symptoms + numbered list   → index

For non-pure ICL ratios the test question type is rotated across tasks using
sample_idx to average out the format-interaction effect that arises at the
assistant-header hook position (causal attention from last token into ICL).
"""

import random
from datasets import load_dataset

TASKS            = ["MCQ", "PubMedQA", "Symptom2Disease"]
N_VECTOR_SAMPLES = 20
SEED             = 42

# (n_mcq, n_pubmed, n_s2d) — must sum to 3
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

PURE_RATIOS: list[tuple[int, int, int]] = [(3, 0, 0), (0, 3, 0), (0, 0, 3)]

_DISEASE_LIST:    list[str] = []
_DISEASE_OPTIONS: str       = ""


def load_pools() -> dict[str, dict]:
    """Load ICL and test pools for all three tasks from HuggingFace.

    MedMCQA:         train/validation splits (Medicine subject only)
    PubMedQA:        single train split — divided manually at index 700
    Symptom2Disease: train/test splits

    Calls init_disease_list before returning so _DISEASE_LIST is populated.
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

    pools = {
        "MCQ":             {"icl": mcq_icl,    "test": mcq_test},
        "PubMedQA":        {"icl": pubmed_icl,  "test": pubmed_test},
        "Symptom2Disease": {"icl": s2d_icl,     "test": s2d_test},
    }
    init_disease_list(pools)
    return pools


def init_disease_list(pools: dict) -> None:
    """Build sorted disease list and numbered options string for S2D formatting."""
    global _DISEASE_LIST, _DISEASE_OPTIONS
    all_ex = pools["Symptom2Disease"]["icl"] + pools["Symptom2Disease"]["test"]
    _DISEASE_LIST    = sorted({ex["output_text"].lower() for ex in all_ex})
    _DISEASE_OPTIONS = "  ".join(
        f"{i + 1}) {d}" for i, d in enumerate(_DISEASE_LIST)
    )


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
    else:
        idx = _DISEASE_LIST.index(ex["output_text"].lower()) + 1
        inp = (
            f"Question: {ex['input_text']}\n"
            f"{_DISEASE_OPTIONS}\n"
            "Answer:"
        )
        out = str(idx)
    return inp, out


def format_test_input(task: str, ex: dict) -> str:
    """Format test question with no options — format cue comes from ICL examples only."""
    if task in ("MCQ", "PubMedQA"):
        return f"Question: {ex['question']}\nAnswer:"
    else:
        return f"Question: {ex['input_text']}\nAnswer:"


def build_messages(
    pools: dict,
    rng: random.Random,
    counts: tuple[int, int, int],
    sample_idx: int = 0,
) -> list[dict]:
    """ICL messages for the given per-task example counts + a test question.

    For pure conditions the test question is drawn from that task's own pool so
    it matches the expected format. For mixed conditions the test question type
    rotates with sample_idx to average out the format-interaction effect at the
    assistant-header hook position.
    """
    is_pure = counts in PURE_RATIOS
    if is_pure:
        pure_task  = TASKS[counts.index(max(counts))]
        test_ex    = rng.choice(pools[pure_task]["test"])
        test_input = format_test_input(pure_task, test_ex)
    else:
        test_task  = TASKS[sample_idx % len(TASKS)]
        test_ex    = rng.choice(pools[test_task]["test"])
        test_input = format_test_input(test_task, test_ex)

    all_examples: list[tuple[str, str]] = []
    for task, n in zip(TASKS, counts):
        if n > 0:
            for ex in rng.sample(pools[task]["icl"], n):
                all_examples.append(format_icl_example(task, ex))
    rng.shuffle(all_examples)

    messages: list[dict] = []
    for inp, out in all_examples:
        messages.append({"role": "user",      "content": inp})
        messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": test_input})
    return messages
