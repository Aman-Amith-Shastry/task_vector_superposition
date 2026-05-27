"""
Data generation for the format-varying arithmetic experiment.

Three tasks with identical semantic content (single-step integer addition) but
structurally different output formats:

  Direct        "4 + 7 A:"                       → "11"   (number)
  MCQ           "4 + 7 = ? A) 9  B) 11  C) 13  D) 15 A:" → "B"    (letter)
  Verification  "4 + 7 = 11. Correct? A:"         → "yes"/"no"

All three end the user turn with "A:" so the pre-answer structural token is
identical across conditions — analogous to the "=" hook in Hendel et al.
Content is generated programmatically; no HuggingFace dataset required.
"""

import random

TASKS            = ["Direct", "MCQ", "Verification"]
N_ICL_POOL       = 200
N_TEST_POOL      = 50
N_VECTOR_SAMPLES = 20
SEED             = 42

# (n_direct, n_mcq, n_verification) — must sum to 3
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

_LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------
# Internal generation helpers
# --------------------------------------------------------------------------

def _gen_base(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [{"op1": rng.randint(1, 49), "op2": rng.randint(1, 49)} for _ in range(n)]


def _mcq_options(answer: int, rng: random.Random) -> tuple[list[int], int]:
    """Return (four_options, correct_idx) with plausible shuffled distractors."""
    offsets = [-2, -1, 1, 2, -3, 3, -5, 5, 10, -10]
    wrong: list[int] = []
    for off in rng.sample(offsets, len(offsets)):
        w = answer + off
        if w > 0 and w != answer and w not in wrong:
            wrong.append(w)
        if len(wrong) == 3:
            break
    opts = wrong + [answer]
    rng.shuffle(opts)
    return opts, opts.index(answer)


def _as_direct(base: dict) -> dict:
    return {**base, "answer": base["op1"] + base["op2"]}


def _as_mcq(base: dict, seed: int) -> dict:
    answer      = base["op1"] + base["op2"]
    opts, idx   = _mcq_options(answer, random.Random(seed))
    return {**base, "answer": answer, "options": opts, "correct_idx": idx}


def _as_verification(base: dict, seed: int) -> dict:
    answer = base["op1"] + base["op2"]
    rng    = random.Random(seed)
    if rng.random() < 0.5:
        return {**base, "answer": answer, "shown": answer, "label": "yes"}
    offsets = [-2, -1, 1, 2, -3, 3]
    for off in rng.sample(offsets, len(offsets)):
        w = answer + off
        if w > 0:
            return {**base, "answer": answer, "shown": w, "label": "no"}
    return {**base, "answer": answer, "shown": answer + 1, "label": "no"}


# --------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------

def load_pools() -> dict[str, dict]:
    """Generate ICL and test pools for all three format tasks."""
    base_icl  = _gen_base(N_ICL_POOL, SEED)
    base_test = _gen_base(N_TEST_POOL, SEED + 1)

    pools: dict[str, dict] = {}
    for i, task in enumerate(TASKS):
        seed_off = i * 1000
        if task == "Direct":
            pools[task] = {
                "icl":  [_as_direct(b) for b in base_icl],
                "test": [_as_direct(b) for b in base_test],
            }
        elif task == "MCQ":
            pools[task] = {
                "icl":  [_as_mcq(b, seed_off + j)             for j, b in enumerate(base_icl)],
                "test": [_as_mcq(b, seed_off + N_ICL_POOL + j) for j, b in enumerate(base_test)],
            }
        else:  # Verification
            pools[task] = {
                "icl":  [_as_verification(b, seed_off + j)             for j, b in enumerate(base_icl)],
                "test": [_as_verification(b, seed_off + N_ICL_POOL + j) for j, b in enumerate(base_test)],
            }
    return pools


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def format_icl_example(task: str, ex: dict) -> tuple[str, str]:
    """Return (user_content, assistant_content) for a single ICL example."""
    if task == "Direct":
        return f"{ex['op1']} + {ex['op2']} A:", str(ex["answer"])
    elif task == "MCQ":
        opts = "  ".join(f"{l}) {v}" for l, v in zip(_LETTERS, ex["options"]))
        return f"{ex['op1']} + {ex['op2']} = ? {opts} A:", _LETTERS[ex["correct_idx"]]
    else:
        return f"{ex['op1']} + {ex['op2']} = {ex['shown']}. Correct? A:", ex["label"]


def format_test_input(task: str, ex: dict) -> str:
    """User content for the held-out test question."""
    if task == "Direct":
        return f"{ex['op1']} + {ex['op2']} A:"
    elif task == "MCQ":
        opts = "  ".join(f"{l}) {v}" for l, v in zip(_LETTERS, ex["options"]))
        return f"{ex['op1']} + {ex['op2']} = ? {opts} A:"
    else:
        return f"{ex['op1']} + {ex['op2']} = {ex['shown']}. Correct? A:"


# --------------------------------------------------------------------------
# build_messages
# --------------------------------------------------------------------------

def build_messages(
    pools: dict,
    rng: random.Random,
    counts: tuple[int, int, int],
    sample_idx: int = 0,
    rotate_test: bool = False,
) -> list[dict]:
    """ICL messages for the given per-task counts + a held-out test question.

    By default, pure conditions use a matching test question format and mixed
    conditions rotate with sample_idx.  Set rotate_test=True to rotate for
    all conditions uniformly, so the test question contributes equally to every
    centroid and the contrast vector purely reflects the ICL composition.
    """
    is_pure   = counts in PURE_RATIOS
    if rotate_test:
        test_task = TASKS[sample_idx % len(TASKS)]
    else:
        test_task = TASKS[counts.index(max(counts))] if is_pure else TASKS[sample_idx % len(TASKS)]

    test_ex    = rng.choice(pools[test_task]["test"])
    test_input = format_test_input(test_task, test_ex)

    all_examples: list[tuple[str, str]] = []
    for task, n in zip(TASKS, counts):
        for ex in rng.sample(pools[task]["icl"], n):
            all_examples.append(format_icl_example(task, ex))
    rng.shuffle(all_examples)

    messages: list[dict] = []
    for inp, out in all_examples:
        messages.append({"role": "user",      "content": inp})
        messages.append({"role": "assistant", "content": out})
    messages.append({"role": "user", "content": test_input})
    return messages
