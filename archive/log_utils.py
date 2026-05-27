"""
Shared logging utility for ICL example tracing.

Each collection function calls log_run_header() once at the start, then
log_icl_sample() for every (condition, sample_index) pair so that the exact
questions used in each forward pass are recorded in icl_examples.log.
"""

from datetime import datetime

LOG_FILE = "icl_examples.log"


def _extract_question(content: str) -> str:
    first_line = content.split("\n")[0]
    return first_line[len("Question: "):] if first_line.startswith("Question: ") else first_line


def log_run_header(experiment: str) -> None:
    """Write a timestamped section header for a new experiment run."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 70}\n")
        f.write(f"EXPERIMENT : {experiment}\n")
        f.write(f"TIME       : {ts}\n")
        f.write(f"{'=' * 70}\n\n")


def log_icl_sample(
    condition: str,
    sample_idx: int,
    messages: list[dict],
    layer: int | None = None,
) -> None:
    """Append one sample's ICL context and test question to the log file.

    Parses the message list directly so no extra metadata needs to be passed:
      - system messages are logged with a [SYS] tag (first 120 chars)
      - consecutive user/assistant pairs are logged as [ICL n] with the answer
      - the final user message (no following assistant turn) is logged as [TEST]
    """
    header = f"--- {condition} | sample {sample_idx}"
    if layer is not None:
        header += f" | layer {layer}"
    header += " ---"

    lines = [header]
    i, icl_count = 0, 0

    while i < len(messages):
        msg = messages[i]
        role = msg["role"]

        if role == "system":
            lines.append(f"  [SYS] {msg['content'][:120]}")
            i += 1

        elif role == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            q   = _extract_question(msg["content"])[:80]
            ans = messages[i + 1]["content"]
            icl_count += 1
            lines.append(f"  [ICL {icl_count}] {q!r} → {ans}")
            i += 2

        elif role == "user":
            q = _extract_question(msg["content"])[:80]
            lines.append(f"  [TEST] {q!r}")
            i += 1

        else:
            i += 1

    lines.append("")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
