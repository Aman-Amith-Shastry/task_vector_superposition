"""
Extract and save mean contrast vectors for format-varying arithmetic
(Direct / MCQ / Verification) at each candidate injection layer.

A contrast vector isolates what 3-shot ICL context *adds* to the residual
stream compared to the bare zero-shot question:

  θ_format = mean[ act(3-shot ICL + test_q) − act(test_q alone) ]

Both activations use the same test question (last message of the ICL prompt),
so content effects cancel and the vector captures format/task-context signal.
Saved as:

  vectors/arith_contrast_<Format>_layer<l>.npy                    mean  shape: (hidden_dim,)
  vectors/arith_contrast_<Format>_layer<l>_samples.npy            all   shape: (N_EXTRACT, hidden_dim)
  (non-default models get a MODEL_TAG inserted: arith_contrast_<Format>_<MODEL_TAG>_layer<l>.npy)

The samples file is used by decompose_task_mixture.py for bootstrap confidence intervals.

Model is taken from --model (default Llama-3.2-3B-Instruct). The default model's
files stay untagged for backward compatibility with existing results; any other
model gets its vectors tagged with MODEL_TAG so they don't clobber the default's.

Usage:
  python experiments/task_vector_injection/extract_arithmetic_vectors.py
  python experiments/task_vector_injection/extract_arithmetic_vectors.py --model Qwen/Qwen2.5-3B-Instruct
"""

import sys
import os

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "data"))

# Parse --model BEFORE importing local_model (model loads at import time)
import argparse as _ap
_mp = _ap.ArgumentParser(add_help=False)
_mp.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                 help="HuggingFace model ID to use.")
_mp.add_argument("--quantize", default="", choices=["", "int8", "int4"],
                 help="Quantize weights via quanto (recommended for 8B on MPS).")
_margs, _ = _mp.parse_known_args()
os.environ["TASK_VECTOR_MODEL"]    = _margs.model
os.environ["TASK_VECTOR_QUANTIZE"] = _margs.quantize

import re
def _make_model_tag(model_id: str) -> str:
    size = (re.search(r'(\d+\.?\d*[Bb])', model_id) or type("", (), {"group": lambda s, n: model_id.split("/")[-1]})()).group(1).upper()
    name = model_id.lower()
    if "qwen"    in name: return f"Qwen-{size}"
    if "llama"   in name: return f"Llama-{size}"
    if "mistral" in name: return f"Mistral-{size}"
    if "gemma"   in name: return f"Gemma-{size}"
    return model_id.split("/")[-1]
MODEL_TAG   = _make_model_tag(_margs.model)
_IS_DEFAULT = _margs.model == "meta-llama/Llama-3.2-3B-Instruct"

import random
import numpy as np

from local_model import get_activations_all_layers, n_layers
import data_arithmetic_formats as data

# The Llama-3.2-3B-Instruct (28L) layer set is pinned exactly as originally chosen,
# so re-running extraction for the default model reproduces the existing vectors/
# results bit-for-bit. Other models fall back to the same relative-depth fractions
# used by sweep_arithmetic.py, rounded to that model's own depth (n_layers).
_LAYER_FRACS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.75]
LAYERS = (
    [1, 3, 5, 8, 10, 14, 18, 20] if _IS_DEFAULT
    else sorted(set(max(1, round(f * n_layers)) for f in _LAYER_FRACS))
)
N_EXTRACT    = 50
OUT_DIR      = os.path.join(_dir, "vectors")


def _vec_path(fmt: str, layer: int, suffix: str = "") -> str:
    if _IS_DEFAULT:
        return os.path.join(OUT_DIR, f"arith_contrast_{fmt}_layer{layer}{suffix}.npy")
    return os.path.join(OUT_DIR, f"arith_contrast_{fmt}_{MODEL_TAG}_layer{layer}{suffix}.npy")


def extract_contrast_vectors(pools: dict, fmt: str, layers: list[int]) -> dict[int, np.ndarray]:
    """Contrast vectors at each layer for format `fmt`, averaged over N_EXTRACT samples.

    For each sample: build the ICL prompt and extract the last user message as the
    matching zero-shot prompt. Subtract zero-shot activation from ICL activation so
    content effects cancel and only the ICL-context contribution remains.
    Both prompts are processed in one forward pass each (all layers simultaneously).
    """
    pure_ratio = data.PURE_RATIOS[data.TASKS.index(fmt)]
    contrasts: dict[int, list] = {l: [] for l in layers}

    for i in range(N_EXTRACT):
        icl_msgs = data.build_messages(pools, random.Random(i), pure_ratio, sample_idx=i, rotate_test=True)
        zs_msgs  = [icl_msgs[-1]]  # same test question, no ICL context

        icl_acts = get_activations_all_layers(icl_msgs, layers)
        zs_acts  = get_activations_all_layers(zs_msgs,  layers)

        for l in layers:
            diff = icl_acts[l].float().numpy() - zs_acts[l].float().numpy()
            contrasts[l].append(diff)

    return {l: np.stack(contrasts[l]) for l in layers}   # (N_EXTRACT, hidden_dim)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Model: {_margs.model}  ({n_layers} layers, tag={MODEL_TAG})")
    print(f"Layers: {LAYERS}")
    print("Generating arithmetic pools...")
    pools = data.load_pools()

    for fmt in data.TASKS:
        print(f"\n{fmt}:")
        print(f"  Extracting layers {LAYERS} (2 passes × {N_EXTRACT} samples)...", end=" ", flush=True)
        samples = extract_contrast_vectors(pools, fmt, LAYERS)  # {l: (N_EXTRACT, hidden_dim)}
        for l, s in samples.items():
            np.save(_vec_path(fmt, l),            s.mean(axis=0))
            np.save(_vec_path(fmt, l, "_samples"), s)
        print(f"saved  shape={next(iter(samples.values())).shape}")

    print("\nDone. Contrast vectors saved to:", OUT_DIR)
