"""
Extract and save mean contrast vectors for format-varying arithmetic
(Direct / MCQ / Verification) at each candidate injection layer.

A contrast vector isolates what 3-shot ICL context *adds* to the residual
stream compared to the bare zero-shot question:

  θ_format = mean[ act(3-shot ICL + test_q) − act(test_q alone) ]

Both activations use the same test question (last message of the ICL prompt),
so content effects cancel and the vector captures format/task-context signal.
Saved as:

  vectors/arith_contrast_<Format>_layer<l>.npy   shape: (hidden_dim,)

Usage:
  python task_vector_injection/extract_arithmetic_vectors.py
"""

import sys
import os

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "layer_stratification"))

import random
import numpy as np

from local_model import get_activations_all_layers
import data_arithmetic_formats as data

LAYERS    = [1, 3, 5, 8, 10, 14, 18, 20]
N_EXTRACT = 20
OUT_DIR   = os.path.join(_dir, "vectors")


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
        icl_msgs = data.build_messages(pools, random.Random(i), pure_ratio, sample_idx=i)
        zs_msgs  = [icl_msgs[-1]]  # same test question, no ICL context

        icl_acts = get_activations_all_layers(icl_msgs, layers)
        zs_acts  = get_activations_all_layers(zs_msgs,  layers)

        for l in layers:
            diff = icl_acts[l].float().numpy() - zs_acts[l].float().numpy()
            contrasts[l].append(diff)

    return {l: np.mean(contrasts[l], axis=0) for l in layers}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating arithmetic pools...")
    pools = data.load_pools()

    for fmt in data.TASKS:
        print(f"\n{fmt}:")
        print(f"  Extracting layers {LAYERS} (2 passes × {N_EXTRACT} samples)...", end=" ", flush=True)
        vecs = extract_contrast_vectors(pools, fmt, LAYERS)
        for l, vec in vecs.items():
            path = os.path.join(OUT_DIR, f"arith_contrast_{fmt}_layer{l}.npy")
            np.save(path, vec)
        print(f"saved  shape={next(iter(vecs.values())).shape}")

    print("\nDone. Contrast vectors saved to:", OUT_DIR)
