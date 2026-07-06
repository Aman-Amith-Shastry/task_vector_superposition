"""
Extract and save contrast vectors for MMLU semantic domains
(History / Law / ML) at each candidate injection layer.

Contrast vector:
  θ_domain = mean(act(ICL + test_q)) − act(test_q alone)
  Removes test-question content, leaving only the ICL-context shift.

Saved as:
  vectors/<domain>_layer<l>.npy          mean   shape: (hidden_dim,)
  vectors/<domain>_layer<l>_samples.npy  all    shape: (N_EXTRACT, hidden_dim)

The samples file is used by decompose_task_mixture.py for bootstrap CIs.

Usage:
  python experiments/task_vector_injection/extract_vectors.py
"""

import sys, os

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "data"))

import random
import numpy as np

from local_model import get_activations_all_layers
import data_semantic_domains as data

LAYERS    = [1, 3, 5, 8, 10, 14, 18, 20]
N_EXTRACT = 50
OUT_DIR   = os.path.join(_dir, "vectors")


def extract_contrast_vectors(pools: dict, domain: str,
                              layers: list[int]) -> dict[int, np.ndarray]:
    """Return {layer: (N_EXTRACT, hidden_dim)} of individual contrast vectors."""
    pure_ratio = data.PURE_RATIOS[data.TASKS.index(domain)]
    accum: dict[int, list] = {l: [] for l in layers}
    for i in range(N_EXTRACT):
        rng      = random.Random(i)
        msgs     = data.build_messages(pools, rng, pure_ratio, sample_idx=i,
                                       rotate_test=True)
        icl_acts = get_activations_all_layers(msgs,      layers)
        zs_acts  = get_activations_all_layers(msgs[-1:], layers)
        for l in layers:
            accum[l].append(icl_acts[l].float().numpy() - zs_acts[l].float().numpy())
    return {l: np.stack(accum[l]) for l in layers}   # (N_EXTRACT, hidden_dim)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading MMLU pools...")
    pools = data.load_pools()

    for domain in data.TASKS:
        print(f"\n{domain}:")
        samples = extract_contrast_vectors(pools, domain, LAYERS)
        for layer, s in samples.items():
            np.save(os.path.join(OUT_DIR, f"{domain}_layer{layer}.npy"),         s.mean(axis=0))
            np.save(os.path.join(OUT_DIR, f"{domain}_layer{layer}_samples.npy"), s)
            print(f"  Layer {layer}  shape={s.shape}")

    print("\nDone. Vectors saved to:", OUT_DIR)
