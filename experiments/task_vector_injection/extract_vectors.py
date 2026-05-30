"""
Extract and save contrast vectors and domain-residual vectors for MMLU
semantic domains (History / Law / ML) at each candidate injection layer.

Contrast vector:
  θ_domain = mean(act(ICL + test_q)) − mean(act(test_q alone))
  Removes test-question content; still contains the shared ICL structural
  signal common to all domains (turn-taking format, attention patterns).

Residual vector:
  θ_residual = θ_domain − θ_common,  where θ_common = mean over all domains
  Removes the shared ICL component, leaving only what is unique to each
  domain. Use this for injection when you want domain-specific steering
  without the general "be in ICL mode" effect.

Saved as:
  vectors/<domain>_layer<l>.npy           contrast vector
  vectors/<domain>_residual_layer<l>.npy  domain-specific residual

Usage:
  python task_vector_injection/extract_vectors.py
"""

import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
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


def extract_mean_contrast_vector(pools: dict, domain: str, layers: list[int]) -> dict[int, np.ndarray]:
    """Average contrast vectors for `domain` at each layer over N_EXTRACT ICL prompts."""
    pure_ratio = data.PURE_RATIOS[data.TASKS.index(domain)]
    accum: dict[int, list] = {l: [] for l in layers}
    for i in range(N_EXTRACT):
        rng      = random.Random(i)
        msgs     = data.build_messages(pools, rng, pure_ratio, sample_idx=i, rotate_test=True)
        icl_acts = get_activations_all_layers(msgs,      layers)
        zs_acts  = get_activations_all_layers(msgs[-1:], layers)
        for l in layers:
            diff = icl_acts[l].float().numpy() - zs_acts[l].float().numpy()
            accum[l].append(diff)
    return {l: np.mean(accum[l], axis=0) for l in layers}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading MMLU pools...")
    pools = data.load_pools()

    # --- Step 1: extract contrast vectors for all domains ---
    contrast: dict[str, dict[int, np.ndarray]] = {}
    for domain in data.TASKS:
        print(f"\n{domain} (contrast vectors):")
        contrast[domain] = extract_mean_contrast_vector(pools, domain, LAYERS)
        for layer, vec in contrast[domain].items():
            path = os.path.join(OUT_DIR, f"{domain}_layer{layer}.npy")
            np.save(path, vec)
            print(f"  Layer {layer}... saved  shape={vec.shape}")

    # --- Step 2: compute and save domain-residual vectors ---
    print("\nComputing residual vectors (contrast − common):")
    for layer in LAYERS:
        common = np.mean([contrast[d][layer] for d in data.TASKS], axis=0)
        for domain in data.TASKS:
            residual = contrast[domain][layer] - common
            path = os.path.join(OUT_DIR, f"{domain}_residual_layer{layer}.npy")
            np.save(path, residual)
        print(f"  Layer {layer}... saved residuals for {data.TASKS}")

    print("\nDone. Vectors saved to:", OUT_DIR)
