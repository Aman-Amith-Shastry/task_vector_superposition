"""
ICL ratio sweep for arithmetic format tasks (Direct / MCQ / Verification).

Convenience wrapper around sweep_icl_ratios.py --experiment arithmetic.

Usage:
  python layer_stratification/sweep_arithmetic_ratios.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv += ["--experiment", "arithmetic"]

from sweep_icl_ratios import (   # noqa: E402
    TASKS, RATIOS, PURE_RATIOS, LAYER, OUT_PREFIX,
    load_pools, collect_all_ratios, fit_and_project, plot_ratio_sweep,
)
import numpy as np

if __name__ == "__main__":
    print("Experiment: arithmetic (Direct / MCQ / Verification)")
    print("Loading data...")
    pools = load_pools()

    print(f"\nCollecting contrast vectors at layer {LAYER}...")
    vectors        = collect_all_ratios(pools)
    _, projected   = fit_and_project(vectors)
    pure_centroids = {p: projected[p].mean(axis=0) for p in PURE_RATIOS}

    print("\n=== Ratio sweep results ===")
    for ratio in RATIOS:
        actual    = projected[ratio].mean(axis=0)
        total     = sum(ratio)
        predicted = sum(
            (n / total) * pure_centroids[p]
            for n, p in zip(ratio, PURE_RATIOS)
        )
        dist  = np.linalg.norm(actual - predicted)
        label = "-".join(map(str, ratio))
        print(f"  {label:>8}: |actual − predicted| = {dist:.4f}")

    plot_ratio_sweep(projected)
