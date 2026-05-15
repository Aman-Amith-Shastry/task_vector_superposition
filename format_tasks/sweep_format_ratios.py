"""
ICL ratio sweep for format-distinct tasks (MCQ / PubMedQA / Symptom2Disease).

Convenience wrapper around sweep_icl_ratios.py --experiment format_tasks.
Run directly or use the unified entry point:

  python sweep_format_ratios.py
  python sweep_icl_ratios.py --experiment format_tasks   # equivalent
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv += ["--experiment", "format_tasks"]

from sweep_icl_ratios import (   # noqa: E402  (import after argv patch)
    TASKS, RATIOS, PURE_RATIOS, LAYER, OUT_PREFIX,
    load_pools, collect_all_ratios, fit_and_project, plot_ratio_sweep,
)

if __name__ == "__main__":
    print("Experiment: format_tasks")
    print("Loading data...")
    pools = load_pools()

    print(f"\nCollecting contrast vectors at layer {LAYER}...")
    vectors        = collect_all_ratios(pools)
    _, projected   = fit_and_project(vectors)
    pure_centroids = {p: projected[p].mean(axis=0) for p in PURE_RATIOS}

    print("\n=== Ratio sweep results ===")
    for ratio in RATIOS:
        import numpy as np
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
