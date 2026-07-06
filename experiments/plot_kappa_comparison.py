"""
Cross-model κ comparison plot.

Reads arithmetic_kappa_<MODEL_TAG>.json (or mmlu_kappa_<MODEL_TAG>.json)
files produced by sweep_arithmetic.py / sweep_mmlu.py and plots mean κ ± std
as a function of relative layer depth (layer / n_layers) for each model.

A horizontal dashed line at κ=1 marks perfect linear superposition.

Usage:
    python experiments/plot_kappa_comparison.py
    python experiments/plot_kappa_comparison.py --glob "mmlu_kappa_*.json"
"""

import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import glob as _glob
import numpy as np
import matplotlib.pyplot as plt

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--glob", default="arithmetic_kappa_*.json",
                     help="Glob pattern for JSON files (relative to results/ dir).")
_parser.add_argument("--clip", type=float, default=2.5,
                     help="Clip |κ| beyond this value before averaging (removes outliers).")
_args, _ = _parser.parse_known_args()

_results_dir = os.path.join(_root, "results")

# Model display order and line styles
_MODEL_STYLES = {
    "Llama-1B": dict(color="#4878CF", marker="o", linestyle="-",  label="Llama-3.2-1B"),
    "Llama-3B": dict(color="#6ACC65", marker="s", linestyle="--", label="Llama-3.2-3B"),
    "Llama-8B": dict(color="#D65F5F", marker="^", linestyle="-.", label="Llama-3.1-8B"),
    "Qwen-3B":  dict(color="#E07B39", marker="D", linestyle=":",  label="Qwen-2.5-3B"),
    "Gemma-2B": dict(color="#8E44AD", marker="v", linestyle="-.", label="Gemma-2-2B"),
}


def load_kappa_file(path: str, clip: float) -> dict:
    with open(path) as f:
        data = json.load(f)
    n_layers = data["n_layers"]
    model    = data["model"]
    points   = []   # (relative_depth, mean_kappa, std_kappa)
    for layer_str, kappas in data["kappa"].items():
        if not kappas:
            continue
        arr = np.clip(np.array(kappas), -clip, clip)
        rel = int(layer_str) / n_layers
        points.append((rel, float(arr.mean()), float(arr.std())))
    points.sort()
    return {"model": model, "n_layers": n_layers, "points": points}


def model_tag(model_id: str) -> str:
    import re
    m    = re.search(r'(\d+\.?\d*[Bb])', model_id)
    size = m.group(1).upper() if m else model_id.split("/")[-1]
    name = model_id.lower()
    if "qwen"    in name: return f"Qwen-{size}"
    if "llama"   in name: return f"Llama-{size}"
    if "mistral" in name: return f"Mistral-{size}"
    if "gemma"   in name: return f"Gemma-{size}"
    return size


def main():
    pattern  = os.path.join(_results_dir, _args.glob)
    paths    = sorted(_glob.glob(pattern))
    if not paths:
        print(f"No files matched: {pattern}")
        print("Run experiments/sweep_arithmetic.py (or sweep_mmlu.py) for each model first.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.6, label="κ = 1 (perfect linearity)")
    ax.axhspan(0.75, 1.25, color="gray", alpha=0.07)   # ±25% band

    for path in paths:
        data = load_kappa_file(path, _args.clip)
        tag  = model_tag(data["model"])
        style = _MODEL_STYLES.get(tag, dict(color="black", marker="o", linestyle="-", label=tag))

        xs    = [p[0] for p in data["points"]]
        means = [p[1] for p in data["points"]]
        stds  = [p[2] for p in data["points"]]

        ax.plot(xs, means, marker=style["marker"], linestyle=style["linestyle"],
                color=style["color"], label=style["label"], linewidth=2, markersize=7)
        ax.fill_between(xs,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=style["color"], alpha=0.15)

    ax.set_xlabel("Relative layer depth  (layer / n_layers)", fontsize=11)
    ax.set_ylabel("Mean κ  (linear superposition fidelity)", fontsize=11)
    ax.set_title(
        "κ vs. relative layer depth across model scales\n"
        "Shaded band = ±1 std across mixed ratios  ·  Grey zone = κ ∈ [0.75, 1.25]",
        fontsize=10,
    )
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    prefix = _args.glob.split("_kappa_")[0] if "_kappa_" in _args.glob else "kappa"
    out = os.path.join(_results_dir, f"{prefix}_kappa_comparison_across_models.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
