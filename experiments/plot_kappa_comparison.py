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
_parser.add_argument("--combined", action="store_true",
                     help="Emit one wide 1x3 figure (all task groups) with a shared legend.")
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


_GROUP_ORDER = ["arithmetic", "entity", "mmlu"]
_GROUP_TITLE = {"arithmetic": "Arithmetic", "entity": "Entity", "mmlu": "MMLU"}
# Fixed model order so the shared legend and line ordering are stable.
_MODEL_ORDER = ["Llama-1B", "Llama-3B", "Llama-8B", "Qwen-3B", "Gemma-2B"]


def _draw_group(ax, group, clip, first):
    """Plot one task group's cross-model kappa-vs-depth panel. Returns
    {tag: Line2D} handles from the first panel (for a shared legend)."""
    ax.axhline(1.0, color="gray", lw=1.4, ls="--", alpha=0.7,
               label="$\\kappa = 1$ (perfect linearity)" if first else None)
    ax.axhspan(0.75, 1.25, color="gray", alpha=0.07,
               label="$\\kappa \\in [0.75, 1.25]$" if first else None)

    by_tag = {}
    for path in sorted(_glob.glob(os.path.join(_results_dir, f"{group}_kappa_*.json"))):
        data = load_kappa_file(path, clip)
        by_tag[model_tag(data["model"])] = data

    for tag in _MODEL_ORDER:
        if tag not in by_tag:
            continue
        data = by_tag[tag]
        style = _MODEL_STYLES.get(tag, dict(color="black", marker="o",
                                            linestyle="-", label=tag))
        xs = [p[0] for p in data["points"]]
        means = [p[1] for p in data["points"]]
        stds = [p[2] for p in data["points"]]
        ax.plot(xs, means, marker=style["marker"], linestyle=style["linestyle"],
                color=style["color"], label=style["label"] if first else None,
                linewidth=2.4, markersize=8)
        ax.fill_between(xs, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=style["color"], alpha=0.15)

    ax.set_title(_GROUP_TITLE[group], fontsize=19)
    ax.set_xlabel("Relative layer depth", fontsize=15)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=13)
    ax.grid(axis="y", alpha=0.3)


def main_combined():
    """One wide 1x3 figure (Arithmetic / Entity / MMLU) sharing a y-axis and a
    single legend below, sized so every label is legible at print scale."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for i, (ax, group) in enumerate(zip(axes, _GROUP_ORDER)):
        _draw_group(ax, group, _args.clip, first=(i == 0))
    axes[0].set_ylabel("Mean $\\kappa$  (linear superposition fidelity)", fontsize=15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=13.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.3)
    fig.suptitle("$\\kappa$ vs. relative layer depth across model scales   "
                 "(shaded band = $\\pm1$ std across mixed ratios)", fontsize=17)
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    out = os.path.join(_results_dir, "kappa_comparison_all_groups.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved → {out}")


def main():
    if getattr(_args, "combined", False):
        main_combined()
        return
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
