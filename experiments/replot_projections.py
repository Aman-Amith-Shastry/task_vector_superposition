"""Regenerate the appendix LDA layer-sweep projection figures from the cached
per-sample contrast vectors (ratio_samples/<group>_<tag>.npz), with no model
reload. Same underlying data as sweep_{arithmetic,entity,mmlu}.py, restyled for
print legibility:

  * 2 columns x 4 rows (portrait) so each of the 8 swept layers gets a wide
    panel, rendered near full text width in the one-column appendix;
  * large fonts (titles/axes/ticks/legend) that survive down-scaling;
  * the per-point "d=.. kappa=.." text labels removed (those numbers live in
    Table 1 and the Appendix kappa tables) so the geometry is unobstructed;
  * one shared legend, higher dpi.

Writes results/<group>_layer_sweep_projections_<tag>_rotated.png (same filenames
the LaTeX already references).

    python experiments/replot_projections.py --all
    python experiments/replot_projections.py --group arithmetic --model Llama-3B
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pure-task base colors (identical across groups in the original scripts); mixed
# ratios are the ternary blend, exactly as sweep_*.py computes them.
PURE_RGB = np.array([
    [0.27, 0.51, 0.71],   # steelblue
    [0.18, 0.55, 0.34],   # seagreen
    [0.85, 0.55, 0.13],   # darkorange
])
PURE_RATIOS = [(3, 0, 0), (0, 3, 0), (0, 0, 3)]

# Task label per pure-ratio index, matching each sweep script's comments/captions.
GROUP_TASKS = {
    "arithmetic": ["Direct", "MCQ", "Verification"],
    "entity":     ["Capital", "Currency", "Language"],
    "mmlu":       ["History", "Law", "ML"],
}
GROUP_TITLE = {"arithmetic": "Arithmetic", "entity": "Entity", "mmlu": "MMLU"}
MODEL_TITLE = {"Gemma-2B": "Gemma 2 2B", "Llama-3B": "Llama 3.2 3B",
               "Qwen-3B": "Qwen 2.5 3B", "Llama-1B": "Llama 3.2 1B",
               "Llama-8B": "Llama 3.1 8B"}


def ratio_color(ratio):
    w = np.array(ratio, dtype=float)
    w = w / w.sum()
    return np.clip(w @ PURE_RGB, 0, 1)


def load_samples(group, tag):
    """results[ratio_tuple][layer_int] = (n, hidden) array, plus sorted layers."""
    path = os.path.join(ROOT, "ratio_samples", f"{group}_{tag}.npz")
    d = np.load(path)
    results, layers = {}, set()
    for key in d.files:
        lay_s, rat_s = key.split("|")
        lay = int(lay_s)
        rat = tuple(int(x) for x in rat_s.split("-"))
        results.setdefault(rat, {})[lay] = d[key]
        layers.add(lay)
    return results, sorted(layers)


def n_layers_for(group, tag):
    path = os.path.join(ROOT, "results", f"{group}_kappa_{tag}.json")
    if os.path.exists(path):
        return json.load(open(path)).get("n_layers")
    return None


def plot_one(group, tag):
    tasks = GROUP_TASKS[group]
    results, layers = load_samples(group, tag)
    nL = n_layers_for(group, tag)
    ratios = list(results.keys())

    n_cols, n_rows = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 15))
    axes = axes.flatten()

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]

        X_pure = np.vstack([results[p][layer] for p in PURE_RATIOS])
        y_pure = np.concatenate([np.full(len(results[p][layer]), i)
                                 for i, p in enumerate(PURE_RATIOS)])
        lda = LinearDiscriminantAnalysis(n_components=2).fit(X_pure, y_pure)

        pure_centroids = {}
        for ratio in ratios:
            proj = lda.transform(results[ratio][layer])
            color = ratio_color(ratio)
            is_pure = ratio in PURE_RATIOS
            ax.scatter(proj[:, 0], proj[:, 1], color=color,
                       s=42 if is_pure else 24,
                       alpha=0.8 if is_pure else 0.45, zorder=3, linewidths=0)
            if is_pure:
                c = proj.mean(axis=0)
                pure_centroids[ratio] = c
                ax.scatter(*c, color=color, s=300, edgecolors="black",
                           linewidths=1.6, zorder=6)
                ax.annotate(tasks[PURE_RATIOS.index(ratio)], c,
                            textcoords="offset points", xytext=(8, 6),
                            fontsize=15, fontweight="bold")

        for ratio in ratios:
            if ratio in PURE_RATIOS:
                continue
            total = sum(ratio)
            centroid = lda.transform(results[ratio][layer]).mean(axis=0)
            predicted = sum((n / total) * pure_centroids[p]
                            for n, p in zip(ratio, PURE_RATIOS))
            ax.plot([predicted[0], centroid[0]], [predicted[1], centroid[1]],
                    color="0.35", linewidth=1.3, linestyle=":", zorder=4)
            ax.scatter(*centroid, color=ratio_color(ratio), s=150,
                       edgecolors="black", linewidths=1.0, zorder=5)
            ax.scatter(*predicted, color="gray", marker="x", s=150,
                       linewidths=2.6, zorder=5)

        ax.margins(0.14)   # breathing room so vertex labels never clip the frame
        ttl = f"Layer {layer}" + (f"  (depth {layer / nL:.2f})" if nL else "")
        ax.set_title(ttl, fontsize=21)
        ax.set_xlabel("LDA 1", fontsize=18)
        ax.set_ylabel("LDA 2", fontsize=18)
        ax.tick_params(labelsize=14)

    for j in range(len(layers), len(axes)):
        axes[j].set_visible(False)

    handles = [plt.scatter([], [], color=ratio_color(p), s=170,
                           edgecolors="black", linewidths=1.2, label=tasks[i])
               for i, p in enumerate(PURE_RATIOS)]
    handles += [
        plt.scatter([], [], color="dimgray", s=150, edgecolors="black",
                    linewidths=1.0, label="Actual mixed centroid"),
        plt.scatter([], [], color="gray", marker="x", s=150, linewidths=2.6,
                    label="Predicted mixed centroid"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=17,
               framealpha=0.9, columnspacing=1.6, bbox_to_anchor=(0.5, 0.004))

    fig.suptitle(f"{GROUP_TITLE[group]} — {MODEL_TITLE.get(tag, tag)}: "
                 f"LDA projections of contrast vectors by layer",
                 fontsize=23, y=0.995)
    fig.tight_layout(rect=[0, 0.075, 1, 0.975])

    out = os.path.join(ROOT, "results",
                       f"{group}_layer_sweep_projections_{tag}_rotated.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUP_TASKS))
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.all:
        for path in sorted(glob.glob(os.path.join(ROOT, "ratio_samples", "*.npz"))):
            base = os.path.basename(path)[:-4]
            group, tag = base.split("_", 1)
            plot_one(group, tag)
    else:
        plot_one(a.group, a.model)


if __name__ == "__main__":
    main()
