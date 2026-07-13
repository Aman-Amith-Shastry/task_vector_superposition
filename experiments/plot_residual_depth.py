"""
Visualize the orthogonal residual of the linear prediction as a function of
(relative) layer depth, one panel per task group, one colored line per model.

For each model x group we plot the full-sample orthogonal residual
sin_pred = mean sin(theta(full-sample d_act, d_pred)) as a solid line, and the
split-half sampling-noise floor (n/2) as a thin dashed line of the same color.
The full-sample residual is the minimum-variance estimate of the true orthogonal
component's magnitude, so it shows the depth trend most cleanly; the floor is
drawn for scale. The message: the residual is small early and grows with depth in
every cell (linear superposition is geometrically tightest early), while whether
that growth is a statistically resolvable orthogonal component -- the size-matched
(n/2 vs n/2) test reported in the split-half table -- depends on the group's
separability (the floor).

Reads results/<group>_kappa_<Model>.json (needs the "split_half" block).

    python plot_residual_depth.py            # -> results/orthogonal_residual_depth.png
    python plot_residual_depth.py --show
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt

_FNAME = re.compile(r"(?P<group>.+?)_kappa_(?P<model>.+)\.json$")

# 95% normal-approximation multiplier. The band is point +/- Z * bootstrap SE
# (results/bootstrap_ci_<group>_<model>.json, boot_std) rather than the stored
# percentile interval: for this angle-of-mean statistic, resampling with
# replacement biases the raw percentile CI (it inflates the full-sample residual
# via Jensen and deflates the split-half floor via duplicated points), so the
# percentile interval does not bracket the point estimate. The bootstrap SE is
# robust to that location bias, giving an honest band centered on the plotted
# minimum-variance point.
Z95 = 1.959963984540054

# Fixed order + pretty names so panels and colors are consistent.
GROUP_ORDER = ["arithmetic", "entity", "mmlu"]
GROUP_TITLE = {"arithmetic": "Arithmetic", "entity": "Entity", "mmlu": "MMLU"}
MODEL_TITLE = {"Gemma-2B": "Gemma 2 2B", "Llama-3B": "Llama 3.2 3B",
               "Qwen-3B": "Qwen 2.5 3B", "Llama-1B": "Llama 3.2 1B",
               "Llama-8B": "Llama 3.1 8B"}
MODEL_ORDER = ["Gemma-2B", "Llama-3B", "Qwen-3B", "Llama-1B", "Llama-8B"]
MODEL_COLOR = {"Gemma-2B": "#1b9e77", "Llama-3B": "#d95f02", "Qwen-3B": "#7570b3",
               "Llama-1B": "#e7298a", "Llama-8B": "#66a61e"}


def _boot_se(root, group, model):
    """{layer_index: {"full": se, "floor": se}} from the bootstrap CI file.

    se is the bootstrap standard error (boot_std) of the ratio-aggregated
    quantity at that layer. Returns {} when no bootstrap file is present, so
    older/partial runs simply plot without a band.
    """
    path = os.path.join(root, "results", f"bootstrap_ci_{group}_{model}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        b = json.load(fh)
    out = {}
    for L, lay in b.get("layers", {}).items():
        out[int(L)] = {stat: float(lay[stat]["boot_std"])
                       for stat in ("full", "floor") if stat in lay}
    return out


def _band(point, se):
    """95% bootstrap-SE band around a plotted point, clamped to [0, 1]."""
    pt, s = np.asarray(point), np.asarray(se)
    return np.clip(pt - Z95 * s, 0.0, 1.0), np.clip(pt + Z95 * s, 0.0, 1.0)


def _series(path, group, model, root):
    """Return arrays for one file, with bootstrap-SE bands on full and floor.

    (rel_depth, sin_pred, floor, full_lo, full_hi, floor_lo, floor_hi).
    The *_lo/*_hi are None when no bootstrap CI file exists for this pair, in
    which case the caller draws the lines with no band, exactly as before.
    """
    with open(path) as fh:
        data = json.load(fh)
    sh = data.get("split_half", {})
    if not sh:
        return None
    n_layers = data.get("n_layers") or (max(int(l) for l in sh) + 1)
    layers = sorted(int(l) for l in sh)
    se = _boot_se(root, group, model)
    rel, sp, fl, se_full, se_floor = [], [], [], [], []
    has_ci = bool(se)
    for L in layers:
        pr = sh[str(L)]["per_ratio"]
        rel.append(L / n_layers)
        # full-sample residual: minimum-variance estimate of the orthogonal
        # component's magnitude, so the depth trend is not flattened by the n/2
        # sampling noise. (The size-matched resolvability test lives in the table.)
        sp.append(float(np.nanmean([v["sin_pred"] for v in pr.values()])))
        fl.append(float(np.nanmean([v["sin_floor_mean"] for v in pr.values()])))
        if L in se and "full" in se[L] and "floor" in se[L]:
            se_full.append(se[L]["full"])
            se_floor.append(se[L]["floor"])
        else:
            has_ci = False
    rel, sp, fl = np.array(rel), np.array(sp), np.array(fl)
    if has_ci:
        full_lo, full_hi = _band(sp, se_full)
        floor_lo, floor_hi = _band(fl, se_floor)
    else:
        full_lo = full_hi = floor_lo = floor_hi = None
    return rel, sp, fl, full_lo, full_hi, floor_lo, floor_hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/*_kappa_*.json")
    ap.add_argument("--out", default="results/orthogonal_residual_depth.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = args.glob if os.path.isabs(args.glob) else os.path.join(root, args.glob)

    # group -> {model: (rel, sp, fl)}
    cells = {g: {} for g in GROUP_ORDER}
    for p in glob.glob(pattern):
        m = _FNAME.search(os.path.basename(p))
        if not m:
            continue
        g, model = m.group("group"), m.group("model")
        if g not in cells:
            continue
        s = _series(p, g, model, root)
        if s is not None:
            cells[g][model] = s

    # Only plot models with complete coverage across every task group, so a
    # partially-run model (e.g. one with only MMLU split-half data) does not
    # appear as a lone line in a single panel.
    complete = [m for m in MODEL_ORDER if all(m in cells[gg] for gg in GROUP_ORDER)]

    fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(13, 4.2), sharey=True)
    for ax, g in zip(axes, GROUP_ORDER):
        models = [m for m in complete if m in cells[g]]
        for model in models:
            rel, sp, fl, full_lo, full_hi, floor_lo, floor_hi = cells[g][model]
            c = MODEL_COLOR.get(model, None)
            # Bootstrap 95% SE bands, when the run provides them: a solid-fill
            # band on the full-sample residual and a fainter one on the floor.
            if full_lo is not None:
                ax.fill_between(rel, full_lo, full_hi, color=c, alpha=0.15, lw=0)
                ax.fill_between(rel, floor_lo, floor_hi, color=c, alpha=0.07, lw=0)
            ax.plot(rel, sp, "-o", color=c, ms=4, lw=1.8,
                    label=MODEL_TITLE.get(model, model))
            ax.plot(rel, fl, "--", color=c, lw=1.0, alpha=0.55)
        ax.set_title(GROUP_TITLE[g])
        ax.set_xlabel("relative layer depth")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"orthogonal residual  $\sin\theta(d_{\mathrm{act}}, d_{\mathrm{pred}})$")
    fig.tight_layout()

    # One shared legend for all three panels (identical model set), placed below
    # the axes so it never overlaps the curves.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=len(labels), fontsize=9, framealpha=0.9)

    # One shared note explaining the dashed lines, below the legend.
    fig.text(0.5, -0.06,
             "Solid: full-sample orthogonal residual (minimum-variance magnitude "
             "estimate).  Dashed (same color): split-half sampling-noise floor "
             "(n/2), shown for scale.  Shaded: 95% bootstrap-SE bands (darker = "
             "residual, fainter = floor).  Size-matched resolvability test: see "
             "the split-half table.",
             ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.22)

    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
