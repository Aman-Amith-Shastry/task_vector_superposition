"""
Resolvability (SNR) of the orthogonal residual as a function of relative layer
depth: the ratio of the size-matched residual to the split-half noise floor,

    R^l_r = tilde{s}^l_r / phi^l_r        (both estimated at n/2, so it is fair)

where tilde{s} is the *half-sample* residual (sin_pred_half) and phi is the
split-half floor (sin_floor_mean). This is deliberately NOT the full-sample
residual that the magnitude figure (plot_residual_depth.py) draws as its solid
line -- the magnitude figure answers "how big is the deviation from linearity",
this one answers "is that deviation resolvable above sampling noise, and does the
resolvability grow with depth".

Reading the curve (null is 1, not 0, because the residual is signal+noise in
quadrature, tilde{s}^2 ~= s_true^2 + phi^2):

    rising with depth      signal outpaces the floor -> deviation is real AND
                           increasingly resolvable (both claims hold)
    flat, well above 1     signal grows in lockstep with the floor -> the
                           magnitude claim still holds; only the *acceleration*
                           of resolvability is absent
    flat at ~1             tilde{s} ~= phi everywhere -> no resolvable signal; the
                           raw-residual growth would be a floor artifact
    falling                floor grows faster than the residual -> deep layers
                           are LESS resolvable

Per JSON cell we form R at each layer from the ratio-averaged numerator and
denominator (mean sin_pred_half / mean sin_floor_mean over the mixed ratios),
which is more stable than averaging the per-ratio ratio_to_floor field when a
single ratio's floor is tiny. Cells are then aggregated ACROSS MODELS onto a
common relative-depth grid (linear interpolation), one line per task group, with
a band for the across-model spread. Relative depth (layer / n_layers) is the
standard common x-axis for models of differing absolute depth, and R is
dimensionless, so the cross-model average is well posed.

Reads results/<group>_kappa_<Model>.json (needs the "split_half" block).

    python plot_residual_ratio_depth.py          # -> results/orthogonal_residual_ratio_depth.png
    python plot_residual_ratio_depth.py --facet  # also ..._by_model.png (per-model panels)
    python plot_residual_ratio_depth.py --per-model   # faint per-model lines under aggregate
    python plot_residual_ratio_depth.py --show
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt

_FNAME = re.compile(r"(?P<group>.+?)_kappa_(?P<model>.+)\.json$")

GROUP_ORDER = ["arithmetic", "entity", "mmlu"]
GROUP_TITLE = {"arithmetic": "Arithmetic", "entity": "Entity", "mmlu": "MMLU"}
GROUP_COLOR = {"arithmetic": "#d95f02", "entity": "#1b9e77", "mmlu": "#7570b3"}

# Per-model styling for the facet plot (matches plot_residual_depth.py).
MODEL_TITLE = {"Gemma-2B": "Gemma 2 2B", "Llama-3B": "Llama 3.2 3B",
               "Qwen-3B": "Qwen 2.5 3B", "Llama-1B": "Llama 3.2 1B",
               "Llama-8B": "Llama 3.1 8B"}
MODEL_ORDER = ["Gemma-2B", "Llama-3B", "Qwen-3B", "Llama-1B", "Llama-8B"]
MODEL_COLOR = {"Gemma-2B": "#1b9e77", "Llama-3B": "#d95f02", "Qwen-3B": "#7570b3",
               "Llama-1B": "#e7298a", "Llama-8B": "#66a61e"}

# Common relative-depth grid all models are interpolated onto before averaging.
GRID = np.linspace(0.05, 1.0, 20)

# 95% normal-approximation multiplier for the per-model bootstrap band. The band
# is R +/- Z * bootstrap SE (results/bootstrap_ci_<group>_<model>.json, the
# "ratio" boot_std) rather than the stored percentile interval: resampling with
# replacement biases the raw percentile CI for this angle-of-mean ratio, so it
# does not bracket the point estimate; the bootstrap SE does not carry that
# location bias, giving an honest band centered on the plotted R.
Z95 = 1.959963984540054


def _boot_ratio_se(root, group, model):
    """{rel_depth: se} for the aggregated resolvability ratio R at each layer.

    Reads the "ratio" boot_std from results/bootstrap_ci_<group>_<model>.json and
    maps each layer index to its relative depth (layer / n_layers, with n_layers
    taken from the matching kappa file), matching how ``_series`` builds its
    x-axis. Returns {} when either file is missing.
    """
    boot_path = os.path.join(root, "results", f"bootstrap_ci_{group}_{model}.json")
    kappa_path = os.path.join(root, "results", f"{group}_kappa_{model}.json")
    if not (os.path.exists(boot_path) and os.path.exists(kappa_path)):
        return {}
    with open(boot_path) as fh:
        b = json.load(fh)
    with open(kappa_path) as fh:
        k = json.load(fh)
    sh = k.get("split_half", {})
    n_layers = k.get("n_layers") or (max(int(l) for l in sh) + 1 if sh else 1)
    return {int(L) / n_layers: float(lay["ratio"]["boot_std"])
            for L, lay in b.get("layers", {}).items() if "ratio" in lay}


def _series(path):
    """Return (rel_depth, ratio) arrays for one file, or None.

    ratio = mean(sin_pred_half) / mean(sin_floor_mean) over the mixed ratios at
    each swept layer -- i.e. the size-matched (half-sample) residual over the
    split-half floor, the fair SNR.
    """
    with open(path) as fh:
        data = json.load(fh)
    sh = data.get("split_half", {})
    if not sh:
        return None
    n_layers = data.get("n_layers") or (max(int(l) for l in sh) + 1)
    layers = sorted(int(l) for l in sh)
    rel, ratio = [], []
    for L in layers:
        pr = sh[str(L)]["per_ratio"]
        sph = np.array([v["sin_pred_half"] for v in pr.values()], dtype=float)
        fl = np.array([v["sin_floor_mean"] for v in pr.values()], dtype=float)
        num = float(np.nanmean(sph))
        den = float(np.nanmean(fl))
        if not (den > 0):
            continue
        rel.append(L / n_layers)
        ratio.append(num / den)
    if len(rel) < 2:
        return None
    return np.array(rel), np.array(ratio)


def _interp_to_grid(rel, y):
    """Linear-interpolate a per-model curve onto GRID, NaN outside its support."""
    out = np.interp(GRID, rel, y, left=np.nan, right=np.nan)
    out[(GRID < rel.min()) | (GRID > rel.max())] = np.nan
    return out


def _aggregate(series):
    """Mean and across-model std of a list of (rel, ratio) curves on GRID.

    Returns (mask, mean, std) all on GRID; mask marks grid points covered by at
    least one model. std is only meaningful where >=2 models overlap.
    """
    stack = np.vstack([_interp_to_grid(rel, r) for rel, r in series])
    n_ok = np.sum(~np.isnan(stack), axis=0)
    mask = n_ok >= 1
    mean = np.full(GRID.shape, np.nan)
    std = np.full(GRID.shape, np.nan)
    mean[mask] = np.nanmean(stack[:, mask], axis=0)
    std[mask] = np.nanstd(stack[:, mask], axis=0)
    band = np.where(n_ok >= 2, std, 0.0)
    return mask, mean, band


def _collect(pattern):
    """pattern -> {group: {model: (rel, ratio)}}, keeping model identity."""
    cells = {g: {} for g in GROUP_ORDER}
    for p in glob.glob(pattern):
        m = _FNAME.search(os.path.basename(p))
        if not m or m.group("group") not in cells:
            continue
        s = _series(p)
        if s is not None:
            cells[m.group("group")][m.group("model")] = s
    return cells


def _plot_aggregate(cells, out, per_model, show):
    """Single panel: one aggregate line per group (optionally faint per-model)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(1.0, color="0.4", lw=1.0, ls=":", zorder=1)
    ax.text(0.012, 1.0, "noise floor (R = 1)", color="0.35", fontsize=7.5,
            va="bottom", ha="left")

    for g in GROUP_ORDER:
        series = list(cells[g].values())
        if not series:
            continue
        c = GROUP_COLOR[g]
        if per_model:
            for rel, r in series:
                ax.plot(rel, r, color=c, lw=0.8, alpha=0.25, zorder=2)
        mask, mean, band = _aggregate(series)
        ax.fill_between(GRID[mask], (mean - band)[mask], (mean + band)[mask],
                        color=c, alpha=0.15, lw=0, zorder=3)
        ax.plot(GRID[mask], mean[mask], "-o", color=c, ms=4, lw=2.0, zorder=4,
                label=f"{GROUP_TITLE[g]}  (n={len(series)})")

    ax.set_xlabel("relative layer depth")
    ax.set_ylabel(r"resolvability  $R = \tilde{s}^{\,l}_r / \phi^{\,l}_r$"
                  "\n(half-sample residual / noise floor)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9,
              title="aggregated across models")
    fig.text(0.5, -0.03,
             "Line: mean over models of the size-matched residual-to-floor ratio "
             "at matched relative depth; band: across-model std.  R>1 is signal "
             "above the sampling-noise floor; R=1 is the null.",
             ha="center", fontsize=7.5, wrap=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_facet(cells, out, show, root):
    """Small multiples: one panel per group, per-model lines + bold aggregate."""
    fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(13, 4.2), sharey=True)
    # shared top from the data so tall Entity peaks are not clipped
    ymax = max((r.max() for g in GROUP_ORDER for _, r in cells[g].values()),
               default=2.0)
    for ax, g in zip(axes, GROUP_ORDER):
        ax.axhline(1.0, color="0.4", lw=1.0, ls=":", zorder=1)
        models = [m for m in MODEL_ORDER if m in cells[g]]
        for model in models:
            rel, r = cells[g][model]
            c = MODEL_COLOR.get(model)
            # 95% bootstrap-SE band on the resolvability ratio, when available.
            se_map = _boot_ratio_se(root, g, model)
            if se_map:
                se = np.array([se_map.get(x, np.nan) for x in rel])
                ax.fill_between(rel, np.clip(r - Z95 * se, 0.0, None),
                                r + Z95 * se, color=c, alpha=0.13, lw=0, zorder=2)
            ax.plot(rel, r, "-o", color=c, ms=3, lw=1.3,
                    alpha=0.9, zorder=3, label=MODEL_TITLE.get(model, model))
        series = [cells[g][m] for m in models]
        if series:
            mask, mean, _ = _aggregate(series)
            ax.plot(GRID[mask], mean[mask], color="0.15", lw=2.6, zorder=4,
                    label="across-model mean")
        ax.set_title(GROUP_TITLE[g])
        ax.set_xlabel("relative layer depth")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax * 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"resolvability  $R = \tilde{s}^{\,l}_r / \phi^{\,l}_r$")
    axes[0].text(0.012, 1.0, "R = 1", color="0.35", fontsize=7.5,
                 va="bottom", ha="left")
    fig.tight_layout()

    # One shared legend for all three panels (identical model set + the
    # across-model mean), placed below the axes so it never overlaps the curves.
    handles, labels, seen = [], [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=len(labels), fontsize=9, framealpha=0.9)

    fig.text(0.5, -0.06,
             "Per-model size-matched residual-to-floor ratio (half-sample "
             "residual / split-half noise floor) vs. relative depth; shaded: 95% "
             "bootstrap-SE band; bold: across-model mean.  R>1 is resolvable "
             "signal above the floor; R=1 (dotted) is the null.",
             ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/*_kappa_*.json")
    ap.add_argument("--out", default="results/orthogonal_residual_ratio_depth.png")
    ap.add_argument("--facet", action="store_true",
                    help="also write the faceted per-model plot "
                         "(<out stem>_by_model.png)")
    ap.add_argument("--per-model", action="store_true",
                    help="draw faint per-model curves under the aggregate line")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = args.glob if os.path.isabs(args.glob) else os.path.join(root, args.glob)
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    cells = _collect(pattern)          # {group: {model: (rel, ratio)}}
    if not any(cells[g] for g in GROUP_ORDER):
        print(f"No result files with split-half data matched {pattern!r}.")
        return

    _plot_aggregate(cells, out, args.per_model, args.show)
    if args.facet:
        stem, ext = os.path.splitext(out)
        _plot_facet(cells, f"{stem}_by_model{ext}", args.show, root)


if __name__ == "__main__":
    main()
