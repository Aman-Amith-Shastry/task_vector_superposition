"""
Aggregate the split-half / orthogonal-residual results across every
results/<group>_kappa_<Model>.json into the numbers that populate the paper's
split-half table (tab:split_half), plus an optional per-layer grid.

Everything is recomputed from the per-ratio fields stored in each JSON
(``split_half[layer]["per_ratio"][ratio] = {sin_pred, sin_floor_mean,
sin_pred_half, ...}``) and the per-layer ``kappa`` lists, so runs produced
*before* the size-matched ``within_floor`` change are re-scored under the new
definition without re-running the sweeps.

Two within-floor definitions are reported per cell:

  within_floor       size-matched: mean sin_pred_half <= mean sin_floor  (n/2 vs
                     n/2, the fair, reported test)
  within_floor_full  best case: mean sin_pred (full n) <= mean sin_floor (n/2),
                     generous by construction

"early" / "deep" = mean over the two shallowest / two deepest swept layers, the
convention used in the table caption.

Usage
-----
    python aggregate_split_half.py                 # all results/*_kappa_*.json
    python aggregate_split_half.py --grid          # also dump the per-layer grid
    python aggregate_split_half.py --latex         # emit LaTeX table rows
    python aggregate_split_half.py --glob 'results/arithmetic_kappa_*.json'
"""

import argparse
import glob
import json
import os
import re

import numpy as np

# Filename convention: results/<group>_kappa_<Model>.json
_FNAME = re.compile(r"(?P<group>.+?)_kappa_(?P<model>.+)\.json$")


def _cell_from_json(path):
    """Recompute the per-layer and summary numbers for one results file."""
    with open(path) as fh:
        data = json.load(fh)

    kappa = data.get("kappa", {})
    split_half = data.get("split_half", {})
    if not split_half:
        return None

    layers = sorted(int(l) for l in split_half)
    per_layer = {}
    for L in layers:
        sh = split_half[str(L)]
        pr = sh["per_ratio"]
        sp = np.array([v["sin_pred"] for v in pr.values()], dtype=float)
        fl = np.array([v["sin_floor_mean"] for v in pr.values()], dtype=float)
        sph = np.array([v["sin_pred_half"] for v in pr.values()], dtype=float)

        sp_m = float(np.nanmean(sp))
        fl_m = float(np.nanmean(fl))
        sph_m = float(np.nanmean(sph))
        kappas = kappa.get(str(L))
        per_layer[L] = {
            "kappa_mean":        float(np.mean(kappas)) if kappas else float("nan"),
            "sin_pred_mean":     sp_m,
            "sin_pred_half_mean": sph_m,
            "sin_floor_mean":    fl_m,
            # reported (size-matched) and best-case (full-sample) flags
            "within_floor":      bool(sph_m <= fl_m),
            "within_floor_full": bool(sp_m <= fl_m),
        }

    n = len(layers)
    lo = layers[: min(2, n)]        # two shallowest swept layers
    hi = layers[-min(2, n):]        # two deepest swept layers

    def _avg(ls, key):
        return float(np.mean([per_layer[L][key] for L in ls]))

    kmeans = [per_layer[L]["kappa_mean"] for L in layers]
    return {
        "layers":       layers,
        "per_layer":    per_layer,
        "kappa_lo":     float(np.nanmin(kmeans)),
        "kappa_hi":     float(np.nanmax(kmeans)),
        "early_pred":   _avg(lo, "sin_pred_mean"),
        "early_pred_half": _avg(lo, "sin_pred_half_mean"),
        "early_floor":  _avg(lo, "sin_floor_mean"),
        "deep_pred":    _avg(hi, "sin_pred_mean"),
        "deep_pred_half": _avg(hi, "sin_pred_half_mean"),
        "deep_floor":   _avg(hi, "sin_floor_mean"),
        "within_count":      sum(per_layer[L]["within_floor"] for L in layers),
        "within_count_full": sum(per_layer[L]["within_floor_full"] for L in layers),
        "n_layers_swept":    n,
    }


def _collect(paths):
    cells = []
    for p in sorted(paths):
        m = _FNAME.search(os.path.basename(p))
        if not m:
            continue
        cell = _cell_from_json(p)
        if cell is None:
            continue
        cell["group"] = m.group("group")
        cell["model"] = m.group("model")
        cell["path"] = p
        cells.append(cell)
    return cells


def _print_summary(cells):
    hdr = (f"{'Group':<12} {'Model':<10} {'kappa range':>13}  "
           f"{'early sinP(floor)':>19}  {'deep sinP(floor)':>19}  "
           f"{'within/full':>12}")
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        early = f"{c['early_pred']:.2f}({c['early_floor']:.2f})"
        deep = f"{c['deep_pred']:.2f}({c['deep_floor']:.2f})"
        within = f"{c['within_count']}/{c['within_count_full']}/{c['n_layers_swept']}"
        print(f"{c['group']:<12} {c['model']:<10} "
              f"{c['kappa_lo']:.2f}-{c['kappa_hi']:.2f}".ljust(38)[:38]
              + f"  {early:>19}  {deep:>19}  {within:>12}")
    print("\nColumns: early/deep sinP = full-sample sin_pred (floor = split-half "
          "noise floor); within/full = size-matched within-floor / best-case "
          "within-floor / layers swept.")


def _print_grid(cells):
    for c in cells:
        print(f"\n=== {c['group']} / {c['model']}  ({c['path']}) ===")
        print(f"{'layer':>6} {'kappa':>7} {'sinP':>7} {'sinP_half':>10} "
              f"{'floor':>7} {'within':>7} {'w_full':>7}")
        for L in c["layers"]:
            r = c["per_layer"][L]
            print(f"{L:>6} {r['kappa_mean']:>7.3f} {r['sin_pred_mean']:>7.3f} "
                  f"{r['sin_pred_half_mean']:>10.3f} {r['sin_floor_mean']:>7.3f} "
                  f"{str(r['within_floor']):>7} {str(r['within_floor_full']):>7}")


def _print_latex(cells):
    print("% auto-generated split-half rows (aggregate_split_half.py --latex)")
    print("% cols: Group & Model & kappa range & early sin(floor) & "
          "deep sin(floor) & within-floor")
    print("% sin = full-sample residual (sin_pred), the minimum-variance magnitude "
          "estimate; floor = n/2 split-half noise floor shown for scale")
    print("% within = size-matched (n/2 vs n/2) resolvability count, sin_pred_half "
          "<= floor")
    for c in cells:
        early = f"{c['early_pred']:.2f}\\,({c['early_floor']:.2f})"
        deep = f"{c['deep_pred']:.2f}\\,({c['deep_floor']:.2f})"
        within = f"{c['within_count']}/{c['n_layers_swept']}"
        print(f"{c['group']} & {c['model']} & "
              f"{c['kappa_lo']:.2f}--{c['kappa_hi']:.2f} & "
              f"{early} & {deep} & {within} \\\\")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/*_kappa_*.json",
                    help="glob for result files (default: results/*_kappa_*.json)")
    ap.add_argument("--grid", action="store_true", help="also print per-layer grid")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table rows")
    args = ap.parse_args()

    # Resolve glob relative to the repo root, not the experiments/ dir.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = args.glob if os.path.isabs(args.glob) else os.path.join(root, args.glob)
    paths = glob.glob(pattern)
    cells = _collect(paths)
    if not cells:
        print(f"No result files with split-half data matched {pattern!r}.")
        return

    _print_summary(cells)
    if args.grid:
        _print_grid(cells)
    if args.latex:
        print()
        _print_latex(cells)


if __name__ == "__main__":
    main()
