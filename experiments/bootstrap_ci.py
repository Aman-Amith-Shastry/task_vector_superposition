"""
Bootstrap 95% confidence intervals for the orthogonal-residual quantities.

For every mixed ratio separately, we resample its stored contrast vectors with
replacement (n = the number of samples stored for that task group) and recompute
the four quantities calibrated in Appendix B/C:

  * full  : full-sample residual  sin θ(d_act, d_pred), d_act = μ_r − center
  * half  : size-matched (n/2 half-sample) residual, mean over inner split-halves
  * floor : split-half sampling-noise floor, mean over inner split-halves
  * ratio : half / floor

The geometry (center, pure-task centroids, and hence the predicted displacement
d_pred) is held at its full-sample value throughout, exactly as in
``kappa_null.split_half_consistency`` — only the mixed ratio's own centroid is
re-estimated, so the interval reflects the sampling variance of that centroid.

Each bootstrap iteration b:
  1. Resample n vectors with replacement from the ratio's vectors  -> X*
  2. Compute the ratio centroid  μ* = mean(X*)
  3. Compute the predicted displacement d_pred (pures fixed at full sample)
  4. Compute the ratio's residual(s): full on μ*, and floor/half from an inner
     split-half over X*.

Percentile CIs are taken over the n_boot iterations. Results are reported both
per ratio and aggregated (mean over ratios) per layer, mirroring the split-half
tables. Raw bootstrap draws are also saved (.npz) so intervals can be recomputed
without rerunning. No efficiency compression is applied.

Usage:
    python experiments/bootstrap_ci.py                 # all groups x models
    python experiments/bootstrap_ci.py --groups arithmetic --models Gemma-2B
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kappa_null import _sin_angle  # identical geometry as the point estimates

PURE_RATIOS = [(3, 0, 0), (0, 3, 0), (0, 0, 3)]
SCALE_FLOOR = 0.05  # skip a mixed ratio when ||d_pred|| <= SCALE_FLOOR * mean pure radius

GROUPS = ["arithmetic", "entity", "mmlu"]
MODELS = ["Llama-8B", "Llama-3B", "Qwen-3B", "Gemma-2B", "Llama-1B"]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ci_block(point, draws, ci):
    """Summary dict {point, boot_mean, boot_std, ci_lo, ci_hi} for a draw vector."""
    lo, hi = np.nanpercentile(draws, ci)
    return {
        "point": float(point),
        "boot_mean": float(np.nanmean(draws)),
        "boot_std": float(np.nanstd(draws)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def _d_pred(rt, pure_c, center):
    total = float(sum(rt))
    return np.sum([(k / total) * pure_c[p] for k, p in zip(rt, PURE_RATIOS)], axis=0) - center


def _sin_rows(A, B):
    """Row-wise sin theta between A (S, r) and B (r,) or (S, r); nan on ~0 rows.

    Vectorized twin of ``kappa_null._sin_angle`` (same sqrt(1 - cos^2), same
    1e-12 zero-norm guard), applied to a stack of displacement vectors at once.
    """
    A = np.asarray(A, dtype=float)
    B = np.broadcast_to(np.asarray(B, dtype=float), A.shape)
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    dot = np.einsum("ij,ij->i", A, B)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = dot / (na * nb)
    sin = np.sqrt(np.clip(1.0 - cos ** 2, 0.0, 1.0))
    sin[(na < 1e-12) | (nb < 1e-12)] = np.nan
    return sin


def _residuals_from_sample(X, d_pred, center, rng, inner_splits):
    """Full residual (on the whole sample) and mean floor / half over inner splits.

    The inner split-half loop is vectorized: we draw ``inner_splits`` independent
    random equipartitions at once and form each half-mean as a sparse weighted sum
    ``W @ X``, so the calibration is a couple of matmuls rather than a Python loop.
    Statistically identical to the per-split loop; only the RNG draw order differs.
    """
    n = X.shape[0]
    half = n // 2
    S = inner_splits
    full = _sin_angle(X.mean(0) - center, d_pred)

    perm = np.argsort(rng.random((S, n)), axis=1)     # S independent permutations
    rows = np.arange(S)[:, None]
    Wa = np.zeros((S, n)); Wa[rows, perm[:, :half]] = 1.0 / half
    Wb = np.zeros((S, n)); Wb[rows, perm[:, half:2 * half]] = 1.0 / half
    A = Wa @ X - center                                # (S, r) half-A displacements
    B = Wb @ X - center                                # (S, r) half-B displacements

    floor = _sin_rows(A, B)
    half_pred = np.concatenate([_sin_rows(A, d_pred), _sin_rows(B, d_pred)])
    return full, float(np.nanmean(floor)), float(np.nanmean(half_pred))


def _bootstrap_ratio(X, d_pred, center, rng, n_boot, inner_splits):
    """Point estimate + per-iteration bootstrap draws for one mixed ratio.

    Every bootstrap resample is a subset (with repetition) of X's rows, so all
    displacements live in the affine subspace spanned by {X rows, center, d_pred}.
    We project into an orthonormal basis of that <= n+2 dim space once up front:
    angles are exactly preserved (orthonormal columns preserve norms and dots),
    but the inner split-half calibration then runs in ~n dims instead of the full
    hidden size, which is what makes many inner splits affordable.
    """
    n = X.shape[0]
    M = np.column_stack([X.T, center[:, None], d_pred[:, None]])  # (dim, n+2)
    Q, _ = np.linalg.qr(M)                                        # (dim, r) orthonormal
    X = X @ Q                                                     # project into span
    center = center @ Q
    d_pred = d_pred @ Q

    full_pt, floor_pt, half_pt = _residuals_from_sample(X, d_pred, center, rng, inner_splits)

    full_b = np.empty(n_boot, dtype=float)
    floor_b = np.empty(n_boot, dtype=float)
    half_b = np.empty(n_boot, dtype=float)
    for bi in range(n_boot):
        idx = rng.integers(0, n, size=n)              # 1. resample n with replacement
        Xb = X[idx]                                    # 2./3./4. residuals on the resample
        f, fl, hp = _residuals_from_sample(Xb, d_pred, center, rng, inner_splits)
        full_b[bi], floor_b[bi], half_b[bi] = f, fl, hp
    ratio_b = np.divide(half_b, floor_b, out=np.full(n_boot, np.nan), where=floor_b > 0)
    ratio_pt = half_pt / floor_pt if floor_pt > 0 else float("nan")

    return {
        "point": {"full": full_pt, "half": half_pt, "floor": floor_pt, "ratio": ratio_pt},
        "draws": {"full": full_b, "half": half_b, "floor": floor_b, "ratio": ratio_b},
    }


def process_pair(group, model, n_boot, inner_splits, seed, ci):
    path = os.path.join(_ROOT, "ratio_samples", f"{group}_{model}.npz")
    if not os.path.exists(path):
        print(f"  [skip] missing {path}")
        return None
    data = np.load(path)

    # key format "{layer}|{a-b-c}" -> layers[layer][ratio_tuple] = X (n, dim)
    layers = {}
    for key in data.files:
        lay_s, rat_s = key.split("|")
        rt = tuple(int(x) for x in rat_s.split("-"))
        layers.setdefault(int(lay_s), {})[rt] = np.asarray(data[key], dtype=float)

    rng = np.random.default_rng(seed)
    out_layers = {}
    draws_npz = {}

    for layer in sorted(layers):
        samples = layers[layer]
        pure_c = {p: samples[p].mean(0) for p in PURE_RATIOS}
        center = np.mean([pure_c[p] for p in PURE_RATIOS], axis=0)
        scale = float(np.mean([np.linalg.norm(pure_c[p] - center) for p in PURE_RATIOS]))

        per_ratio = {}
        # aligned per-iteration draws, one column per mixed ratio, for aggregation
        agg = {k: [] for k in ("full", "half", "floor")}
        n_samples = None
        for rt, X in samples.items():
            if rt in PURE_RATIOS:
                continue
            d_pred = _d_pred(rt, pure_c, center)
            if np.linalg.norm(d_pred) <= SCALE_FLOOR * scale:
                continue  # undefined prediction (e.g. 1-1-1)
            n_samples = X.shape[0]
            res = _bootstrap_ratio(X, d_pred, center, rng, n_boot, inner_splits)
            rstr = "-".join(map(str, rt))
            per_ratio[rstr] = {
                stat: _ci_block(res["point"][stat], res["draws"][stat], ci)
                for stat in ("full", "half", "floor", "ratio")
            }
            for stat in ("full", "half", "floor"):
                agg[stat].append(res["draws"][stat])
            for stat in ("full", "half", "floor", "ratio"):
                draws_npz[f"{layer}|{rstr}|{stat}"] = res["draws"][stat].astype(np.float32)

        if not per_ratio:
            continue

        # aggregate over ratios: mean of the per-ratio draws at each iteration
        full_agg = np.mean(agg["full"], axis=0)
        half_agg = np.mean(agg["half"], axis=0)
        floor_agg = np.mean(agg["floor"], axis=0)
        ratio_agg = np.divide(half_agg, floor_agg,
                              out=np.full_like(half_agg, np.nan), where=floor_agg > 0)
        full_pt = float(np.mean([per_ratio[r]["full"]["point"] for r in per_ratio]))
        half_pt = float(np.mean([per_ratio[r]["half"]["point"] for r in per_ratio]))
        floor_pt = float(np.mean([per_ratio[r]["floor"]["point"] for r in per_ratio]))
        ratio_pt = half_pt / floor_pt if floor_pt > 0 else float("nan")

        out_layers[str(layer)] = {
            "n_samples": int(n_samples),
            "n_mixed": len(per_ratio),
            "full": _ci_block(full_pt, full_agg, ci),
            "half": _ci_block(half_pt, half_agg, ci),
            "floor": _ci_block(floor_pt, floor_agg, ci),
            "ratio": _ci_block(ratio_pt, ratio_agg, ci),
            "per_ratio": per_ratio,
        }
        for stat, arr in (("full", full_agg), ("half", half_agg),
                          ("floor", floor_agg), ("ratio", ratio_agg)):
            draws_npz[f"{layer}|{stat}"] = arr.astype(np.float32)

        print(f"    layer {layer:2d}: full {full_pt:.3f} [{out_layers[str(layer)]['full']['ci_lo']:.3f}, "
              f"{out_layers[str(layer)]['full']['ci_hi']:.3f}]  "
              f"floor {floor_pt:.3f} [{out_layers[str(layer)]['floor']['ci_lo']:.3f}, "
              f"{out_layers[str(layer)]['floor']['ci_hi']:.3f}]  "
              f"ratio {ratio_pt:.3f} [{out_layers[str(layer)]['ratio']['ci_lo']:.3f}, "
              f"{out_layers[str(layer)]['ratio']['ci_hi']:.3f}]")

    result = {
        "model": model,
        "group": group,
        "n_boot": n_boot,
        "inner_splits": inner_splits,
        "ci_percentiles": list(ci),
        "seed": seed,
        "layers": out_layers,
    }
    out_json = os.path.join(_ROOT, "results", f"bootstrap_ci_{group}_{model}.json")
    out_npz = os.path.join(_ROOT, "results", f"bootstrap_ci_{group}_{model}.npz")
    with open(out_json, "w") as fo:
        json.dump(result, fo, indent=2)
    np.savez(out_npz, **draws_npz)
    print(f"  saved {os.path.relpath(out_json, _ROOT)} and .npz")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=GROUPS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--inner-splits", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ci", type=float, nargs=2, default=(2.5, 97.5))
    args = ap.parse_args()

    for group in args.groups:
        for model in args.models:
            print(f"[{group} / {model}]")
            process_pair(group, model, args.n_boot, args.inner_splits, args.seed, tuple(args.ci))


if __name__ == "__main__":
    main()
