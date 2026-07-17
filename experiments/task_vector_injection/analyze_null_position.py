"""Locate the residual-removal divergence within the random-orthogonal null.

Handles two run layouts, both comparing --- always w.r.t. the *clean few-shot*
logits --- the ablation ``KL_test`` against the random norm-matched orthogonal
null ``KL_null``:

  A. nulldist_<group>_<tag>.npz  (null_distribution.py, centroid geometry)
       test_kl : (R, L)          one ablation number per (ratio, layer)
       null_kl : (R, L, D)       D draws, each already a mean over questions
     -> per (ratio, layer) we place the single test value inside its D-point null.

  B. <group>_<tag>.npz           (inject_orthogonal_projection.py --mode both)
       test_kl : (S, R, L)  or   test_logits (S,R,L,V)
       null_kl : (S, R, L, D) or null_logits (S,R,L,V)   [live per-question geometry]
     -> per cell we place KL_test in that cell's own D-draw null, then average
        over samples per (ratio, layer).

Reported per (ratio, layer) and pooled per layer: mean KL_test, null mean/std,
z-score, percentile of test inside the null, and the paired ``test<null`` fraction.

Outputs:
  nullpos_<group>_<tag>.json

Usage:
  env/bin/python experiments/task_vector_injection/analyze_null_position.py \
      --group arithmetic --stem nulldist
"""
import argparse, json, os
import numpy as np

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))


def _logsoftmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _kl(log_p, log_q):
    return (np.exp(log_p) * (log_p - log_q)).sum(-1)


def _kls_from_logits(fs, arm):                     # fs:(S,R,V) arm:(S,R,L,V)->(S,R,L)
    S, R, L, V = arm.shape
    lf = _logsoftmax(fs.astype(np.float32))
    out = np.empty((S, R, L))
    for i in range(S):
        for j in range(R):
            for k in range(L):
                out[i, j, k] = _kl(lf[i, j], _logsoftmax(arm[i, j, k].astype(np.float32)))
    return out


def _stats(kl_test, kl_null):
    """kl_test broadcasts over the trailing draw axis of kl_null (…,D)."""
    n_mean = kl_null.mean(-1)
    n_std  = kl_null.std(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z    = (kl_test - n_mean) / n_std
        pct  = 100.0 * np.mean(kl_null < kl_test[..., None], axis=-1)
        frac = 100.0 * np.mean(kl_test[..., None] < kl_null, axis=-1)
    return n_mean, n_std, z, pct, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--model-tag", default="Llama-3B")
    ap.add_argument("--stem", default="", help="Filename prefix, e.g. 'nulldist' -> "
                    "nulldist_<group>_<tag>.npz. Empty = <group>_<tag>.npz.")
    ap.add_argument("--dir", default=os.path.join(_root, "causal_projection"))
    args = ap.parse_args()

    pre  = f"{args.stem}_" if args.stem else ""
    stem = os.path.join(args.dir, f"{pre}{args.group}_{args.model_tag}")
    npz  = np.load(stem + ".npz")
    files = set(npz.files)
    layers   = list(map(int, npz["layers"]))
    ratios   = list(map(str, npz["ratios"]))
    meta = json.load(open(stem + ".json"))
    n_layers = meta["n_layers"]
    depth    = [l / n_layers for l in layers]

    # ---- assemble kl_test (…,) and kl_null (…, D) with a leading sample axis or not ----
    if "test_kl" in files and npz["test_kl"].ndim == 2:
        layout, sample_axis = "centroid", False          # (R,L) / (R,L,D)
        kl_test = npz["test_kl"].astype(np.float64)       # (R,L)
        kl_null = npz["null_kl"].astype(np.float64)       # (R,L,D)
    else:
        layout, sample_axis = "per-cell", True            # (S,R,L) / (S,R,L,D)
        kl_test = (npz["test_kl"].astype(np.float64) if "test_kl" in files
                   else _kls_from_logits(npz["fewshot_logits"], npz["test_logits"]))
        if "null_kl" in files:
            kl_null = npz["null_kl"].astype(np.float64)
        elif "null_logits" in files:
            kl_null = _kls_from_logits(npz["fewshot_logits"], npz["null_logits"])[..., None]
        else:
            raise SystemExit("run has no null arm.")

    # optional full-ICL-removal reference: header replaced by the cached zero-shot header
    kl_zs = npz["zs_kl"].astype(np.float64) if "zs_kl" in files else None   # (R,L), centroid only

    n_draws = kl_null.shape[-1]
    n_mean, n_std, z, pct, frac = _stats(kl_test, kl_null)
    valid = np.isfinite(kl_test) & np.isfinite(n_mean)

    def _agg(a, mask):
        return float(np.nanmean(np.where(mask, a, np.nan))) if np.any(mask) else float("nan")

    # index helpers so both layouts share the reporting code
    R, L = len(ratios), len(layers)
    def cell(arr, j, k):     # -> array over samples (or 0-d) for (ratio j, layer k)
        return arr[:, j, k] if sample_axis else arr[j, k]

    per_ratio = []
    for j, rat in enumerate(ratios):
        rows = []
        for k, l in enumerate(layers):
            mk = cell(valid, j, k)
            rows.append({
                "depth": round(depth[k], 4), "layer": l,
                "mean_kl_test": _agg(cell(kl_test, j, k), mk),
                "mean_kl_zs":   (float(kl_zs[j, k]) if kl_zs is not None else None),
                "null_mean":    _agg(cell(n_mean, j, k), mk),
                "null_std":     _agg(cell(n_std, j, k), mk),
                "z":            _agg(cell(z, j, k), mk),
                "pctile":       _agg(cell(pct, j, k), mk),
                "frac_test_lt_null": _agg(cell(frac, j, k), mk),
            })
        per_ratio.append({"ratio": rat, "per_layer": rows})

    pooled = []
    for k, l in enumerate(layers):
        mk = valid[:, :, k] if sample_axis else valid[:, k]
        kt = kl_test[:, :, k] if sample_axis else kl_test[:, k]
        nm = n_mean[:, :, k] if sample_axis else n_mean[:, k]
        mt = _agg(kt, mk); mn = _agg(nm, mk)
        mz = float(np.nanmean(kl_zs[:, k])) if kl_zs is not None else None
        pooled.append({
            "depth": round(depth[k], 4), "layer": l,
            "mean_kl_test": mt, "mean_kl_null": mn, "mean_kl_zs": mz,
            "null_over_test": (mn / mt if mt > 0 else None),
            "test_over_zs": (mt / mz if (mz is not None and mz > 0) else None),
            "z":      _agg(z[:, :, k] if sample_axis else z[:, k], mk),
            "pctile": _agg(pct[:, :, k] if sample_axis else pct[:, k], mk),
            "frac_test_lt_null": _agg(frac[:, :, k] if sample_axis else frac[:, k], mk),
        })

    out = {
        "group": args.group, "model": meta["model"], "model_tag": args.model_tag,
        "layout": layout, "n_layers": n_layers, "n_ratios": R, "n_draws": n_draws,
        "n_samples": (int(kl_test.shape[0]) if sample_axis else meta.get("n_samples")),
        "seed": meta.get("seed", meta.get("null_seed")),
        "reference": "both KL divergences w.r.t. clean few-shot logits",
        "reduction": ("centroid: test scalar located in D-draw null per (ratio,layer)"
                      if layout == "centroid"
                      else "per cell: test in its D-draw null, then mean over samples"),
        "columns": {
            "z": "(KL_test - null_mean)/null_std",
            "pctile": "percentile of KL_test inside the null (0=below all draws)",
            "frac_test_lt_null": "% of null draws larger than KL_test",
        },
        "per_ratio": per_ratio, "pooled_over_ratios_per_layer": pooled,
    }
    if "null_resid_cos" in files:
        out["null_resid_cos_absmean"] = float(np.nanmean(np.abs(npz["null_resid_cos"])))

    dst = os.path.join(args.dir, f"nullpos_{args.group}_{args.model_tag}.json")
    json.dump(out, open(dst, "w"), indent=2)

    # ---- report ----
    print(f"[{args.group} / {args.model_tag}]  layout={layout}  R={R} L={L}  D_null={n_draws}"
          + (f"   |cos(g,resid)|~{out['null_resid_cos_absmean']:.3f}"
             if "null_resid_cos_absmean" in out else ""))
    if n_draws == 1:
        print("  (single-draw run: per-cell null is one point; z/pctile degenerate.)")
    head = (f"{'depth':>6}{'layer':>6}{'KL_test':>10}{'null_mean':>10}{'null_std':>9}"
            f"{'z':>7}{'pctile':>8}{'test<null%':>11}")
    for blk in per_ratio:
        print(f"\nratio {blk['ratio']}")
        print(head)
        for r in blk["per_layer"]:
            print(f"{r['depth']:6.2f}{r['layer']:6d}{r['mean_kl_test']:10.3f}{r['null_mean']:10.3f}"
                  f"{r['null_std']:9.3f}{r['z']:7.2f}{r['pctile']:7.0f}%{r['frac_test_lt_null']:10.0f}%")
    print("\npooled over ratios, per layer:")
    _zc = "" if kl_zs is None else f"{'KL_zs':>9}{'test/zs':>9}"
    print(f"{'depth':>6}{'layer':>6}{'KL_test':>10}{'KL_null':>10}{'null/test':>10}"
          f"{'z':>7}{'pctile':>8}{'test<null%':>11}" + _zc)
    for r in pooled:
        nt = r["null_over_test"]
        line = (f"{r['depth']:6.2f}{r['layer']:6d}{r['mean_kl_test']:10.3f}{r['mean_kl_null']:10.3f}"
                f"{(nt if nt is not None else float('nan')):10.2f}{r['z']:7.2f}"
                f"{r['pctile']:7.0f}%{r['frac_test_lt_null']:10.0f}%")
        if kl_zs is not None:
            tz = r["test_over_zs"]
            line += f"{r['mean_kl_zs']:9.3f}{(tz if tz is not None else float('nan')):9.2f}"
        print(line)
    print(f"\nstored -> {os.path.relpath(dst, _root)}")


if __name__ == "__main__":
    main()
