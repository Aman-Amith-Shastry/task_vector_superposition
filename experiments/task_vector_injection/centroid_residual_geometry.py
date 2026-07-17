"""Centroid-level orthogonal residual geometry, for the residual-ablation section.

The systematic (ratio-level) residual: for each mixed ratio r and layer l we take the
*centroid* of the per-question contrasts, decompose its displacement from the tri-center
against the convex prediction, and keep the orthogonal part

    d_c      = mean_i c_r^l(x_i) - mu^l
    v_perp,c = d_c - kappa_c * d_hat_r^l ,    kappa_c = <d_c, d_hat> / <d_hat, d_hat>

This is the vector that ``null_distribution.py`` ablates, and it differs from the
per-question residual v_perp(x_i) = c_r^l(x_i) - mu^l - kappa_i * d_hat: the per-question
version is 3-17x larger because it carries question-specific idiosyncrasy that averages
out of the centroid. Only the centroid part is a systematic deviation from the convex
prediction, so it is what a linearity claim should be tested on.

Centroids are fit with the first EVAL_HOLDOUT questions withheld -- those are the ones
null_distribution.py ablates, and including them would let v_perp,c carry a slice of the
residual it is subtracted from. Reported rho_c therefore describes the same held-out
vector that gets causally tested.

Pure numpy over cached samples -- no model load, so MMLU is available here
even though its causal null could not be run.

Reads:  ratio_samples/<group>_<tag>.npz,
        causal_projection/relmag_<group>_<tag>.json   (for ||h||)
Writes: causal_projection/centroid_geom_<tag>.json
"""
import json, os
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUPS      = ["arithmetic", "entity", "mmlu"]
PURE_RATIOS = [(3, 0, 0), (0, 3, 0), (0, 0, 3)]
EVAL_HOLDOUT = 10          # = null_distribution.py --samples; questions 0..9 are ablated


def _mixed_ratios(group: str):
    import importlib, sys
    sys.path.insert(0, os.path.join(_root, "data"))
    mod = {"arithmetic": "data_arithmetic_formats",
           "entity":     "data_entity_attribute",
           "mmlu":       "data_semantic_domains"}[group]
    d = importlib.import_module(mod)

    def _degenerate(r):
        w = np.array(r, dtype=float) / sum(r)
        return np.allclose(w, w.mean())

    return [r for r in d.RATIOS if r not in d.PURE_RATIOS and not _degenerate(r)]


def geometry(group: str, tag: str):
    samp = np.load(os.path.join(_root, "ratio_samples",   f"{group}_{tag}.npz"))
    rel  = json.load(open(os.path.join(_root, "causal_projection", f"relmag_{group}_{tag}.json")))

    layers   = rel["layers"]
    n_layers = rel["n_layers"]
    h_norm   = {r["layer"]: r["h_norm"] for r in rel["pooled_over_ratios_per_layer"]}
    mixed    = _mixed_ratios(group)
    fit      = slice(EVAL_HOLDOUT, None)

    out = []
    for l in layers:
        pure = {p: samp[f"{l}|{'-'.join(map(str, p))}"][fit].astype(np.float64).mean(0)
                for p in PURE_RATIOS}
        m    = np.mean([pure[p] for p in PURE_RATIOS], axis=0)

        rn, kaps = [], []
        for r in mixed:
            total  = float(sum(r))
            c_pred = np.sum([(n / total) * pure[p] for n, p in zip(r, PURE_RATIOS)], axis=0)
            d_pred = c_pred - m
            denom  = float(d_pred @ d_pred)
            c_act  = samp[f"{l}|{'-'.join(map(str, r))}"][fit].astype(np.float64).mean(0)
            d_act  = c_act - m
            kap    = float(d_act @ d_pred) / denom
            resid  = d_act - kap * d_pred
            rn.append(float(np.linalg.norm(resid)))
            kaps.append(kap)

        out.append({
            "layer": l, "depth": round(l / n_layers, 4),
            "h_norm": h_norm[l],
            "vperp_c_norm": float(np.mean(rn)),
            "rho_c": float(np.mean(rn) / h_norm[l]),
            "kappa_c": float(np.mean(kaps)),
        })
    return out, n_layers


def main():
    tag = "Llama-3B"
    res = {}
    for g in GROUPS:
        res[g], n_layers = geometry(g, tag)

    dst = os.path.join(_root, "causal_projection", f"centroid_geom_{tag}.json")
    json.dump({
        "model_tag": tag, "n_layers": n_layers,
        "definitions": {
            "vperp_c_norm": "||v_perp,c||, orthogonal part of the ratio-centroid displacement",
            "rho_c": "||v_perp,c|| / ||h||, ambient-relative magnitude of the systematic residual",
            "kappa_c": "projection coefficient of the centroid displacement onto the convex prediction",
            "h_norm": "||h_l(p_r(x))|| few-shot boundary activation norm (from relmag run)",
        },
        "per_group": res,
    }, open(dst, "w"), indent=2)

    print(f"{'depth':>6}" + "".join(f"{g:>24}" for g in GROUPS))
    print(f"{'':>6}" + "".join(f"{'rho_c':>8}{'|v|':>8}{'kap_c':>8}" for _ in GROUPS))
    for k in range(len(res["arithmetic"])):
        line = f"{res['arithmetic'][k]['depth']:6.2f}"
        for g in GROUPS:
            r = res[g][k]
            line += f"{r['rho_c']:8.3f}{r['vperp_c_norm']:8.2f}{r['kappa_c']:8.2f}"
        print(line)
    print(f"\nstored -> {os.path.relpath(dst, _root)}")


if __name__ == "__main__":
    main()
