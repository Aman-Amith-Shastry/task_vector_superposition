"""Relative residual magnitude at the injection point.

The null arm norm-matches the random vector to the true orthogonal residual
``v_perp = d_act - proj`` --- but "matched magnitude" is only meaningful relative
to the *ambient* residual-stream state that is being perturbed. So for each
(group, layer) we report the perturbation as a fraction of the ambient activation:

    ||h||        = mean over test questions (and ratios) of the norm of the actual
                   few-shot hidden state  h_l(p_r(x))  at the assistant-boundary
                   token --- the exact state the patch overwrites.
    ||v_perp||   = mean over the same cells of the orthogonal residual norm
                   |d_act - proj|  (stored as ``resid_norm`` by the injection run).
    rel          = ||v_perp|| / ||h||     (perturbation as a fraction of ambient)

We also carry the mean test-arm divergence KL(fs||test) per layer (residual
removed), so the relative magnitude and the behavioural effect sit side by side
and we can see whether the residual grows relative to the whole activation with
depth.

``||h||`` is recomputed here (it was never stored) by re-running the *same*
questions/ratios/layers as the on-disk injection run --- reproducible from the
seeds (``random.Random(i)``, ``sample_idx=i``, ``rotate_test=True``). We verify
the recomputed few-shot logits match the stored ones, so the prompts are identical.

Reads:  causal_projection/<group>_<tag>.npz   (fewshot_logits, test_logits, resid_norm)
Writes: causal_projection/relmag_<group>_<tag>.npz / .json

Usage:
  env/bin/python experiments/task_vector_injection/relative_residual_magnitude.py \
      --group arithmetic
"""
import os
import sys
import json
import random
import argparse

import numpy as np

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "data"))

import importlib
import inject_orthogonal_projection as inj

model, tokenizer, n_layers = inj.model, inj.tokenizer, inj.n_layers


def _logsoftmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="arithmetic", choices=list(inj._GROUPS))
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--quantize", default="", choices=["", "int8", "int4"])
    ap.add_argument("--dir", default=os.path.join(_root, "causal_projection"))
    args = ap.parse_args()

    model_tag = inj._make_model_tag(args.model)
    data = importlib.import_module(inj._GROUPS[args.group])
    PURE_RATIOS = data.PURE_RATIOS

    stem = os.path.join(args.dir, f"{args.group}_{model_tag}")
    npz  = np.load(stem + ".npz")
    meta = json.load(open(stem + ".json"))
    layers = list(map(int, npz["layers"]))
    ratio_strs = list(map(str, npz["ratios"]))
    S = int(meta["n_samples"])
    R, L = len(ratio_strs), len(layers)

    fs_logits  = npz["fewshot_logits"].astype(np.float32)     # (S,R,V)
    resid_norm = npz["resid_norm"].astype(np.float64)         # (S,R,L) = ||v_perp||
    have_test  = "test_logits" in set(npz.files)
    test_logits = npz["test_logits"].astype(np.float32) if have_test else None  # (S,R,L,V)

    # rebuild the mixed-ratio order exactly as the injection run did
    def _is_degenerate(r):
        w = np.array(r, dtype=float); w = w / w.sum()
        return np.allclose(w, w.mean())
    mixed_all = [r for r in data.RATIOS if r not in PURE_RATIOS and not _is_degenerate(r)]
    mixed = [r for r in mixed_all if "-".join(map(str, r)) in set(ratio_strs)]
    assert ["-".join(map(str, r)) for r in mixed] == ratio_strs, "ratio order mismatch"

    print(f"Model: {args.model}  ({model_tag}, {n_layers} layers)")
    print(f"Group: {args.group}   Layers: {layers}   S={S} R={R}")
    print("Loading pools...")
    pools = data.load_pools()

    h_norm = np.full((S, R, L), np.nan, np.float64)   # ||h_icl|| at the injection point
    fs_max_dlogit = 0.0                                # prompt-identity check

    for i in range(S):
        for j, ratio in enumerate(mixed):
            rng = random.Random(i)
            icl_msgs = data.build_messages(pools, rng, ratio, sample_idx=i, rotate_test=True)
            acts, lg = inj._capture(icl_msgs, layers)
            fs_max_dlogit = max(fs_max_dlogit,
                                float(np.max(np.abs(lg.astype(np.float32) - fs_logits[i, j]))))
            for k, l in enumerate(layers):
                h_norm[i, j, k] = float(np.linalg.norm(acts[l].astype(np.float64)))
        print(f"  captured question {i + 1}/{S}   (max|Δ fs-logit| so far = {fs_max_dlogit:.2e})")

    # test-arm KL(fs||test), per cell
    kl_test = np.full((S, R, L), np.nan, np.float64)
    if have_test:
        for i in range(S):
            for j in range(R):
                lf = _logsoftmax(fs_logits[i, j])
                for k in range(L):
                    lq = _logsoftmax(test_logits[i, j, k])
                    kl_test[i, j, k] = float((np.exp(lf) * (lf - lq)).sum())

    # per-cell relative magnitude, then aggregate to (group, layer)
    rel_cell = resid_norm / h_norm                                   # (S,R,L)

    depth = [l / n_layers for l in layers]
    # aggregate over questions AND ratios -> one row per layer
    h_layer      = np.nanmean(h_norm.reshape(-1, L), axis=0)         # ||h|| per layer
    vperp_layer  = np.nanmean(resid_norm.reshape(-1, L), axis=0)     # ||v_perp|| per layer
    rel_ratio_of_means = vperp_layer / h_layer                       # mean||v||/mean||h||
    rel_mean_of_ratios = np.nanmean(rel_cell.reshape(-1, L), axis=0) # mean(||v||/||h||)
    rel_std            = np.nanstd(rel_cell.reshape(-1, L), axis=0)
    kl_layer     = np.nanmean(kl_test.reshape(-1, L), axis=0) if have_test else np.full(L, np.nan)

    # also keep the per-ratio breakdown
    per_ratio = []
    for j, rat in enumerate(ratio_strs):
        rows = []
        for k, l in enumerate(layers):
            rows.append({
                "depth": round(depth[k], 4), "layer": l,
                "h_norm": float(np.nanmean(h_norm[:, j, k])),
                "vperp_norm": float(np.nanmean(resid_norm[:, j, k])),
                "rel": float(np.nanmean(rel_cell[:, j, k])),
                "mean_kl_test": float(np.nanmean(kl_test[:, j, k])) if have_test else None,
            })
        per_ratio.append({"ratio": rat, "per_layer": rows})

    pooled = []
    for k, l in enumerate(layers):
        pooled.append({
            "depth": round(depth[k], 4), "layer": l,
            "h_norm": float(h_layer[k]),
            "vperp_norm": float(vperp_layer[k]),
            "rel_ratio_of_means": float(rel_ratio_of_means[k]),
            "rel_mean_of_ratios": float(rel_mean_of_ratios[k]),
            "rel_std": float(rel_std[k]),
            "mean_kl_test": float(kl_layer[k]) if have_test else None,
        })

    out = {
        "group": args.group, "model": args.model, "model_tag": model_tag,
        "n_layers": n_layers, "layers": layers, "depth": depth,
        "ratios": ratio_strs, "n_samples": S, "n_ratios": R,
        "fs_logit_identity_max_abs_diff": fs_max_dlogit,
        "definitions": {
            "h_norm": "||h_l(p_r(x))|| ambient few-shot hidden state at assistant-boundary token, mean over questions(&ratios)",
            "vperp_norm": "||d_act - proj|| orthogonal residual norm (the null-matched perturbation)",
            "rel_ratio_of_means": "mean||v_perp|| / mean||h||  per layer",
            "rel_mean_of_ratios": "mean over cells of ||v_perp||/||h|| per layer",
            "mean_kl_test": "KL(fs||test) residual-removed divergence, mean over cells",
        },
        "pooled_over_ratios_per_layer": pooled,
        "per_ratio": per_ratio,
    }
    dst = os.path.join(args.dir, f"relmag_{args.group}_{model_tag}")
    json.dump(out, open(dst + ".json", "w"), indent=2)
    np.savez_compressed(
        dst + ".npz",
        layers=np.array(layers), depth=np.array(depth), ratios=np.array(ratio_strs),
        h_norm=h_norm.astype(np.float32), vperp_norm=resid_norm.astype(np.float32),
        rel_cell=rel_cell.astype(np.float32),
        kl_test=kl_test.astype(np.float32) if have_test else np.zeros(0, np.float32),
        h_layer=h_layer.astype(np.float32), vperp_layer=vperp_layer.astype(np.float32),
        rel_ratio_of_means=rel_ratio_of_means.astype(np.float32),
        rel_mean_of_ratios=rel_mean_of_ratios.astype(np.float32),
        kl_layer=kl_layer.astype(np.float32),
    )

    # ---- report ----
    print(f"\n[{args.group} / {model_tag}]  prompt-identity check: max|Δ fs-logit| = {fs_max_dlogit:.2e}"
          + ("  (0 => identical prompts to the stored run)" if fs_max_dlogit == 0 else ""))
    print(f"{'depth':>6}{'layer':>6}{'||h||':>10}{'||v_perp||':>12}{'rel=v/h':>10}"
          f"{'rel_std':>9}{'meanKL_test':>13}")
    for r in pooled:
        kl = r["mean_kl_test"]
        print(f"{r['depth']:6.2f}{r['layer']:6d}{r['h_norm']:10.2f}{r['vperp_norm']:12.3f}"
              f"{r['rel_ratio_of_means']:10.4f}{r['rel_std']:9.4f}"
              f"{(kl if kl is not None else float('nan')):13.3f}")
    print(f"\nstored -> {os.path.relpath(dst + '.npz', _root)}, {os.path.relpath(dst + '.json', _root)}")


if __name__ == "__main__":
    main()
