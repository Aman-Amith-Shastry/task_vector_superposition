"""Null distribution for the orthogonal-residual ablation, centroid geometry.

Unlike ``inject_orthogonal_projection.py`` (which derives the residual from each
question's *live* contrast), here the geometry is fixed per ``(ratio, layer)`` and
precomputed from disk --- the model is only run to obtain the few-shot and
zero-shot *activations* of each question:

  * pure-ratio centroids  (ratio_centroids/<group>_<tag>.npz)  ->  m, d_pred
  * the ratio's actual centroid = mean of its per-sample contrasts
                              (ratio_samples/<group>_<tag>.npz) ->  d_act, proj, residual

Per ``(ratio, layer)``:

    m       = mean of the three pure-ratio contrast centroids
    d_pred  = c_pred - m           (c_pred = convex combo of pures for the ratio)
    d_act   = c_act  - m           (c_act  = actual centroid of the ratio)
    proj    = (d_act . d_pred / |d_pred|^2) d_pred
    residual= d_act - proj         (the orthogonal-to-prediction component, ⊥ d_pred)

We then draw ``--draws`` random vectors ``g`` that are also ⊥ d_pred and
norm-matched to ``|residual|`` (precomputed, no model needed).

Patch (assistant-header token, layer l, injected into the *few-shot* prompt so
the ICL context is preserved --- an in-context ablation). Both arms *remove* a
vector of norm ``|residual|`` from the actual few-shot header activation ``h_fs``,
so they are matched in displacement magnitude and differ only in direction:

    test  header = h_fs - residual           (the true orthogonal residual removed)
    null  header = h_fs - g                   (a random ⊥ d_pred, norm-matched
                                               vector removed instead)

with ``h_fs`` the question's actual few-shot header activation. Reference for every
KL is the clean few-shot logits of that question. If the residual's *direction*
carries task information, removing it should disrupt the output more than removing
a random direction of the same size, i.e. ``KL_test`` sits in the upper tail of the
null.

Structure the user asked for: the null distribution has one point per *draw*, and
each draw's statistic is the KL **averaged over the S questions**:

    null_kl[ratio, layer, draw] = mean_q KL( fs_logits[q] || patched(h_fs_q - g_draw) )
    test_kl[ratio, layer]       = mean_q KL( fs_logits[q] || patched(h_fs_q - residual) )

The same ``g_draw`` is applied to every question, so a draw is one consistent
random residual. For speed the ``draws`` are run as one batched forward per
(question, layer) via ``logits_to_keep=1`` --- numerically identical to looping
draws on the outside, S x cheaper.

Outputs (in ``causal_projection/``):
  nulldist_<group>_<tag>.npz   null_kl (R,L,D), test_kl (R,L), per-question raw, geometry
  nulldist_<group>_<tag>.json  metadata

Then locate test within the null with:
  env/bin/python experiments/task_vector_injection/analyze_null_position.py \
      --group arithmetic --stem nulldist

Usage:
  env/bin/python experiments/task_vector_injection/null_distribution.py \
      --group arithmetic --samples 10 --draws 100
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

# Importing the injection module loads the model (honouring --model/--quantize on
# argv) and gives us the shared helpers: _capture, _patched_logits[_batch],
# _load_geometry, _pred_contrast, _logsoftmax, _kl_rows, _make_model_tag.
import inject_orthogonal_projection as inj

model, tokenizer, n_layers = inj.model, inj.tokenizer, inj.n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="arithmetic", choices=list(inj._GROUPS))
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--quantize", default="", choices=["", "int8", "int4"])
    ap.add_argument("--samples", type=int, default=10, help="Questions per ratio (inner loop).")
    ap.add_argument("--draws", type=int, default=100, help="Random orthogonal residuals (null size).")
    ap.add_argument("--seed", type=int, default=0, help="Base seed for the random residuals.")
    ap.add_argument("--chunk", type=int, default=50,
                    help="Draws per batched null forward. Activation memory scales with "
                         "chunk*seq_len, so long-prompt groups (mmlu) need a small value "
                         "(e.g. 4) to fit; short-prompt groups run all draws in one batch.")
    ap.add_argument("--ratios", nargs="+", default=None, help="Restrict to these ratio strings.")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Explicit layer indices to sweep (must be a subset of the precomputed "
                         "swept layers). Default: the standard depth fractions. "
                         "e.g. --layers 1  to check just the shallowest layer.")
    ap.add_argument("--zs-only", action="store_true",
                    help="Compute only the zero-shot-replacement arm and merge it into an "
                         "existing nulldist npz, reusing its test/null arms. ~1/31 the cost "
                         "of a full run. Refuses to merge if layers/ratios/samples disagree.")
    ap.add_argument("--out-dir", default=os.path.join(_root, "causal_projection"))
    args = ap.parse_args()

    model_tag = inj._make_model_tag(args.model)
    data = importlib.import_module(inj._GROUPS[args.group])
    PURE_RATIOS = data.PURE_RATIOS
    H = int(model.config.hidden_size)

    default_layers = sorted(set(max(1, round(f * n_layers)) for f in inj._LAYER_FRACS))
    if args.layers:
        layers = sorted(set(args.layers))
        missing = [l for l in layers if l not in default_layers]
        if missing:
            raise SystemExit(f"No precomputed geometry for layers {missing}; "
                             f"choose a subset of {default_layers}.")
    else:
        layers = default_layers

    def _is_degenerate(r):
        w = np.array(r, dtype=float); w = w / w.sum()
        return np.allclose(w, w.mean())
    mixed = [r for r in data.RATIOS if r not in PURE_RATIOS and not _is_degenerate(r)]
    if args.ratios:
        want = set(args.ratios)
        mixed = [r for r in mixed if "-".join(map(str, r)) in want]
    if not mixed:
        raise SystemExit("No mixed ratios selected.")

    S, R, L, D = args.samples, len(mixed), len(layers), args.draws
    ratio_strs = ["-".join(map(str, r)) for r in mixed]

    print(f"Model: {args.model}  ({model_tag}, {n_layers} layers)")
    print(f"Group: {args.group}   Layers: {layers}")
    print(f"Mixed ratios: {ratio_strs}")
    print(f"Questions (inner): {S}   Draws (null size): {D}")

    # ----- zs-only: verify the target run matches before spending any compute -----
    merge_stem = os.path.join(args.out_dir, f"nulldist_{args.group}_{model_tag}")
    if args.zs_only:
        if not os.path.exists(merge_stem + ".npz"):
            raise SystemExit(f"--zs-only needs an existing run to merge into: {merge_stem}.npz")
        prev = np.load(merge_stem + ".npz")
        prev_layers = list(map(int, prev["layers"]))
        prev_ratios = list(map(str, prev["ratios"]))
        prev_meta   = json.load(open(merge_stem + ".json"))
        if prev_layers != layers:
            raise SystemExit(f"Layer mismatch: run has {prev_layers}, this invocation {layers}. "
                             "The zs arm is indexed by (ratio, layer, question) and must align.")
        if prev_ratios != ratio_strs:
            raise SystemExit(f"Ratio mismatch: run has {prev_ratios}, this invocation {ratio_strs}.")
        if prev_meta.get("n_samples") != S:
            raise SystemExit(f"Sample mismatch: run used n_samples={prev_meta.get('n_samples')}, "
                             f"this invocation {S}. Questions are indexed by sample idx.")
        print(f"--zs-only: merging into {os.path.basename(merge_stem)}.npz "
              f"(reusing its {prev_meta.get('n_draws')}-draw null and test arms)")

    # ----- precompute geometry per (ratio, layer) from disk -----
    if args.zs_only:
        resid_g = G_g = proj_g = None
        resid_norm = proj_norm = dpred_norm = coef = null_cos = None
    else:
        # Questions 0..S-1 are the ones ablated below, so they are withheld from
        # every centroid: the removed vector is fit only on questions S..n-1.
        tri_center, pure, samp, n_fit = inj._fit_geometry(
            args.group, model_tag, layers, PURE_RATIOS, skip=S)
        print(f"Centroids fit on {n_fit} held-out questions "
              f"(evaluating on questions 0..{S - 1}, disjoint)")

        proj_g   = {}                                          # (j,k) -> proj vector
        resid_g  = {}                                          # (j,k) -> true residual vector
        G_g      = {}                                          # (j,k) -> (D,H) random residuals
        resid_norm = np.full((R, L), np.nan, np.float32)
        proj_norm  = np.full((R, L), np.nan, np.float32)
        dpred_norm = np.full((R, L), np.nan, np.float32)
        coef       = np.full((R, L), np.nan, np.float32)       # kappa (along-prediction)
        null_cos   = np.full((R, L), np.nan, np.float32)       # mean |cos(g, residual)|

        for j, ratio in enumerate(mixed):
            for k, l in enumerate(layers):
                m      = tri_center[l]
                c_pred = inj._pred_contrast(pure[l], ratio, PURE_RATIOS)
                d_pred = c_pred - m
                denom  = float(d_pred @ d_pred)
                c_act  = samp[f"{l}|{ratio_strs[j]}"][S:].astype(np.float64).mean(0)  # held-out centroid
                d_act  = c_act - m
                kap    = float(d_act @ d_pred) / denom
                proj   = kap * d_pred
                resid  = d_act - proj                          # ⊥ d_pred
                rn     = float(np.linalg.norm(resid))

                rs = np.random.RandomState(args.seed + (j * L + k))
                G  = rs.standard_normal((D, H))
                G -= (G @ d_pred / denom)[:, None] * d_pred[None, :]        # ⊥ d_pred
                Gn = np.linalg.norm(G, axis=1, keepdims=True)
                np.divide(G, Gn, out=G, where=Gn > 0.0)
                G *= rn                                                     # norm-match
                cos = (G @ resid) / (np.linalg.norm(G, axis=1) * rn + 1e-12)

                proj_g[(j, k)]      = proj
                resid_g[(j, k)]     = resid
                G_g[(j, k)]         = G
                resid_norm[j, k]    = rn
                proj_norm[j, k]     = float(np.linalg.norm(proj))
                dpred_norm[j, k]    = float(np.sqrt(denom))
                coef[j, k]          = kap
                null_cos[j, k]      = float(np.mean(np.abs(cos)))

    # ----- forward passes: per-question zero-shot activations + few-shot logprobs -----
    print("Loading pools...")
    pools = data.load_pools()

    hfs      = [[None] * R for _ in range(S)]              # hfs[i][j][l] : few-shot header act
    hzs      = [None] * S                                  # hzs[i][l]    : zero-shot header act
    fs_lsm   = [[None] * R for _ in range(S)]              # clean few-shot log-probs (V,)
    icl_msgs = [[None] * R for _ in range(S)]
    for i in range(S):
        # Zero-shot header: the held-out question alone (no exemplars). rotate_test=True
        # fixes the question by sample index, so this is shared across ratios.
        zs_msgs = data.build_messages(pools, random.Random(i), mixed[0], sample_idx=i,
                                      rotate_test=True)[-1:]   # user test turn only
        hzs[i], _ = inj._capture(zs_msgs, layers)
        for j, ratio in enumerate(mixed):
            msgs = data.build_messages(pools, random.Random(i), ratio,
                                       sample_idx=i, rotate_test=True)
            acts, lg = inj._capture(msgs, layers)
            hfs[i][j]      = acts                           # per-layer header activations
            fs_lsm[i][j]   = inj._logsoftmax(lg)
            icl_msgs[i][j] = msgs
        print(f"  captured question {i + 1}/{S}")

    # ----- patched evals: per (ratio, layer) draw distribution, averaged over questions -----
    null_kl      = np.zeros((R, L, D), np.float64)
    test_kl      = np.zeros((R, L), np.float64)
    null_kl_perq = np.zeros((R, L, S, D), np.float32)
    test_kl_perq = np.zeros((R, L, S), np.float32)
    zs_kl        = np.zeros((R, L), np.float64)            # full-ICL-removal reference
    zs_kl_perq   = np.zeros((R, L, S), np.float32)

    import time
    for j, ratio in enumerate(mixed):
        t0 = time.time()
        for k, l in enumerate(layers):
            resid = resid_g[(j, k)] if not args.zs_only else None
            G     = G_g[(j, k)]     if not args.zs_only else None
            for i in range(S):
                hf    = hfs[i][j][l]                        # (H,) actual few-shot header act
                flsm  = fs_lsm[i][j]
                msgs  = icl_msgs[i][j]
                # reference: replace the header outright with its zero-shot counterpart
                # (removes the whole ICL displacement, not just the residual)
                lz = inj._patched_logits(msgs, l, hzs[i][l].astype(np.float32))
                zs_kl_perq[j, k, i] = inj._kl_rows(flsm, inj._logsoftmax(lz))
                if args.zs_only:
                    continue
                # test: remove the true residual
                lt = inj._patched_logits(msgs, l, (hf - resid).astype(np.float32))
                test_kl_perq[j, k, i] = inj._kl_rows(flsm, inj._logsoftmax(lt))
                # null: remove D random norm-matched orthogonal vectors, batched forwards
                patch_mat = (hf[None, :] - G).astype(np.float32)       # (D, H)
                lgs = inj._patched_logits_batch(msgs, l, patch_mat,
                                                chunk=args.chunk)      # (D, V)
                null_kl_perq[j, k, i] = inj._kl_rows(flsm, inj._logsoftmax(lgs))
            zs_kl[j, k] = zs_kl_perq[j, k].mean()          # mean over questions
            if not args.zs_only:
                test_kl[j, k] = test_kl_perq[j, k].mean()  # mean over questions
                null_kl[j, k] = null_kl_perq[j, k].mean(0) # (D,) mean over questions per draw
        dt = time.time() - t0
        if args.zs_only:
            print(f"  evaluated ratio {ratio_strs[j]} in {dt:.0f}s  "
                  f"(deepest: KL_zs={zs_kl[j, -1]:.3f})   [~{dt * (R - j - 1):.0f}s left]")
        else:
            pctl = 100.0 * np.mean(null_kl[j, -1] < test_kl[j, -1])   # test's rank in the D draws
            print(f"  evaluated ratio {ratio_strs[j]} in {dt:.0f}s  "
                  f"(deepest: KL_test={test_kl[j, -1]:.3f}, pctile in {D}-draw null = {pctl:.0f}%, "
                  f"KL_zs={zs_kl[j, -1]:.3f})"
                  f"   [~{dt * (R - j - 1):.0f}s left]")

    # ----- save -----
    os.makedirs(args.out_dir, exist_ok=True)
    depth = [l / n_layers for l in layers]
    stem  = merge_stem

    if args.zs_only:
        # Merge: keep every existing array, add only the zs arm. Guards above already
        # verified layers/ratios/n_samples align, so indices are comparable.
        merged = {k: prev[k] for k in prev.files}
        merged["zs_kl"]      = zs_kl.astype(np.float32)
        merged["zs_kl_perq"] = zs_kl_perq
        np.savez_compressed(stem + ".npz", **merged)
        prev_meta["zs_arm"] = (
            "zs_kl = KL(fs || few-shot forward with the header replaced by that question's "
            "cached zero-shot header h_zs) -- removes the entire ICL displacement "
            "d = h_fs - h_zs, so it upper-references the residual-only ablation test_kl")
        prev_meta["zs_arm_provenance"] = (
            "computed by a later --zs-only pass and merged in; test/null arms untouched. "
            "Questions are reproduced deterministically from the same seeds, and the "
            "clean few-shot reference logits come from the same _capture code path.")
        json.dump(prev_meta, open(stem + ".json", "w"), indent=2)
        print(f"\nmerged zs arm -> {os.path.relpath(stem + '.npz', _root)}")
        print("pooled over ratios, per layer:")
        print(f"{'depth':>6}{'layer':>6}{'KL_zs':>10}")
        for k, l in enumerate(layers):
            print(f"{depth[k]:6.2f}{l:6d}{zs_kl[:, k].mean():10.3f}")
        return

    np.savez_compressed(
        stem + ".npz",
        layers=np.array(layers), depth=np.array(depth), ratios=np.array(ratio_strs),
        null_kl=null_kl.astype(np.float32), test_kl=test_kl.astype(np.float32),
        null_kl_perq=null_kl_perq, test_kl_perq=test_kl_perq,
        zs_kl=zs_kl.astype(np.float32), zs_kl_perq=zs_kl_perq,
        resid_norm=resid_norm, proj_norm=proj_norm, dpred_norm=dpred_norm,
        coef=coef, null_resid_cos=null_cos,
        n_draws=np.array(D), seed=np.array(args.seed),
    )
    meta = {
        "experiment": "null_distribution", "group": args.group,
        "model": args.model, "model_tag": model_tag,
        "n_layers": n_layers, "layers": layers, "depth": depth, "ratios": ratio_strs,
        "n_samples": S, "n_draws": D, "seed": args.seed,
        "patch_base": "actual few-shot header activation h_fs; test = h_fs - residual, "
                      "null = h_fs - g (matched-removal: both remove a norm-|residual| vector)",
        "null_construction": "g drawn ⊥ d_pred and norm-matched to |residual|; test and null "
                             "differ only in direction, not displacement magnitude",
        "kl_reference": "clean few-shot logits per question, forward KL(fs||patched)",
        "zs_arm": "zs_kl = KL(fs || few-shot forward with the header replaced by that "
                  "question's cached zero-shot header h_zs) -- removes the entire ICL "
                  "displacement d = h_fs - h_zs, so it upper-references the residual-only "
                  "ablation test_kl (which removes only the component of d orthogonal to d_pred)",
        "geometry_source": "residual from actual centroids (ratio_samples means); "
                           "d_pred/m from pure-ratio centroids",
        "reduction": "each draw = mean KL over questions; null dist has n_draws points/(ratio,layer)",
        "null_resid_cos_absmean": float(np.nanmean(np.abs(null_cos))),
    }
    json.dump(meta, open(stem + ".json", "w"), indent=2)

    print(f"\nnull |cos(g,residual)| mean = {meta['null_resid_cos_absmean']:.3f} "
          f"(≈0 confirms random orthogonal directions)")
    print(f"stored -> {os.path.relpath(stem + '.npz', _root)}, "
          f"{os.path.relpath(stem + '.json', _root)}")


if __name__ == "__main__":
    main()
