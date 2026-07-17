"""Compute + store KL divergences and gold-token logits per relative depth for a
causal-projection run produced by ``inject_orthogonal_projection.py``.

For each cell (sample x ratio x layer) we compare the residual-ablated few-shot
distribution (``test_logits``) against the clean few-shot distribution
(``fewshot_logits``):

    KL(fs||test)  forward KL, few-shot as reference (coverage of ICL by the ablation)
    KL(test||fs)  reverse KL

and record the gold-answer token's logit under test / few-shot / zero-shot. All
per-layer numbers reported are means over the S*R cells.

The gold answer per sample is derived per group and *validated* against the stored
question string:
  * arithmetic : parsed from the question (Direct number / MCQ letter / yes-no).
  * mmlu       : replayed from the data module (rng.choice), asserting the
                 reconstructed question matches the one saved in the run.

Outputs (next to the run, in ``causal_projection/``):
  kl_<group>_<MODEL_TAG>.npz   per-cell arrays for downstream / per-ratio analysis
  kl_<group>_<MODEL_TAG>.json  per-layer summary table + endpoints + baseline

Usage:
    env/bin/python experiments/task_vector_injection/analyze_projection_kl.py --group mmlu
    env/bin/python experiments/task_vector_injection/analyze_projection_kl.py --group arithmetic --model-tag Llama-3B
"""
import argparse, json, os, random, re, sys
import numpy as np
from transformers import AutoTokenizer

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))
sys.path.insert(0, os.path.join(_root, "data"))

_LET = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------
# Gold-answer derivation (per group), validated against the stored question
# --------------------------------------------------------------------------
def _gold_arithmetic(questions):
    golds = []
    for q in questions:
        op1, op2 = map(int, re.match(r"\s*(\d+)\s*\+\s*(\d+)", q).groups())
        s = op1 + op2
        if "= ?" in q and "A)" in q:                                    # MCQ
            opts = [int(x) for x in re.findall(r"[A-D]\)\s*(\d+)", q)]
            golds.append(_LET[opts.index(s)])
        elif re.search(r"=\s*(\d+)\.\s*Correct", q):                    # verification
            shown = int(re.search(r"=\s*(\d+)\.\s*Correct", q).group(1))
            golds.append("yes" if shown == s else "no")
        else:                                                           # direct
            golds.append(str(s))
    return golds


def _gold_mmlu(questions):
    import data_semantic_domains as D
    pools = D.load_pools()
    golds = []
    for i, q in enumerate(questions):
        task = D.TASKS[i % len(D.TASKS)]
        ex   = random.Random(i).choice(pools[task]["test"])            # same draw as the run
        if D._format_question(ex) != q:
            raise AssertionError(f"sample {i}: reconstructed question does not match "
                                 f"the stored one — gold cannot be trusted.")
        golds.append(D._LETTERS[ex["answer"]])
    return golds


_GOLD = {"arithmetic": _gold_arithmetic, "mmlu": _gold_mmlu}


def _gold_token_id(tok, ans):
    # Answer follows the header "\n\n" -> no leading space; first non-space token.
    ids = tok.encode(ans, add_special_tokens=False)
    for i in ids:
        if tok.decode([i]).strip():
            return i
    return ids[0]


# --------------------------------------------------------------------------
def _logsoftmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _kl(logP, logQ):                      # KL(P||Q) over the last axis
    return (np.exp(logP) * (logP - logQ)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(_GOLD))
    ap.add_argument("--model-tag", default="Llama-3B")
    ap.add_argument("--dir", default=os.path.join(_root, "causal_projection"))
    args = ap.parse_args()

    stem = os.path.join(args.dir, f"{args.group}_{args.model_tag}")
    npz  = np.load(stem + ".npz")
    meta = json.load(open(stem + ".json"))

    have_test = "test_logits" in npz.files
    have_null = "null_logits" in npz.files
    if not (have_test or have_null):
        raise SystemExit(f"run has mode={meta['mode']!r}; no test_logits/null_logits.")

    layers   = list(map(int, npz["layers"]))
    n_layers = meta["n_layers"]
    ratios   = list(map(str, npz["ratios"]))
    zs = npz["zeroshot_logits"].astype(np.float32)      # (S,V)
    fs = npz["fewshot_logits"].astype(np.float32)       # (S,R,V)
    ref = npz["test_logits"] if have_test else npz["null_logits"]
    S, R, L, V = ref.shape

    tok   = AutoTokenizer.from_pretrained(meta["model"])
    quest = [s["question"] for s in meta["samples"]]
    golds = _GOLD[args.group](quest)
    gids  = [_gold_token_id(tok, g) for g in golds]

    fs_argmax = fs.argmax(-1)
    hit = float(np.mean([fs_argmax[i, j] == gids[i] for i in range(S) for j in range(R)]))

    lz, lf = _logsoftmax(zs), _logsoftmax(fs)

    def per_cell(arm):                       # KL(fs||arm) + gold logit, shape (S,R,L)
        X  = npz[arm + "_logits"].astype(np.float32)
        lX = _logsoftmax(X)
        kl = np.empty((S, R, L)); gold = np.empty((S, R, L))
        for i in range(S):
            for j in range(R):
                for k in range(L):
                    kl[i, j, k]   = _kl(lf[i, j], lX[i, j, k])
                    gold[i, j, k] = X[i, j, k, gids[i]]
        return kl, gold

    arms = {}
    if have_test: arms["test"] = per_cell("test")
    if have_null: arms["null"] = per_cell("null")

    gold_fs = np.array([[fs[i, j, gids[i]] for j in range(R)] for i in range(S)])   # (S,R)
    gold_zs = np.array([zs[i, gids[i]] for i in range(S)])                          # (S,)
    base_kl = np.array([[_kl(lf[i, j], lz[i]) for j in range(R)] for i in range(S)])# (S,R)

    depth = [l / n_layers for l in layers]

    # ---- per-ratio summary: average over samples for each ratio (no ratio pooling) ----
    per_ratio = []
    for j, rat in enumerate(ratios):
        rows = []
        for k, l in enumerate(layers):
            row = {"depth": round(depth[k], 4), "layer": l}
            for arm, (kl, gold) in arms.items():
                row[f"kl_fs_{arm}"]       = float(kl[:, j, k].mean())
                row[f"gold_logit_{arm}"]  = float(gold[:, j, k].mean())
                row[f"delta_{arm}_vs_fs"] = float((gold[:, j, k] - gold_fs[:, j]).mean())
            rows.append(row)
        per_ratio.append({
            "ratio": rat,
            "gold_logit_fewshot":  float(gold_fs[:, j].mean()),
            "gold_logit_zeroshot": float(gold_zs.mean()),
            "baseline_kl_fs_zs":   float(base_kl[:, j].mean()),
            "per_layer": rows,
        })

    summary = {
        "group": args.group, "model": meta["model"], "model_tag": args.model_tag,
        "n_layers": n_layers, "n_samples": S, "n_ratios": R,
        "arms": list(arms), "reduction": "mean over samples, per ratio (ratios not pooled)",
        "null_seed": meta.get("null_seed"),
        "gold_is_fewshot_argmax_frac": hit, "gold_answers": golds,
        "per_ratio": per_ratio,
    }
    if have_null:
        summary["null_resid_cos_absmean"] = float(np.nanmean(np.abs(npz["null_resid_cos"])))

    out_npz  = os.path.join(args.dir, f"kl_{args.group}_{args.model_tag}.npz")
    out_json = os.path.join(args.dir, f"kl_{args.group}_{args.model_tag}.json")
    store = dict(layers=np.array(layers), depth=np.array(depth), ratios=np.array(ratios),
                 gold_logit_fs=gold_fs, gold_logit_zs=gold_zs,
                 baseline_kl_fs_zs=base_kl, gold_token_ids=np.array(gids))
    for arm, (kl, gold) in arms.items():
        store[f"kl_fs_{arm}"] = kl
        store[f"gold_logit_{arm}"] = gold
    np.savez_compressed(out_npz, **store)
    json.dump(summary, open(out_json, "w"), indent=2)

    # ---- report: one block per ratio, rows = layers, means over samples ----
    print(f"[{args.group} / {args.model_tag}]  S={S} R={R} L={L}   arms={list(arms)}   "
          f"gold==few-shot argmax: {hit*100:.0f}%")
    if have_null:
        print(f"null |cos(g, residual)| mean = {summary['null_resid_cos_absmean']:.3f} "
              f"(≈0 confirms random orthogonal direction)")
    cols = []
    if have_test: cols += ["kl_fs_test", "gold_logit_test"]
    if have_null: cols += ["kl_fs_null", "gold_logit_null"]
    head = f"{'depth':>6}{'layer':>6}" + "".join(f"{c:>17}" for c in cols)
    for blk in per_ratio:
        print(f"\nratio {blk['ratio']}   fs gold={blk['gold_logit_fewshot']:.2f}  "
              f"zs gold={blk['gold_logit_zeroshot']:.2f}  "
              f"baseKL(fs||zs)={blk['baseline_kl_fs_zs']:.2f}")
        print(head)
        for r in blk["per_layer"]:
            print(f"{r['depth']:6.2f}{r['layer']:6d}" + "".join(f"{r[c]:17.3f}" for c in cols))
    print(f"\nstored -> {os.path.relpath(out_npz, _root)}, {os.path.relpath(out_json, _root)}")


if __name__ == "__main__":
    main()
