"""
Causal test of the orthogonal residual via activation patching.

We patch the residual-stream state at the assistant-header token --- the exact
spot the contrast vectors were extracted from (the last-token output of
``model.model.layers[l]`` with ``add_generation_prompt=True``) --- and read the
resulting next-token logits. Two modes:

  control : Patch the *few-shot* prompt's header activation with
            ``h_zs + c``, where ``c = h_icl - h_zs`` is the live contrast vector
            (h_icl from the ICL prompt, h_zs from the zero-shot prompt, 2 forward
            passes as always). Since ``h_zs + c == h_icl`` this must reproduce the
            clean few-shot logits to within floating-point precision --- a sanity
            check that the patching machinery is faithful. Reported as the max /
            mean |Δlogit| against the unpatched few-shot run.

  test    : Patch the *few-shot* prompt's header activation with the ICL
            displacement projected onto the predicted convex direction, i.e. with
            the orthogonal-to-the-prediction component removed. Concretely, with
            the tri-center ``m`` (mean of the pure-ratio contrast centroids) and
            the predicted displacement ``d_pred = c_pred - m`` (c_pred = the
            ratio's convex combination of pure centroids):

                d_act = c - m                            (live displacement)
                proj  = (d_act . d_pred / |d_pred|^2) d_pred
                patch = h_zs + m + proj

            Because the clean few-shot header is ``h_icl = h_zs + m + d_act``, the
            patched header differs from it by *exactly* the orthogonal residual
            ``d_act - proj`` and nothing else --- and the ICL context is preserved
            in both runs. Comparing the patched logits to the clean few-shot logits
            is therefore an in-context ablation that isolates the causal effect of
            the orthogonal residual (no zero-shot-vs-full-context confound).

  null    : Same few-shot in-context patch as ``test``, but instead of dropping
            the residual we replace it with a *random* vector ``g`` that is also
            orthogonal to the predicted convex displacement ``d_pred`` and
            norm-matched to the true residual (``|g| = |d_act - proj|``):

                patch = h_zs + m + proj + g

            This is the magnitude/direction control for ``test``: if removing the
            residual (``test``) and swapping it for a random same-size orthogonal
            vector (``null``) perturb the few-shot logits by the same amount, the
            residual's *direction* carries no behaviour --- only its magnitude in
            the orthogonal subspace matters. If ``null`` deviates more, the true
            residual points somewhere specific. Seeded per cell for reproducibility.

The tri-center and predicted contrast come from the stored pure-ratio centroids
in ``ratio_centroids/<group>_<MODEL_TAG>.npz`` (no model needed for the geometry;
only the live contrast ``c`` is computed at run time). Both modes run at the same
8 relative depths (SWEEP_LAYERS) used everywhere else, for whichever model is
loaded.

Logits, questions, and per-cell geometry are saved under ``causal_projection/``
for reproducibility and downstream analysis (e.g. JS divergence between test and
full-ICL distributions).

Usage:
    python experiments/task_vector_injection/inject_orthogonal_projection.py
    python experiments/task_vector_injection/inject_orthogonal_projection.py \
        --model meta-llama/Llama-3.1-8B-Instruct --quantize int8 --group entity
    python experiments/task_vector_injection/inject_orthogonal_projection.py \
        --group mmlu --samples 20 --mode test
"""

import sys, os

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "data"))
sys.path.insert(0, os.path.dirname(_dir))   # experiments/ (unused here but consistent)

# Parse --model / --quantize BEFORE importing local_model (model loads at import).
import argparse as _ap
_mp = _ap.ArgumentParser(add_help=False, allow_abbrev=False)
_mp.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                 help="HuggingFace model ID to use.")
_mp.add_argument("--quantize", default="", choices=["", "int8", "int4"],
                 help="Quantize weights via quanto (recommended for 8B on MPS).")
_margs, _ = _mp.parse_known_args()
os.environ["TASK_VECTOR_MODEL"]    = _margs.model
os.environ["TASK_VECTOR_QUANTIZE"] = _margs.quantize

import re
import json
import random
import importlib

import numpy as np
import torch

from local_model import model, tokenizer, n_layers


def _make_model_tag(model_id: str) -> str:
    m = re.search(r'(\d+\.?\d*[Bb])', model_id)
    size = m.group(1).upper() if m else model_id.split("/")[-1]
    name = model_id.lower()
    if "qwen"    in name: return f"Qwen-{size}"
    if "llama"   in name: return f"Llama-{size}"
    if "mistral" in name: return f"Mistral-{size}"
    if "gemma"   in name: return f"Gemma-{size}"
    return model_id.split("/")[-1]


# group -> (data module, centroid-file tag)
_GROUPS = {
    "arithmetic": "data_arithmetic_formats",
    "entity":     "data_entity_attribute",
    "mmlu":       "data_semantic_domains",
}

# Fixed fractions of model depth --- identical to the sweep / extraction scripts,
# so SWEEP_LAYERS matches the layers stored in ratio_centroids/.
_LAYER_FRACS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.75]


# --------------------------------------------------------------------------
# Activation capture and patching at the assistant-header token
# --------------------------------------------------------------------------

def _capture(messages: list[dict], layers: list[int]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """One forward pass: last-token residual state at each layer + next-token logits.

    Mirrors ``local_model.get_activations_all_layers`` (same hook point, the
    ``model.model.layers[l]`` output at the final/assistant-header token) but also
    returns the clean logits from the same pass. Everything is float32 on CPU.
    """
    buf: dict[int, np.ndarray] = {}
    handles = []
    for l in layers:
        def _hook(module, inp, out, _l=l):
            h = out[0] if isinstance(out, tuple) else out
            v = h[0, -1, :] if h.dim() == 3 else h[-1, :]
            buf[_l] = v.detach().float().cpu().numpy()
        handles.append(model.model.layers[l].register_forward_hook(_hook))
    try:
        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].detach().float().cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    return buf, logits


def _patched_logits(messages: list[dict], layer: int, patch_vec: np.ndarray) -> np.ndarray:
    """Next-token logits after *replacing* the header activation at `layer`.

    The forward hook overwrites the last-token output of ``model.model.layers[layer]``
    with `patch_vec` (cast to the model's device/dtype) and lets it propagate.
    """
    pv = torch.from_numpy(np.ascontiguousarray(patch_vec))

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = pv.to(device=h.device, dtype=h.dtype)
        if h.dim() == 3:
            h[:, -1, :] = v
        else:
            h[-1, :] = v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs, logits_to_keep=1,
                           use_cache=False).logits[0, -1, :].detach().float().cpu().numpy()
    finally:
        handle.remove()
    return logits


def _patched_logits_batch(messages: list[dict], layer: int,
                          patch_mat: np.ndarray, chunk: int = 50) -> np.ndarray:
    """Last-token logits for a *batch* of header patches at `layer`.

    ``patch_mat`` is ``(D, hidden)``: one forward per chunk with batch ``D``,
    overwriting each row's last-token output of ``model.model.layers[layer]``
    with the corresponding patch, returning the last-token logits ``(D, V)``.
    Uses ``logits_to_keep=1`` so memory scales with ``D*V`` (final position
    only), not ``D*seq*V`` --- this is what makes 100 draws/cell one cheap
    forward instead of 100.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc  = tokenizer(text, return_tensors="pt")
    outs = []
    for s in range(0, len(patch_mat), chunk):
        pv = torch.from_numpy(np.ascontiguousarray(patch_mat[s:s + chunk]))   # (b, hidden)

        def _hook(module, inp, out, pv=pv):
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1, :] = pv.to(device=h.device, dtype=h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        handle = model.model.layers[layer].register_forward_hook(_hook)
        try:
            b = pv.shape[0]
            inputs = {k: v.repeat(b, 1).to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                lg = model(**inputs, logits_to_keep=1,
                           use_cache=False).logits[:, -1, :].detach().float().cpu().numpy()
        finally:
            handle.remove()
        outs.append(lg)
    return np.concatenate(outs, 0)                                            # (D, V)


def _logsoftmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _kl_rows(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """KL(p||q) over the last axis; broadcasts a single ``log_p`` over rows of ``log_q``."""
    return (np.exp(log_p) * (log_p - log_q)).sum(-1)


# --------------------------------------------------------------------------
# Geometry loaded from stored pure-ratio centroids
# --------------------------------------------------------------------------

def _load_geometry(group_tag: str, model_tag: str, layers: list[int], pure_ratios):
    """Return per-layer tri-center and a predicted-contrast lookup from centroids.

    tri_center[l]           : mean of the three pure-ratio contrast centroids
    pred_contrast[l][ratio] : convex combination of the pure centroids for `ratio`
    """
    path = os.path.join(_root, "ratio_centroids", f"{group_tag}_{model_tag}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing pure-ratio centroids: {path}. Run the sweep for this "
            f"model/group first (it writes ratio_centroids/)."
        )
    data = np.load(path)

    def _key(l, r):
        return f"{l}|{'-'.join(map(str, r))}"

    tri_center: dict[int, np.ndarray] = {}
    pure: dict[int, dict] = {}
    for l in layers:
        for p in pure_ratios:
            if _key(l, p) not in data.files:
                raise KeyError(f"{_key(l, p)} not in {path}; layer set mismatch.")
        pure[l] = {p: data[_key(l, p)].astype(np.float64) for p in pure_ratios}
        tri_center[l] = np.mean([pure[l][p] for p in pure_ratios], axis=0)
    return tri_center, pure


def _fit_geometry(group_tag: str, model_tag: str, layers: list[int], pure_ratios,
                  skip: int = 0):
    """Same contract as _load_geometry, but centroids are refit from per-question
    samples with the first `skip` questions withheld.

    Questions 0..skip-1 are the ones the ablation is evaluated on. Withholding
    them keeps every fitted quantity independent of the activations the fitted
    vector is later subtracted from; otherwise each centroid carries a 1/n slice
    of the very residual being removed. skip=0 reproduces _load_geometry (the
    cached centroids are exactly the mean over all samples).

    Returns (tri_center, pure, samp, n_fit) -- samp is handed back so callers can
    slice the mixed-ratio centroids from the same array and holdout.
    """
    path = os.path.join(_root, "ratio_samples", f"{group_tag}_{model_tag}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing per-sample contrasts: {path}. Run the sweep for this "
            f"model/group first (it writes ratio_samples/)."
        )
    samp = np.load(path)

    def _key(l, r):
        return f"{l}|{'-'.join(map(str, r))}"

    n_avail = samp[_key(layers[0], pure_ratios[0])].shape[0]
    n_fit   = n_avail - skip
    if n_fit < 2:
        raise SystemExit(
            f"Holdout leaves {n_fit} question(s) to fit centroids from "
            f"({n_avail} available, {skip} withheld for evaluation). Need >= 2."
        )

    tri_center: dict[int, np.ndarray] = {}
    pure: dict[int, dict] = {}
    for l in layers:
        for p in pure_ratios:
            if _key(l, p) not in samp.files:
                raise KeyError(f"{_key(l, p)} not in {path}; layer set mismatch.")
        pure[l] = {p: samp[_key(l, p)][skip:].astype(np.float64).mean(0)
                   for p in pure_ratios}
        tri_center[l] = np.mean([pure[l][p] for p in pure_ratios], axis=0)
    return tri_center, pure, samp, n_fit


def _pred_contrast(pure_l: dict, ratio, pure_ratios) -> np.ndarray:
    total = float(sum(ratio))
    return np.sum([(n / total) * pure_l[p] for n, p in zip(ratio, pure_ratios)], axis=0)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = _ap.ArgumentParser()
    ap.add_argument("--mode", default="both",
                    choices=["control", "test", "null", "both", "all"],
                    help="both = test + null (the paired arms); all adds control.")
    ap.add_argument("--null-seed", type=int, default=0,
                    help="Base seed for the null random orthogonal residual.")
    ap.add_argument("--null-draws", type=int, default=1,
                    help="Random orthogonal norm-matched residuals drawn per "
                         "(sample,ratio,layer) to form the null distribution. "
                         "1 (default) stores full null logits (back-compat); >1 "
                         "runs the draws as one batched forward and stores per-draw "
                         "KL(fs||null) scalars only (full logits would be ~D x bigger).")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--quantize", default="", choices=["", "int8", "int4"])
    ap.add_argument("--group", default="arithmetic", choices=list(_GROUPS))
    ap.add_argument("--samples", type=int, default=10,
                    help="Number of test partitions (held-out questions) to run.")
    ap.add_argument("--ratios", nargs="+", default=None,
                    help="Subset of mixed ratios as a-b-c strings (default: all "
                         "mixed ratios with a well-defined prediction).")
    ap.add_argument("--out-dir", default=os.path.join(_root, "causal_projection"))
    args = ap.parse_args()

    model_tag = _make_model_tag(args.model)
    data = importlib.import_module(_GROUPS[args.group])
    PURE_RATIOS = data.PURE_RATIOS

    layers = sorted(set(max(1, round(f * n_layers)) for f in _LAYER_FRACS))

    # Mixed ratios with a well-defined predicted direction (drop pures + 1-1-1).
    def _is_degenerate(r):
        w = np.array(r, dtype=float)
        w = w / w.sum()
        return np.allclose(w, w.mean())          # uniform weights -> d_pred == 0
    mixed = [r for r in data.RATIOS if r not in PURE_RATIOS and not _is_degenerate(r)]
    if args.ratios:
        want = set(args.ratios)
        mixed = [r for r in mixed if "-".join(map(str, r)) in want]
    if not mixed:
        raise SystemExit("No mixed ratios selected.")

    print(f"Model: {args.model}  ({model_tag}, {n_layers} layers)")
    print(f"Group: {args.group}   Layers: {layers}")
    print(f"Mixed ratios: {['-'.join(map(str, r)) for r in mixed]}")
    print(f"Samples: {args.samples}   Mode: {args.mode}")

    tri_center, pure = _load_geometry(args.group, model_tag, layers, PURE_RATIOS)

    print("Loading pools...")
    pools = data.load_pools()

    S, R, L = args.samples, len(mixed), len(layers)
    V = int(model.config.vocab_size)

    do_ctrl = args.mode in ("control", "all")
    do_test = args.mode in ("test", "both", "all")
    do_null = args.mode in ("null", "both", "all")
    D_null  = max(1, args.null_draws)
    multi   = do_null and D_null > 1        # per-draw KL storage instead of full logits

    fewshot_logits  = np.zeros((S, R, V), dtype=np.float16)
    zeroshot_logits = np.zeros((S, V),    dtype=np.float16)
    control_logits  = np.zeros((S, R, L, V), dtype=np.float16) if do_ctrl else None
    test_logits     = np.zeros((S, R, L, V), dtype=np.float16) if do_test else None
    null_logits     = np.zeros((S, R, L, V), dtype=np.float16) if (do_null and not multi) else None
    null_kl         = np.full((S, R, L, D_null), np.nan, dtype=np.float32) if multi else None
    test_kl         = np.full((S, R, L), np.nan, dtype=np.float32) if (multi and do_test) else None
    null_resid_cos  = np.full((S, R, L), np.nan, dtype=np.float32)  # |cos(g, true residual)| (mean over draws)

    ctrl_absdiff_max  = np.full((S, R, L), np.nan, dtype=np.float32)
    ctrl_absdiff_mean = np.full((S, R, L), np.nan, dtype=np.float32)
    coef       = np.full((S, R, L), np.nan, dtype=np.float32)   # along-prediction coeff (== kappa)
    dact_norm  = np.full((S, R, L), np.nan, dtype=np.float32)
    proj_norm  = np.full((S, R, L), np.nan, dtype=np.float32)
    resid_norm = np.full((S, R, L), np.nan, dtype=np.float32)   # |d_act - proj|
    dpred_norm = np.full((S, R, L), np.nan, dtype=np.float32)

    samples_meta = []

    for i in range(S):
        # Zero-shot activations + logits: the held-out question is fixed by
        # sample index (rotate_test=True), so this is shared across ratios.
        rng0     = random.Random(i)
        zs_msgs  = data.build_messages(pools, rng0, mixed[0], sample_idx=i,
                                       rotate_test=True)[-1:]  # user test turn only
        zs_acts, zs_logits = _capture(zs_msgs, layers)
        zeroshot_logits[i] = zs_logits.astype(np.float16)
        samples_meta.append({"idx": i, "question": zs_msgs[-1]["content"]})

        for j, ratio in enumerate(mixed):
            rng      = random.Random(i)
            icl_msgs = data.build_messages(pools, rng, ratio, sample_idx=i, rotate_test=True)
            icl_acts, icl_logits = _capture(icl_msgs, layers)
            fewshot_logits[i, j] = icl_logits.astype(np.float16)
            fs_lsm = _logsoftmax(icl_logits) if multi else None   # clean few-shot log-probs (V,)

            for k, l in enumerate(layers):
                h_icl = icl_acts[l].astype(np.float64)
                h_zs  = zs_acts[l].astype(np.float64)
                c     = h_icl - h_zs                      # live contrast vector

                # --- control: h_zs + c == h_icl -> must reproduce clean logits ---
                if do_ctrl:
                    patch_ctrl = (h_zs + c).astype(np.float32)
                    lg = _patched_logits(icl_msgs, l, patch_ctrl)
                    control_logits[i, j, k] = lg.astype(np.float16)
                    d = np.abs(lg - icl_logits)
                    ctrl_absdiff_max[i, j, k]  = float(d.max())
                    ctrl_absdiff_mean[i, j, k] = float(d.mean())

                # --- test: strip the orthogonal residual, patch the few-shot run ---
                m       = tri_center[l]
                c_pred  = _pred_contrast(pure[l], ratio, PURE_RATIOS)
                d_pred  = c_pred - m
                d_act   = c - m
                denom   = float(d_pred @ d_pred)
                kap     = float(d_act @ d_pred) / denom
                proj    = kap * d_pred

                coef[i, j, k]       = kap
                dact_norm[i, j, k]  = float(np.linalg.norm(d_act))
                proj_norm[i, j, k]  = float(np.linalg.norm(proj))
                resid_norm[i, j, k] = float(np.linalg.norm(d_act - proj))
                dpred_norm[i, j, k] = float(np.sqrt(denom))

                if do_test:
                    patch_test = (h_zs + m + proj).astype(np.float32)
                    lg = _patched_logits(icl_msgs, l, patch_test)
                    test_logits[i, j, k] = lg.astype(np.float16)
                    if multi:                                # KL(fs||test) for the comparison
                        test_kl[i, j, k] = _kl_rows(fs_lsm, _logsoftmax(lg))

                # --- null: replace the residual with a random vector that is
                #     also orthogonal to d_pred and norm-matched to |d_act-proj| ---
                if do_null:
                    resid_vec = d_act - proj                 # true residual, ⊥ d_pred
                    rn = float(np.linalg.norm(resid_vec))
                    base_seed = args.null_seed + ((i * R + j) * L + k)
                    if multi:
                        # D_null random directions ⊥ d_pred, each norm-matched to |resid|
                        rs = np.random.RandomState(base_seed)
                        G  = rs.standard_normal((D_null, h_zs.shape[0]))
                        G -= (G @ d_pred / denom)[:, None] * d_pred[None, :]   # ⊥ d_pred
                        Gn = np.linalg.norm(G, axis=1, keepdims=True)
                        np.divide(G, Gn, out=G, where=Gn > 0.0)
                        G *= rn                                                # norm-match
                        cos = (G @ resid_vec) / (np.linalg.norm(G, axis=1) * rn + 1e-12)
                        null_resid_cos[i, j, k] = float(np.mean(np.abs(cos)))
                        patch_mat = ((h_zs + m + proj)[None, :] + G).astype(np.float32)
                        lgs = _patched_logits_batch(icl_msgs, l, patch_mat)    # (D, V)
                        null_kl[i, j, k] = _kl_rows(fs_lsm, _logsoftmax(lgs))  # (D,)
                    else:
                        rs = np.random.RandomState(base_seed)
                        g  = rs.standard_normal(h_zs.shape[0])
                        g  = g - (float(g @ d_pred) / denom) * d_pred   # ⊥ predicted displacement
                        gn = float(np.linalg.norm(g))
                        if gn > 0.0:
                            g = g / gn * rn                             # norm-match to residual
                        denom_cos = np.linalg.norm(g) * rn
                        null_resid_cos[i, j, k] = (float(g @ resid_vec) / denom_cos
                                                   if denom_cos > 0 else np.nan)
                        patch_null = (h_zs + m + proj + g).astype(np.float32)
                        lg = _patched_logits(icl_msgs, l, patch_null)
                        null_logits[i, j, k] = lg.astype(np.float16)

        print(f"  sample {i + 1}/{S} done"
              + (f"   ctrl max|Δ|={np.nanmax(ctrl_absdiff_max[i]):.2e}" if do_ctrl else "")
              + (f"   null cos(g,resid)~{np.nanmean(np.abs(null_resid_cos[i])):.3f}"
                 if do_null else ""))

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    ratio_strs = ["-".join(map(str, r)) for r in mixed]

    npz_payload = dict(
        layers=np.array(layers),
        ratios=np.array(ratio_strs),
        fewshot_logits=fewshot_logits,
        zeroshot_logits=zeroshot_logits,
        coef=coef, dact_norm=dact_norm, proj_norm=proj_norm,
        resid_norm=resid_norm, dpred_norm=dpred_norm,
    )
    if do_ctrl:
        npz_payload.update(control_logits=control_logits,
                           control_absdiff_max=ctrl_absdiff_max,
                           control_absdiff_mean=ctrl_absdiff_mean)
    if do_test:
        npz_payload.update(test_logits=test_logits)
    if do_null:
        npz_payload.update(null_resid_cos=null_resid_cos,
                           null_seed=np.array(args.null_seed),
                           null_draws=np.array(D_null))
        if multi:
            npz_payload.update(null_kl=null_kl)
            if do_test:
                npz_payload.update(test_kl=test_kl)
        else:
            npz_payload.update(null_logits=null_logits)

    npz_path  = os.path.join(args.out_dir, f"{args.group}_{model_tag}.npz")
    json_path = os.path.join(args.out_dir, f"{args.group}_{model_tag}.json")
    np.savez_compressed(npz_path, **npz_payload)

    meta = {
        "model": args.model, "model_tag": model_tag, "group": args.group,
        "n_layers": n_layers, "layers": layers, "ratios": ratio_strs,
        "n_samples": S, "mode": args.mode, "vocab_size": V,
        "null_seed": (args.null_seed if do_null else None),
        "null_draws": (D_null if do_null else None),
        "patch_point": "model.model.layers[l] output, assistant-header token",
        "control_check": {
            "max_abs_logit_diff":  (float(np.nanmax(ctrl_absdiff_max))  if do_ctrl else None),
            "mean_abs_logit_diff": (float(np.nanmean(ctrl_absdiff_mean)) if do_ctrl else None),
        },
        "samples": samples_meta,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved logits  -> {os.path.relpath(npz_path, _root)}")
    print(f"Saved metadata -> {os.path.relpath(json_path, _root)}")
    if do_ctrl:
        print(f"Control check: max |Δlogit| = {np.nanmax(ctrl_absdiff_max):.3e}, "
              f"mean |Δlogit| = {np.nanmean(ctrl_absdiff_mean):.3e} "
              f"(should be ~floating-point precision).")


if __name__ == "__main__":
    main()
