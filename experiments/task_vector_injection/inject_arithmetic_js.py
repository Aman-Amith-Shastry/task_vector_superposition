"""Distribution-level behavioral test for arithmetic format-vector injection.

Instead of scoring a single gold answer, we compare the model's full first-token
distribution under injection to its distribution under actual ICL:

  Z          first-token dist for a bare held-out addition prompt (zero-shot)
  P_ICL_g    first-token dist for a task-g ICL prompt + that addition prompt
  Q_inject_t first-token dist for the same bare addition prompt with arithmetic
             contrast vector t injected at the assistant-header token of `layer`

The arithmetic task group has three output formats over addition problems:
Direct numeric answers, MCQ letter answers, and yes/no verification answers.

Confusion matrix (per held-out addition example):
  M[t][g] = JS(Q_inject_t , P_ICL_g)        # Jensen-Shannon, nats, in [0, ln2]

JS is LOWER when the distributions are closer, so if injection reproduces the
matching task's ICL behavior, the diagonal (t == g) should be the row MINIMUM.

We double-center M to strip the two task-agnostic main effects (injection-t's
overall closeness; ICL-g's overall closeness), leaving the interaction. After
centering, a task-matching effect shows as a NEGATIVE, row-minimum diagonal.
Significance: bootstrap over held-out addition examples (default 1000 resamples)
for a 95% CI on each double-centered cell; the diagonal is significant if its CI
upper bound < 0.

Pure ICL tasks are used by default; pass --all_ratios to include mixed ratios.

Usage:
  python experiments/task_vector_injection/inject_arithmetic_js.py --layers 18 --alphas 4
  python experiments/task_vector_injection/inject_arithmetic_js.py --layers 18 --alphas 1 2 4 6 --test_task MCQ
"""

import sys, os, argparse, random, itertools

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "data"))

os.environ.setdefault("TASK_VECTOR_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
os.environ.setdefault("TASK_VECTOR_QUANTIZE", "")

_p = argparse.ArgumentParser()
_p.add_argument(
    "--model",
    default=os.environ["TASK_VECTOR_MODEL"],
    help="HuggingFace model ID. Must match the model used to run "
    "extract_arithmetic_vectors.py, since vectors are looked up by MODEL_TAG.",
)
_p.add_argument(
    "--layers",
    type=int,
    nargs="+",
    default=[18],
    help="Layers to inject at. Default: 18.",
)
_p.add_argument("--alphas", type=float, nargs="+", default=[4.0])
_p.add_argument(
    "--n_icl",
    type=int,
    default=4,
    help="ICL demo samplings to average per (example, task) for P_ICL_g.",
)
_p.add_argument(
    "--n_examples",
    type=int,
    default=None,
    help="How many held-out arithmetic examples to average over (default: all).",
)
_p.add_argument(
    "--test_task",
    choices=["Direct", "MCQ", "Verification"],
    default="MCQ",
    help="Held-out test-question format used for zero-shot, ICL, and ceiling prompts.",
)
_p.add_argument("--n_bootstrap", type=int, default=1000)
_p.add_argument(
    "--n_perm",
    type=int,
    default=10000,
    help="Monte-Carlo permutation draws for the task-matching p-value.",
)
_p.add_argument(
    "--full",
    action="store_true",
    help="Also print the full per-cell JS matrices, not just the per-layer summary.",
)
_p.add_argument(
    "--all_ratios",
    action="store_true",
    help="Use all 10 ICL ratios (pure + mixed) instead of the 3 pure tasks. "
    "Injects the weighted sum Σ w_i v_i and compares to real ratio-r ICL.",
)
_p.add_argument(
    "--ceiling",
    action="store_true",
    help="Ceiling diagnostic: JS among the real P_ICL(r) distributions "
    "themselves (no injection). Measures how distinguishable the ratios "
    "are behaviorally — the upper bound on what injection could reproduce.",
)
_args = _p.parse_args()
os.environ["TASK_VECTOR_MODEL"] = _args.model

import re
def _make_model_tag(model_id: str) -> str:
    size = (re.search(r'(\d+\.?\d*[Bb])', model_id) or type("", (), {"group": lambda s, n: model_id.split("/")[-1]})()).group(1).upper()
    name = model_id.lower()
    if "qwen"    in name: return f"Qwen-{size}"
    if "llama"   in name: return f"Llama-{size}"
    if "mistral" in name: return f"Mistral-{size}"
    if "gemma"   in name: return f"Gemma-{size}"
    return model_id.split("/")[-1]
MODEL_TAG   = _make_model_tag(_args.model)
_IS_DEFAULT = _args.model == "meta-llama/Llama-3.2-3B-Instruct"

import numpy as np
import torch
from local_model import model, tokenizer, _device
import data_arithmetic_formats as data

TASKS = data.TASKS
VEC_DIR = os.path.join(_dir, "vectors")


def load_vec(task: str, layer: int) -> torch.Tensor:
    fname = (f"arith_contrast_{task}_layer{layer}.npy" if _IS_DEFAULT
             else f"arith_contrast_{task}_{MODEL_TAG}_layer{layer}.npy")
    path = os.path.join(VEC_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No stored vector at {path}. Run extract_arithmetic_vectors.py --model {_args.model} first."
        )
    return torch.from_numpy(np.load(path)).float()


def first_token_dist(messages, task_vector=None, layer=None, alpha=0.0) -> np.ndarray:
    """Full softmax over the vocab at the first answer token (assistant header).

    If a vector is given, alpha*vector is added at the header (last token) — the
    same additive injection used elsewhere. No continuation is appended, so the
    header is the last position [-1].
    """
    handle = None
    if task_vector is not None and alpha != 0.0:
        tv = task_vector.to(_device)

        def _hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            if hidden.dim() == 3:
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * tv
            else:
                hidden[-1, :] = hidden[-1, :] + alpha * tv
            return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

        handle = model.model.layers[layer].register_forward_hook(_hook)

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(_device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
    finally:
        if handle is not None:
            handle.remove()

    return torch.softmax(logits, dim=-1).float().cpu().numpy()


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (natural log, in [0, ln2]). Softmax outputs are
    strictly positive; eps just guards the logs numerically."""
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * kl_pm + 0.5 * kl_qm


def ratio_label(r):
    return "-".join(map(str, r))


def inject_vector(ratio, vecs):
    """Weighted sum Σ (r_i / Σr) · v_i — the linear combination for this ratio.
    For a pure ratio this is just the single task vector."""
    total = sum(ratio)
    return sum((n / total) * vecs[TASKS[i]] for i, n in enumerate(ratio))


def build_mixed_icl(pools, rng, ratio, test_rec, test_task):
    """ICL prompt with arithmetic demos in the given task ratio plus a held-out
    addition question in `test_task` format as the final user turn.

    Handles pure and mixed ratios (pure = 3 demos of one format). Demo examples
    with the same operand pair as the held-out question are excluded so the model
    cannot copy a demonstrated addition answer.
    """
    demos = []
    for task, n in zip(TASKS, ratio):
        pool = [
            ex
            for ex in pools[task]["icl"]
            if (ex["op1"], ex["op2"]) != (test_rec["op1"], test_rec["op2"])
        ]
        for rec in rng.sample(pool, n):
            demos.append(data.format_icl_example(task, rec))
    rng.shuffle(demos)
    msgs = []
    for u, a in demos:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": data.format_test_input(test_task, test_rec)})
    return msgs


def double_center(mat: np.ndarray) -> np.ndarray:
    """Remove additive row and column main effects, leaving the interaction."""
    return (
        mat
        - mat.mean(axis=1, keepdims=True)
        - mat.mean(axis=0, keepdims=True)
        + mat.mean()
    )


def simplex_distance(r1, r2):
    """Euclidean distance between two ratios as normalized weight vectors."""
    w1 = np.array(r1, float) / sum(r1)
    w2 = np.array(r2, float) / sum(r2)
    return float(np.linalg.norm(w1 - w2))


def ceiling_diagnostic(pools, examples, ratio_set, n_icl, test_task):
    """How distinguishable are the real P_ICL(r) distributions from each other?

    Computes the mean (over held-out addition examples) JS matrix among the
    P_ICL(r) distributions for a fixed `test_task` prompt format. There is no
    injection, so the diagnostic is independent of layer/alpha. Returns
    (mean_js, simplex_dist, corr), where corr is the correlation between
    off-diagonal JS and simplex distance. A strong positive corr means behavioral
    distinguishability tracks task composition; uniformly tiny JS means a low
    behavioral ceiling (injection cannot reproduce what ICL barely differentiates).
    """
    R = len(ratio_set)
    mats = []
    for rec in examples:
        P = []
        for ri, r in enumerate(ratio_set):
            dists = [
                first_token_dist(
                    build_mixed_icl(
                        pools, random.Random(1000 * ri + s), r, rec, test_task
                    )
                )
                for s in range(n_icl)
            ]
            P.append(np.mean(dists, axis=0))
        M = np.array([[js_divergence(P[i], P[j]) for j in range(R)] for i in range(R)])
        mats.append(M)
    mean_js = np.mean(mats, axis=0)
    dist = np.array(
        [
            [simplex_distance(ratio_set[i], ratio_set[j]) for j in range(R)]
            for i in range(R)
        ]
    )
    iu = np.triu_indices(R, k=1)
    corr = float(np.corrcoef(mean_js[iu], dist[iu])[0, 1])
    return mean_js, dist, corr


def per_datapoint_matrices(
    pools, examples, vecs, ratio_set, layer, alpha, n_icl, test_task
):
    """Return list of R×R raw JS matrices M[r][r'] = JS(Q_inject(r), P_ICL(r')),
    one per held-out addition example, where R = len(ratio_set).

    Q_inject(r) injects the weighted-sum vector Σ w_i v_i; P_ICL(r') is the real
    ratio-r' ICL output distribution for the same `test_task` prompt format,
    averaged over n_icl demo samplings.
    """
    mats = []
    for rec in examples:
        prompt = [{"role": "user", "content": data.format_test_input(test_task, rec)}]
        # P_ICL(r'): average over n_icl demo samplings of the real ratio-r' ICL prompt
        P = {}
        for ri, r in enumerate(ratio_set):
            dists = []
            for s in range(n_icl):
                rng = random.Random(1000 * ri + s)
                dists.append(
                    first_token_dist(build_mixed_icl(pools, rng, r, rec, test_task))
                )
            P[r] = np.mean(dists, axis=0)
        # Q_inject(r): inject the weighted-sum vector for ratio r
        Q = {
            r: first_token_dist(prompt, inject_vector(r, vecs), layer, alpha)
            for r in ratio_set
        }
        M = np.array(
            [[js_divergence(Q[r], P[rp]) for rp in ratio_set] for r in ratio_set]
        )
        mats.append(M)
    return mats


def bootstrap_double_centered(mats, n_bootstrap):
    """Point estimate + 95% CI (per cell) of the double-centered mean matrix,
    resampling datapoints with replacement."""
    arr = np.stack(mats)  # (n_datapoints, R, R)
    R = arr.shape[1]
    point = double_center(arr.mean(axis=0))
    n = len(arr)
    rng = np.random.default_rng(0)
    boot = np.empty((n_bootstrap, R, R))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot[b] = double_center(arr[idx].mean(axis=0))
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return point, lo, hi, arr.mean(axis=0), boot


def mean_diag_stat(point, boot):
    """Mean of the double-centered diagonal (NEGATIVE = task-matching) + 95% CI,
    where the CI is taken over the bootstrap distribution of that same mean."""
    idx = np.arange(point.shape[0])
    pt = float(point[idx, idx].mean())
    bd = boot[:, idx, idx].mean(axis=1)
    return pt, float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))


def permutation_test(mats, n_perm=10000, seed=0):
    """One-sided Monte-Carlo permutation test that the matched (identity) trace is
    the LOWEST (closest in JS). Per datapoint, the column labels are exchangeable
    under the null (injection is ratio-agnostic), so each null draw assigns every
    datapoint an independent uniform column permutation and sums the traces.

    The trace cancels row/column main effects by construction (each permutation
    uses every row and column once), so this runs on the raw JS matrices.
    Permutations are sampled (not enumerated), so this scales to any R — including
    the 10 ratios (10! is far too many to enumerate).

    Returns (observed_trace, null_mean_trace, p_value).
    """
    arr = np.stack(mats)  # (n_datapoints, R, R)
    n, R, _ = arr.shape
    idx = np.arange(R)
    t_obs = float(arr[:, idx, idx].sum())  # identity (matched) trace, pooled

    rng = np.random.default_rng(seed)
    # Independent random column permutation per (draw, datapoint) via argsort trick.
    perms = np.argsort(rng.random((n_perm, n, R)), axis=2)  # (n_perm, n, R)
    c_idx = np.arange(n)[None, :, None]
    t_idx = idx[None, None, :]
    gathered = arr[c_idx, t_idx, perms]  # (n_perm, n, R): M[c, t, π(t)]
    t_null = gathered.sum(axis=(1, 2))  # pooled trace per draw

    count = int((t_null <= t_obs).sum())  # one-sided: matched should be lowest
    p = (1 + count) / (1 + n_perm)  # add-one correction
    return t_obs, float(t_null.mean()), p


def print_results(point, lo, hi, raw_mean, labels):
    short = [l[:8] for l in labels]

    print("  Raw mean JS(Q_inject(r) || P_ICL(r'))  [lower = closer; * = row min]:")
    print("      inject \\ ICL   " + "  ".join(f"{s:>8}" for s in short))
    for ri, r in enumerate(labels):
        amin = int(np.argmin(raw_mean[ri]))
        cells = "  ".join(
            f"{raw_mean[ri, ci]:>7.4f}{'*' if ci == amin else ' '}"
            for ci in range(len(labels))
        )
        print(f"      {r:>12}   {cells}")

    diag_ok = sum(int(np.argmin(point[i])) == i for i in range(len(labels)))
    diag_sig = sum(int(hi[i, i] < 0) for i in range(len(labels)))
    print(
        f"\n  Diagonal is row-minimum: {diag_ok}/{len(labels)}   "
        f"Diagonal CI upper bound < 0: {diag_sig}/{len(labels)}"
    )


if __name__ == "__main__":
    layers = _args.layers

    ratio_set = list(data.RATIOS) if _args.all_ratios else list(data.PURE_RATIOS)
    labels = [ratio_label(r) for r in ratio_set]
    R = len(ratio_set)

    print(
        f"JS injection test | {'all 10 ratios' if _args.all_ratios else '3 pure tasks'} | "
        f"test_task={_args.test_task} | layers {layers} | alphas {_args.alphas} | n_icl={_args.n_icl} | "
        f"n_bootstrap={_args.n_bootstrap} | n_perm={_args.n_perm}"
    )

    data_pools = data.load_pools()
    examples = data_pools[_args.test_task]["test"]
    if _args.n_examples:
        examples = examples[: _args.n_examples]
    print(f"Test examples ({_args.test_task} format): {len(examples)}\n")

    if _args.ceiling:
        print(
            "Ceiling diagnostic: JS among real P_ICL(r) distributions (no injection)\n"
        )
        mean_js, dist, corr = ceiling_diagnostic(
            data_pools, examples, ratio_set, _args.n_icl, _args.test_task
        )

        print("  Mean JS(P_ICL(r) || P_ICL(r'))  [0 = identical behavior]:")
        print("      " + "  ".join(f"{l:>8}" for l in labels))
        for i, l in enumerate(labels):
            print(f"  {l:>8}  " + "  ".join(f"{mean_js[i, j]:>8.4f}" for j in range(R)))

        iu = np.triu_indices(R, k=1)
        pairs = sorted(
            ((mean_js[i, j], labels[i], labels[j], dist[i, j]) for i, j in zip(*iu)),
            key=lambda x: x[0],
        )
        print("\n  Most similar ratio pairs (smallest JS):")
        for js, a, b, d in pairs[:5]:
            print(f"    {a} ~ {b}:  JS={js:.4f}  (simplex dist {d:.2f})")
        print("  Most distinct ratio pairs (largest JS):")
        for js, a, b, d in pairs[-5:]:
            print(f"    {a} vs {b}:  JS={js:.4f}  (simplex dist {d:.2f})")

        print(
            f"\n  Off-diagonal JS range: [{mean_js[iu].min():.4f}, {mean_js[iu].max():.4f}]  "
            f"mean {mean_js[iu].mean():.4f}"
        )
        print(f"  Corr(JS, simplex distance) = {corr:+.3f}")
        print(
            "\n  High corr + a real JS range => behavioral distinguishability tracks "
            "composition (real ceiling to reproduce). Uniformly tiny JS => low ceiling: "
            "the model barely differentiates ratios behaviorally, so no alpha will help."
        )
        sys.exit(0)

    curve = []  # (layer, alpha, mean_diag, ci_lo, ci_hi, rowmin, sig, perm_p)
    for layer in layers:
        try:
            vecs = {t: load_vec(t, layer) for t in TASKS}
        except FileNotFoundError:
            print(
                f"layer {layer:>2}: SKIP (no stored vector - re-run extract_arithmetic_vectors.py)"
            )
            continue
        for alpha in _args.alphas:
            mats = per_datapoint_matrices(
                data_pools,
                examples,
                vecs,
                ratio_set,
                layer,
                alpha,
                _args.n_icl,
                _args.test_task,
            )
            point, lo, hi, raw_mean, boot = bootstrap_double_centered(
                mats, _args.n_bootstrap
            )
            mdiag, mlo, mhi = mean_diag_stat(point, boot)
            rowmin = sum(int(np.argmin(point[i])) == i for i in range(R))
            sig = mhi < 0
            t_obs, t_null, perm_p = permutation_test(mats, _args.n_perm)
            curve.append((layer, alpha, mdiag, mlo, mhi, rowmin, sig, perm_p))
            print(
                f"layer {layer:>2}, alpha {alpha:>4}: mean centered diag = "
                f"{mdiag:>+7.4f} [{mlo:>+7.4f}, {mhi:>+7.4f}]  "
                f"row-min {rowmin}/{R}  boot-sig: {'YES' if sig else 'no'}  "
                f"perm-p: {perm_p:.4f}  (trace obs {t_obs:.3f} vs null {t_null:.3f})"
            )
            if _args.full:
                print_results(point, lo, hi, raw_mean, labels)
                print()

    # Depth curve: bootstrap effect size (CI) + permutation p-value per layer
    print("\n=== Task-matching vs depth ===")
    print(
        "  bootstrap meanDiag (effect size, more negative = stronger) | permutation p (significance)"
    )
    print(
        f"{'layer':>6} {'alpha':>6} {'meanDiag':>10} {'95% CI':>22} {'row-min':>8} {'perm-p':>8}"
    )
    for layer, alpha, mdiag, mlo, mhi, rowmin, sig, perm_p in curve:
        star = "*" if perm_p < 0.05 else " "
        print(
            f"{layer:>6} {alpha:>6} {mdiag:>+10.4f}  [{mlo:>+7.4f}, {mhi:>+7.4f}]  "
            f"{str(rowmin) + '/' + str(R):>8} {perm_p:>7.4f}{star}"
        )

    print(
        "\nPermutation p < 0.05 (*): injecting the (weighted-sum) vector for ratio r "
        "reproduces real ratio-r ICL behavior better than chance relabeling. "
        "Bootstrap meanDiag/CI gives the effect size."
    )
    if _args.all_ratios:
        print(
            "Note: with 10 ratios, adjacent ratios (e.g. 2-1-0 vs 1-2-0) are genuinely "
            "similar, so row-min/10 will be < 10 even when the effect is real — lean on the "
            "permutation p (pooled trace), which stays valid because random relabelings pair "
            "dissimilar ratios."
        )
