"""
Infer the task mixture of an arbitrary ICL prompt by decomposing its contrast
vector as a linear combination of the stored pure-task centroids.

Supports two experiments:
  arithmetic  Direct / MCQ / Verification  (format-varying)
  mmlu        History / Law / ML           (semantic-varying, shared format)

Method:
  Reparametrised OLS enforcing sum-to-1:
    x_probe - c_anchor ≈ Σ αᵢ·(cᵢ - c_anchor)
    α_anchor = 1 - Σ αᵢ

  Important: centroid extraction and probe extraction must use the same
  rotate_test protocol. Mismatches cause systematic bias even after
  zero-shot subtraction, because the ICL-test interaction term is not
  removed by the contrast.

Usage:
  python experiments/task_vector_injection/decompose_task_mixture.py
  python experiments/task_vector_injection/decompose_task_mixture.py --experiment mmlu
  python experiments/task_vector_injection/decompose_task_mixture.py --ratio 2 1 0
  python experiments/task_vector_injection/decompose_task_mixture.py --layer 5
"""

import sys, os, argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--experiment", choices=["arithmetic", "mmlu", "entity"], default="arithmetic")
_parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                     help="HuggingFace model ID to use.")
_parser.add_argument("--quantize", default="", choices=["", "int8", "int4"],
                     help="Quantize weights via quanto (recommended for 8B on MPS).")
_parser.add_argument("--layer", type=int, default=None,
                     help="Layer to use. Defaults: 8 for arithmetic, 14 for mmlu, 8 for entity.")
_parser.add_argument("--ratio", type=int, nargs=3, default=None,
                     metavar=("N_0", "N_1", "N_2"),
                     help="Single ICL ratio to probe, e.g. --ratio 2 1 0.")
_parser.add_argument("--sweep", action="store_true",
                     help="Probe all 10 simplex ratios from data.RATIOS. "
                          "Overridden by --ratio if both are given.")
_parser.add_argument("--select_layer", action="store_true",
                     help="Scan all stored layers, run pure conditions at each, "
                          "and report norm spread, condition number, and mean L2. "
                          "Prints a summary table and recommends the best layer.")
_parser.add_argument("--n_samples", type=int, default=50,
                     help="Samples to average for the probe contrast vector.")
_parser.add_argument("--ci", action="store_true",
                     help="Report empirical 95%% CI via bootstrap resampling of the "
                          "probe contrast vectors. No additional forward passes needed.")
_parser.add_argument("--n_bootstrap", type=int, default=1000,
                     help="Number of bootstrap resamples for CI estimation.")
_parser.add_argument("--noise", action="store_true",
                     help="Permutation baseline: replace stored centroids with random "
                          "Gaussian vectors of the same per-task norm. Runs n_bootstrap "
                          "independent draws and reports null L2 mean ± std alongside "
                          "the actual L2, giving an empirical p-value.")
_args = _parser.parse_args()

_dir  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_dir))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "data"))

os.environ["TASK_VECTOR_MODEL"]    = _args.model
os.environ["TASK_VECTOR_QUANTIZE"] = _args.quantize

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

import random
import numpy as np
from local_model import get_activations_all_layers

if _args.experiment == "arithmetic":
    import data_arithmetic_formats as data
    DEFAULT_LAYER = 3
elif _args.experiment == "entity":
    import data_entity_attribute as data
    DEFAULT_LAYER = 8
else:
    import data_semantic_domains as data
    DEFAULT_LAYER = 14

LAYER   = _args.layer if _args.layer is not None else DEFAULT_LAYER
VEC_DIR = os.path.join(_dir, "vectors")
TASKS   = data.TASKS


# --------------------------------------------------------------------------
# Load stored centroids
# --------------------------------------------------------------------------

def _centroid_path(task: str, layer: int, samples: bool = False) -> str:
    suffix = "_samples" if samples else ""
    tag = "" if _IS_DEFAULT else f"_{MODEL_TAG}"
    if _args.experiment == "arithmetic":
        return os.path.join(VEC_DIR, f"arith_contrast_{task}{tag}_layer{layer}{suffix}.npy")
    if _args.experiment == "entity":
        return os.path.join(VEC_DIR, f"entity_contrast_{task}{tag}_layer{layer}{suffix}.npy")
    return os.path.join(VEC_DIR, f"{task}{tag}_layer{layer}{suffix}.npy")


def load_centroid(task: str, layer: int) -> np.ndarray:
    path = _centroid_path(task, layer)
    if not os.path.exists(path):
        script = {"arithmetic": "extract_arithmetic_vectors.py",
                  "entity":     "extract_entity.py"}.get(_args.experiment, "extract_vectors.py")
        raise FileNotFoundError(
            f"No stored vector at {path}.\n"
            f"Run experiments/task_vector_injection/{script} first."
        )
    return np.load(path)


# --------------------------------------------------------------------------
# Layer selection helpers
# --------------------------------------------------------------------------

def stored_layers() -> list[int]:
    """Return sorted list of layers for which ALL tasks have stored vectors."""
    import re
    tag = "" if _IS_DEFAULT else re.escape(MODEL_TAG) + "_"
    pattern = {
        "arithmetic": rf"arith_contrast_\w+?_{tag}layer(\d+)\.npy" if tag else r"arith_contrast_\w+_layer(\d+)\.npy",
        "entity":     rf"entity_contrast_\w+?_{tag}layer(\d+)\.npy" if tag else r"entity_contrast_\w+_layer(\d+)\.npy",
    }.get(_args.experiment, rf"[A-Za-z]+_{tag}layer(\d+)\.npy" if tag else r"[A-Za-z]+_layer(\d+)\.npy")
    layers = set()
    for fname in os.listdir(VEC_DIR):
        m = re.fullmatch(pattern, fname)
        if m:
            layers.add(int(m.group(1)))
    # Keep only layers where every task has a file
    return sorted(l for l in layers
                  if all(os.path.exists(_centroid_path(t, l)) for t in TASKS))


def layer_diagnostics(centroids: dict) -> tuple[float, float]:
    """Return (norm_spread, condition_number) for a set of centroids."""
    norms       = [np.linalg.norm(centroids[t]) for t in TASKS]
    norm_spread = max(norms) - min(norms)
    c           = [centroids[t] for t in TASKS]
    X           = np.column_stack([ci - c[-1] for ci in c[:-1]])
    return norm_spread, float(np.linalg.cond(X))


# --------------------------------------------------------------------------
# Probe contrast vector
# --------------------------------------------------------------------------

def extract_contrast_vector(pools, rng, ratio, sample_idx, layer) -> np.ndarray:
    msgs     = data.build_messages(pools, rng, ratio, sample_idx=sample_idx,
                                   rotate_test=True)
    icl_acts = get_activations_all_layers(msgs,      [layer])
    zs_acts  = get_activations_all_layers(msgs[-1:], [layer])
    return (icl_acts[layer] - zs_acts[layer]).float().numpy()


def individual_contrast_vectors(pools, ratio, layer, n_samples) -> np.ndarray:
    """Return (n_samples, hidden_dim) array of individual contrast vectors."""
    vecs = []
    for i in range(n_samples):
        rng = random.Random(i + 9999)   # offset from centroid extraction seeds
        vecs.append(extract_contrast_vector(pools, rng, ratio, i, layer))
    return np.stack(vecs)


def mean_contrast_vector(pools, ratio, layer, n_samples) -> np.ndarray:
    return individual_contrast_vectors(pools, ratio, layer, n_samples).mean(axis=0)


def bootstrap_ci(vecs: np.ndarray, centroids: dict,
                 n_bootstrap: int = 1000) -> dict[str, tuple[float, float]]:
    """Empirical 95% CI for each α via bootstrap resampling.

    Resamples rows of `vecs` (n_samples, hidden_dim) with replacement,
    computes the mean of each resample, decomposes it, and collects the
    distribution of α values. Returns {task: (lower_2.5%, upper_97.5%)}.
    No additional forward passes required.
    """
    n    = len(vecs)
    rng  = np.random.default_rng(0)
    boot = np.array([
        list(decompose(vecs[rng.integers(0, n, size=n)].mean(axis=0), centroids).values())
        for _ in range(n_bootstrap)
    ])   # (n_bootstrap, n_tasks)
    return {
        t: (float(np.percentile(boot[:, i], 2.5)),
            float(np.percentile(boot[:, i], 97.5)))
        for i, t in enumerate(TASKS)
    }


# --------------------------------------------------------------------------
# Noise baseline
# --------------------------------------------------------------------------

def make_noise_centroids(real_centroids: dict, rng: np.random.Generator) -> dict:
    """Replace task-specific residuals with random Gaussian noise, preserving the
    common ICL direction shared by all real centroids.

    Each real centroid = common + task_residual. We keep common and replace
    task_residual with a random vector of the same norm. This tests whether the
    task-specific geometry matters, rather than whether having any ICL signal at all
    matters (which pure zero-mean noise would conflate).
    """
    hidden_dim = next(iter(real_centroids.values())).shape[0]
    common     = np.mean(list(real_centroids.values()), axis=0)
    noise      = {}
    for t, c in real_centroids.items():
        residual      = c - common
        residual_norm = np.linalg.norm(residual)
        v             = rng.standard_normal(hidden_dim).astype(np.float32)
        noise[t]      = common + v * (residual_norm / np.linalg.norm(v))
    return noise


def null_l2_distribution(x: np.ndarray, real_centroids: dict,
                          true_weights: dict, n_draws: int) -> np.ndarray:
    """L2 errors from n_draws independent random-centroid decompositions."""
    rng = np.random.default_rng(42)
    l2s = []
    for _ in range(n_draws):
        alpha = decompose(x, make_noise_centroids(real_centroids, rng))
        l2s.append(sum((alpha[t] - true_weights[t])**2 for t in TASKS) ** 0.5)
    return np.array(l2s)


# --------------------------------------------------------------------------
# Reparametrised OLS: coefficients sum to 1
# --------------------------------------------------------------------------

def decompose(x: np.ndarray, centroids: dict) -> dict[str, float]:
    c      = [centroids[t] for t in TASKS]
    anchor = c[-1]
    X      = np.column_stack([ci - anchor for ci in c[:-1]])
    y      = x - anchor
    a, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha  = list(a) + [1.0 - sum(a)]
    return {t: float(v) for t, v in zip(TASKS, alpha)}


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def print_result(alpha: dict, true_weights: dict,
                 ci: dict[str, tuple[float, float]] | None = None) -> None:
    for task, a in alpha.items():
        true_w = true_weights[task]
        ci_str = (f"  95% CI [{ci[task][0]:+.3f}, {ci[task][1]:+.3f}]"
                  if ci else "")
        print(f"    {task:<14} α={a:+.3f}  (true {true_w:.2f}){ci_str}")
    l2 = sum((alpha[t] - true_weights[t])**2 for t in TASKS) ** 0.5
    print(f"    L2 error: {l2:.4f}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_probe(pools, ratio, centroids) -> None:
    total        = sum(ratio)
    true_weights = {t: ratio[i] / total for i, t in enumerate(TASKS)}
    label        = "-".join(map(str, ratio))

    print(f"\nRatio {label}  "
          f"({', '.join(f'{t}={w:.2f}' for t, w in true_weights.items())})")
    print(f"  Extracting probe at layer {LAYER} ({_args.n_samples} samples)...",
          end=" ", flush=True)

    vecs  = individual_contrast_vectors(pools, ratio, LAYER, _args.n_samples)
    x     = vecs.mean(axis=0)
    alpha = decompose(x, centroids)
    print("done")

    ci = bootstrap_ci(vecs, centroids, _args.n_bootstrap) if _args.ci else None
    if ci:
        print(f"  Bootstrap CI ({_args.n_bootstrap} resamples)...")
    print_result(alpha, true_weights, ci)

    if _args.noise:
        null = null_l2_distribution(x, centroids, true_weights, _args.n_bootstrap)
        actual_l2 = sum((alpha[t] - true_weights[t])**2 for t in TASKS) ** 0.5
        p_value   = float((null <= actual_l2).mean())
        print(f"  Null L2 ({_args.n_bootstrap} random-centroid draws): "
              f"mean={null.mean():.4f}  std={null.std():.4f}  "
              f"p={p_value:.4f}")


if __name__ == "__main__":
    print(f"Experiment : {_args.experiment}  |  Tasks: {TASKS}")

    print("\nLoading data pools...")
    pools = data.load_pools()

    if _args.select_layer:
        layers = stored_layers()
        if not layers:
            print("No stored vectors found. Run the extraction script first.")
            sys.exit(1)

        print(f"\nScanning {len(layers)} stored layers: {layers}")
        print(f"Running pure conditions ({_args.n_samples} samples each)...\n")
        print(f"{'Layer':>6}  {'Norm spread':>12}  {'Cond #':>8}  {'Mean L2 (pure)':>15}")
        print("─" * 50)

        best_layer, best_l2 = None, float("inf")
        for layer in layers:
            centroids    = {t: load_centroid(t, layer) for t in TASKS}
            spread, cond = layer_diagnostics(centroids)
            l2s = []
            for pure_ratio in data.PURE_RATIOS:
                total        = sum(pure_ratio)
                true_weights = {t: pure_ratio[i] / total for i, t in enumerate(TASKS)}
                x            = mean_contrast_vector(pools, pure_ratio, layer, _args.n_samples)
                alpha        = decompose(x, centroids)
                l2s.append(sum((alpha[t] - true_weights[t])**2 for t in TASKS) ** 0.5)
            mean_l2 = float(np.mean(l2s))
            print(f"{layer:>6}  {spread:>12.4f}  {cond:>8.2f}  {mean_l2:>15.4f}")
            if mean_l2 < best_l2:
                best_l2, best_layer = mean_l2, layer

        print("─" * 50)
        print(f"\nBest layer: {best_layer}  (mean pure-condition L2 = {best_l2:.4f})")

    else:
        print(f"Layer      : {LAYER}  |  n_samples: {_args.n_samples}")
        print(f"Loading stored centroids at layer {LAYER}...")
        centroids = {t: load_centroid(t, LAYER) for t in TASKS}

        if _args.ratio:
            ratios = [tuple(_args.ratio)]
        elif _args.sweep:
            ratios = list(data.RATIOS)
        else:
            ratios = list(data.PURE_RATIOS) + [(1, 1, 1)]
        for ratio in ratios:
            run_probe(pools, ratio, centroids)

    print("\nDone.")
