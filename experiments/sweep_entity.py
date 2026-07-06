"""
Layer-by-layer scatter of contrast vector projections for the semantic-relation
entity→attribute group (Capital / Currency / Language) across all ICL ratios.

Mirrors sweep_arithmetic.py / sweep_mmlu.py but uses data_entity_attribute. All
three tasks share the same surface form (country in, short answer out); any
separability reflects the semantic *relation* taught by the demonstrations, not
format. This is the semantic-relation counterpart to the format-varying
arithmetic group and the semantic-domain MMLU group.

κ is computed in the original contrast-vector space (LDA is used only for the
2-D scatter); the cross-model κ JSON feeds plot_kappa_comparison.py.

Layers are chosen as fixed fractions of model depth so results are comparable
across model sizes.

Usage:
    python experiments/sweep_entity.py
    python experiments/sweep_entity.py --model meta-llama/Llama-3.2-1B-Instruct
    python experiments/sweep_entity.py --model meta-llama/Llama-3.1-8B-Instruct --quantize int8
"""

import json
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "data"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # experiments/ (kappa_null)

# Parse --model BEFORE importing local_model (model loads at import time)
import argparse as _ap
_mp = _ap.ArgumentParser(add_help=False)
_mp.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                 help="HuggingFace model ID to use.")
_mp.add_argument("--quantize", default="", choices=["", "int8", "int4"],
                 help="Quantize weights via quanto (recommended for 8B on MPS).")
_mp.add_argument("--null-check", action="store_true",
                 help="Also compute the exact 720-permutation null for κ (per layer).")
_margs, _ = _mp.parse_known_args()
os.environ["TASK_VECTOR_MODEL"]    = _margs.model
os.environ["TASK_VECTOR_QUANTIZE"] = _margs.quantize

import re
def _make_model_tag(model_id: str) -> str:
    size = (re.search(r'(\d+\.?\d*[Bb])', model_id) or type("", (), {"group": lambda s, n: model_id.split("/")[-1]})()).group(1).upper()
    name = model_id.lower()
    if "qwen"    in name: return f"Qwen-{size}"
    if "llama"   in name: return f"Llama-{size}"
    if "mistral" in name: return f"Mistral-{size}"
    if "gemma"   in name: return f"Gemma-{size}"
    return model_id.split("/")[-1]
MODEL_TAG  = _make_model_tag(_margs.model)
NULL_CHECK = _margs.null_check

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from local_model import get_activations_all_layers, n_layers
from kappa_null import kappa_permutation_null
from data_entity_attribute import (
    TASKS, RATIOS, PURE_RATIOS, N_VECTOR_SAMPLES, load_pools, build_messages,
    format_test_input,
)

# Fixed fractions of model depth — comparable across 1B (16L), 3B (28L), 8B (32L)
_LAYER_FRACS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.75]
SWEEP_LAYERS = sorted(set(max(1, round(f * n_layers)) for f in _LAYER_FRACS))
N_SAMPLES    = 50

print(f"Model: {_margs.model}  ({n_layers} layers)")
print(f"Sweep layers: {SWEEP_LAYERS}")

# Base RGB for each pure task — used for ternary colour mixing
_PURE_RGB = np.array([
    [0.27, 0.51, 0.71],   # steelblue  → Capital
    [0.18, 0.55, 0.34],   # seagreen   → Currency
    [0.85, 0.55, 0.13],   # darkorange → Language
])


def ratio_color(ratio: tuple) -> np.ndarray:
    """Blend the three pure colours proportionally to the ratio weights."""
    w = np.array(ratio, dtype=float)
    w /= w.sum()
    return np.clip(w @ _PURE_RGB, 0, 1)


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------

def collect_contrasts(
    pools,
    layers: list[int],
    ratios: list[tuple],
    n_samples: int,
) -> dict:
    """
    Returns results[ratio][layer]: np.ndarray shape (n_samples, hidden_dim).

    Zero-shot activations are precomputed once per sample (not once per ratio),
    since rotate_test=True makes the test question identical across all ratios
    for a given sample index. The test input here is a bare country name.
    """
    print("  Precomputing zero-shot activations...", end=" ", flush=True)
    zs_cache: dict[int, dict[int, torch.Tensor]] = {}
    for i in range(n_samples):
        # Reproduce exactly what build_messages does for the test question:
        # test_task is fixed by sample_idx (rotate_test=True) and the test record
        # is the first rng call, so it is ratio-independent.
        rng       = random.Random(i)
        test_task = TASKS[i % len(TASKS)]
        test_rec  = rng.choice(pools[test_task]["test"])
        zs_msgs   = [{"role": "user", "content": format_test_input(test_task, test_rec)}]
        zs_cache[i] = get_activations_all_layers(zs_msgs, layers)
    print("done")

    results = {r: {l: [] for l in layers} for r in ratios}

    for ratio in ratios:
        label = "-".join(map(str, ratio))
        print(f"  {label} ({n_samples} samples)...", end=" ", flush=True)
        for i in range(n_samples):
            rng      = random.Random(i)
            msgs     = build_messages(pools, rng, ratio, sample_idx=i, rotate_test=True)
            icl_acts = get_activations_all_layers(msgs, layers)
            for l in layers:
                diff = (icl_acts[l] - zs_cache[i][l]).float().numpy()
                results[ratio][l].append(diff)
        print("done")

    for ratio in ratios:
        for l in layers:
            results[ratio][l] = np.stack(results[ratio][l])   # (n, hidden)

    return results


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_layer_grid(results: dict, layers: list[int]) -> None:
    n_cols = 4
    n_rows = (len(layers) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    # kappa_data[layer] = list of valid κ values across mixed ratios
    kappa_data: dict[str, list[float]] = {}
    # kappa_null_data[layer] = exact permutation-null summary (only if NULL_CHECK)
    kappa_null_data: dict[str, dict] = {}
    # ratio_centroids_out["{layer}|{ratio}"] = original-space actual centroid μ_r^l
    # for all 10 ratios; persisted (see below) so ρ / κ / cosine can be recomputed
    # offline without re-running the model.
    ratio_centroids_out: dict[str, np.ndarray] = {}

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]

        # --- Fit LDA on pure conditions ---
        X_pure = np.vstack([results[p][layer] for p in PURE_RATIOS])
        y_pure = np.repeat(np.arange(len(PURE_RATIOS)), N_SAMPLES)
        lda    = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(X_pure, y_pure)

        pure_centroids: dict[tuple, np.ndarray] = {}

        # --- Plot all ratios ---
        for ratio in RATIOS:
            proj     = lda.transform(results[ratio][layer])  # (n, 2)
            color    = ratio_color(ratio)
            is_pure  = ratio in PURE_RATIOS

            ax.scatter(
                proj[:, 0], proj[:, 1],
                color=color,
                s=28 if is_pure else 14,
                alpha=0.75 if is_pure else 0.45,
                zorder=3,
            )

            centroid = proj.mean(axis=0)

            if is_pure:
                pure_centroids[ratio] = centroid
                ax.scatter(*centroid, color=color, s=140,
                           edgecolors="black", linewidths=0.9, zorder=6)
                task_name = TASKS[PURE_RATIOS.index(ratio)]
                ax.annotate(task_name, centroid,
                            textcoords="offset points", xytext=(5, 4), fontsize=7)

        # Triangle centre — geometric mean of pure centroids (2-D, for plotting only)
        tri_center = np.mean([pure_centroids[p] for p in PURE_RATIOS], axis=0)

        # --- Original-space quantities for κ (LDA used only for scatter plots) ---
        # d_predicted lies in the span of the pure centroids (the LDA plane), so
        # projecting to 2-D loses no information about the predicted direction.
        # However d_actual may drift orthogonal to that plane; computing κ in the
        # original space catches those deviations, making it the stronger test.
        orig_centroids_map = {r: results[r][layer].mean(axis=0) for r in RATIOS}
        for _r, _c in orig_centroids_map.items():
            ratio_centroids_out[f"{layer}|{'-'.join(map(str, _r))}"] = _c.astype(np.float32)
        pure_orig       = {p: orig_centroids_map[p] for p in PURE_RATIOS}
        tri_center_orig = np.mean(list(pure_orig.values()), axis=0)
        tri_scale_orig  = np.mean([np.linalg.norm(pure_orig[p] - tri_center_orig)
                                    for p in PURE_RATIOS])

        layer_kappas = []
        layer_pairs  = []   # (d_act, d_pred) for mixed ratios with valid prediction
        for ratio in RATIOS:
            if ratio in PURE_RATIOS:
                continue
            proj      = lda.transform(results[ratio][layer])
            color     = ratio_color(ratio)
            centroid  = proj.mean(axis=0)

            total     = sum(ratio)
            predicted = sum(
                (n / total) * pure_centroids[p]
                for n, p in zip(ratio, PURE_RATIOS)
            )

            # κ computed in the original contrast-vector space (not LDA 2-D).
            # Undefined (shown as "—") when predicted ≈ triangle centre (e.g. 1-1-1).
            orig_c    = orig_centroids_map[ratio]
            orig_p    = sum((n / total) * pure_orig[p] for n, p in zip(ratio, PURE_RATIOS))
            d_act_o   = orig_c - tri_center_orig
            d_pred_o  = orig_p - tri_center_orig
            denom     = np.dot(d_pred_o, d_pred_o)
            kappa_str = "—"
            kappa     = None
            if np.sqrt(denom) > 0.05 * tri_scale_orig:
                kappa     = float(np.dot(d_act_o, d_pred_o) / denom)
                kappa_str = f"{kappa:.2f}"
                layer_kappas.append(kappa)
                layer_pairs.append((d_act_o, d_pred_o))
            dist = np.linalg.norm(centroid - predicted)   # 2-D visual distance for annotation

            ax.scatter(*centroid,  color=color, s=70,
                       edgecolors="black", linewidths=0.6, zorder=5)
            ax.scatter(*predicted, color="gray", s=60,
                       marker="x", linewidths=1.4, zorder=5)
            ax.plot(
                [predicted[0], centroid[0]],
                [predicted[1], centroid[1]],
                color="gray", linewidth=0.7, linestyle=":",
            )
            label = "-".join(map(str, ratio))
            ax.annotate(f"{label}\nd={dist:.1f} κ={kappa_str}", centroid,
                        textcoords="offset points", xytext=(4, -16),
                        fontsize=6, color="dimgray")

        kappa_data[str(layer)] = layer_kappas

        if NULL_CHECK and len(layer_pairs) >= 2:
            null = kappa_permutation_null(layer_pairs)
            kappa_null_data[str(layer)] = null
            print(f"    layer {layer}: mean κ = {null['S_obs']:.3f}  "
                  f"exact perm-p = {null['p_value']:.4g}  "
                  f"(null mean {null['null_mean']:.3f}, n_perm {null['n_perm']})")

        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_xlabel("LDA 1", fontsize=8)
        ax.set_ylabel("LDA 2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused panels
    for ax_idx in range(len(layers), len(axes)):
        axes[ax_idx].set_visible(False)

    # Shared legend
    legend_handles = [
        plt.scatter([], [], color=ratio_color(p), s=60,
                    edgecolors="black", linewidths=0.6, label=TASKS[i])
        for i, p in enumerate(PURE_RATIOS)
    ] + [
        plt.scatter([], [], color="gray", marker="x", s=60,
                    linewidths=1.4, label="Predicted mixed centroid"),
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=9, ncol=2,
               framealpha=0.9)

    fig.suptitle(
        "Entity→attribute (country) — LDA projections by layer\n"
        "Large dots = pure-task centroids  ·  Small dots = individual samples  "
        "·  × = ratio-predicted centroid",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    _results_dir = os.path.join(_root, "results")
    os.makedirs(_results_dir, exist_ok=True)
    out = os.path.join(_results_dir, f"entity_layer_sweep_projections_{MODEL_TAG}_rotated.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved → {out}")

    # Save κ data for cross-model comparison plot
    json_out = os.path.join(_results_dir, f"entity_kappa_{MODEL_TAG}.json")
    with open(json_out, "w") as f:
        json.dump({
            "model":      _margs.model,
            "n_layers":   n_layers,
            "kappa":      kappa_data,        # {layer_str: [kappa, ...]}
            "kappa_null": kappa_null_data,   # {layer_str: {S_obs, p_value, ...}}
        }, f, indent=2)
    print(f"Saved κ data → {json_out}")

    # Persist the actual per-ratio centroids so ρ / κ / cosine can be recomputed
    # offline without re-running the model. Kept out of vectors/ (which holds the
    # pure-task injection vectors); ratio_centroids/ holds all 10 ratios per layer.
    _rc_dir = os.path.join(_root, "ratio_centroids")
    os.makedirs(_rc_dir, exist_ok=True)
    _rc_out = os.path.join(_rc_dir, f"entity_{MODEL_TAG}.npz")
    np.savez(_rc_out, **ratio_centroids_out)
    print(f"Saved ratio centroids → {_rc_out}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading entity→attribute pools...")
    pools = load_pools()

    print(
        f"\nCollecting contrast vectors: {len(RATIOS)} ratios × "
        f"{N_SAMPLES} samples × 2 passes, capturing {len(SWEEP_LAYERS)} layers each.\n"
    )
    results = collect_contrasts(pools, SWEEP_LAYERS, RATIOS, N_SAMPLES)
    plot_layer_grid(results, SWEEP_LAYERS)
