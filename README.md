# Beyond First-Order Task Vector Superposition: Orthogonal Residuals in In-Context Learning

An empirical study of whether In-Context Learning (ICL) examples induce **linearly superposed task representations** in transformer residual streams, how faithfully that first-order description holds **across depth**, and whether the structure it misses is **behaviorally meaningful**.

When in-context examples span multiple tasks, LLMs represent and perform all of them at once — *task superposition*. Prior work posits that this superposition is a **convex combination** of individual task vectors, but this had not been directly verified at the activation level in instruction-tuned models, nor had the fidelity and consequences of that approximation been examined. We study it three ways:

1. **Geometry (κ).** Mixed-task contrast-vector centroids land on the ratio-weighted convex combination of the pure-task centroids (κ ≈ 1), calibrated against an **exact permutation null** — significant at the floor *p* = 1/720 for every model, task group, and layer. κ is a scalar projection coefficient, however — blind to any component orthogonal to the predicted direction — so κ ≈ 1 establishes a **dominant linear structure**, not perfect collinearity, and it stays near-constant across depth, unable to reveal how the geometry changes layer to layer.
2. **The orthogonal residual.** The component κ ignores is measured directly and calibrated against a **split-half sampling-noise floor**. Its *magnitude* grows with depth, while its *resolvability* above the floor peaks in the **middle layers** — locating where the linear approximation is most faithful. Whether a resolvable orthogonal component emerges at all is governed by **task separability**, not depth alone, yielding three regimes: *residual-driven*, *floor-driven*, and *floor-limited*.
3. **Causal ablation.** Is the orthogonal residual behaviorally load-bearing, or geometric noise? We extract each ratio's systematic residual from held-out questions and surgically remove it from an evaluated question's own few-shot activation, comparing the resulting shift in the model's next-token distribution against a norm-matched null of random orthogonal directions. In the deeper layers the residual's removal changes the output distribution far more than a random direction of the same size does — including for the task group (MMLU) whose residual is smallest and statistically *unresolvable* against the split-half floor, showing that magnitude, resolvability, and causal importance are three distinct properties.

Inspired by [*Everything Everywhere All at Once* (Xiong et al., 2024)](https://arxiv.org/abs/2410.05603), which established task superposition **behaviorally** in pretrained models. Behavioral superposition is a claim about *outputs* and does not by itself entail that the underlying representations **combine linearly**; we test the representations directly, and then test whether what the linear approximation misses actually matters.

**Models (three families, 1B–8B):** Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Gemma-2-2B, Qwen-2.5-3B (instruction-tuned variants). The causal-ablation result (Result 3) is reported on Llama-3.2-3B.

---

## Core methodology

**Task vectors** are the raw activations of the assistant header in the residual stream.
**Contrast vectors** isolate the ICL signal by subtracting the zero-shot baseline:

```
contrast(ICL) = activation(ICL + test_q) − activation(test_q alone)
```

This removes the test question's independent contribution, leaving only the shift caused by the ICL context. The zero-shot activation is precomputed once per sample and reused across all ratios.

**κ (linear superposition fidelity)** quantifies how closely the actual mixed centroid tracks the ratio-weighted linear combination of pure-task centroids. For a given in-context task ratio:
* `predicted` = the ratio-weighted average of the pure-task centroids
* `actual` = the true centroid of the samples at that in-context ratio
* `center` = the arithmetic mean of the pure-task centroids

```
κ = dot(actual − center, predicted − center) / ‖predicted − center‖²
```

κ = 1 means the predicted convex direction exactly accounts for the observed displacement's projection onto it — a **projection coefficient**, not a similarity measure, so it says nothing about any component orthogonal to that direction. κ is signed and unbounded by design, so it *could* be negative or ≫ 1. κ is undefined for the equal 1-1-1 ratio (predicted = center, denominator = 0) and excluded there. All κ is computed in the **original activation space**, not the LDA projection used for the scatter plots.

**Exact permutation null.** κ ≈ 1 could in principle be an artifact of the metric's scale, so we calibrate it. For the *K* = 6 mixed ratios with a well-defined predicted displacement (the three pure ratios form the basis; 1-1-1 is excluded), we re-pair each observed displacement with a permuted prediction and score the mean κ. With 6! = 720 relabelings we enumerate the null **exactly**; the observed pairing beating all others gives the floor *p* = 1/720 ≈ 1.4 × 10⁻³. See [`experiments/kappa_null.py`](experiments/kappa_null.py).

**Orthogonal residual + split-half noise floor.** κ pins only the component of the observed displacement *along* the predicted convex direction; it is blind to any *orthogonal* deviation. We measure that residual directly as the sine of the angle between the observed and predicted displacements, `sinθ(d, d̂)`. In finite samples this can never reach 0 (the centroid carries sampling noise), so comparing it to 0 is uninformative. Instead we calibrate it against a **split-half sampling-noise floor** φ: repeatedly split a ratio's samples in half, form a displacement from each half, and take the angle between them (holding the pure-task basis at its full-sample value, 500 splits). φ is the best agreement the data can resolve — no prediction should align with `d` more closely than an independent estimate of `d` aligns with itself.

* The **magnitude** of the residual is reported with the full-sample `sinθ` (minimum-variance, so it tracks the trend across depth without being flattened by noise).
* **Resolvability** is tested with a *size-matched* residual on the same *n*/2 half-samples as the floor, so the comparison is fair. The **resolvability ratio** `R = s̃/φ` has `R = 1` as the floor: `R > 1` marks a statistically resolvable orthogonal component; `R ≤ 1` means the mixed centroid is indistinguishable from the exact convex-predicted point given sampling noise. See [`experiments/aggregate_split_half.py`](experiments/aggregate_split_half.py) and [`experiments/bootstrap_ci.py`](experiments/bootstrap_ci.py).

**Systematic orthogonal residual `v⊥(r)` (for causal ablation).** For each mixed ratio, the *average* orthogonal residual across held-out questions — one vector per (ratio, layer) — is the component of the tri-center-relative displacement `(c̄ᵣ − μ)` left over after subtracting its κ-scaled projection onto the predicted direction: `v⊥(r) := (c̄ᵣ − μ) − κ(r)·d̂ᵣ`, so `v⊥(r) ⊥ d̂ᵣ` by construction. Its relative magnitude is `ρ := ‖v⊥(r)‖ / ‖h‖`. This is distinct from the per-sample `sinθ` used for the magnitude/resolvability analysis above — it is a single systematic direction per ratio, extracted so it can later be causally removed. See [`experiments/task_vector_injection/centroid_residual_geometry.py`](experiments/task_vector_injection/centroid_residual_geometry.py).

---

## Task groups

Three groups probe superposition under different kinds of task difference, each varying one source of variation while holding the others fixed.

| Group | Tasks | Shared surface form | Varies… |
|---|---|---|---|
| **Arithmetic** | Direct (`42`) · MCQ (`B`) · Verification (`True`) | same arithmetic inputs, same underlying relation | **output format** only (early-layer signal) |
| **Entity** | Capital (`France→Paris`) · Currency (`France→Euro`) · Language (`France→French`) | a bare country name; identical format across tasks | **semantic relation**, under an underspecified prompt |
| **MMLU** | History · Law · ML | identical A/B/C/D format | **semantic domain** only (hardest case) |

In-context ratios sweep the 10 points of the 3-task simplex summing to 3: three pures (3-0-0, 0-3-0, 0-0-3), six 2-1-0 mixes, and the equal 1-1-1.

---

## Result 1 — Geometry: κ ≈ 1 everywhere, rejected against the exact null

**Mean κ across all mixed ratios and swept layers** (— = not run; Llama-8B/MMLU omitted for hardware):

| Model | Arithmetic | Entity | MMLU |
|---|---|---|---|
| Llama-3.2-1B | 1.051 | 0.983 | 0.977 |
| Llama-3.2-3B | 0.994 | 1.016 | 1.006 |
| Llama-3.1-8B | 0.997 | 0.930 | — |
| Gemma-2-2B | 1.014 | 1.038 | 0.980 |
| Qwen-2.5-3B | 1.051 | 1.033 | 0.964 |

**In every cell above, at every one of the 8 swept layers, the observed mean κ exceeds all 720 permutations** → floor *p* = 1/720 ≈ 1.4 × 10⁻³, against a null centered near 0. The convex-combination hypothesis is rejected in favor of a genuine dominant linear structure across all models and task groups.

### Cross-model κ vs. relative layer depth

Mean κ ± 1 std across mixed ratios at each relative depth; grey band = κ ∈ [0.75, 1.25].

| Arithmetic | Entity | MMLU |
|---|---|---|
| ![κ arithmetic](results/arithmetic_kappa_comparison_across_models.png) | ![κ entity](results/entity_kappa_comparison_across_models.png) | ![κ mmlu](results/mmlu_kappa_comparison_across_models.png) |

Format-varying **Arithmetic** shows the largest κ-variance growth with depth; semantic **MMLU** holds κ ≈ 1 nearly flat. Qwen and Gemma replicate the Llama pattern despite different architectures, training data, and tokenizers — supporting cross-family generalizability.

The **mean** κ, however, stays near unity at every depth in every cell. As a scalar projection it is scale-accurate but *flat across depth* — it establishes that a dominant linear structure describes the geometry, but not how that geometry changes layer to layer. That question is carried entirely by the component κ ignores: the **orthogonal residual**, analyzed next.

### Layer-sweep projections (LDA, 8 relative depths)

Large dots = pure-task centroids, small dots = samples, × = ratio-predicted centroid, dotted line = predicted→actual gap, κ annotated per mixed ratio.

**Arithmetic**

| Llama-1B | Llama-3B | Llama-8B | Gemma-2B | Qwen-3B |
|---|---|---|---|---|
| ![](results/arithmetic_layer_sweep_projections_Llama-1B_rotated.png) | ![](results/arithmetic_layer_sweep_projections_Llama-3B_rotated.png) | ![](results/arithmetic_layer_sweep_projections_Llama-8B_rotated.png) | ![](results/arithmetic_layer_sweep_projections_Gemma-2B_rotated.png) | ![](results/arithmetic_layer_sweep_projections_Qwen-3B_rotated.png) |

**Entity**

| Llama-1B | Llama-3B | Llama-8B | Gemma-2B | Qwen-3B |
|---|---|---|---|---|
| ![](results/entity_layer_sweep_projections_Llama-1B_rotated.png) | ![](results/entity_layer_sweep_projections_Llama-3B_rotated.png) | ![](results/entity_layer_sweep_projections_Llama-8B_rotated.png) | ![](results/entity_layer_sweep_projections_Gemma-2B_rotated.png) | ![](results/entity_layer_sweep_projections_Qwen-3B_rotated.png) |

**MMLU**

| Llama-1B | Llama-3B | Gemma-2B | Qwen-3B |
|---|---|---|---|
| ![](results/mmlu_layer_sweep_projections_Llama-1B_rotated.png) | ![](results/mmlu_layer_sweep_projections_Llama-3B_rotated.png) | ![](results/mmlu_layer_sweep_projections_Gemma-2B_rotated.png) | ![](results/mmlu_layer_sweep_projections_Qwen-3B_rotated.png) |

---

## Result 2 — The orthogonal residual: where linearity strains, and why

κ confirms the mixed centroid lies *along* the predicted convex direction, but a centroid can project with unit coefficient and still sit *off* the predicted point. We measure that orthogonal residual `sinθ(d, d̂)` directly and calibrate it against the split-half sampling-noise floor φ (see *Core methodology*).

**Two findings.**
1. The residual's **magnitude** is smallest in the early layers (sinθ ≈ 0.30–0.49, at or below the floor) and substantially larger by the deepest layers (≈ 0.55–0.83) — the linear approximation is geometrically **tightest early**.
2. Whether the deeper residual is *statistically resolvable* above the floor is governed by **task separability, not depth alone**: for well-separated groups the pure-task displacements sharpen with depth, shrinking φ until the residual clears it; for entangled MMLU (three subjects sharing one A/B/C/D format) the floor stays high, so resolvability only *approaches* the floor with depth — clearing it marginally and only at the deepest layers in the two best-separated models (Gemma-2B, Qwen-3B; R ≈ 1.05–1.09), while two of the five models stay at or below it throughout.

This is about *resolving power*, not a breakdown of linearity — κ stays ≈ 1 in every case, and the convex combination remains the dominant structure at all depths. Whether the unresolved residual still *matters behaviorally* is exactly what Result 3 tests.

### Residual magnitude and resolvability vs. depth

Left: full-sample residual magnitude (solid) vs. its split-half floor (dashed), per model. Right: resolvability ratio `R = s̃/φ` aggregated across models per group; dashed line = floor (R = 1), above which the orthogonal component is resolvable.

| Residual magnitude (+ noise floor) | Resolvability `R = s̃/φ` |
|---|---|
| ![orthogonal residual vs depth](results/orthogonal_residual_depth.png) | ![resolvability ratio vs depth](results/orthogonal_residual_ratio_depth_by_model.png) |

The residual's magnitude grows with depth **everywhere**, but its resolvability does **not**: it peaks in the **middle layers** for the well-separated groups and decays back toward the floor by the deepest layers, while MMLU only *approaches* the floor with depth — clearing it marginally at the deepest layers in two models (Gemma-2B, Qwen-3B; R ≈ 1.05–1.09) and staying at or below it throughout in the others.

### One mechanism, three regimes

Resolvability `R = s̃/φ` has **two levers**: it rises when the orthogonal residual *grows* (numerator up) and when task separability *sharpens*, shrinking the floor (denominator down). Decomposing the ascending run of `R` up to its peak — per-model least-squares slopes of the residual and the floor vs. relative depth, averaged across the five models:

| Group | Mean peak depth | Residual ROC | Floor ROC | Regime |
|---|---|---|---|---|
| **Arithmetic** | 0.38 | **+1.0** | 0.0 | **Residual-driven** — an emerging orthogonal component rises into a near-static floor |
| **Entity** | 0.56 | +0.2 | **−1.3** | **Floor-driven** — a collapsing floor exposes an almost-flat residual as separation sharpens |
| **MMLU** | 0.73 | +0.5 | +0.2 | **Floor-limited** — residual grows as elsewhere, but the floor never falls, so R never forms a mid-network peak; it clears the floor only marginally at the deepest layers (Gemma-2B, Qwen-3B; R ≈ 1.05–1.09), staying at or below it throughout in two of five models |

(ROC = rate of change per unit relative depth.) A resolvable orthogonal component thus emerges only where a **rising residual meets a falling floor** — a mid-depth phenomenon for separable geometries, and one that surfaces at most marginally — only at the deepest layers, and only in some models — for entangled ones, even though all three groups preserve their dominant linear structure equally well (κ ≈ 1).

---

## Result 3 — Causal ablation: the orthogonal residual is behaviorally load-bearing

Statistical resolvability tells us whether the residual is distinguishable from noise, not whether it *matters* to the model's behavior. We test this causally on Llama-3.2-3B: for each mixed ratio and layer, extract the systematic orthogonal residual `v⊥(r)` from held-out questions (see *Core methodology*), then surgically remove it from an *evaluated* question's own few-shot activation at the same ratio/layer/assistant-boundary state, and measure how much the model's next-token distribution shifts.

**Protocol.** For each of the 6 non-degenerate mixed ratios (1-1-1 excluded, since κ is undefined there), `v⊥(r)` is fit on a disjoint estimation pool (40 questions for Arithmetic/Entity, 20 for MMLU) and evaluated on `n = 10` held-out questions never used for estimation. The ablation is compared against a null of `D = 30` random directions `g ~ N(0, I)`, each projected orthogonal to the predicted direction `d̂ᵣ` and rescaled to `‖v⊥(r)‖` — matching magnitude and orthogonality, differing only in *which* orthogonal direction is removed. The test statistic is `KL(few-shot ‖ ablated)`. A **faithfulness control** — overwriting the boundary state with the unmodified activation — is a verified no-op (max|Δlogit| = 0, KL = 0), confirming the patching mechanism itself introduces no artifact.

**Findings, at three-quarter depth (layer 21 of Llama-3.2-3B):**
* The relative residual magnitude ρ (which is ≈ 0.003–0.012 at the shallowest probed layer) grows to **0.162 (Arithmetic), 0.242 (Entity), 0.067 (MMLU)**, while κ stays within [0.85, 1.09] throughout — the dominant linear structure is preserved even as the orthogonal component grows.
* **The true ablation beats all 30 null draws in all 6 mixed ratios in all 3 task groups** (smallest one-sided *p* = 1/31 ≈ 0.032 per ratio).
* Pooled KL divergence far exceeds the null mean: **Arithmetic 0.044 nats vs. 0.005 (8.0× the largest null draw), Entity 0.297 vs. 0.042 (4.7×), MMLU 0.0036 vs. 0.0004 (6.5×)**.
* **Negative control:** at the shallowest probed layer, the ablation's percentile within the null distribution scatters from the 3rd to the 87th percentile across ratios — indistinguishable from random — confirming the deep-layer effect is a genuine depth-dependent phenomenon, not a patching artifact.
* **MMLU dissociation:** MMLU has the smallest ρ and the κ closest to 1, and its residual sits at or below the split-half noise floor (statistically *unresolvable*, per Result 2) — yet its ablation still clears the null in all 6 ratios by 6.5×. Magnitude, statistical resolvability, and causal importance are three separate properties: a direction can be too small to distinguish from sampling noise and still be the direction the model's output actually depends on.

### Ablation vs. null, across depth

Solid = true ablation KL, dashed = null mean, shaded band = full range of the 30 null draws, filled/hollow markers indicate significance; log scale. MMLU is evaluated only at layer 21 due to compute cost.

![residual ablation vs null, by depth](results/residual_ablation_centroid_Llama-3B.png)

---

## Project structure

```
.
├── local_model.py                  # model loading; set TASK_VECTOR_MODEL / TASK_VECTOR_QUANTIZE env vars
├── requirements.txt
├── data/
│   ├── data_arithmetic_formats.py      # Direct / MCQ / Verification pools
│   ├── data_entity_attribute.py        # Capital / Currency / Language (country-attribute) pools
│   └── data_semantic_domains.py        # History / Law / ML MMLU pools
├── experiments/
│   ├── sweep_arithmetic.py             # ratio + layer sweep, arithmetic (κ + split-half + permutation null)
│   ├── sweep_entity.py                 # ratio + layer sweep, entity attribution
│   ├── sweep_mmlu.py                   # ratio + layer sweep, MMLU semantic domains
│   ├── kappa_null.py                   # exact 720-permutation null for κ
│   ├── plot_kappa_comparison.py        # cross-model κ comparison plots
│   ├── aggregate_split_half.py         # orthogonal residual + split-half noise floor (magnitude, resolvability)
│   ├── bootstrap_ci.py                 # bootstrap CIs for the resolvability ratio R
│   ├── plot_residual_depth.py          # residual magnitude vs. depth (+ floor), per group
│   ├── plot_residual_ratio_depth.py    # resolvability R = s̃/φ vs. depth, aggregated across models
│   └── task_vector_injection/
│       ├── extract_arithmetic.py               # pure-task contrast-vector extraction, arithmetic
│       ├── extract_entity.py                   # pure-task contrast-vector extraction, entity
│       ├── extract_vectors.py                  # pure-task contrast-vector extraction, MMLU
│       ├── inject_orthogonal_projection.py      # core patching mechanism + geometry fitting (imported, not run directly)
│       ├── centroid_residual_geometry.py        # fits v_perp(r), rho, kappa per (ratio, layer) from held-out questions
│       ├── null_distribution.py                 # runs the true ablation + D=30 norm-matched random-direction null
│       ├── analyze_projection_kl.py             # KL(few-shot || ablated) + gold-token logits per depth
│       ├── analyze_null_position.py             # percentile of the true ablation within its null distribution
│       ├── relative_residual_magnitude.py       # relative residual magnitude rho = ||v_perp|| / ||h||
│       ├── plot_residual_ablation_centroid.py   # Figure: ablation KL vs. null, by depth (main causal-ablation figure)
│       ├── plot_residual_ablation.py            # supplementary: residual magnitude / KL panels vs. depth
│       └── plot_residual_ablation_diff.py       # supplementary: (Arithmetic − MMLU), (Entity − MMLU) KL gap vs. depth
├── causal_projection/                  # cached geometry, null draws, and KL results for the ablation pipeline
├── results/                            # output figures + κ JSON (κ, split-half residual/floor, permutation null)
├── ratio_centroids/                    # cached mixed-ratio centroids (for downstream metrics)
└── archive/                            # earlier exploratory experiments
```

---

## Usage

```bash
# --- κ layer/ratio sweeps (writes results/<group>_kappa_<model>.json + figures) ---
python experiments/sweep_arithmetic.py                                             # default Llama-3.2-3B, fp16
python experiments/sweep_arithmetic.py --model meta-llama/Llama-3.2-1B-Instruct
python experiments/sweep_arithmetic.py --model meta-llama/Llama-3.1-8B-Instruct --quantize int8
python experiments/sweep_entity.py     --model google/gemma-2-2b-it
python experiments/sweep_mmlu.py       --model Qwen/Qwen2.5-3B-Instruct

# --- cross-model κ comparison plots (reads results/) ---
python experiments/plot_kappa_comparison.py                          # → arithmetic_kappa_comparison_across_models.png
python experiments/plot_kappa_comparison.py --glob "entity_kappa_*.json"
python experiments/plot_kappa_comparison.py --glob "mmlu_kappa_*.json"

# --- orthogonal residual + split-half noise floor (reads results/*_kappa_*.json) ---
python experiments/aggregate_split_half.py --grid                    # residual/floor summary table (per layer)
python experiments/bootstrap_ci.py                                   # bootstrap CIs for resolvability R = s̃/φ
python experiments/plot_residual_depth.py                            # → results/orthogonal_residual_depth.png
python experiments/plot_residual_ratio_depth.py --per-model          # → results/orthogonal_residual_ratio_depth_by_model.png

# --- causal ablation of the orthogonal residual (writes causal_projection/*.json) ---
python experiments/task_vector_injection/centroid_residual_geometry.py                        # fit v_perp(r), rho, kappa for all groups
python experiments/task_vector_injection/null_distribution.py --group arithmetic --draws 30   # true ablation + null draws
python experiments/task_vector_injection/analyze_projection_kl.py --group arithmetic          # KL(few-shot || ablated)
python experiments/task_vector_injection/analyze_null_position.py --group arithmetic          # percentile within null
python experiments/task_vector_injection/relative_residual_magnitude.py --group arithmetic    # rho = ||v_perp|| / ||h||
python experiments/task_vector_injection/plot_residual_ablation_centroid.py                   # → results/residual_ablation_centroid_<model>.png
```

Set the model with `--model` or the `TASK_VECTOR_MODEL` environment variable. Outputs (PNG figures and κ / ablation JSON) are written to `results/` and `causal_projection/` respectively.

---

## Background

Early exploratory work (archived) tested the same superposition hypothesis on medical-specialty tasks (Medicine / Surgery / Pharmacology from MedMCQA) and format-distinct clinical tasks (MCQ / PubMedQA / Symptom2Disease). Those runs established the initial observation of linear superposition and that optimal layer depth differs by task type (early for format, middle for semantics). The arithmetic, entity, and MMLU experiments here add tighter controls, the quantitative κ framework with an exact null, the depth-resolved orthogonal-residual analysis against a split-half noise floor, and a causal ablation framework establishing that the residual left over by the linear approximation is behaviorally meaningful.
