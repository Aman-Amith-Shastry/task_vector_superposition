# Empirical Verification of Linear Task Superposition in LLMs

An empirical study of whether In-Context Learning (ICL) examples induce **linearly superposed task representations** in transformer residual streams, whether that linearity is quantifiable and statistically significant, and whether it is **causally functional**.

When in-context examples span multiple tasks, LLMs represent and perform all of them at once — *task superposition*. Prior work posits that this superposition is a **convex combination** of individual task vectors, but this had not been directly verified at the activation level in instruction-tuned models. We verify it three ways:

1. **Geometry (κ).** Mixed-task contrast-vector centroids land on the ratio-weighted convex combination of the pure-task centroids (κ ≈ 1), calibrated against an **exact permutation null** — significant at the floor *p* = 1/720 for every model, task group, and layer.
2. **Recoverability (OLS).** The true mixture ratio of a mixed-task prompt is recoverable from the pure-task centroids via reparameterized OLS, with no prior knowledge of the task distribution.
3. **Causality (injection).** Injecting the convex combination of pure-task vectors into a task-free prompt shifts model behavior toward the injected ratio's ICL distribution — significant at permutation *p* = 0.0001 in every condition, in two model families.

Inspired by [*Everything Everywhere All at Once* (Xiong et al., 2024)](https://arxiv.org/abs/2410.05603), which established task superposition **behaviorally** in pretrained models. Behavioral superposition is a claim about *outputs* and does not by itself entail that the underlying representations **combine linearly**; we test the representations directly.

**Models (three families, 1B–8B):** Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Gemma-2-2B, Qwen-2.5-3B (instruction-tuned variants).

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

κ = 1 means perfect linear combination; κ = 0 means orthogonal (no linear tracking); κ is signed and unbounded by design, so it *could* be negative or ≫ 1. κ is undefined for the equal 1-1-1 ratio (predicted = center, denominator = 0) and excluded there. All κ is computed in the **original activation space**, not the LDA projection used for the scatter plots.

**Exact permutation null.** κ ≈ 1 could in principle be an artifact of the metric's scale, so we calibrate it. For the *K* = 6 mixed ratios with a well-defined predicted displacement (the three pure ratios form the basis; 1-1-1 is excluded), we re-pair each observed displacement with a permuted prediction and score the mean κ. With 6! = 720 relabelings we enumerate the null **exactly**; the observed pairing beating all others gives the floor *p* = 1/720 ≈ 1.4 × 10⁻³. See [`experiments/kappa_null.py`](experiments/kappa_null.py).

---

## Task groups

Three groups probe superposition under different kinds of task difference.

| Group | Tasks | Shared surface form | Tests separability of… |
|---|---|---|---|
| **Arithmetic** | Direct (`42`) · MCQ (`B`) · Verification (`True`) | same arithmetic inputs, different output *format* | **format** structure (early-layer signal) |
| **Entity** | Capital (`France→Paris`) · Currency (`France→Euro`) · Language (`France→French`) | a bare country name; no scaffolding | **semantic relation** under an underspecified prompt |
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

**In every cell above, at every one of the 8 swept layers, the observed mean κ exceeds all 720 permutations** → floor *p* = 1/720 ≈ 1.4 × 10⁻³, against a null centered near 0. The convex-combination hypothesis is rejected in favor of genuine linear superposition across all models and task groups.

### Cross-model κ vs. relative layer depth

Mean κ ± 1 std across mixed ratios at each relative depth; grey band = κ ∈ [0.75, 1.25].

| Arithmetic | Entity | MMLU |
|---|---|---|
| ![κ arithmetic](results/arithmetic_kappa_comparison_across_models.png) | ![κ entity](results/entity_kappa_comparison_across_models.png) | ![κ mmlu](results/mmlu_kappa_comparison_across_models.png) |

Format-varying **Arithmetic** shows the largest κ-variance growth with depth; semantic **MMLU** holds κ ≈ 1 nearly flat. Qwen and Gemma replicate the Llama pattern despite different architectures, training data, and tokenizers — supporting cross-family generalizability.

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

## Result 2 — Recoverability: inferring the mixture ratio via OLS

If task vectors superpose linearly, the map ICL ratio → contrast vector is invertible: given a contrast vector from an unknown prompt, recover the mixing coefficients α. For a probe `x` and pure-task centroids `c₁, c₂, c₃`, solve

```
x ≈ α₁·c₁ + α₂·c₂ + α₃·c₃    subject to  α₁ + α₂ + α₃ = 1
```

via **reparameterized OLS** (eliminate one variable to enforce sum-to-1 exactly, *without* imposing non-negativity — so non-negativity of the recovered α is a *testable outcome*, not a baked-in constraint):

```
x − c₃ ≈ α₁·(c₁ − c₃) + α₂·(c₂ − c₃);   α₃ = 1 − α₁ − α₂
```

The decomposition layer is chosen **only from the three pure ratios** (lowest reconstruction error); mixed ratios are held out for evaluation. Recovered coefficients (Llama-3.2-3B) plotted against ground truth — points on the identity line = exact recovery:

| Arithmetic (layer 3) | Entity (layer 18) | MMLU (layer 18) |
|---|---|---|
| ![OLS arithmetic](results/ols_arith_expected.png) | ![OLS entity](results/ols_entity_expected.png) | ![OLS mmlu](results/ols_mmlu_expected.png) |

**Per-ratio L2 error across task groups:**

![OLS L2 error](results/ols_l2_error_chart.png)

Recovery is strong across all groups; MMLU is the most accurate and uniform, while Entity shows larger deviations on mixed ratios (semantic heterogeneity under a bare-country prompt mixes less cleanly than shared-format tasks). A noise-centroid permutation baseline confirms recovery is driven by task-specific geometry, not the shared component. Full per-ratio coefficient tables with 95% bootstrap CIs are in the paper appendix.

---

## Result 3 — Causality: injecting the convex combination steers behavior

Geometry alone is correlational. We test whether the convex combination is **causally sufficient**: reconstruct `v_inj(r) = Σ wₖ·ĉₖ` from pure-task vectors, add `α·v_inj(r)` to the residual stream of an otherwise task-free prompt, and measure whether the first-token distribution shifts toward the ratio-*r* ICL distribution (Jensen–Shannon confusion matrix, **double-centered** to remove row/column main effects; a genuine match is a *negative* diagonal). Significance is an exact-logic permutation test over ratio relabelings (10,000 draws, add-one corrected). The injection layer and α are fixed on the **pure ratios only** — never on the mixed-ratio outcomes under test.

**Injection results (mean double-centered JS diagonal, 95% CI; more negative = stronger task-matching):**

| Group | Format | Llama-3.2-3B (*l*=18) | perm-*p* | Qwen-2.5-3B (*l*=27) | perm-*p* |
|---|---|---|---|---|---|
| Entity | — | −0.0232 [−0.0300, −0.0170] | 0.0001 | −0.0012 [−0.0014, −0.0009] | 0.0001 |
| Arithmetic | Direct | −0.0251 [−0.0283, −0.0224] | 0.0001 | −0.0196 [−0.0247, −0.0146] | 0.0001 |
| Arithmetic | MCQ | −0.0449 [−0.0467, −0.0431] | 0.0001 | −0.0170 [−0.0243, −0.0102] | 0.0001 |
| Arithmetic | Verification | −0.0753 [−0.0783, −0.0724] | 0.0001 | −0.0091 [−0.0107, −0.0079] | 0.0001 |

Every condition is significant at *p* = 0.0001 in **both** model families. Qwen's magnitudes are smaller because its first-token distributions occupy a more compressed JS range, so each model is compared against its own null, not across models by raw magnitude. α = 4 throughout, except Qwen Direct (α = 6). Increasing α past its optimum *degrades* task-matching — a signature of genuine steering rather than a monotone artifact.

**Injection depth ≠ OLS decomposition layer.** OLS recovery is most accurate at a *shallow* layer (arithmetic *l*=3), but injecting there is causally inert — the geometry is readable early but not yet wired into the output computation:

| Format | Injection at OLS layer *l*=3 | *p* |
|---|---|---|
| Direct | −0.0030 [−0.0036, −0.0023] | 0.0001 |
| MCQ | +0.0008 [+0.0006, +0.0010] | 1.0000 |
| Verification | +0.0003 [+0.0002, +0.0004] | 1.0000 |

At *l*=3 the effect is an order of magnitude smaller (Direct) or non-significant (MCQ, Verification); at *l*=18 every format is decisively significant (table above). **OLS recovery accuracy does not necessarily correlate with injection efficacy** — the two criteria probe different (static-geometric vs. causal) properties.

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
│   ├── sweep_arithmetic.py             # ratio + layer sweep, arithmetic (κ + permutation null)
│   ├── sweep_entity.py                 # ratio + layer sweep, entity attribution
│   ├── sweep_mmlu.py                   # ratio + layer sweep, MMLU semantic domains
│   ├── kappa_null.py                   # exact 720-permutation null for κ
│   ├── plot_kappa_comparison.py        # cross-model κ comparison plots
│   └── task_vector_injection/
│       ├── extract_arithmetic.py           # pure-task contrast-vector extraction, arithmetic
│       ├── extract_entity.py               # pure-task contrast-vector extraction, entity
│       ├── extract_vectors.py              # pure-task contrast-vector extraction, MMLU
│       ├── decompose_task_mixture.py       # inverse superposition (reparameterized OLS)
│       ├── inject_arithmetic_js.py         # causal injection + JS confusion test, arithmetic
│       ├── inject_entity_js.py             # causal injection + JS confusion test, entity
│       └── inject_entity_behavioral.py     # behavioral (answer-distribution) injection check
├── results/                            # output figures + κ JSON (κ values, permutation null)
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

# --- OLS mixture recovery ---
python experiments/task_vector_injection/decompose_task_mixture.py --select_layer          # find best layer on pure ratios
python experiments/task_vector_injection/decompose_task_mixture.py --layer 3 --sweep --ci   # all 10 ratios + bootstrap CIs
python experiments/task_vector_injection/decompose_task_mixture.py --experiment mmlu --layer 18 --sweep --noise

# --- causal injection (JS confusion + double-centered permutation test) ---
python experiments/task_vector_injection/inject_arithmetic_js.py --layers 18 --alphas 4 --all_ratios
python experiments/task_vector_injection/inject_entity_js.py     --layer 18 --alphas 4 --all_ratios
```

Set the model with `--model` or the `TASK_VECTOR_MODEL` environment variable. Outputs (PNG figures and κ JSON) are written to `results/`.

---

## Background

Early exploratory work (archived) tested the same superposition hypothesis on medical-specialty tasks (Medicine / Surgery / Pharmacology from MedMCQA) and format-distinct clinical tasks (MCQ / PubMedQA / Symptom2Disease). Those runs established the initial observation of linear superposition and that optimal layer depth differs by task type (early for format, middle for semantics). The arithmetic, entity, and MMLU experiments here add tighter controls, the quantitative κ framework with an exact null, and the causal injection test.
