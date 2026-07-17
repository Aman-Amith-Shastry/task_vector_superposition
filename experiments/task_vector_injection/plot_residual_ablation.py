"""Figure for the residual-ablation section (4.4).

Three panels vs relative layer depth, one line per task group (Llama-3B):
  (a) relative residual magnitude  ||v_perp|| / ||h||          (grows with depth)
  (b) KL(few-shot || ablated)      residual removed            (Arith, Entity > MMLU)
  (c) KL difference vs MMLU        (Arith - MMLU), (Entity - MMLU), with 0 as null

Panels (b) and (c) carry 95% clustered-bootstrap bands. The resampling unit is the
question (n=10): the same question index is shared across all 6 mixed ratios
(correlated within a question). For each of B iterations we draw 10 question
indices with replacement once and reuse that draw across all 6 ratios (and across
layers), averaging the 6*10 = 60 KL values into one bootstrap mean. Panel (b) is
the per-group band; panel (c) is the difference band, with the two groups in each
contrast resampled INDEPENDENTLY (unpaired: the question pools differ across
groups). Solid lines are point estimates (mean of all 60 cells); in (c) a filled
marker means the CI clears 0 at that layer, a hollow marker means it straddles 0.

Reads causal_projection/relmag_<group>_Llama-3B.{json,npz}, writes
results/residual_ablation_Llama-3B.pdf (+ .png).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = "Llama-3B"
GROUPS = [("arithmetic", "Arithmetic", "#0072B2", "o"),
          ("entity",     "Entity",     "#D55E00", "s"),
          ("mmlu",       "MMLU",       "#009E73", "^")]
CONTRASTS = [("arithmetic", "Arith$-$MMLU", "#0072B2", "o"),
             ("entity",     "Entity$-$MMLU", "#D55E00", "s")]
B, SEED = 20000, 0

def load_rel(g):
    d = json.load(open(os.path.join(_root, "causal_projection", f"relmag_{g}_{TAG}.json")))
    p = d["pooled_over_ratios_per_layer"]
    return (np.array([r["depth"] for r in p]),
            np.array([r["rel_ratio_of_means"] for r in p]))

def load_kl(g):
    d = np.load(os.path.join(_root, "causal_projection", f"relmag_{g}_{TAG}.npz"))
    return d["kl_test"].astype(np.float64), d["depth"].astype(float)  # (S,R,L), (L,)

kl = {g: load_kl(g)[0] for g, *_ in GROUPS}
depth = load_kl("arithmetic")[1]
L = kl["arithmetic"].shape[2]

# one bootstrap-mean matrix (B,L) per group, independent draws across groups
rng = np.random.default_rng(SEED)
boot, point = {}, {}
for g, *_ in GROUPS:
    S = kl[g].shape[0]
    idx = rng.integers(0, S, size=(B, S))
    boot[g] = kl[g][idx].mean(axis=(1, 2))          # (B,L)
    point[g] = kl[g].reshape(-1, L).mean(0)         # (L,)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.0, 2.35))

# (a) relative residual magnitude
for g, label, c, mk in GROUPS:
    dpt, rel = load_rel(g)
    axA.plot(dpt, rel, marker=mk, color=c, label=label, lw=1.5, ms=3.5)
axA.set_ylabel(r"$\|v_\perp\| \,/\, \|\mathbf{h}\|$")
axA.set_title("(a) Residual magnitude", fontsize=8.5)
axA.set_ylim(0, None)
axA.legend(frameon=False, fontsize=7, loc="upper left")

# (b) per-group ablation KL with bootstrap band
for g, label, c, mk in GROUPS:
    lo, hi = np.percentile(boot[g], [2.5, 97.5], axis=0)
    axB.fill_between(depth, lo, hi, color=c, alpha=0.16, linewidth=0)
    axB.plot(depth, point[g], marker=mk, color=c, lw=1.5, ms=3.5)
axB.set_ylabel(r"$\mathrm{KL}(\mathrm{few\text{-}shot}\,\|\,\mathrm{ablated})$ (nats)")
axB.set_title("(b) Ablation effect", fontsize=8.5)
axB.set_ylim(0, None)

# (c) difference vs MMLU with bootstrap band, 0 as null; filled marker = CI>0
axC.axhline(0, color="0.4", lw=1.0, ls="--", zorder=1)
for g, label, c, mk in CONTRASTS:
    diff = point[g] - point["mmlu"]
    lo, hi = np.percentile(boot[g] - boot["mmlu"], [2.5, 97.5], axis=0)
    sig = lo > 0
    axC.fill_between(depth, lo, hi, color=c, alpha=0.16, linewidth=0, zorder=2)
    axC.plot(depth, diff, color=c, lw=1.5, zorder=3, label=label)
    axC.scatter(depth[sig], diff[sig], s=22, marker=mk, color=c, zorder=4)
    axC.scatter(depth[~sig], diff[~sig], s=22, marker=mk, facecolors="white",
                edgecolors=c, linewidths=1.1, zorder=4)
axC.set_ylabel(r"$\Delta\,\mathrm{KL}$ vs. MMLU (nats)")
axC.set_title("(c) Difference vs. MMLU", fontsize=8.5)
axC.legend(frameon=False, fontsize=7, loc="upper left")

for ax in (axA, axB, axC):
    ax.set_xlabel("Relative layer depth")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.margins(x=0.03)

fig.tight_layout(w_pad=1.0)
out = os.path.join(_root, "results", "residual_ablation_Llama-3B")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", bbox_inches="tight")
print("wrote", os.path.relpath(out + ".pdf", _root), "and .png")

# deepest-layer summary for the text
print(f"\ndeepest layer  [B={B} clustered bootstrap, 95% CI]:")
for g, label, *_ in GROUPS:
    lo, hi = np.percentile(boot[g], [2.5, 97.5], axis=0)
    print(f"  {label:10s} KL={point[g][-1]:.3f}  CI=[{lo[-1]:.3f}, {hi[-1]:.3f}]")
for g, label, *_ in CONTRASTS:
    lo, hi = np.percentile(boot[g] - boot["mmlu"], [2.5, 97.5], axis=0)
    d = point[g][-1] - point["mmlu"][-1]
    print(f"  {label:14s} Δ={d:.3f}  CI=[{lo[-1]:.3f}, {hi[-1]:.3f}]")
