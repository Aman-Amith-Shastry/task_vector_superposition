"""Difference-of-KL figure for the residual-ablation section (4.4).

Single panel vs relative layer depth: the average residual-ablation KL of each
well-separated group minus MMLU's, (Arithmetic - MMLU) and (Entity - MMLU), with
95% clustered-bootstrap CIs. The dashed line at 0 is the null; a filled marker
means the CI clears 0 at that layer, a hollow marker means it straddles 0.

Questions differ across groups, so the two groups in each contrast are resampled
INDEPENDENTLY (unpaired two-sample bootstrap). Within a group the resampling unit
is the question (shared across the 6 ratios), matching the single-group band.

Reads causal_projection/relmag_<group>_Llama-3B.npz, writes
results/residual_ablation_diff_Llama-3B.pdf (+ .png).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = "Llama-3B"
B, SEED = 20000, 0
# each contrast is <group> - MMLU; reuse the group's color/marker from the main fig
CONTRASTS = [("arithmetic", "Arithmetic − MMLU", "#0072B2", "o"),
             ("entity",     "Entity − MMLU",     "#D55E00", "s")]

def load(g):
    d = np.load(os.path.join(_root, "causal_projection", f"relmag_{g}_{TAG}.npz"))
    return d["kl_test"].astype(np.float64), d["layers"].astype(int), d["depth"].astype(float)

kl = {g: load(g)[0] for g in ["arithmetic", "entity", "mmlu"]}
layers, depth = load("arithmetic")[1], load("arithmetic")[2]
L = kl["arithmetic"].shape[2]

def boot_means(g, rng):                      # (B,L) bootstrap mean KL per layer
    S = kl[g].shape[0]
    idx = rng.integers(0, S, size=(B, S))
    return kl[g][idx].mean(axis=(1, 2))

rng = np.random.default_rng(SEED)
boot = {g: boot_means(g, rng) for g in ["arithmetic", "entity", "mmlu"]}
point = {g: kl[g].reshape(-1, L).mean(0) for g in ["arithmetic", "entity", "mmlu"]}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})
fig, ax = plt.subplots(figsize=(3.6, 2.5))
ax.axhline(0, color="0.4", lw=1.0, ls="--", zorder=1)

for g, label, c, mk in CONTRASTS:
    diff = point[g] - point["mmlu"]
    dboot = boot[g] - boot["mmlu"]
    lo, hi = np.percentile(dboot, [2.5, 97.5], axis=0)
    sig = lo > 0                             # CI entirely above the null
    ax.fill_between(depth, lo, hi, color=c, alpha=0.16, linewidth=0, zorder=2)
    ax.plot(depth, diff, color=c, lw=1.6, zorder=3, label=label)
    # filled markers where significant, hollow where not
    ax.scatter(depth[sig],  diff[sig],  s=26, marker=mk, color=c,
               edgecolors=c, zorder=4)
    ax.scatter(depth[~sig], diff[~sig], s=26, marker=mk, facecolors="white",
               edgecolors=c, linewidths=1.2, zorder=4)

ax.set_xlabel("Relative layer depth")
ax.set_ylabel(r"$\Delta\,\mathrm{KL}$ vs. MMLU (nats)")
ax.set_title("Residual-ablation KL, group $-$ MMLU", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(True, alpha=0.25, lw=0.5)
ax.margins(x=0.03)

fig.tight_layout()
out = os.path.join(_root, "results", "residual_ablation_diff_Llama-3B")
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", bbox_inches="tight")
print("wrote", os.path.relpath(out + ".pdf", _root), "and .png")

print(f"\n[B={B} clustered bootstrap, 95% CI; filled marker = CI>0]")
for g, label, *_ in CONTRASTS:
    diff = point[g] - point["mmlu"]
    dboot = boot[g] - boot["mmlu"]
    lo, hi = np.percentile(dboot, [2.5, 97.5], axis=0)
    print(f"\n{label}")
    for k in range(L):
        print(f"  depth={depth[k]:.2f}  diff={diff[k]:>7.3f}  CI=[{lo[k]:>7.3f},{hi[k]:>7.3f}]"
              f"  {'sig' if lo[k] > 0 else ''}")
