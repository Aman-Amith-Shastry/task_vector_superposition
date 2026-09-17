"""Figure for the residual-ablation section (4.4), centroid-residual version.

One panel per task group (a, b, c), vs relative layer depth (Llama-3B):
  KL(few-shot || ablated) for the centroid-residual ablation, against the
  norm-matched random-orthogonal null. The null band spans min..max of the D=30
  draws (the full support of the null, so "test above the band" == "test above
  every draw"); the dashed line is the null mean. Ablation markers are filled
  where the test exceeds all D draws.

The rho = ||v_perp(r)|| / ||h|| magnitude panel was cut -- those per-layer values
are tabulated in the appendix (tab:residual_ablation_full), and the figure reads
better carrying only the causal claim.

Reads causal_projection/nulldist_<group>_<tag>.npz
Writes results/residual_ablation_centroid_<tag>.pdf (+ .png)
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
NULL_GROUPS = ["arithmetic", "entity", "mmlu"]   # groups with a causal null run


def main():
    # One null panel per group: the groups' KL scales overlap (Entity's null band at
    # depth 0.75 spans 0.031-0.065, straddling Arithmetic's test value of 0.040), so a
    # shared axis would read as though Arithmetic's ablation sits inside a null band.
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5))
    null_axes = {"arithmetic": axes[0], "entity": axes[1], "mmlu": axes[2]}
    panel_tag = {"arithmetic": "(a)", "entity": "(b)", "mmlu": "(c)"}

    # ---- (a), (b), (c) ablation vs null, one panel per group ----
    for g, label, color, mk in GROUPS:
        if g not in NULL_GROUPS:
            continue
        ax2   = null_axes[g]
        npz   = np.load(os.path.join(_root, "causal_projection", f"nulldist_{g}_{TAG}.npz"))
        meta  = json.load(open(os.path.join(_root, "causal_projection", f"nulldist_{g}_{TAG}.json")))
        depth = [l / meta["n_layers"] for l in map(int, npz["layers"])]
        test  = npz["test_kl"].astype(np.float64)          # (R, L)
        null  = npz["null_kl"].astype(np.float64)          # (R, L, D)

        t_l  = test.mean(0)                                 # pooled over ratios
        n_l  = null.mean(0)                                 # (L, D)
        lo, hi, mu = n_l.min(1), n_l.max(1), n_l.mean(1)
        # per-layer: does the test exceed every draw, in every ratio?
        allcell = [(null[:, k, :] < test[:, k, None]).all() for k in range(len(depth))]

        if len(depth) == 1:
            # Single swept layer (e.g. a --layers 21 run): a band/curve is undefined, so draw the null's
            # full support as a vertical range at that depth and the ablation as a point.
            d0 = depth[0]
            ax2.vlines(d0, lo[0], hi[0], color=color, lw=6, alpha=0.20,
                       label="null range (30 draws)")
            ax2.plot(d0, mu[0], marker="_", ms=12, color=color, ls="none",
                     alpha=0.8, label="null mean")
            ax2.plot(d0, t_l[0], marker=mk, ms=7, color=color, ls="none", zorder=4,
                     mfc=(color if allcell[0] else "white"), mec=color, label="ablation")
            ax2.set_xlim(d0 - 0.25, d0 + 0.25)
            # Autoscale on 3 near-identical x-values leaves no headroom and the legend
            # covers the ablation point; set the range from the data with decade padding.
            ax2.set_ylim(lo[0] / 3.0, t_l[0] * 6.0)
        else:
            ax2.fill_between(depth, lo, hi, color=color, alpha=0.20, lw=0,
                             label="null range (30 draws)")
            ax2.plot(depth, mu, color=color, ls="--", lw=1.0, alpha=0.8, label="null mean")
            ax2.plot(depth, t_l, color=color, lw=1.8, zorder=3, label="ablation")
            for k, d in enumerate(depth):
                ax2.plot(d, t_l[k], marker=mk, ms=5, color=color, zorder=4,
                         mfc=(color if allcell[k] else "white"), mec=color)
        ax2.set_yscale("log")
        ax2.set_xlabel("relative layer depth")
        ax2.set_ylabel(r"KL(few-shot $\|$ ablated)  [nats]")
        ax2.set_title(f"{panel_tag[g]} {label}: ablation vs. null", fontsize=10)
        ax2.grid(alpha=0.3, lw=0.5, which="both")
        ax2.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        dst = os.path.join(_root, "results", f"residual_ablation_centroid_{TAG}.{ext}")
        fig.savefig(dst, dpi=200, bbox_inches="tight")
        print(f"stored -> {os.path.relpath(dst, _root)}")


if __name__ == "__main__":
    main()
