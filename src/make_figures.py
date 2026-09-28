"""Generate the letter's two figures from the results files.

Usage: python make_figures.py <results_dir> <output_dir>
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "legend.fontsize": 7, "legend.framealpha": 0.9,
    "figure.dpi": 300, "pdf.fonttype": 42,
})

res_dir = sys.argv[1]
out = sys.argv[2]
res = json.load(open(os.path.join(res_dir, "results.json")))
W = res["control"]["W"]


def nlab(key):
    """Label-count string taken from results.json, never hard-coded."""
    return f"{res[key]['labels_mean']:,.0f}"


# ---- Fig 1: per-window empty-set counts, frozen model, seed 0 ----
counts = res["detect"]["counts_seed0"]     # list per batch (2..10)
k_crit = res["detect"]["k_crit"]
fig, ax = plt.subplots(figsize=(3.45, 1.02))
x0 = 0
bounds = [0]
allc = []
for c in counts:
    allc.extend(c)
    x0 += len(c)
    bounds.append(x0)
ax.bar(range(len(allc)), allc, width=1.0, color="#4878a8", edgecolor="none")
ax.axhline(k_crit, color="#c44e52", lw=1.0, ls="--",
           label=f"critical value $k_{{\\delta}}={k_crit}$")
ax.axhline(W * 0.1, color="#555555", lw=0.8, ls=":",
           label=f"stationary mean $W\\alpha={W * 0.1:.0f}$")
label_floor = 50
for i, b in enumerate(bounds[:-1]):
    # separators stop below the label band so they never cross a label
    ax.axvline(b - 0.5, ymin=0, ymax=label_floor / 84, color="k", lw=0.5,
               alpha=0.35)
    # alternate label heights, and nudge the narrow batch pairs apart,
    # so that adjacent labels do not collide
    x_mid = (bounds[i] + bounds[i + 1]) / 2
    if bounds[i + 1] - bounds[i] <= 5:
        x_mid += -2.5 if i % 2 == 0 else 2.5
    ax.text(x_mid, 51 if i % 2 == 0 else 65, f"{i+2}", ha="center",
            va="bottom", fontsize=6.0)
ax.set_xlim(-1, len(allc))
ax.set_ylim(0, 84)
ax.set_yticks([0, 15, 30, 45])
ax.set_xlabel(f"window index, batches 2 to 10 (W = {W})")
ax.set_ylabel("empty sets\nper window")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
          frameon=False, borderaxespad=0.0, handlelength=1.6,
          columnspacing=1.0)
fig.tight_layout(pad=0.3)
fig.savefig(f"{out}/fig_detection.pdf")
plt.close(fig)

# ---- Fig 2: per-batch accuracy ----
fig, ax = plt.subplots(figsize=(3.45, 1.32))
bx = np.arange(2, 11)
styles = {
    "static": ("#777777", "s", "static (0 labels)"),
    "periodic-40": ("#4878a8", "^", f"privileged, B=40 ({nlab('periodic-40')} labels)"),
    "cadence-10": ("#c44e52", "o", f"CADENCE, B=10 ({nlab('cadence-10')} labels)"),
    "oracle": ("#55a868", "d", f"full relabeling ({nlab('oracle')} labels)"),
}
for k, (c, mk, lab) in styles.items():
    mu = np.array(res[k]["per_batch_mean"])
    sd = np.array(res[k]["per_batch_std"])
    ax.plot(bx, mu, marker=mk, ms=3.5, color=c, label=lab)
    ax.fill_between(bx, np.maximum(mu - sd, 0.0), np.minimum(mu + sd, 1.0),
                    color=c, alpha=0.15, lw=0)
ax.set_xlabel("batch (chronological, months 3 to 36)")
ax.set_ylabel("accuracy")
ax.set_xticks(bx)
ax.set_ylim(0.15, 1.03)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
          frameon=False, borderaxespad=0.0, handlelength=1.3,
          columnspacing=0.9, handletextpad=0.4, fontsize=6.2)
fig.tight_layout(pad=0.3)
fig.savefig(f"{out}/fig_accuracy.pdf")
plt.close(fig)
print("figures written")
