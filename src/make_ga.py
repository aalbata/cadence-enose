"""Generate the graphical abstract from results.json (no hand-typed numbers).

Usage: python make_ga.py <results.json> <output_dir>
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

res = json.load(open(sys.argv[1]))
out = sys.argv[2]

plt.rcParams.update({
    "font.size": 8, "font.family": "sans-serif",
    "axes.linewidth": 0.6, "figure.dpi": 300, "pdf.fonttype": 42,
})

static_acc = res["static"]["mean_acc"]
full_acc = res["oracle"]["mean_acc"]
cad_acc = res["cadence-10"]["mean_acc"]
full_lab = res["oracle"]["labels_mean"]
cad_lab = res["cadence-10"]["labels_mean"]
ratio = full_lab / cad_lab
k_crit = res["detect"]["k_crit"]
counts = res["detect"]["counts_seed0"]
W = res["control"]["W"]
fired_min = min(sum(1 for b in range(9) if res["detect"]["first_alarm"][b][s] is not None)
                for s in range(len(res["detect"]["first_alarm"][0])))
ctl_alarms = res["control"]["alarm_windows"]
ctl_windows = res["control"]["total_windows"]
n_seeds = len(res["detect"]["first_alarm"][0])

# Artwork on a 285-pt-high canvas so that submission-system headers have
# about 32 pt of clear space above the title.
canvas_height = 285 / 72
art_height = 3.45
y_scale = art_height / canvas_height
fig = plt.figure(figsize=(6.6, canvas_height))

fig.text(0.02, 0.965 * y_scale, "CADENCE", color="#b03030", fontsize=11.5,
         ha="left", va="top")
fig.text(0.145, 0.960 * y_scale,
         "a label-free drift alarm with certified pre-adaptation "
         "false-alarm control for gas sensor arrays",
         fontsize=8.2, ha="left", va="top")

# ---- left: frozen-model empty-set counts ----
axl = fig.add_axes([0.105, 0.17 * y_scale, 0.30, 0.60 * y_scale])
allc = [c for batch in counts for c in batch]
axl.bar(range(len(allc)), allc, width=1.0, color="#4878a8",
        edgecolor="none")
axl.axhline(k_crit, color="#c44e52", lw=1.1, ls="--")
axl.axhline(W * 0.1, color="#555555", lw=0.9, ls=":")
axl.text(len(allc) * 0.03, k_crit + 2,
         f"alarm threshold = {k_crit}", color="#c44e52", fontsize=7,
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
axl.text(len(allc) * 0.50, 7.5, f"nominal stationary mean = {W * 0.1:.0f}",
         color="#444444", fontsize=6.2, ha="center",
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
axl.set_xlim(-1, len(allc))
axl.set_ylim(0, 52)
axl.set_yticks([0, 15, 30, 45])
axl.set_xticks([])
axl.set_ylabel(f"empty conformal sets\nper {W}-sample window", fontsize=7)
axl.set_xlabel("replayed benchmark stream, batches 2 to 10", fontsize=6.4)
axl.set_title("empty-set counts under drift", fontsize=8.5, pad=4)

# ---- middle: flow boxes ----
axm = fig.add_axes([0.415, 0.10 * y_scale, 0.28, 0.72 * y_scale])
axm.set_xlim(0, 1)
axm.set_ylim(0, 1)
axm.axis("off")


def box(y, h, text, fc, fontsize=6.8):
    axm.add_patch(FancyBboxPatch((0.04, y), 0.92, h,
                                 boxstyle="round,pad=0.02",
                                 fc=fc, ec="#999999", lw=0.7))
    axm.text(0.5, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize)


# boxes from top to bottom: (height, text, face colour, font size); the
# column is laid out from these heights with one fixed gap between the
# visible edges of consecutive boxes, so that both arrows have the same
# length, and it is centred vertically in the panel
PAD = 0.02   # rounded-frame padding of box(), which extends the visible box
GAP = 0.075  # distance between the visible edges of consecutive boxes
boxes = [
    (0.19, "empty prediction set? per sample,\none subtraction, one comparison,\n"
     "one conditional increment, and\na window-position count", "#eef2f7", 6.2),
    (0.30, f"count $\\geq$ {k_crit} in a window\nraises the alarm\n"
     "false-alarm probability\n$\\leq 10^{-3}$ per window\n"
     "certified under i.i.d.\nstationarity\npre-adaptation epoch",
     "#fdeeee", 5.9),
    (0.20, "query 10 labels, retrain,\nresume monitoring", "#eef7ee", 6.8),
]
column = sum(h for h, *_ in boxes) + 2 * PAD * len(boxes) + GAP * (len(boxes) - 1)
top_edge = 0.5 + column / 2
edges = []
for h, text, fc, fs in boxes:
    y = top_edge - PAD - h
    box(y, h, text, fc, fontsize=fs)
    edges.append((top_edge, y - PAD))
    top_edge = y - PAD - GAP
for (_, lower_edge), (upper_edge, _) in zip(edges[:-1], edges[1:]):
    axm.add_patch(FancyArrowPatch((0.5, lower_edge), (0.5, upper_edge),
                                  arrowstyle="-|>", mutation_scale=9,
                                  shrinkA=1, shrinkB=1,
                                  color="#666666", lw=0.9))

# ---- right: accuracy bars ----
axr = fig.add_axes([0.71, 0.17 * y_scale, 0.27, 0.53 * y_scale])
names = ["static (0 labels)",
         f"full relabeling ({full_lab:,.0f} labels)",
         f"CADENCE ({cad_lab:.0f} labels)"]
vals = [static_acc, full_acc, cad_acc]
cols = ["#8a8a8a", "#5a9a5a", "#c44e52"]
yy = [2, 1, 0]
axr.barh(yy, vals, color=cols, height=0.5)
for y, v, nm, c in zip(yy, vals, names, cols):
    axr.text(0.012, y + 0.38, nm, va="bottom", ha="left", fontsize=6.4,
             color=c)
    axr.text(v + 0.02, y, f"{v:.3f}", va="center", ha="left", fontsize=7.5)
axr.set_xlim(0, 1.13)
axr.set_ylim(-0.45, 3.05)
axr.set_yticks([])
axr.set_xticks([0, 0.5, 1.0])
axr.set_xticklabels(["0", "0.5", "1.0"], fontsize=6.5)
axr.set_xlabel("mean accuracy, months 3 to 36", fontsize=7)
axr.set_title("alarm-gated recalibration\non the benchmark replay\n"
              f"{static_acc:.3f} to {cad_acc:.3f} at "
              f"{ratio:.0f}$\\times$ fewer labels",
              fontsize=7.2, color="#b03030", pad=3)
for s in ["top", "right"]:
    axr.spines[s].set_visible(False)

fig.text(0.5, 0.022 * y_scale,
         f"UCI gas sensor drift benchmark replay ({n_seeds} seeds), alarms in "
         f"{fired_min} of 9 drifted batches, {ctl_alarms} false alarms in "
         f"{ctl_windows:,} stationary windows",
         ha="center", fontsize=6.9, color="#333333")

fig.savefig(f"{out}/graphical_abstract.pdf")
plt.close(fig)
print("wrote", f"{out}/graphical_abstract.pdf")
