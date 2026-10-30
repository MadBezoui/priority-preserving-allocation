"""Draw the ramp-limit example (figures/ramp_geometry.pdf)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RED, BLUE, BLACK, GRAY = "#ff0000", "#0000ff", "#000000", "#808080"
DASH = (0, (2.5, 1.5))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# d = (1, 1), C = 1, r = (0.1, 0.1).
R = .1


def panel(ax, letter, prev):
    ax.add_patch(Polygon([(0, 0), (1, 0), (0, 1)], closed=True, facecolor=GRAY, alpha=.15, edgecolor="none"))
    ax.plot([0, 1], [1, 0], color=BLACK, linewidth=1.6, solid_capstyle="butt", zorder=2)
    lo = [max(0, p - R) for p in prev]
    hi = [min(1, p + R) for p in prev]
    ax.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], facecolor=GRAY, alpha=.35, edgecolor="none", zorder=1))
    ax.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], facecolor="none", edgecolor=BLACK, linewidth=.9, zorder=3))
    ax.plot(*prev, "o", markersize=6, markerfacecolor="white", markeredgecolor=BLACK, markeredgewidth=.9, zorder=5)
    ax.set_xlim(0, 1.15), ax.set_ylim(0, 1.15), ax.set_aspect("equal")
    ax.set_xticks([0, .5, 1]), ax.set_yticks([0, .5, 1])
    ax.set_xticklabels(["0", "0.5", "1"]), ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("$x_1$", color=RED), ax.set_ylabel("$x_2$", color=BLUE, rotation=0, labelpad=8)
    ax.tick_params(labelsize=9, length=3, width=.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.7)
    ax.text(-.3, 1.03, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


fig = plt.figure(figsize=(7.4, 2.85))
axA = fig.add_axes([.07, .17, .38, .78])
axB = fig.add_axes([.58, .17, .38, .78])

# A: from (0.2, 0.8) the reachable priority point is (0.3, 0.7), then (0.4, 0.6), towards (1, 0).
panel(axA, "A", (.2, .8))
steps = [(.3, .7), (.4, .6)]
axA.plot([.1, .3], [.9, .7], color=BLUE, linewidth=2.4, solid_capstyle="butt", zorder=3)
axA.add_patch(FancyArrowPatch(steps[0], steps[1], arrowstyle="-|>", mutation_scale=7, color=BLUE,
                              linewidth=.9, linestyle=DASH, shrinkA=4, shrinkB=4, zorder=4))
axA.plot(*steps[0], "o", markersize=6, color=BLUE, markeredgecolor=BLACK, markeredgewidth=.5, zorder=5)
axA.plot(*steps[1], "o", markersize=4.5, markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=.9, zorder=5)
axA.plot(1, 0, "D", markersize=5.5, color=BLACK, zorder=5, clip_on=False)
axA.text(.08, .8, "$x_{t-1}$", ha="right", va="center", fontsize=10)
axA.text(.36, .76, "$x_t$", ha="left", va="center", fontsize=10, color=BLUE)
axA.text(.46, .66, "$x_{t+1}$", ha="left", va="center", fontsize=10, color=BLUE)
axA.text(1.02, .1, r"$x_t^\star$", ha="center", va="bottom", fontsize=10)

# B: from (1, 1) the total cannot fall below 1.8 > C = 1.
panel(axB, "B", (1, 1))
axB.plot([.8, 1.0], [1.0, .8], color=BLACK, linewidth=.9, linestyle=DASH, zorder=4)
axB.plot(.9, .9, "o", markersize=4, color=BLACK, zorder=5)
axB.text(.92, .76, "$x_1+x_2\\geq1.8$", ha="center", va="top", fontsize=9.5)
axB.text(.3, .3, "$x_1+x_2\\leq1$", ha="center", va="center", fontsize=9.5)
axB.text(.97, 1.07, "$x_{t-1}$", ha="right", va="bottom", fontsize=10)

out = ROOT / "figures" / "ramp_geometry.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=.02, metadata={"Creator": None, "Producer": None, "CreationDate": None})
