"""Draw the priority-face example (figures/priority_geometry.pdf)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RED, BLUE, MAGENTA, BLACK, GRAY = "#ff0000", "#0000ff", "#ff00ff", "#000000", "#808080"
DASH = (0, (2.5, 1.5))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# Panel A: d = (2,2,3,3,1,1), C = 7, tiers {1,2}, {3,4}, {5,6}.
REQUEST = {1: 2, 2: 2, 3: 3, 4: 3, 5: 1, 6: 1}
SERVED = {1: 2, 2: 2, 3: 1.5, 4: 1.5, 5: 0, 6: 0}
TIER = {1: RED, 2: RED, 3: BLUE, 4: BLUE, 5: MAGENTA, 6: MAGENTA}
LEAF_X = {1: .7, 2: 1.55, 3: 2.75, 4: 3.6, 5: 4.8, 6: 5.65}
W, H, BASE = .6, .5, .55
NODE = (3.175, 2.7, .3, .62)           # centre, bottom, height, width per unit
# Panel B: the marginal tier at gain 0.25.
PREV, TARGET, CAND, PROJ, EXACT = (.4, 2.6), (3, 3), (1.05, 2.7), (.675, 2.325), (1.5, 1.5)

fig = plt.figure(figsize=(7.4, 2.85))
axA = fig.add_axes([0, .13, .56, .85])
axB = fig.add_axes([.66, .2, .3, .76])

# Panel A
axA.set_xlim(0, 7.3), axA.set_ylim(-.45, 3.2), axA.axis("off")
xc, y0, nh, sc = NODE
for i, x in LEAF_X.items():
    axA.plot([xc, x], [y0, BASE + REQUEST[i] * H], color=GRAY, linewidth=.9, zorder=0)
left = xc - 7 * sc / 2
for amount, color in [(4, RED), (3, BLUE)]:
    axA.add_patch(Rectangle((left, y0), amount * sc, nh, facecolor=color, edgecolor="none"))
    left += amount * sc
axA.add_patch(Rectangle((xc - 7 * sc / 2, y0), 7 * sc, nh, facecolor="none", edgecolor=BLACK, linewidth=.9))
for k in range(1, 7):
    xk = xc - 7 * sc / 2 + k * sc
    axA.plot([xk, xk], [y0, y0 + nh], color="white", linewidth=.8)
axA.text(xc + 7 * sc / 2 + .18, y0 + nh / 2, "$C=7$", va="center", fontsize=10)
for i, x in LEAF_X.items():
    axA.add_patch(Rectangle((x - W / 2, BASE), W, REQUEST[i] * H, facecolor="white",
                            edgecolor=TIER[i], linewidth=.9, linestyle=DASH))
    if SERVED[i]:
        axA.add_patch(Rectangle((x - W / 2, BASE), W, SERVED[i] * H, facecolor=TIER[i], edgecolor=TIER[i], linewidth=.9))
    axA.text(x, BASE - .1, str(i), ha="center", va="top", fontsize=9, color=TIER[i])
axA.plot([.2, 6.15], [BASE, BASE], color=BLACK, linewidth=.8)
ya = BASE + 1.5 * H + .1
axA.add_patch(FancyArrowPatch((LEAF_X[3], ya), (LEAF_X[4], ya), arrowstyle="<|-|>", mutation_scale=7,
                              connectionstyle="arc3,rad=-0.55", color=BLACK, linewidth=.9, zorder=5))
axA.text((LEAF_X[3] + LEAF_X[4]) / 2, BASE - .52, "$x_3+x_4=3$", ha="center", va="top", fontsize=9.5)
axA.text(0, 3.2, "A", fontsize=11, fontweight="bold", va="top")

# Panel B
axB.plot([0, 3], [3, 0], color=BLACK, linewidth=1.6, solid_capstyle="butt", zorder=2)
axB.plot([PREV[0], TARGET[0]], [PREV[1], TARGET[1]], color=GRAY, linewidth=.9, linestyle=DASH, zorder=1)
axB.add_patch(FancyArrowPatch(CAND, PROJ, arrowstyle="-|>", mutation_scale=8, color=BLUE,
                              linewidth=1.1, shrinkA=3, shrinkB=4, zorder=3))
axB.plot(*PREV, "o", markersize=6, markerfacecolor="white", markeredgecolor=BLACK, markeredgewidth=.9, zorder=4)
axB.plot(*CAND, "s", markersize=5.5, markerfacecolor="white", markeredgecolor=BLACK, markeredgewidth=.9, zorder=4)
axB.plot(*PROJ, "o", markersize=6, color=BLUE, markeredgecolor=BLACK, markeredgewidth=.5, zorder=4)
axB.plot(*EXACT, "D", markersize=5.5, color=BLACK, zorder=4)
axB.plot(*TARGET, "x", markersize=6, color=BLACK, markeredgewidth=1.1, zorder=4)
for (x, y), label, dx, dy, ha, color in [(PREV, "$x_{t-1}$", -.12, -.17, "right", BLACK),
                                        (CAND, "$v_t$", .03, .22, "center", BLACK),
                                        (PROJ, "$x_t$", -.14, -.16, "right", BLUE),
                                        (EXACT, r"$x_t^\star$", .16, .1, "left", BLACK),
                                        (TARGET, "$d_t$", 0, -.24, "center", BLACK)]:
    axB.text(x + dx, y + dy, label, ha=ha, va="center", fontsize=10, color=color)
axB.set_xlim(0, 3.25), axB.set_ylim(0, 3.25), axB.set_aspect("equal")
axB.set_xticks([0, 1, 2, 3]), axB.set_yticks([0, 1, 2, 3])
axB.set_xlabel("$x_3$"), axB.set_ylabel("$x_4$", rotation=0, labelpad=8)
axB.tick_params(labelsize=9, length=3, width=.6)
for side in ("top", "right"):
    axB.spines[side].set_visible(False)
for side in ("left", "bottom"):
    axB.spines[side].set_linewidth(.7)
axB.text(-.33, 1.03, "B", transform=axB.transAxes, fontsize=11, fontweight="bold", va="bottom")

handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=c) for c in (RED, BLUE, MAGENTA)]
handles += [Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=BLACK, linestyle=DASH),
            Rectangle((0, 0), 1, 1, facecolor=GRAY, edgecolor=GRAY)]
fig.legend(handles, ["Tier 1", "Tier 2", "Tier 3", "Request", "Service"], loc="lower left",
           bbox_to_anchor=(.0, -.04), ncol=5, frameon=False, fontsize=9, handlelength=1.1, columnspacing=1.2)

out = ROOT / "figures" / "priority_geometry.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=.02, metadata={"Creator": None, "Producer": None, "CreationDate": None})
