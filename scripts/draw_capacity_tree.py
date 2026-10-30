"""Draw the capacity-tree example (figures/hierarchical_service.pdf)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RED, BLUE, BLACK, GRAY = "#ff0000", "#0000ff", "#000000", "#808080"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# Tiers {1,2,3,4} and {5,6,7,8}, branches {1,2,5,6} and {3,4,7,8}.
REQUEST = {1: 2, 2: 2, 3: .25, 4: .25, 5: .25, 6: .25, 7: 2, 8: 2}
SERVED = {1: 1, 2: 1, 3: .25, 4: .25, 5: 0, 6: 0, 7: .75, 8: .75}
TIER = {i: RED if i <= 4 else BLUE for i in REQUEST}
LEAF_X = {1: 1.6, 2: 2.5, 5: 3.7, 6: 4.6, 3: 7.6, 4: 8.5, 7: 9.7, 8: 10.6}
W, H, BASE = .62, .78, .55               # bar width, height per unit, baseline
NODE_H, SCALE = .36, .9                   # capacity bar height, width per unit


def capacity_node(ax, xc, y, cap, parts):
    x0 = xc - cap * SCALE / 2
    for amount, color in parts:
        ax.add_patch(Rectangle((x0, y), amount * SCALE, NODE_H, facecolor=color, edgecolor="none"))
        x0 += amount * SCALE
    ax.add_patch(Rectangle((xc - cap * SCALE / 2, y), cap * SCALE, NODE_H, facecolor="none",
                           edgecolor=BLACK, linewidth=.9))
    for k in range(1, int(cap)):
        xk = xc - cap * SCALE / 2 + k * SCALE
        ax.plot([xk, xk], [y, y + NODE_H], color="white", linewidth=.8)
    ax.text(xc + cap * SCALE / 2 + .18, y + NODE_H / 2, f"$c={cap:g}$", va="center", ha="left", fontsize=10)


def flexible(ax, a, b, label):
    ya, yb = BASE + SERVED[a] * H, BASE + SERVED[b] * H
    ax.add_patch(FancyArrowPatch((LEAF_X[a], ya + .1), (LEAF_X[b], yb + .1), arrowstyle="<|-|>",
                                 mutation_scale=7, connectionstyle="arc3,rad=-0.55",
                                 color=BLACK, linewidth=.9, zorder=5))
    ax.text((LEAF_X[a] + LEAF_X[b]) / 2, BASE - .52, label, ha="center", va="top", fontsize=9.5)


fig, ax = plt.subplots(figsize=(8.8, 3.5))
ax.set_xlim(0, 13.4), ax.set_ylim(-.35, 5.05), ax.axis("off")

root, branches = (6.1, 4.35), {"A": (3.1, 3.05, 2, [(2, RED)]), "B": (9.1, 3.05, 3, [(.5, RED), (1.5, BLUE)])}
for xc, y, cap, parts in branches.values():
    ax.plot([root[0], xc], [root[1], y + NODE_H], color=GRAY, linewidth=.9, zorder=0)
capacity_node(ax, *root, 4, [(2.5, RED), (1.5, BLUE)])

members = {"A": (1, 2, 5, 6), "B": (3, 4, 7, 8)}
for key, (xc, y, cap, parts) in branches.items():
    capacity_node(ax, xc, y, cap, parts)
    for i in members[key]:
        ax.plot([xc, LEAF_X[i]], [y, BASE + REQUEST[i] * H], color=GRAY, linewidth=.9, zorder=0)

for i, x in LEAF_X.items():
    ax.add_patch(Rectangle((x - W / 2, BASE), W, REQUEST[i] * H, facecolor="white",
                           edgecolor=TIER[i], linewidth=.9, linestyle=(0, (2.5, 1.5))))
    if SERVED[i]:
        ax.add_patch(Rectangle((x - W / 2, BASE), W, SERVED[i] * H, facecolor=TIER[i], edgecolor=TIER[i], linewidth=.9))
    ax.text(x, BASE - .1, str(i), ha="center", va="top", fontsize=9, color=TIER[i])
ax.plot([.9, 11.3], [BASE, BASE], color=BLACK, linewidth=.8)
flexible(ax, 1, 2, "$x_1+x_2=2$")
flexible(ax, 7, 8, "$x_7+x_8=1.5$")

# Legend
lx, ly = 11.9, 2.0
items = [(dict(facecolor=RED, edgecolor=RED), "Tier 1"), (dict(facecolor=BLUE, edgecolor=BLUE), "Tier 2"),
         (dict(facecolor="white", edgecolor=BLACK, linestyle=(0, (2.5, 1.5))), "Request"),
         (dict(facecolor=GRAY, edgecolor=GRAY), "Service")]
for k, (style, label) in enumerate(items):
    yk = ly - k * .42
    ax.add_patch(Rectangle((lx, yk - .13), .3, .26, linewidth=.9, **style))
    ax.text(lx + .45, yk, label, va="center", fontsize=9)

out = ROOT / "figures" / "hierarchical_service.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=.02, metadata={"Creator": None, "Producer": None, "CreationDate": None})
