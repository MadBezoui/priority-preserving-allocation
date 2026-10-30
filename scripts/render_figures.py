"""Render the paper's main-grid figures, schematic figures and table fragments.

Reads the saved results and writes figures/*.pdf, results/paper_primary.tex and
results/paper_numbers.tex. The comparison figures and table come from
analyze_comparisons.py, the weighted, horizon and ramp figures from
render_extensions.py.
"""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
BLUE, RED, MAGENTA, GREEN, BLACK = "#0000ff", "#ff0000", "#ff00ff", "#00ff00", "#000000"
GRAY = "#808080"
COLORS = dict(exact=BLACK, feedback=BLUE, delayed_hard=RED,
              delayed_priority=MAGENTA, filtered_exact=GREEN)
MARKERS = dict(exact="s", feedback="o", delayed_hard="^",
               delayed_priority="X", filtered_exact="D")
NAMES = dict(exact="Exact", feedback="Feedback", delayed_hard="Hard filter",
             delayed_priority="Priority filter", filtered_exact="Smoothed delayed")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "lines.linewidth": 1.9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
SCHEMATICS = ("draw_priority_geometry.py", "draw_capacity_tree.py", "draw_ramp_geometry.py")


def read(name):
    return pd.read_csv(R / name)


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight",
                metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None})
    plt.close(fig)

def boot(values, seed=20261018):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = a[rng.integers(0, len(a), (5000, len(a)))].mean(axis=1)
    return a.mean(), *np.quantile(samples, [.025, .975])



def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", pad=12)
    ax.grid(axis="y", color=GRAY, alpha=.17, linewidth=.6)
    ax.set_axisbelow(True)



def line(ax, x, y, method):
    return ax.plot(x, y, color=COLORS[method], marker=MARKERS[method],
                   markersize=5, markeredgecolor=BLACK, markeredgewidth=.45,
                   label=NAMES[method])[0]



def legend(fig, methods, y=-.02, ncol=3):
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m],
                      markeredgecolor=BLACK, markeredgewidth=.45,
                      label=NAMES[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y),
               ncol=ncol, frameon=False, columnspacing=1.8, handlelength=2.3)



def primary_comparison(df):
    base = df[(df.dataset == "electricity") & (df.ratio == .9) & (df.delay == 4)]
    methods = ["exact", "feedback", "filtered_exact", "delayed_priority", "delayed_hard"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    fig.subplots_adjust(left=.21, bottom=.18, right=.98, top=.85, wspace=.22)
    jitter = np.linspace(-.13, .13, 30)
    for ax, metric, title, unit in zip(axes, ["variation", "quadratic_shortfall"],
                                      ["Movement", "Shortfall"],
                                      [r"$V$", r"$\overline{Q}$"]):
        panel(ax, "A" if metric == "variation" else "B", title)
        ax.grid(False)
        ax.grid(axis="x", color=GRAY, alpha=.17, linewidth=.6)
        for j, m in enumerate(methods):
            values = base[base.method == m].sort_values("block")[metric].to_numpy()
            mean, low, high = boot(values)
            ax.scatter(values, j + jitter, color=GRAY, alpha=.25, s=14, edgecolors="none", zorder=2)
            ax.errorbar(mean, j, xerr=[[mean-low], [high-mean]], color=COLORS[m],
                        fmt=MARKERS[m], markersize=7, markeredgecolor=BLACK,
                        markeredgewidth=.5, capsize=4, linewidth=2.3, zorder=4)
        ax.set_xlabel(unit)
    axes[0].set_yticks(range(5), [NAMES[m] for m in methods])
    axes[0].invert_yaxis()
    save(fig, "primary_tradeoff")



def gain(df):
    gains = [.1, .25, .5, .75, 1.]
    methods = ["alpha_010", "feedback", "alpha_050", "alpha_075", "exact"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    fig.subplots_adjust(left=.10, right=.98, top=.85, bottom=.27, wspace=.32)
    for ax, metric, title in zip(axes, ["variation", "quadratic_gap"], ["Movement", "Excess shortfall"]):
        panel(ax, "A" if metric == "variation" else "B", title)
        for ratio, color, marker in [(.6, RED, "s"), (.9, BLUE, "o"), (1.2, MAGENTA, "^")]:
            data = df[(df.dataset == "electricity") & (df.delay == 0) & (df.ratio == ratio)]
            means = [data[data.method == m][metric].mean() for m in methods]
            if ratio == .9:
                for _, block in data.groupby("block"):
                    ax.plot(gains, [block[block.method == m][metric].iloc[0] for m in methods],
                            color=GRAY, alpha=.12, linewidth=.7, zorder=1)
            ax.plot(gains, means, color=color, marker=marker, markersize=5,
                    markeredgecolor=BLACK, markeredgewidth=.4, label=rf"$\rho={ratio}$", zorder=4)
        ax.set_xticks(gains)
        ax.set_xlabel(r"Gain $\alpha$")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(r"$V$")
    axes[1].set_ylabel(r"$\overline{Q}-\overline{Q}^\star$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    save(fig, "gain_sensitivity")



def command_age(df):
    s = df[(df.dataset == "electricity") & (df.ratio == .9)].groupby(["delay", "method"]).mean(numeric_only=True).reset_index()
    methods = ["exact", "feedback", "delayed_priority", "filtered_exact", "delayed_hard"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65))
    fig.subplots_adjust(left=.10, right=.98, top=.86, bottom=.30, wspace=.35)
    panel(axes[0], "A", "Priority deficit")
    h = s[s.method == "delayed_hard"]
    line(axes[0], h.delay, h.priority_regret * 100, "delayed_hard")
    axes[0].plot(h.delay, np.zeros(len(h)), color=BLACK, marker="s", markersize=5)
    axes[0].text(.02, .10, r"Projection onto $F_t$", transform=axes[0].transAxes, fontsize=9)
    axes[0].set_ylabel("Deficit / training total (%)")
    panel(axes[1], "B", "Excess shortfall")
    for m in methods:
        a = s[s.method == m]
        line(axes[1], a.delay, a.quadratic_gap, m)
    axes[1].set_ylabel(r"$\overline{Q}-\overline{Q}^\star$")
    for ax in axes:
        ax.set_xlabel("Command age (observations)")
        ax.set_xticks([0, 1, 4, 12])
        ax.set_ylim(bottom=-.001)
    legend(fig, methods, y=-.015)
    save(fig, "delay")



def runtime_recovery():
    rt, rec = read("runtime.csv"), read("recovery.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.subplots_adjust(left=.105, right=.98, top=.85, bottom=.23, wspace=.43)
    panel(axes[0], "A", "Evaluation time")
    for m, color, marker, name in [("exact", BLACK, "s", "Exact"), ("feedback", BLUE, "o", "Feedback"),
                                  ("generic_qp", RED, "^", "SLSQP")]:
        group = rt[rt.method == m].groupby("n").microseconds
        med, hi = group.median(), group.quantile(.95)
        axes[0].fill_between(med.index, med.values, hi.values, color=GRAY, alpha=.15)
        axes[0].plot(med.index, med.values, color=color, marker=marker, markersize=5, label=name)
    axes[0].set(xscale="log", yscale="log", xlabel="Recipients", ylabel=r"Time ($\mu$s)")
    axes[0].legend(frameon=False, loc="upper left")
    panel(axes[1], "B", "Recovery to a fixed optimum")
    for a, color, marker in [(.1, BLACK, "s"), (.25, BLUE, "o"), (.5, RED, "^"), (.75, MAGENTA, "D")]:
        z = rec[rec.alpha == a]
        axes[1].semilogy(z.t, np.maximum(z.bound, 1e-14), color=GRAY, alpha=.45,
                          linestyle=(0, (4, 3)), linewidth=1)
        axes[1].semilogy(z.t, np.maximum(z.error, 1e-14), color=color,
                          marker=marker, markevery=10, markersize=3.5, label=rf"$\alpha={a}$")
    axes[1].set(xlabel="Updates after perturbation", ylabel=r"Error $\|x_t-x^\star\|_2$")
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.27), ncol=2, fontsize=8)
    save(fig, "runtime_recovery")



def example_trace():
    name = "example_trajectories.npz"
    ex = np.load(R / name)
    d, c = ex["electricity_demand"], ex["electricity_capacity"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
    fig.subplots_adjust(left=.11, right=.98, top=.92, bottom=.12, hspace=.53)
    axes[0].plot(d.sum(axis=1), color=BLACK, linewidth=1.2, label="Requests")
    axes[0].plot(c, color=GRAY, linewidth=1.5, linestyle=(0, (4, 3)), label="Capacity")
    axes[0].plot(ex["electricity_feedback"].sum(axis=1), color=BLUE, label="Priority service")
    axes[0].set_ylabel("Total units")
    axes[0].legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=8, borderaxespad=0)
    for m in ["exact", "feedback", "delayed_hard"]:
        a = ex["electricity_" + m]
        axes[1].plot(a[:, 12], color=COLORS[m], label=NAMES[m], linewidth=1.3)
    for m in ["exact", "feedback"]:
        a = ex["electricity_" + m]
        # The first implemented allocation equals the common exact warm start.
        increments = np.r_[0., np.abs(np.diff(a, axis=0)).sum(axis=1) / 32]
        axes[2].plot(np.cumsum(increments), color=COLORS[m], label=NAMES[m])
    for ax, letter, title in zip(axes, "ABC", ["Common service totals", "Redistribution within tier 2", "Accumulated movement"]):
        panel(ax, letter, title)
        ax.axvspan(112, 224, color=GRAY, alpha=.12, linewidth=0)
    axes[1].set_ylabel("Recipient 13 units")
    axes[1].legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=8, borderaxespad=0)
    axes[2].set_ylabel(r"Cumulative $\ell_1$ / 32")
    axes[2].set_xlabel("15-minute observation within the first block")
    axes[2].legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    axes[2].set_xlim(0, 335)
    save(fig, "example_trace")



def paper_tables(df):
    primary = df[(df.dataset == "electricity") & (df.ratio == .9) & (df.delay == 4)]
    rows = []
    names = dict(NAMES, uniform="No priority", no_projection="No projection")
    for method in ["exact", "feedback", "delayed_priority", "filtered_exact", "delayed_hard", "uniform", "no_projection"]:
        z = primary[primary.method == method].mean(numeric_only=True)
        rows.append(f"{names[method]} & {100*z.priority_regret:.3f} & {z.quadratic_shortfall:.4f} & {z.variation:.4f} & {100*z.capacity_violation:.3f} " + r"\\")
    text = r"\begin{tabular}{lrrrr}\toprule" + "\n" + r"Method & $100R_1$ & $\overline{Q}$ & $V$ & Excess (\%)\\\midrule" + "\n" + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}" + "\n"
    (R / "paper_primary.tex").write_text(text)
    timing = read("runtime.csv").groupby(["n", "method"]).microseconds.median()
    macros = {
        "ExactThirtyTwo": f"{timing.loc[32, 'exact']:.1f}",
        "FeedbackThirtyTwo": f"{timing.loc[32, 'feedback']:.1f}",
        "SlsqpThirtyTwo": f"{timing.loc[32, 'generic_qp']:.1f}",
        "ExactLarge": f"{timing.loc[2048, 'exact']:.1f}",
        "FeedbackLarge": f"{timing.loc[2048, 'feedback']:.1f}",
    }
    (R / "paper_numbers.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in macros.items()) + "\n")


def main():
    F.mkdir(exist_ok=True)
    for script in SCHEMATICS:
        runpy.run_path(str(Path(__file__).with_name(script)))
    df = read("replay.csv")
    primary_comparison(df)
    gain(df)
    command_age(df)
    runtime_recovery()
    example_trace()
    paper_tables(df)


if __name__ == "__main__":
    main()
