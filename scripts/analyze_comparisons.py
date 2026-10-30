"""Summaries, table fragment and figures for the comparison experiments.

Reads results/comparisons/*.csv and writes
  results/comparisons/summary.json      every summary number
  results/comparisons/table_external.tex  avoidable-cost table
  figures/pareto.pdf, figures/horizon_ratio.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abo.comparison import paired_bootstrap  # noqa: E402

RES = ROOT / "results" / "comparisons"
FIG = ROOT / "figures"
MU = (1.0, 3.0, 9.0)
BLUE, RED, MAGENTA, GREEN, CYAN = "#0000ff", "#ff0000", "#ff00ff", "#00ff00", "#00ffff"
BLACK, WHITE, GRAY = "#000000", "#ffffff", "#808080"
STYLE = {"exact": (BLACK, "s"), "feedback": (BLUE, "o"), "ema": (CYAN, "D"), "robd": (MAGENTA, "^"),
         "l1": (RED, "v"), "mpc-persistence": (WHITE, "h"), "mpc-oracle": (GREEN, "*")}
NAMES = {"feedback": "Feedback", "ema": "Exponential smoothing", "robd": "Regularized OBD",
         "l1": "Absolute penalty", "mpc-oracle": "MPC, oracle", "mpc-persistence": "MPC, persistence",
         "exact": "Exact"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "lines.linewidth": 1.9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def panel(ax, title=None):
    if title:
        ax.set_title(title, loc="left", pad=12)
    ax.grid(axis="y", color=GRAY, alpha=.17, linewidth=.6)
    ax.set_axisbelow(True)


def mark(ax, x, y, fam, ls="none", **kw):
    color, marker = STYLE[fam]
    line = BLACK if color == WHITE else color
    return ax.plot(x, y, ls=ls, marker=marker, color=line, mfc=color, mec=BLACK, mew=.45,
                   ms=7 if marker == "*" else 5, lw=1.5, label=NAMES[fam], **kw)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight",
                metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None})
    plt.close(fig)


def family(row):
    if row["rule"] == "mpc":
        p = json.loads(row["param"])
        return "mpc-" + p["forecast"]
    if row["rule"] == "feedback" and json.loads(row["param"])["alpha"] == 1.0:
        return "exact"
    return row["rule"]


def load_rules():
    r = pd.read_csv(RES / "rules.csv")
    r["family"] = r.apply(family, axis=1)
    return r


def excess_at(agg, col, levels):
    """Excess shortfall interpolated at common movement levels."""
    tab = {}
    for fam in ["feedback", "ema", "robd", "l1"]:
        a = agg[agg.family == fam].sort_values(col)
        tab[fam] = {}
        for v in levels:
            if a[col].min() <= v <= a[col].max():
                if fam == "robd":
                    # lower envelope of the two-parameter family
                    best = np.inf
                    for i in range(len(a)):
                        for j in range(len(a)):
                            v1, v2 = a[col].iloc[i], a[col].iloc[j]
                            if v1 <= v <= v2 and v2 > v1:
                                w = (v - v1) / (v2 - v1)
                                best = min(best, (1 - w) * a.dQ.iloc[i] + w * a.dQ.iloc[j])
                    tab[fam][v] = best
                else:
                    tab[fam][v] = float(np.interp(v, a[col], a.dQ))
            else:
                tab[fam][v] = None
    return {f: {str(k): v for k, v in d.items()} for f, d in tab.items()}


def pareto(r, summary):
    exact = r[r.family == "exact"].set_index("block")
    agg = r.groupby(["family", "config"])[["Q", "V", "V2", "seconds"]].mean().reset_index()
    qx = exact.Q.mean()
    agg["dQ"] = agg.Q - qx
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    fig.subplots_adjust(left=.10, right=.98, top=.86, bottom=.32, wspace=.30)
    for ax, col, lab, title in [(axes[0], "V", "Absolute movement $V$", "A   Absolute movement"),
                                (axes[1], "V2", "Squared movement $V_2$", "B   Squared movement")]:
        panel(ax, title)
        for fam in ["feedback", "ema", "l1", "robd", "mpc-persistence", "mpc-oracle"]:
            a = agg[agg.family == fam].sort_values(col)
            mark(ax, a[col], a.dQ, fam, ls="-" if fam in ("feedback", "ema", "l1") else "none", zorder=3)
        mark(ax, [exact[col].mean()], [0], "exact", zorder=4)
        ax.set_xlabel(lab)
        ax.set_ylim(bottom=-0.001)
    axes[0].set_ylabel("Excess shortfall over exact")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, 0), ncol=4, frameon=False,
               columnspacing=1.5, handlelength=2.3)
    save(fig, "pareto")

    summary["excess_at_movement"] = excess_at(agg, "V", [0.045, 0.05, 0.06])
    summary["excess_at_squared_movement"] = excess_at(agg, "V2", [0.018])
    summary["pareto_points"] = agg.to_dict(orient="records")
    return agg


def horizon(r, summary):
    comp = pd.read_csv(RES / "competitive.csv")
    clair = pd.read_csv(RES / "clairvoyant.csv")
    exact = r[r.family == "exact"].set_index("block")
    out = {}
    # unavoidable shortfall per block = exact rule shortfall total
    T, n = 336, 32
    qstar = exact.Q * T * n
    rows = []
    for mu in MU:
        c = comp[comp.mu == mu].set_index("block")
        opt = c.excess_opt_upper
        for fam in ["exact", "feedback", "ema", "robd", "l1", "mpc-oracle", "mpc-persistence"]:
            sub = r[r.family == fam]
            for cfg, g in sub.groupby("config"):
                g = g.set_index("block").sort_index()
                ex = g[f"phi_{mu:g}"] - qstar
                rows.append({"mu": mu, "family": fam, "config": cfg, "block": None,
                             "ratio_mean": float((ex / opt).mean()), "ratios": (ex / opt).values})
    df = pd.DataFrame(rows)
    # two-fold selection: choose config minimizing mean ratio on one half, evaluate on other
    halves = [np.arange(0, 15), np.arange(15, 30)]
    sel = []
    for mu in MU:
        for fam in df.family.unique():
            d = df[(df.mu == mu) & (df.family == fam)]
            test_vals = np.zeros(30)
            chosen = []
            for tr, te in [(halves[0], halves[1]), (halves[1], halves[0])]:
                best = min(d.itertuples(), key=lambda z: z.ratios[tr].mean())
                test_vals[te] = best.ratios[te]
                chosen.append(best.config)
            sel.append({"mu": mu, "family": fam, "chosen": chosen, "ratios": test_vals,
                        "mean": float(test_vals.mean())})
    sel = pd.DataFrame(sel)
    # nominal feedback alpha = 1/(1+mu)
    nominal = {}
    for mu in MU:
        c = comp[comp.mu == mu]
        star = clair[clair.mu == mu].set_index("block").phi_star_upper
        rr = (c.excess_feedback / c.excess_opt_upper).values
        nominal[mu] = {"mean": float(rr.mean()), "max": float(rr.max()), "min": float(rr.min()),
                       "gap_pct_total": float((100 * clair[clair.mu == mu].gap / clair[clair.mu == mu].phi_star_upper).mean()),
                       "gap_pct_total_myopic": float((100 * (exact[f"phi_{mu:g}"] - star) / star).mean()),
                       "bound": 1 + mu,
                       "apriori_mean": float(c.bound_myopic_comparator.mean() / c.excess_opt_upper.mean())}
    tests = {}
    for mu in MU:
        fb = sel[(sel.mu == mu) & (sel.family == "feedback")].ratios.iloc[0]
        for fam in ["exact", "ema", "robd", "l1", "mpc-oracle", "mpc-persistence"]:
            o = sel[(sel.mu == mu) & (sel.family == fam)].ratios.iloc[0]
            diff = o - fb
            lo, hi = paired_bootstrap(diff)
            p = float(wilcoxon(o, fb).pvalue) if np.any(diff != 0) else 1.0
            tests[f"{mu:g}:{fam}"] = {"mean_diff": float(diff.mean()), "lo": lo, "hi": hi, "p": p,
                                      "wins_feedback": int((diff > 0).sum())}
    summary["horizon_selected"] = {f"{z.mu:g}:{z.family}": {"mean": z.mean, "chosen": z.chosen}
                                   for z in sel.itertuples()}
    summary["horizon_nominal_feedback"] = {f"{k:g}": v for k, v in nominal.items()}
    c3 = clair[clair.mu == 3]
    summary["certificate_mu3"] = {"gap_mean": float(c3.gap.mean()), "sharp_bound_mean": float(c3.sharp_bound.mean()),
                                  "simple_bound_mean": float(c3.simple_bound.mean()),
                                  "all_below": bool((c3.gap <= c3.sharp_bound).all())}
    summary["horizon_tests"] = tests

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    panel(ax)
    fams = ["exact", "feedback", "ema", "robd", "l1", "mpc-persistence", "mpc-oracle"]
    width = 0.12
    for i, fam in enumerate(fams):
        for k, mu in enumerate(MU):
            v = sel[(sel.mu == mu) & (sel.family == fam)].ratios.iloc[0]
            x = k + (i - 3) * width
            jitter = np.random.default_rng(i).uniform(-0.03, 0.03, v.size)
            ax.scatter(x + jitter, v, color=GRAY, alpha=.35, s=7, edgecolors="none", zorder=1)
            lo, hi = paired_bootstrap(v)
            ax.vlines(x, lo, hi, color=BLACK, lw=1, zorder=2)
            h = mark(ax, [x], [v.mean()], fam, zorder=3)
            if k:
                h[0].set_label(None)
    ax.axhline(1, color=GRAY, lw=.8, ls="--", zorder=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([rf"$\mu={m:g}$" for m in MU])
    ax.set_ylabel("Avoidable cost / clairvoyant")
    ax.set_ylim(0.9, None)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False,
              columnspacing=1.5, handlelength=1.6)
    save(fig, "horizon_ratio")
    return sel


def sensitivity(summary):
    s = pd.read_csv(RES / "sensitivity.csv")
    s["fam"] = s.config.str.split("[", regex=False).str[0]
    out = {}
    for (kind, size), g in s.groupby(["kind", "tier_size"]):
        piv = g.pivot_table(index=["seed", "block"], columns="fam", values=["Q", "V"])
        red = 1 - piv["V"]["feedback"] / piv["V"]["exact"]
        inc = piv["Q"]["feedback"].groupby("seed").mean() / piv["Q"]["exact"].groupby("seed").mean() - 1
        ema_red = 1 - piv["V"]["ema"] / piv["V"]["exact"]
        ema_inc = piv["Q"]["ema"].groupby("seed").mean() / piv["Q"]["exact"].groupby("seed").mean() - 1
        l1_red = 1 - piv["V"]["l1"] / piv["V"]["exact"]
        l1_inc = piv["Q"]["l1"].groupby("seed").mean() / piv["Q"]["exact"].groupby("seed").mean() - 1
        bys = red.groupby("seed").mean()
        out[f"{kind}:{size}"] = {
            "fb_red_mean": float(bys.mean()), "fb_red_min": float(bys.min()), "fb_red_max": float(bys.max()),
            "fb_inc_mean": float(inc.mean()), "fb_inc_min": float(inc.min()), "fb_inc_max": float(inc.max()),
            "ema_red_mean": float(ema_red.groupby("seed").mean().mean()), "ema_inc_mean": float(ema_inc.mean()),
            "l1_red_mean": float(l1_red.groupby("seed").mean().mean()), "l1_inc_mean": float(l1_inc.mean()),
            "R1_max": float(g.R1.max()), "excess_max": float(g.excess.max())}
    summary["sensitivity"] = out


def reach(summary):
    r = pd.read_csv(RES / "reachability.csv")
    g = r.groupby("ramp")[["transitions", "capacity_ok", "request_ok", "both_ok"]].sum()
    summary["reachability"] = {f"{k:g}": {"both": float(v.both_ok / v.transitions),
                                          "request": float(v.request_ok / v.transitions),
                                          "capacity": float(v.capacity_ok / v.transitions)}
                               for k, v in g.iterrows()}


def main():
    FIG.mkdir(exist_ok=True)
    summary = {}
    reach(summary)
    if (RES / "sensitivity.csv").exists():
        sensitivity(summary)
    if (RES / "rules.csv").exists():
        r = load_rules()
        pareto(r, summary)
        horizon(r, summary)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


def external_table():
    """LaTeX fragment for Table tab:external (run after main())."""
    summ = json.loads((RES / "summary.json").read_text())
    r = load_rules()
    secs = r.groupby("family").seconds.mean()
    rows = [("exact", "Exact (myopic)"), ("feedback", "Feedback"), ("ema", "Exponential smoothing"),
            ("robd", "Regularized OBD"), ("l1", "Absolute penalty"),
            ("mpc-persistence", "MPC, persistence"), ("mpc-oracle", "MPC, oracle")]
    def pp(x):
        # percentage points, signed, without a negative zero
        s = f"{100 * x:+.1f}"
        return "$0.0$" if s in ("+0.0", "-0.0") else f"${s}$"

    lines = [r"\begin{tabular}{lrrrrr}\toprule",
             r"Rule & $\mu=1$ & $\mu=3$ & $\mu=9$ & Difference at $\mu=3$ & Time (s)\\\midrule"]
    for fam, name in rows:
        vals = [summ["horizon_selected"][f"{m:g}:{fam}"]["mean"] for m in MU]
        if fam == "feedback":
            diff = "--"
        else:
            t = summ["horizon_tests"][f"3:{fam}"]
            diff = f"{pp(t['mean_diff'])} [{pp(t['lo'])}, {pp(t['hi'])}]"
        lines.append(f"{name} & {vals[0]:.3f} & {vals[1]:.3f} & {vals[2]:.3f} & {diff} & {secs[fam]:.2f}\\\\")
    lines.append(r"\bottomrule\end{tabular}")
    (RES / "table_external.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
    external_table()
