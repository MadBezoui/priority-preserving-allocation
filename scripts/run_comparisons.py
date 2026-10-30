"""Comparison experiments on the 30 electricity blocks.

Outputs (in results/comparisons/):
  rules.csv        one row per block and rule configuration: shortfall,
                   movement (L1 and squared L2), priority deficit, capacity
                   excess and horizon cost at mu in MU_GRID
  clairvoyant.csv  clairvoyant horizon optimum per block and mu, with its
                   objective interval and the two feedback certificates
  sensitivity.csv  random tier assignments and kW units
  reachability.csv frequency of the recursive-feasibility condition
  competitive.csv  avoidable cost of feedback against the clairvoyant bound

Usage: python scripts/run_comparisons.py [--part all|reach|clairvoyant|competitive|sensitivity|rules]
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abo import comparison as core  # noqa: E402

OUT = ROOT / "results" / "comparisons"
MU_GRID = (1.0, 3.0, 9.0)
GAINS = (0.05, 0.1, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0)


def configs():
    cfg = []
    for a in GAINS:
        cfg.append(("feedback", {"alpha": a}))
    for a in GAINS[:-1]:
        cfg.append(("ema", {"alpha": a}))
    for l1 in (1.0, 3.0, 9.0):
        for l2 in (0.25, 1.0, 4.0):
            cfg.append(("robd", {"lam1": l1, "lam2": l2}))
    for k in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        cfg.append(("l1", {"kappa": k}))
    for mu in MU_GRID:
        for w in (4, 12):
            cfg.append(("mpc", {"mu": mu, "window": w, "forecast": "oracle"}))
        cfg.append(("mpc", {"mu": mu, "window": 12, "forecast": "persistence"}))
    return cfg


def label(rule, par):
    if not par:
        return rule
    return rule + "[" + ",".join(f"{k}={v}" for k, v in par.items()) + "]"


def _rules_block(j):
    inst = core.electricity_blocks()[j]
    x0 = core.exact_path(core.Instance(inst.demand[:1], inst.capacity[:1], inst.tiers, inst.scale))[0]
    rows = []
    for rule, par in configs():
        t0 = time.perf_counter()
        X = core.run_rule(inst, rule, **par)
        el = time.perf_counter() - t0
        m = core.metrics(X, inst)
        prev = np.vstack([x0[None, :], X[:-1]])
        m["V2"] = ((X - prev) ** 2).sum() / (X.shape[0] * inst.scale)
        row = {"block": j, "rule": rule, "config": label(rule, par),
               "param": json.dumps(par, sort_keys=True), "seconds": el, **m}
        for mu in MU_GRID:
            row[f"phi_{mu:g}"] = core.horizon_cost(X, inst.demand, x0, mu)
        rows.append(row)
    return rows


def _clair_block(j):
    inst = core.electricity_blocks()[j]
    D, C, tiers = inst.demand, inst.capacity, inst.tiers
    x0 = core.exact_path(core.Instance(D[:1], C[:1], tiers, inst.scale))[0]
    # observed-data increments B_t (Corollary 5), row 0 has no predecessor
    B = np.zeros(D.shape[0])
    B[1:] = np.sqrt(2) * np.abs(np.diff(D, axis=0)).sum(1) + np.abs(np.diff(C))
    rows = []
    for mu in MU_GRID:
        Y, up, low = core.solve_horizon(D, C, tiers, x0, mu, tol=1e-11, max_iter=40000)
        alpha = 1.0 / (1.0 + mu)
        X = core.run_rule(inst, "feedback", alpha=alpha)
        phiX = core.horizon_cost(X, D, x0, mu)
        moves = np.diff(X, axis=0)
        simple = 0.5 * mu ** 2 * (moves ** 2).sum()
        # sharpened a posteriori certificate 0.5 h^T H^{-1} h (Proposition 6)
        h = np.zeros_like(X)
        h[:-1] = mu * (X[:-1] - X[1:])
        sharp = 0.5 * float((h * _solve_H(h, mu)).sum())
        # a priori certificate (new proposition): E_t propagated from E_0 = 0
        q = 1.0 - alpha
        E = np.zeros(D.shape[0])
        for t in range(1, D.shape[0]):
            E[t] = q * (E[t - 1] + B[t])
        star = core.exact_path(inst)
        e = np.linalg.norm(X - star, axis=1)
        apriori = 0.5 * mu ** 2 * ((E[:-1] + E[1:] + B[1:]) ** 2).sum()
        apriori_closed = 0.5 * mu ** 2 * (1 + 2 * mu) ** 2 * (B[1:] ** 2).sum()
        rows.append({"block": j, "mu": mu, "phi_star_upper": up, "phi_star_lower": low,
                     "phi_feedback": phiX, "gap": phiX - up, "sharp_bound": sharp,
                     "simple_bound": simple, "apriori_bound": apriori,
                     "apriori_closed_bound": apriori_closed,
                     "tracking_ok": bool(np.all(e <= E + 1e-9)),
                     "sum_B2": float((B[1:] ** 2).sum())})
    return rows


def _solve_H(h, mu):
    """Apply H^{-1} with H = I + mu D^T D (tridiagonal in time, fixed initial value)."""
    from scipy.linalg import solve_banded
    T = h.shape[0]
    main = 1.0 + 2.0 * mu * np.ones(T)
    main[-1] = 1.0 + mu
    off = -mu * np.ones(T - 1)
    ab = np.zeros((3, T))
    ab[0, 1:] = off
    ab[1] = main
    ab[2, :-1] = off
    return solve_banded((1, 1), ab, h)


def _sens_job(args):
    kind, size, seed = args
    rng = np.random.default_rng(seed)
    if kind == "random_tiers":
        crit = rng.choice(32, size=size, replace=False)
        blocks = core.electricity_blocks(critical=crit)
    else:
        blocks = core.electricity_blocks(units="kw")
    rows = []
    for j, inst in enumerate(blocks):
        for rule, par in [("exact", {}), ("feedback", {"alpha": 0.25}), ("ema", {"alpha": 0.25}),
                          ("l1", {"kappa": 0.1 if kind == "random_tiers" else 0.1 * 2124.6 / 32})]:
            X = core.run_rule(inst, rule, **par)
            m = core.metrics(X, inst)
            rows.append({"kind": kind, "tier_size": size, "seed": seed, "block": j,
                         "config": label(rule, par), **m})
    return rows


def _competitive_block(j):
    inst = core.electricity_blocks()[j]
    D, C, tiers = inst.demand, inst.capacity, inst.tiers
    star = core.exact_path(inst)
    x0 = star[0]
    qstar = 0.5 * ((star - D) ** 2).sum()
    rows = []
    for mu in MU_GRID:
        Y, up, low = core.solve_horizon(D, C, tiers, x0, mu, tol=1e-11, max_iter=40000)
        X = core.run_rule(inst, "feedback", alpha=1 / (1 + mu))
        ex_alg = core.horizon_cost(X, D, x0, mu) - qstar
        hitY = 0.5 * ((Y - D) ** 2).sum() - qstar
        swY = 0.5 * mu * (np.diff(np.vstack([x0, Y]), axis=0) ** 2).sum()
        sw_star = 0.5 * mu * (np.diff(star, axis=0) ** 2).sum()
        B = np.sqrt(2) * np.abs(np.diff(D, axis=0)).sum(1) + np.abs(np.diff(C))
        rows.append({"block": j, "mu": mu, "excess_feedback": ex_alg,
                     "excess_opt_lower": low - qstar, "excess_opt_upper": up - qstar,
                     "bound_opt": hitY + (1 + mu) * swY,
                     "ratio_bound": 1 + mu,
                     "bound_myopic_comparator": (1 + mu) * sw_star,
                     "bound_data": (1 + mu) * 0.5 * mu * (B ** 2).sum()})
    return rows



def reachability():
    rows = []
    for j, inst in enumerate(core.electricity_blocks()):
        D, C = inst.demand, inst.capacity
        dC = np.maximum(C[:-1] - C[1:], 0)
        dd = np.maximum(D[:-1] - D[1:], 0)
        for r in (0.1, 0.5, 2.0):
            cap_ok = dC <= r
            dem_ok = (dd <= r).all(axis=1)
            rows.append({"block": j, "ramp": r, "transitions": len(dC),
                         "capacity_ok": int(cap_ok.sum()), "request_ok": int(dem_ok.sum()),
                         "both_ok": int((cap_ok & dem_ok).sum())})
    return rows


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    meta = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "workers": a.workers,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with Pool(a.workers) as pool:
        if a.part in ("all", "reach"):
            write("reachability.csv", reachability())
        if a.part in ("all", "clairvoyant"):
            write("clairvoyant.csv", [r for rs in pool.map(_clair_block, range(core.N_BLOCKS)) for r in rs])
        if a.part in ("all", "competitive"):
            write("competitive.csv", [r for rs in pool.map(_competitive_block, range(core.N_BLOCKS)) for r in rs])
        if a.part in ("all", "sensitivity"):
            jobs = [("random_tiers", s, 20261100 + 10 * s + k) for s in (4, 8, 16) for k in range(10)]
            jobs.append(("kw", 8, 0))
            write("sensitivity.csv", [r for rs in pool.map(_sens_job, jobs) for r in rs])
        if a.part in ("all", "rules"):
            write("rules.csv", [r for rs in pool.map(_rules_block, range(core.N_BLOCKS)) for r in rs])
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT / f"environment_{a.part}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
