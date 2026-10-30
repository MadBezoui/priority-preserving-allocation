"""Comparison rules on the single-resource priority face.

NumPy implementation of the priority face, the projected feedback rule and the
external comparison rules: exponential smoothing of requests, regularized
online balanced descent, an absolute movement penalty and receding-horizon
control. Projections use bisection on the tier multiplier, independently of
the breakpoint solver in ``abo.allocation``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BLOCK = 336
N_BLOCKS = 30
DROP = (112, 224)
BISECT = 80


@dataclass
class Instance:
    demand: np.ndarray  # (T, n)
    capacity: np.ndarray  # (T,)
    tiers: list  # list of index arrays, highest priority first
    scale: float  # training mean total s


def load_electricity(units: str = "normalized"):
    data = np.load(ROOT / "data/processed/electricity.npz")
    demand = np.asarray(data["demand"], dtype=float)
    critical = np.asarray(data["critical"], dtype=int)
    scale = float(data["train_total"])
    if units == "kw":
        manifest = json.loads((ROOT / "data/processing_manifest.json").read_text())
        means = np.asarray(manifest["electricity"]["training_means_kw"], dtype=float)
        demand = demand * means
        scale = float(means.sum())
    return demand, critical, scale


def tiers_from_critical(n: int, critical) -> list:
    critical = np.asarray(sorted(set(int(i) for i in critical)))
    ordinary = np.asarray([i for i in range(n) if i not in set(critical.tolist())])
    return [critical, ordinary]


def block_starts(total_rows: int, count: int = N_BLOCKS, length: int = BLOCK):
    return [j * (total_rows - length) // (count - 1) for j in range(count)]


def electricity_blocks(rho: float = 0.9, units: str = "normalized", critical=None):
    demand, crit, scale = load_electricity(units)
    if critical is not None:
        crit = critical
    tiers = tiers_from_critical(demand.shape[1], crit)
    out = []
    for start in block_starts(demand.shape[0]):
        cap = np.full(BLOCK, rho * scale)
        cap[DROP[0]:DROP[1]] *= 0.5
        out.append(Instance(demand[start:start + BLOCK].copy(), cap, tiers, scale))
    return out


# ---------------------------------------------------------------- budgets

def budgets(d: np.ndarray, C: np.ndarray, tiers: list) -> np.ndarray:
    """Tier budgets b_k for rows of d (shape (..., n)) and capacities C."""
    d = np.atleast_2d(d)
    C = np.atleast_1d(np.asarray(C, dtype=float))
    rem = C.copy()
    out = np.zeros((d.shape[0], len(tiers)))
    for k, idx in enumerate(tiers):
        tot = d[:, idx].sum(axis=1)
        out[:, k] = np.minimum(tot, np.maximum(rem, 0.0))
        rem = rem - tot
    return out


def _bisect_rows(fun, lo, hi, target):
    for _ in range(BISECT):
        mid = 0.5 * (lo + hi)
        big = fun(mid) > target
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    return 0.5 * (lo + hi)


def project_face(v: np.ndarray, d: np.ndarray, C, tiers: list) -> np.ndarray:
    """Euclidean projection of rows of v onto the priority faces F(d_t, C_t)."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    d = np.atleast_2d(np.asarray(d, dtype=float))
    b = budgets(d, C, tiers)
    x = np.zeros_like(v)
    for k, idx in enumerate(tiers):
        vv, uu, bb = v[:, idx], d[:, idx], b[:, k]
        full = bb >= uu.sum(axis=1) - 1e-14
        lo = (vv - uu).min(axis=1) - 1.0
        hi = vv.max(axis=1) + 1.0
        lam = _bisect_rows(lambda m: np.clip(vv - m[:, None], 0.0, uu).sum(axis=1), lo, hi, bb)
        xk = np.clip(vv - lam[:, None], 0.0, uu)
        xk[full] = uu[full]
        xk[bb <= 0] = 0.0
        x[:, idx] = xk
    return x


def project_face_l1(a: np.ndarray, p: np.ndarray, kappa: float, d, C, tiers):
    """argmin 0.5||x-a||^2 + kappa ||x-p||_1 over the priority face (one row)."""
    a = np.atleast_2d(a)
    p = np.atleast_2d(p)
    d = np.atleast_2d(d)
    b = budgets(d, C, tiers)
    x = np.zeros_like(a)

    def prox(z, pp):
        return np.where(z > pp + kappa, z - kappa, np.where(z < pp - kappa, z + kappa, pp))

    for k, idx in enumerate(tiers):
        aa, pp, uu, bb = a[:, idx], p[:, idx], d[:, idx], b[:, k]
        span = np.abs(aa).max() + np.abs(pp).max() + uu.max() + kappa + 1.0
        lo = np.full(aa.shape[0], -span)
        hi = np.full(aa.shape[0], span)
        f = lambda m: np.clip(prox(aa - m[:, None], pp), 0.0, uu).sum(axis=1)
        lam = _bisect_rows(f, lo, hi, bb)
        xk = np.clip(prox(aa - lam[:, None], pp), 0.0, uu)
        full = bb >= uu.sum(axis=1) - 1e-14
        xk[full] = uu[full]
        xk[bb <= 0] = 0.0
        x[:, idx] = xk
    return x


def exact_path(inst: Instance) -> np.ndarray:
    return project_face(inst.demand, inst.demand, inst.capacity, inst.tiers)


# ---------------------------------------------------------------- online rules

def run_rule(inst: Instance, rule: str, **par) -> np.ndarray:
    """Return the allocation path (T, n). x_{-1} is the exact allocation at t=0."""
    D, C, tiers = inst.demand, inst.capacity, inst.tiers
    T, n = D.shape
    star = exact_path(inst)
    X = np.zeros((T, n))
    prev = star[0].copy()
    ema = D[0].copy()
    for t in range(T):
        if rule == "exact":
            x = star[t]
        elif rule == "feedback":
            a = par["alpha"]
            x = project_face((1 - a) * prev + a * D[t], D[t], C[t], tiers)[0]
        elif rule == "ema":
            a = par["alpha"]
            ema = (1 - a) * ema + a * D[t]
            x = project_face(ema, D[t], C[t], tiers)[0]
        elif rule == "robd":
            l1, l2 = par["lam1"], par["lam2"]
            v = (D[t] + l1 * prev + l2 * star[t]) / (1 + l1 + l2)
            x = project_face(v, D[t], C[t], tiers)[0]
        elif rule == "l1":
            x = project_face_l1(D[t], prev, par["kappa"], D[t], C[t], tiers)[0]
        elif rule == "mpc":
            x = mpc_step(inst, t, prev, par["mu"], par["window"], par["forecast"])
        else:
            raise ValueError(rule)
        X[t] = x
        prev = x
    return X


# ---------------------------------------------------------------- horizon problem

def horizon_cost(X, D, x0, mu):
    diff = np.diff(np.vstack([x0[None, :], X]), axis=0)
    return 0.5 * ((X - D) ** 2).sum() + 0.5 * mu * (diff ** 2).sum()


def horizon_grad(Y, D, x0, mu):
    Z = np.vstack([x0[None, :], Y])
    diff = np.diff(Z, axis=0)
    g = (Y - D) + mu * diff
    g[:-1] -= mu * diff[1:]
    return g


def solve_horizon(D, C, tiers, x0, mu, Y0=None, tol=1e-10, max_iter=20000):
    """Accelerated projected gradient with an a posteriori objective interval.

    Returns (Y, upper, lower). The lower bound uses the normal-cone residual
    argument of Proposition 6 with H >= I.
    """
    L = 1.0 + 4.0 * mu
    beta = (np.sqrt(L) - 1.0) / (np.sqrt(L) + 1.0)
    Y = project_face(D if Y0 is None else Y0, D, C, tiers)
    Z = Y.copy()
    for it in range(max_iter):
        gZ = horizon_grad(Z, D, x0, mu)
        Yn = project_face(Z - gZ / L, D, C, tiers)
        if it % 25 == 0 or it == max_iter - 1:
            r = horizon_grad(Yn, D, x0, mu) + L * (Z - Yn) - gZ
            up = horizon_cost(Yn, D, x0, mu)
            lowb = up - 0.5 * (r ** 2).sum()
            if up - lowb <= tol * max(1.0, up):
                return Yn, up, lowb
        Z = Yn + beta * (Yn - Y)
        Y = Yn
    r = horizon_grad(Y, D, x0, mu) + L * (Z - Y) - horizon_grad(Z, D, x0, mu)
    up = horizon_cost(Y, D, x0, mu)
    return Y, up, up - 0.5 * (r ** 2).sum()


def mpc_step(inst: Instance, t: int, prev, mu, window, forecast):
    D, C, tiers = inst.demand, inst.capacity, inst.tiers
    T = D.shape[0]
    if forecast == "oracle":
        idx = np.arange(t, min(T, t + window))
        Dh, Ch = D[idx], C[idx]
    elif forecast == "persistence":
        Dh = np.repeat(D[t][None, :], window, axis=0)
        Ch = np.full(window, C[t])
    else:
        raise ValueError(forecast)
    Y, _, _ = solve_horizon(Dh, Ch, tiers, prev, mu, tol=1e-8, max_iter=4000)
    return Y[0]


# ---------------------------------------------------------------- metrics

def metrics(X: np.ndarray, inst: Instance) -> dict:
    D, C, tiers, s = inst.demand, inst.capacity, inst.tiers, inst.scale
    T, n = D.shape
    star0 = exact_path(Instance(D[:1], C[:1], tiers, s))[0]
    prev = np.vstack([star0[None, :], X[:-1]])
    Q = 0.5 * ((X - D) ** 2).sum() / (T * n)
    V = np.abs(X - prev).sum() / (T * s)
    I1 = tiers[0]
    R1 = np.maximum(np.minimum(C, D[:, I1].sum(1)) - X[:, I1].sum(1), 0).sum() / (T * s)
    excess = max(0.0, (X.sum(1) - C).max()) / s
    return {"Q": Q, "V": V, "R1": R1, "excess": excess}


def paired_bootstrap(diff, reps=5000, seed=20261029):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    idx = rng.integers(0, diff.size, size=(reps, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
