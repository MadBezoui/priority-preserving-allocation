"""Independent sparse QP check of the clairvoyant horizon optima (OSQP)."""
import csv
import sys
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abo import comparison as core  # noqa: E402

OUT = ROOT / "results" / "comparisons" / "horizon_qp_checks.csv"


def solve_qp(D, C, tiers, x0, mu):
    T, n = D.shape
    N = T * n
    Dm = sp.eye(N) - sp.eye(N, k=-n)  # stacked first differences, y_0 = x0
    P = (sp.eye(N) + mu * Dm.T @ Dm).tocsc()
    rhs0 = np.zeros(N)
    rhs0[:n] = x0
    q = -D.ravel() - mu * (Dm.T @ rhs0)
    b = core.budgets(D, C, tiers)
    rows = []
    for t in range(T):
        for idx in tiers:
            r = np.zeros(N)
            r[t * n + idx] = 1.0
            rows.append(r)
    A = sp.vstack([sp.eye(N), sp.csr_matrix(np.array(rows))]).tocsc()
    l = np.r_[np.zeros(N), b.ravel()]
    u = np.r_[D.ravel(), b.ravel()]
    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=l, u=u, eps_abs=1e-10, eps_rel=1e-10, polishing=True,
            max_iter=200000, verbose=False)
    res = m.solve()
    return res.x.reshape(T, n), res.info.status


def main():
    rows = []
    blocks = core.electricity_blocks()
    for j in (0, 14, 29):
        inst = blocks[j]
        D, C, tiers = inst.demand, inst.capacity, inst.tiers
        x0 = core.exact_path(core.Instance(D[:1], C[:1], tiers, inst.scale))[0]
        for mu in (1.0, 3.0, 9.0):
            Y, up, low = core.solve_horizon(D, C, tiers, x0, mu, tol=1e-11, max_iter=40000)
            Z, status = solve_qp(D, C, tiers, x0, mu)
            Zp = core.project_face(Z, D, C, tiers)  # remove solver-level infeasibility
            rows.append({"block": j, "mu": mu, "apg_upper": up, "apg_lower": low,
                         "osqp_status": status, "osqp_cost": core.horizon_cost(Zp, D, x0, mu),
                         "max_coordinate_difference": float(np.abs(Y - Zp).max())})
            print(rows[-1], flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
