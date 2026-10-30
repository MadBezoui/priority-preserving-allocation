import numpy as np
import osqp
import scipy.sparse as sp

from abo import comparison as cmp
from abo.allocation import feedback, priority_project


def _tiers(n, k):
    idx = np.arange(n)
    return [idx[:k], idx[k:]]


def test_bisection_projection_matches_breakpoint_solver():
    rng = np.random.default_rng(20261030)
    for _ in range(200):
        n = int(rng.integers(2, 20))
        tiers = _tiers(n, int(rng.integers(1, n)))
        d = rng.uniform(0, 2, n)
        d[rng.random(n) < 0.1] = 0
        C = rng.uniform(0, 1.2 * d.sum() + 0.1)
        v = rng.normal(0.5, 1.5, n)
        x = cmp.project_face(v, d, C, tiers)[0]
        y = priority_project(v, d, C, tiers)
        np.testing.assert_allclose(x, y, atol=1e-10)


def test_feedback_rule_matches_package_update():
    rng = np.random.default_rng(1)
    n, T = 12, 40
    tiers = _tiers(n, 4)
    D = rng.uniform(0, 2, (T, n))
    C = np.where(np.arange(T) < 20, 10.0, 5.0)
    inst = cmp.Instance(D, C, tiers, 12.0)
    X = cmp.run_rule(inst, "feedback", alpha=0.25)
    prev = cmp.exact_path(inst)[0]
    for t in range(T):
        prev = feedback(prev, D[t], C[t], tiers, alpha=0.25)
        np.testing.assert_allclose(X[t], prev, atol=1e-9)


def test_absolute_penalty_matches_independent_qp():
    rng = np.random.default_rng(2)
    for _ in range(30):
        n = 10
        tiers = _tiers(n, 3)
        d = rng.uniform(0, 2, n)
        C = rng.uniform(0.2, 1.1) * d.sum()
        p = priority_project(rng.uniform(0, 2, n), d, C, tiers)
        kappa = rng.uniform(0.01, 1.0)
        x = cmp.project_face_l1(d, p, kappa, d, C, tiers)[0]
        b = cmp.budgets(d, C, tiers)[0]
        # variables (x, s): min 0.5||x-d||^2 + kappa*sum(s), s >= |x-p|
        P = sp.block_diag([sp.eye(n), sp.csc_matrix((n, n))]).tocsc()
        q = np.r_[-d, kappa * np.ones(n)]
        rows = [sp.hstack([sp.eye(n), sp.csc_matrix((n, n))]),
                sp.hstack([sp.eye(n), -sp.eye(n)]),
                sp.hstack([-sp.eye(n), -sp.eye(n)])]
        T = np.zeros((len(tiers), 2 * n))
        for k, idx in enumerate(tiers):
            T[k, idx] = 1.0
        A = sp.vstack(rows + [sp.csc_matrix(T)]).tocsc()
        lo = np.r_[np.zeros(n), -np.inf * np.ones(2 * n), b]
        hi = np.r_[d, p, -p, b]
        m = osqp.OSQP()
        m.setup(P=P, q=q, A=A, l=lo, u=hi, eps_abs=1e-11, eps_rel=1e-11,
                polishing=True, max_iter=100000, verbose=False)
        z = m.solve().x[:n]
        obj = lambda y: 0.5 * np.sum((y - d) ** 2) + kappa * np.abs(y - p).sum()
        assert obj(x) <= obj(z) + 1e-7


def test_avoidable_cost_bound_on_random_sequences():
    rng = np.random.default_rng(3)
    n, T = 8, 30
    tiers = _tiers(n, 3)
    for mu in (1.0, 3.0, 9.0):
        D = rng.uniform(0, 2, (T, n))
        C = rng.uniform(4, 10, T)
        inst = cmp.Instance(D, C, tiers, 8.0)
        star = cmp.exact_path(inst)
        x0 = star[0]
        X = cmp.run_rule(inst, "feedback", alpha=1 / (1 + mu))
        Y, up, low = cmp.solve_horizon(D, C, tiers, x0, mu, tol=1e-12, max_iter=60000)
        qstar = 0.5 * ((star - D) ** 2).sum()
        excess_alg = cmp.horizon_cost(X, D, x0, mu) - qstar
        hit = 0.5 * ((Y - D) ** 2).sum() - qstar
        move = 0.5 * mu * (np.diff(np.vstack([x0, Y]), axis=0) ** 2).sum()
        assert excess_alg <= hit + (1 + mu) * move + 1e-7
        assert cmp.horizon_cost(X, D, x0, mu) >= low - 1e-7
