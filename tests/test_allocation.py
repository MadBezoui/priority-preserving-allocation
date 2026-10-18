import numpy as np
import pytest
from scipy.optimize import minimize, linprog
from abo.allocation import capped_simplex, feedback, priority_project, tier_budgets, hard_project


def test_projection_matches_independent_qp():
    rng = np.random.default_rng(20261018)
    for n in [1, 2, 5, 16, 32]:
        for _ in range(20):
            upper = rng.uniform(0, 3, n)
            upper[rng.random(n) < .2] = 0
            v = rng.normal(0, 3, n)
            budget = rng.random()*upper.sum()
            actual = capped_simplex(v, upper, budget)
            if upper.sum() == 0:
                assert np.all(actual == 0)
                continue
            x0 = budget*upper/upper.sum()
            ref = minimize(lambda x: .5*np.sum((x-v)**2), x0,
                jac=lambda x: x-v, bounds=list(zip(np.zeros(n), upper)),
                constraints={"type":"eq", "fun":lambda x: x.sum()-budget,
                             "jac":lambda x: np.ones(n)},
                method="SLSQP", options={"ftol":1e-11, "maxiter":500})
            assert ref.success, ref.message
            np.testing.assert_allclose(actual, ref.x, atol=2e-6)
            assert abs(actual.sum()-budget) < 1e-9


def test_priority_face_matches_sequential_linear_programs():
    rng = np.random.default_rng(88)
    groups = [np.arange(0,4),np.arange(4,8),np.arange(8,12)]
    for _ in range(50):
        d = rng.uniform(0, 3, 12)
        C = rng.uniform(0, 1.4*d.sum())
        eq, rhs = [], []
        for g, b in zip(groups, tier_budgets(d,C,groups)):
            c = np.zeros(12); c[g] = -1
            ref = linprog(c, A_ub=np.ones((1,12)), b_ub=[C],
                A_eq=np.array(eq) if eq else None, b_eq=rhs if rhs else None,
                bounds=list(zip(np.zeros(12),d)), method="highs")
            assert ref.success
            assert abs(-ref.fun-b) < 1e-8
            eq.append(-c); rhs.append(b)


def test_contraction_tracking_and_jumps():
    rng = np.random.default_rng(7)
    groups = [np.arange(4),np.arange(4,12)]
    x, target_old = np.zeros(12), np.zeros(12)
    for _ in range(500):
        d = rng.uniform(0,4,12); C=rng.uniform(0,24)
        a = rng.uniform(.02,1)
        target = priority_project(d,d,C,groups)
        new = feedback(x,d,C,groups,a)
        lhs = np.linalg.norm(new-target)
        bound = (1-a)*(np.linalg.norm(x-target_old)+np.linalg.norm(target-target_old))
        assert lhs <= bound+1e-9
        assert np.max(new-d) <= 1e-9 and np.min(new)>=-1e-9
        assert new.sum() <= C+1e-9
        x,target_old = new,target


def test_nonexpansiveness_and_degenerate_breakpoints():
    rng=np.random.default_rng(2)
    for _ in range(200):
        u=rng.integers(0,4,20).astype(float); b=rng.random()*u.sum()
        v=rng.integers(-2,3,20).astype(float); w=rng.normal(size=20)
        p=capped_simplex(v,u,b); q=capped_simplex(w,u,b)
        assert np.linalg.norm(p-q)<=np.linalg.norm(v-w)+1e-9
        assert abs(p.sum()-b)<1e-9
    np.testing.assert_array_equal(capped_simplex(np.ones(2),np.zeros(2),0),[0,0])


def test_invalid_gain_and_budget():
    with pytest.raises(ValueError): capped_simplex([1],[1],2)
    with pytest.raises(ValueError): feedback(np.zeros(1),np.ones(1),1,[np.array([0])],0)
    with pytest.raises(ValueError): hard_project(np.ones(1),np.ones(1),np.inf)


def test_large_common_offset_preserves_projection_budget():
    np.testing.assert_allclose(capped_simplex([1e16,1e16],[1,1],1),[.5,.5])
