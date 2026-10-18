"""Independent checks for the mathematical additions requested in review."""
import numpy as np
from scipy.optimize import linprog, minimize
from abo.allocation import priority_project, tier_budgets, feedback


def test_observed_increment_bound_and_coordinate_comparative_statics():
    rng=np.random.default_rng(20261019);g=[np.arange(3),np.arange(3,7),np.arange(7,12)]
    for _ in range(500):
        d=rng.uniform(0,4,12);e=rng.uniform(0,4,12);c=rng.uniform(0,48);h=rng.uniform(0,48)
        x=priority_project(d,d,c,g);y=priority_project(e,e,h,g)
        assert np.linalg.norm(x-y)<=np.sqrt(2)*np.abs(d-e).sum()+abs(c-h)+1e-9
        j=rng.integers(12);inc=rng.uniform(0,3);e=d.copy();e[j]+=inc
        y=priority_project(e,e,c,g);delta=y-x
        assert -1e-9<=delta[j]<=inc+1e-9
        assert np.all(np.delete(delta,j)<=1e-9)
        assert np.linalg.norm(delta)<=np.sqrt(2)*inc+1e-9
    # The suggested unit l2 constant fails without a marginal-tier switch.
    g=[np.array([0,1]),np.array([2])];d=np.array([1.,1.,10.]);e=np.array([2.,2.,10.])
    x=priority_project(d,d,5.,g);y=priority_project(e,e,5.,g)
    np.testing.assert_allclose(x,[1,1,3]);np.testing.assert_allclose(y,[2,2,1])
    assert np.linalg.norm(y-x)>np.linalg.norm(e-d)


def test_shifted_ramp_face_against_sequential_lp_and_qp():
    rng=np.random.default_rng(19);g=[np.arange(2),np.arange(2,5),np.arange(5,8)]
    for _ in range(40):
        d=rng.uniform(.2,4,8);p=rng.uniform(0,1,8)*d;r=rng.uniform(.1,1,8)
        L=np.maximum(0,p-r);U=np.minimum(d,p+r);C=L.sum()+rng.random()*(U.sum()-L.sum())
        residual=U-L;budget=tier_budgets(residual,C-L.sum(),g)
        v=rng.normal(size=8);x=L+priority_project(v-L,residual,C-L.sum(),g)
        A=[];b=[]
        for idx,amount in zip(g,budget):
            a=np.zeros(8);a[idx]=1
            sol=linprog(-a,A_ub=np.ones((1,8)),b_ub=[C],A_eq=np.array(A) if A else None,
                b_eq=np.array(b) if b else None,bounds=list(zip(L,U)),method='highs')
            assert sol.success
            np.testing.assert_allclose(sol.x[idx].sum(),L[idx].sum()+amount,atol=1e-8)
            A.append(a);b.append(L[idx].sum()+amount)
        A=np.array(A);b=np.array(b)
        q=minimize(lambda y:.5*np.sum((y-v)**2),x,jac=lambda y:y-v,
            bounds=list(zip(L,U)),constraints={'type':'eq','fun':lambda y:A@y-b,'jac':lambda y:A},
            method='SLSQP',options={'ftol':1e-11,'maxiter':200})
        assert q.success;np.testing.assert_allclose(x,q.x,atol=1e-6)
        assert np.max(np.abs(x-p)-r)<=1e-9


def test_overrelaxed_contraction_and_nonfreezing_zero_gain():
    rng=np.random.default_rng(20);g=[np.arange(4),np.arange(4,10)]
    for _ in range(200):
        d=rng.uniform(0,4,10);c=rng.uniform(0,35);a=rng.uniform(1.001,1.999)
        x=rng.normal(size=10);z=priority_project(d,d,c,g)
        y=priority_project((1-a)*x+a*d,d,c,g)
        assert np.linalg.norm(y-z)<=abs(1-a)*np.linalg.norm(x-z)+1e-9
    d=np.ones(2);p=np.ones(2);g=[np.arange(2)]
    np.testing.assert_allclose(priority_project(p,d,1,g),[.5,.5])


def test_incumbency_and_no_cross_time_demand_order_guarantee():
    g=[np.arange(2)];d=np.array([1.,1.5]);p=np.array([0.,1.])
    x=feedback(p,d,1.,g,.25)
    np.testing.assert_allclose(x,[.0625,.9375])
    assert x[0]>p[0] and x[1]<p[1] and d[0]<d[1]
    np.testing.assert_allclose(priority_project(d,d,1.,g),[.25,.75])
