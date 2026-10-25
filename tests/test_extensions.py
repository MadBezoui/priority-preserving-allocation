"""Independent numerical checks for the October 25 extension algorithms."""
import numpy as np
from scipy.optimize import minimize, linprog, brentq
from abo.allocation import priority_project
from abo.extensions import (weighted_capped_simplex, weighted_priority_project,
    weighted_feedback, reachable_feedback, hierarchical_face, hierarchical_project,
    laminar_matrix, horizon_cost, horizon_gradient, offline_horizon,
    online_horizon_certificate)


def test_weighted_breakpoints_against_independent_qp():
    rng=np.random.default_rng(20261025)
    for _ in range(100):
        u=rng.uniform(.01,3,12);v=rng.normal(0,2,12)
        w=np.exp(rng.uniform(-3,3,12));b=rng.uniform(0,u.sum())
        x=weighted_capped_simplex(v,u,b,w)
        sol=minimize(lambda y:.5*np.sum(w*(y-v)**2),u*b/u.sum(),
            jac=lambda y:w*(y-v),method='SLSQP',bounds=list(zip(np.zeros(12),u)),
            constraints={'type':'eq','fun':lambda y:y.sum()-b,'jac':lambda y:np.ones(12)},
            options={'ftol':1e-11,'maxiter':300})
        assert sol.success
        # An independent scalar root is a coordinate-level reference. SLSQP
        # stops on objective accuracy, which need not imply equal coordinate accuracy.
        lam=brentq(lambda z:np.clip(v-z/w,0,u).sum()-b,
                   np.min(w*(v-u))-1,np.max(w*v)+1,xtol=1e-13)
        np.testing.assert_allclose(x,np.clip(v-lam/w,0,u),atol=1e-10)
        assert abs(.5*np.sum(w*(x-v)**2)-sol.fun)<1e-7
        assert abs(x.sum()-b)<1e-9
    np.testing.assert_array_equal(weighted_capped_simplex([2],[3],1,[4]),[1])
    np.testing.assert_array_equal(weighted_capped_simplex([2],[0],0,[4]),[0])


def test_fixed_weight_contraction_and_proportional_special_case():
    rng=np.random.default_rng(28);g=[np.arange(4),np.arange(4,12)]
    for _ in range(100):
        d=rng.uniform(.1,3,12);w=np.exp(rng.uniform(-2,2,12));c=rng.uniform(0,d.sum())
        p=rng.uniform(0,3,12);star=weighted_priority_project(d,d,c,g,w)
        x=weighted_feedback(p,d,c,g,w,.25)
        norm=lambda z:np.sqrt(np.sum(w*z*z))
        assert norm(x-star)<=.75*norm(p-star)+1e-9
    d=np.array([.5,1,2,3]);g=[np.arange(4)];c=2.
    x=weighted_priority_project(d,d,c,g,1/d)
    np.testing.assert_allclose(x/d,np.full(4,c/d.sum()),atol=1e-12)


def test_reachable_emergency_attains_unavoidable_excess_and_respects_ramps():
    g=[np.arange(2),np.arange(2,4)];p=np.ones(4);r=np.full(4,.1)
    d=np.array([.2,2,2,2]);c=2.
    step=reachable_feedback(p,d,c,g,r)
    assert step.emergency and not step.hard_feasible_set
    np.testing.assert_allclose(step.allocation,np.full(4,.9))
    assert abs(step.capacity_violation-1.6)<1e-12
    assert abs(step.demand_violation-.7)<1e-12
    assert step.ramp_violation<1e-12
    rng=np.random.default_rng(29)
    for _ in range(50):
        d=rng.uniform(.2,3,4);p=rng.uniform(0,2,4);r=rng.uniform(.01,1,4);c=rng.uniform(0,8)
        s=reachable_feedback(p,d,c,g,r)
        assert s.ramp_violation<1e-9
        np.testing.assert_allclose(s.capacity_violation,max(0,s.lower.sum()-c),atol=1e-9)
        np.testing.assert_allclose(s.demand_violation,max(0,np.max(s.lower-d)),atol=1e-9)
        if s.hard_feasible_set:
            assert s.capacity_violation<1e-9 and s.demand_violation<1e-9


def test_recursive_ramp_condition_and_total_ramp_counterexample():
    p=np.array([1.,0.]);d=np.ones(2);r=np.full(2,.1)
    s=reachable_feedback(p,d,.8,[np.arange(2)],r)
    assert .2<=r.sum() and s.emergency
    rng=np.random.default_rng(20261026)
    for _ in range(100):
        n=8;r=rng.uniform(.01,.5,n);d=rng.uniform(.5,2,n)
        C=float(rng.uniform(.1,d.sum()));g=[np.arange(3),np.arange(3,n)]
        p=priority_project(d,d,C,g)
        for t in range(20):
            dn=np.maximum(0,d+rng.uniform(-1,1,n)*r)
            Cn=max(0,C+rng.uniform(-r.min(),r.min()))
            s=reachable_feedback(p,dn,Cn,g,r)
            assert s.hard_feasible_set and s.capacity_violation<1e-9
            assert s.demand_violation<1e-9 and s.ramp_violation<1e-9
            p,d,C=s.allocation,dn,Cn


def test_hierarchy_has_multiple_partial_tiers_and_qp_variational_inequality():
    g=[np.array([0,2]),np.array([1,3])]
    subsets=[np.arange(4),np.array([0,1]),np.array([2,3])]
    d=np.array([3.,3.,1.,3.]);c=np.array([5.,2.,3.])
    face=hierarchical_face(d,c,subsets,g)
    np.testing.assert_allclose(face[3],[3,2])
    assert np.all((face[3]>0)&(face[3]<np.array([d[z].sum() for z in g])))
    rng=np.random.default_rng(30)
    for _ in range(20):
        v=rng.normal(0,3,4);x,info=hierarchical_project(v,d,c,subsets,g,face)
        A,cap,E,b,_=face
        oracle=linprog(x-v,A_ub=A,b_ub=cap,A_eq=E,b_eq=b,
                       bounds=list(zip(np.zeros(4),d)),method='highs')
        assert oracle.success and info['success']
        # Projection optimality checked by an independent linear minimization.
        assert (x-v)@(oracle.x-x)>-2e-6


def test_horizon_gradient_solver_and_gap_certificates_against_dense_qp():
    rng=np.random.default_rng(31);T=5;n=4;g=[np.arange(2),np.arange(2,4)]
    d=rng.uniform(.2,2,(T,n));C=np.array([2.,2.2,1.1,1.2,2.4]);mu=3.
    initial=priority_project(d[0],d[0],C[0],g)
    x,info=offline_horizon(d,C,g,initial,mu,tolerance=1e-11)
    eps=1e-6;point=rng.normal(size=d.shape);j=6
    e=np.zeros(d.size);e[j]=eps;e=e.reshape(d.shape)
    fd=(horizon_cost(point+e,d,initial,mu)-horizon_cost(point-e,d,initial,mu))/(2*eps)
    assert abs(fd-horizon_gradient(point,d,initial,mu).ravel()[j])<1e-7
    rows=[];bud=[]
    for t in range(T):
        star=priority_project(d[t],d[t],C[t],g)
        for group in g:
            row=np.zeros((T,n));row[t,group]=1;rows.append(row.ravel());bud.append(star[group].sum())
    A=np.array(rows);b=np.array(bud)
    sol=minimize(lambda z:horizon_cost(z.reshape(T,n),d,initial,mu),x.ravel(),
        jac=lambda z:horizon_gradient(z.reshape(T,n),d,initial,mu).ravel(),method='SLSQP',
        bounds=list(zip(np.zeros(T*n),d.ravel())),
        constraints={'type':'eq','fun':lambda z:A@z-b,'jac':lambda z:A},
        options={'ftol':1e-11,'maxiter':300})
    assert sol.success
    assert abs(info['objective']-sol.fun)<1e-7
    for alpha in [.25,1.]:
        prev=initial;seq=[]
        for dt,ct in zip(d,C):
            prev=priority_project((1-alpha)*prev+alpha*dt,dt,ct,g);seq.append(prev)
        seq=np.array(seq);gap=horizon_cost(seq,d,initial,mu)-sol.fun
        tight=online_horizon_certificate(seq,d,initial,mu,(1-alpha)/alpha)
        loose=online_horizon_certificate(seq,d,initial,mu,(1-alpha)/alpha,tight=False)
        assert gap<=tight+1e-7 and tight<=loose+1e-10
