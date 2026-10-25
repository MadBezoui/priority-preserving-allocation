"""Weighted allocation, laminar capacities, reachable faces and horizon bounds.

These extensions leave the historical implementation and results untouched.
All solver outputs are floating-point solutions with explicit residual checks.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import linprog, minimize
from scipy import sparse
import osqp
from .allocation import priority_project, tier_budgets


def weighted_capped_simplex(v, upper, budget, weights):
    """Weighted quadratic projection by sorted multiplier breakpoints."""
    v, upper, w = (np.asarray(a, dtype=float) for a in (v, upper, weights))
    if v.ndim != 1 or v.shape != upper.shape or v.shape != w.shape:
        raise ValueError("equal-length vectors required")
    if not all(np.isfinite(a).all() for a in (v, upper, w)):
        raise ValueError("finite vectors required")
    if np.any(upper < 0) or np.any(w <= 0):
        raise ValueError("nonnegative bounds and positive weights required")
    total = float(upper.sum())
    if not np.isfinite(budget) or budget < -1e-10 or budget > total + 1e-10:
        raise ValueError("budget outside feasible interval")
    if budget <= 0:
        return np.zeros_like(v)
    if budget >= total:
        return upper.copy()
    # A shift of w*v by a scalar changes the objective by a constant on sum x=b.
    a = w * v
    a = a - np.max(a)
    breaks = np.r_[a - w * upper, a]
    changes = np.r_[1. / w, -1. / w]
    order = np.argsort(breaks, kind="stable")
    breaks, changes = breaks[order], changes[order]
    slope = np.cumsum(changes)
    values = total - np.r_[0., np.cumsum(slope[:-1] * np.diff(breaks))]
    hit = np.flatnonzero((values[:-1] >= budget) &
                         (values[1:] <= budget) & (slope[:-1] > 0))
    if len(hit):
        j = hit[0]
        lam = breaks[j] + (values[j] - budget) / slope[j]
    else:
        lam = breaks[np.argmin(np.abs(values - budget))]
    x = np.clip((a - lam) / w, 0., upper)
    if abs(float(x.sum()) - budget) > 1e-8 * max(1., total):
        raise ArithmeticError("weighted projection lost budget accuracy")
    return x


def weighted_priority_project(v, demand, capacity, groups, weights):
    d, v, w = (np.asarray(a, dtype=float) for a in (demand, v, weights))
    out = np.zeros_like(d)
    for group, amount in zip(groups, tier_budgets(d, capacity, groups)):
        out[group] = weighted_capped_simplex(v[group], d[group], amount, w[group])
    return out


def weighted_feedback(previous, demand, capacity, groups, weights, alpha=.25):
    if not 0 < alpha <= 1:
        raise ValueError("gain must lie in (0,1]")
    return weighted_priority_project((1-alpha)*previous + alpha*demand,
                                     demand, capacity, groups, weights)


@dataclass
class ReachableStep:
    allocation: np.ndarray
    hard_feasible_set: bool
    emergency: bool
    capacity_violation: float
    demand_violation: float
    ramp_violation: float
    lower: np.ndarray
    upper: np.ndarray


def reachable_feedback(previous, demand, capacity, groups, ramps, alpha=.25):
    """Apply reachable priorities, with explicit minimum-violation emergency.

    If compulsory lower bounds exceed demand or capacity, demand bounds are
    raised to those lower bounds and capacity to their sum. This preserves
    physical ramp limits and attains the least possible demand/capacity excess.
    Such an output is never labeled hard feasible.
    """
    p, d, r = (np.asarray(a, dtype=float) for a in (previous, demand, ramps))
    if p.shape != d.shape or p.shape != r.shape or p.ndim != 1:
        raise ValueError("equal-length vectors required")
    if not all(np.isfinite(a).all() for a in (p,d,r)) or not np.isfinite(capacity):
        raise ValueError("finite inputs required")
    if np.any(p < 0) or np.any(d < 0) or np.any(r < 0) or capacity < 0:
        raise ValueError("nonnegative inputs required")
    if not 0 < alpha <= 1:
        raise ValueError("gain must lie in (0,1]")
    L = np.maximum(0., p-r)
    original_U = np.minimum(d, p+r)
    feasible = bool(np.all(L <= original_U + 1e-12) and L.sum() <= capacity+1e-12)
    soft_d = np.maximum(d,L)
    soft_c = max(float(capacity), float(L.sum()))
    U = np.minimum(soft_d, p+r)
    candidate = (1-alpha)*p + alpha*d
    x = L + priority_project(candidate-L, np.maximum(U-L,0.),
                             max(0.,soft_c-float(L.sum())), groups)
    return ReachableStep(x, feasible, not feasible,
        float(max(0.,x.sum()-capacity)), float(max(0.,np.max(x-d))),
        float(max(0.,np.max(np.abs(x-p)-r))), L, U)


def laminar_matrix(n, subsets):
    """Check a laminar family and return its incidence matrix."""
    sets = [set(map(int,s)) for s in subsets]
    if any(not s or min(s)<0 or max(s)>=n for s in sets):
        raise ValueError("nonempty valid subsets required")
    if len({tuple(sorted(s)) for s in sets}) != len(sets):
        raise ValueError("duplicate subsets")
    for i,a in enumerate(sets):
        for b in sets[i+1:]:
            if a & b and not (a <= b or b <= a):
                raise ValueError("subsets must form a laminar family")
    A = np.zeros((len(sets),n))
    for j,s in enumerate(sets):
        A[j,list(s)] = 1.
    return A


def hierarchical_face(demand, capacities, subsets, groups):
    """Lexicographic tier totals by sequential independent linear programs."""
    d = np.asarray(demand,dtype=float)
    c = np.asarray(capacities,dtype=float)
    A = laminar_matrix(len(d),subsets)
    if c.shape != (len(A),) or np.any(d<0) or np.any(c<0):
        raise ValueError("nonnegative demands and matching capacities required")
    E, b = [], []
    solution = np.zeros_like(d)
    for group in groups:
        row=np.zeros(len(d)); row[group]=1.
        sol=linprog(-row,A_ub=A,b_ub=c,A_eq=np.array(E) if E else None,
                    b_eq=np.array(b) if b else None,bounds=list(zip(np.zeros(len(d)),d)),
                    method="highs")
        if not sol.success:
            raise ArithmeticError("tier linear program failed: "+sol.message)
        amount=float(row@sol.x)
        E.append(row); b.append(amount); solution=sol.x
    return A, c, np.array(E), np.array(b), solution


def hierarchical_project(v, demand, capacities, subsets, groups, face=None):
    """Euclidean projection using a general QP, with no breakpoint speed claim."""
    v=np.asarray(v,dtype=float); d=np.asarray(demand,dtype=float)
    A,c,E,b,start=hierarchical_face(d,capacities,subsets,groups) if face is None else face
    n=len(d)
    matrix=sparse.csc_matrix(np.vstack([np.eye(n),A,E]))
    solver=osqp.OSQP()
    solver.setup(P=sparse.eye(n,format='csc'),q=-v,A=matrix,
        l=np.r_[np.zeros(n),np.full(len(A),-np.inf),b],u=np.r_[d,c,b],
        eps_abs=1e-10,eps_rel=1e-10,max_iter=100000,polishing=True,verbose=False)
    solver.warm_start(x=start)
    sol=solver.solve(raise_error=False)
    if sol.info.status_val != 1:
        raise ArithmeticError('hierarchical QP failed: '+sol.info.status)
    residual=max(0.,float(np.max(A@sol.x-c)),float(np.max(np.abs(E@sol.x-b))),
                 float(np.max(-sol.x)),float(np.max(sol.x-d)))
    if residual>1e-7 or sol.info.dual_res>1e-7:
        raise ArithmeticError(f"hierarchical QP inaccurate, residual={residual}")
    return sol.x, {"success":True,"iterations":int(sol.info.iter),"residual":residual,
                   "dual_residual":float(sol.info.dual_res)}


def horizon_cost(x, demand, initial, mu):
    diff=np.diff(np.vstack([initial,x]),axis=0)
    return float(.5*np.sum((x-demand)**2)+.5*mu*np.sum(diff**2))


def horizon_gradient(x, demand, initial, mu):
    diff=np.diff(np.vstack([initial,x]),axis=0)
    grad=x-demand+mu*diff
    grad[:-1]-=mu*diff[1:]
    return grad


def horizon_project(v, demand, capacity, groups):
    return np.array([priority_project(z,d,c,groups) for z,d,c in zip(v,demand,capacity)])


def offline_horizon(demand, capacity, groups, initial, mu=3., tolerance=1e-7,
                    max_iterations=2000):
    """Accelerated projected gradient with a strong-convexity gap certificate.

    The certificate is for a feasible projected point, using its projection
    normal and the exact quadratic gradient. Floating-point errors are checked
    separately and the returned upper gap is a numerical certificate.
    """
    d=np.asarray(demand,dtype=float); C=np.asarray(capacity,dtype=float)
    if mu<0: raise ValueError("nonnegative movement penalty required")
    L=1.+4.*mu
    beta=(np.sqrt(L)-1.)/(np.sqrt(L)+1.)
    x=horizon_project(d,d,C,groups); z=x.copy()
    for iteration in range(1,max_iterations+1):
        gz=horizon_gradient(z,d,initial,mu)
        y=horizon_project(z-gz/L,d,C,groups)
        residual=horizon_gradient(y,d,initial,mu)-gz+L*(z-y)
        gap=float(.5*np.sum(residual**2))
        value=horizon_cost(y,d,initial,mu)
        if gap <= tolerance*max(1.,value):
            return y,{"iterations":iteration,"objective":value,"gap_bound":gap,
                      "relative_gap_bound":gap/max(1.,value),"converged":True}
        z=y+beta*(y-x); x=y
    raise ArithmeticError(f"offline solver certificate not reached, bound={gap}")


def online_horizon_certificate(x, demand, initial, mu, local_mu=None, tight=True):
    """Hindsight gap bound using the normal of the local proximal decisions.

    local_mu is the actual penalty used to construct x. mu is the common
    horizon evaluation penalty. The tight form uses the full Hessian inverse.
    """
    local_mu=mu if local_mu is None else local_mu
    diff=np.diff(np.vstack([initial,x]),axis=0)
    residual=(mu-local_mu)*diff
    residual[:-1]-=mu*diff[1:]
    if not tight:
        return float(.5*np.sum(residual**2))
    T=len(x)
    band=np.zeros((3,T)); band[1]=1+2*mu; band[1,-1]=1+mu
    band[0,1:]=-mu; band[2,:-1]=-mu
    solution=solve_banded((1,1),band,residual)
    return float(.5*np.sum(residual*solution))
