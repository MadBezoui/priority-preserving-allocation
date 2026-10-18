"""Euclidean projections and a sampled priority-preserving feedback map.

The breakpoint algorithm is a standard continuous quadratic-knapsack method.
This module makes no claim that the primitive or projected gradient is new.
"""
import numpy as np


def capped_simplex(v, upper, budget):
    """Project v onto {0 <= x <= upper, sum(x) = budget}.

    Exact piecewise-linear multiplier search (up to floating-point rounding),
    O(n log n) work and O(n) storage, with no tolerance-driven iteration loop.
    """
    v, upper = np.asarray(v, dtype=float), np.asarray(upper, dtype=float)
    if v.shape != upper.shape or v.ndim != 1:
        raise ValueError("v and upper must be equal-length vectors")
    if not np.isfinite(v).all() or not np.isfinite(upper).all() or np.any(upper < 0):
        raise ValueError("finite vectors and nonnegative upper bounds required")
    total = float(upper.sum())
    if not np.isfinite(budget) or budget < -1e-10 or budget > total + 1e-10:
        raise ValueError("budget outside feasible interval")
    if budget <= 0:
        return np.zeros_like(v)
    if budget >= total:
        return upper.copy()
    # Projection is invariant to adding a common offset to v on a fixed-sum
    # set. Center before subtraction to avoid losing small bounds at 1e16.
    v = v - np.max(v)
    # At lambda=-infinity all coordinates are at their upper bounds.
    # Crossing v_i-u_i activates a slope -1; crossing v_i deactivates it.
    breaks = np.concatenate((v - upper, v))
    changes = np.concatenate((np.ones(len(v)), -np.ones(len(v))))
    order = np.argsort(breaks, kind="stable")
    breaks, changes = breaks[order], changes[order]
    active = np.cumsum(changes)
    values = total - np.r_[0.0, np.cumsum(active[:-1] * np.diff(breaks))]
    crossings = np.flatnonzero((values[:-1] >= budget) &
                              (values[1:] <= budget) & (active[:-1] > 0))
    if not len(crossings):
        # Rounding at a breakpoint: find the nearest, then enforce KKT form.
        lam = breaks[np.argmin(np.abs(values - budget))]
    else:
        j = crossings[0]
        lam = breaks[j] + (values[j] - budget) / active[j]
    result = np.clip(v - lam, 0.0, upper)
    if abs(float(result.sum()) - budget) > 1e-9 * max(1., total):
        raise ArithmeticError("Projection lost budget accuracy; rescale inputs")
    return result


def tier_budgets(demand, capacity, groups):
    if capacity < 0 or not np.isfinite(capacity):
        raise ValueError("finite nonnegative capacity required")
    remaining = capacity
    budgets = []
    for group in groups:
        total = float(np.sum(demand[group]))
        amount = min(total, max(0.0, remaining))
        budgets.append(amount)
        remaining -= amount
    return budgets


def priority_project(v, demand, capacity, groups):
    """Project onto the lexicographically maximal tier-service face.

    Caller must supply a disjoint partition of all coordinate indices.
    """
    out = np.zeros_like(demand, dtype=float)
    for group, budget in zip(groups, tier_budgets(demand, capacity, groups)):
        out[group] = capped_simplex(v[group], demand[group], budget)
    return out


def hard_project(v, demand, capacity):
    """Nearest nonnegative allocation respecting requests and capacity."""
    if capacity < 0 or not np.isfinite(capacity):
        raise ValueError("finite nonnegative capacity required")
    clipped = np.clip(v, 0, demand)
    if clipped.sum() <= capacity:
        return clipped
    return capped_simplex(v, demand, capacity)


def feedback(previous, demand, capacity, groups, alpha=0.25):
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    return priority_project((1-alpha)*previous + alpha*demand,
                            demand, capacity, groups)
