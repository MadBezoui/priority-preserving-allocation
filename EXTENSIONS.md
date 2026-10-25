# Extension experiments, 25 October 2026

This v0.3.0 package extends the archived v0.2.0 baseline. Existing historical outputs are preserved. New results come from constructed allocation scenarios, not an electricity-network or physical deployment model.

## Fixed protocol

The complete parameter choices, input hash, block selection and numerical tolerance are recorded in `results/extensions/protocol.json`. Electricity requests use the original normalization and priority groups. Each rule starts from the same unweighted exact allocation at the first observation. Closed-loop decisions reuse their own previous implemented allocation. An emergency action respects ramps while attaining the smallest unavoidable positive capacity and individual-bound violations. Such actions are recorded as infeasible for the original hard constraints.

- Weighted feedback uses two gains and exponents 0, 1 and 2 with floor 0.05 on positive weight denominators. The common reporting loss remains unweighted. Each of 180 trajectories has 336 observations.
- Closed-loop ramps use limits 0.1, 0.5 and 2, two gains and all 30 electricity blocks, again 180 trajectories of 336 observations.
- Hierarchy scenarios use ten seeds and two gains, eight recipients, two branches and 128 observations. Rank budgets are checked independently and local capacities remain in the QP.
- The horizon experiment uses observations 80 through 175 of the first ten blocks. A common movement price of 3 evaluates the damped and myopic decisions and a clairvoyant sequence with the same initial state and exogenous faces.
- Six fixed-data trajectories measure recovery after capacity falls from 24 to 12 with 32 constant unit requests.

## Numerical protocol amendment

The initial hierarchy projection using SLSQP returned an incompatible-constraints status on a face independently proved feasible by linear programming. This was not recorded as infeasibility. The solver was replaced by OSQP 1.1.3 without changing the experiment scenarios. It uses absolute and relative tolerances of 1e-10, polishing and explicit residual checks. The failed attempt and its source snapshot remain in the local research log.

For weighted reference checks, SLSQP's objective stopping criterion did not imply equally tight coordinate accuracy. An independent scalar Brent root calculation therefore checks coordinates, while the generic QP checks objective agreement.

The offline accelerated projected-gradient solve uses L=1+4*mu, step 1/L and momentum (sqrt(L)-1)/(sqrt(L)+1). A normal-cone residual gives a lower objective bound, stopped at relative width at most 1e-9. All bounds are evaluated in ordinary floating point. They are not enclosures certified by directed rounding. Independent sparse QPs on two full windows agree to numerical tolerance. One approximate sparse solution falls 1.68e-8 below the saved lower bound with a primal residual of 5.55e-10, illustrating the distinction between numerical agreement and an exact arithmetic certificate.

## Verification and interpretation

`scripts/verify_extensions.py` reconstructs all trajectory metrics and constraint events, replays 12 weighted trajectories and 4,032 ramp decisions using an independent Brent solver, checks 2,560 rank calculations, performs 120 independent LP optimality checks, recomputes 20 horizon cost and certificate rows, solves two full horizon QPs independently and replays all six static recoveries. Its saved output is `results/extensions/verification.json`.

The mean within-window horizon gap is about 0.920% for feedback and 3.759% for the myopic rule. These are means of percentages, not percentages of mean costs. They are descriptive outcomes on ten prescribed windows, not a uniform competitive guarantee. At ramp 0.1 in the fixed-data shock, hard feasibility returns after four updates, while unrestricted priority service returns after seven. Only the first event is bounded by the stated recovery theorem.

Reciprocal-request weights at gain 0.25 increase mean marginal-tier Jain index from 0.6742 to 0.9894 while increasing mean unweighted shortfall from 0.25956 to 0.38036. Equality of service fractions is therefore a different objective from minimizing unweighted quadratic shortfall. Reciprocal-square weights do not generally give proportional service.

Code is MIT. Original results and documentation are CC BY 4.0. Original dataset attribution remains in DATA_SOURCES.md. The version-specific DOI is linked from the GitHub release after Zenodo has completed archiving.
