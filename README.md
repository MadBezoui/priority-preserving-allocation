# Resource allocation under strict service priorities with temporal smoothing and ramp limits

Code, data manifests and numerical results for a computational study of divisible-resource allocation under strict service priorities. A projected feedback rule smooths allocations within the set of priority-optimal allocations. The study measures its tradeoff between movement and shortfall, compares it with external smoothing rules and a clairvoyant optimum, and checks its tracking, horizon and feasibility guarantees numerically.

This is a research implementation, not a certified physical controller.

## Installation

Python 3.12 or later. With [`uv`](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

`requirements-lock.txt` pins the package versions of the main experiments. Each `environment_*.json` file records the Python and package versions, platform and elapsed time of the run that produced its results.

## Reproduction

Run every command from the repository root. Single-threaded linear algebra keeps timings comparable:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest -q
.venv/bin/python scripts/download_data.py          # about 250 MB from the UCI repository
.venv/bin/python scripts/run_experiments.py        # main experiment grid
.venv/bin/python scripts/robustness.py
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/verify_outputs.py
.venv/bin/python scripts/diagnostics.py            # fairness, effect conventions, data certificate
.venv/bin/python scripts/run_extensions.py --part all
.venv/bin/python scripts/verify_extensions.py
.venv/bin/python scripts/run_comparisons.py --part all
.venv/bin/python scripts/check_horizon_qp.py
.venv/bin/python scripts/analyze_comparisons.py
.venv/bin/python scripts/render_extensions.py
.venv/bin/python scripts/render_figures.py         # paper figures and table fragments
```

The downloader caches the original archives in `data/raw/`, refuses any archive whose SHA-256 checksum differs from the recorded one and writes `data/processed/`. Allow 1 GB of disk space. No GPU is needed. `run_comparisons.py` takes about 45 minutes on four cores. Timing experiments should run alone:

```sh
.venv/bin/python scripts/run_experiments.py --part runtime
```

Re-running timing changes the measured values. In `results/runtime.csv`, `generic_qp` denotes SLSQP started from the previous allocation. Re-running other experiments can change solver-level rounding and environment metadata. The verification scripts check the saved outputs without regenerating them.

## Model

Nonnegative requests and a capacity define an allocation at each observation. Recipients belong to ordered tiers. Earlier-tier service takes precedence whenever the constraints allow it. The priority face is the set of allocations that attain these lexicographically optimal tier totals. With one capacity, only the partially served tier has allocation freedom. A tree of local capacities can leave several tiers partially served and requires keeping the local constraints.

The feedback rule projects $(1-\alpha)x_{t-1}+\alpha d_t$ onto the current priority face. A gain below one is an exact quadratic movement penalty with price $\mu=(1-\alpha)/\alpha$. The exact rule ($\alpha=1$) uses the same breakpoint projection, so comparisons do not depend on a slow generic solver.

## Experiments

**Main grid** (`run_experiments.py`). 54 disjoint historical blocks of 336 observations (30 electricity, 24 bike-rental), 3 capacity levels, 4 command ages and 10 rules, giving 6,480 scenario summaries, plus 2,240 synthetic records, gain sensitivity, bounded observation errors, ramp-feasibility probes, static recovery, random certificate checks and runtime scaling. The two-class bike trace has singleton tiers and serves as a sanity check. Repeated rows for rules unaffected by command age are not independent replicates.

**Diagnostics** (`diagnostics.py`). Service-ratio fairness in the marginal tier, two conventions for relative effects, robustness by tier, paired timing batches and random checks of the observed-data tracking certificate.

**Weighted, hierarchical and ramp experiments** (`run_extensions.py`). Parameters, input hash and tolerances are in `results/extensions/protocol.json`.

- Weighted feedback: gains 0.25 and 1, weight exponents 0, 1 and 2 with floor 0.05, 180 trajectories of 336 observations. Shortfall is always reported unweighted.
- Closed-loop ramps: limits 0.1, 0.5 and 2, two gains, all 30 electricity blocks. Each decision uses the previously implemented allocation. An emergency action respects ramps and attains the smallest unavoidable capacity and individual-bound violations. Such actions are recorded as infeasible.
- Hierarchical capacities: ten seeds, two gains, eight recipients, two branches, 128 observations. Rank budgets are checked by linear programming, and OSQP 1.1.3 solves the face projections with tolerances 1e-10, polishing and residual checks. SLSQP is not used here because it reports incompatible constraints on faces that linear programming proves feasible.
- Short-window horizon check: observations 80 to 175 of the first ten blocks at price 3.
- Static recovery: six fixed-data trajectories after capacity falls from 24 to 12 with 32 unit requests.

**Comparison rules** (`run_comparisons.py`, module `abo.comparison`). All 30 electricity blocks at capacity ratio 0.9. Every rule projects onto the current priority face:

- feedback, gains 0.05 to 1.
- exponential smoothing of requests followed by projection.
- regularized online balanced descent, lambda1 in {1, 3, 9} and lambda2 in {0.25, 1, 4}.
- an absolute movement penalty kappa ||x - x_prev||_1, kappa from 0.02 to 0.8.
- receding-horizon control over 4 or 12 observations, with oracle or persistence forecasts.

The clairvoyant horizon optimum is computed on each full block at prices 1, 3 and 9 by accelerated projected gradient with an objective interval of relative width at most 1e-11, and checked against OSQP on three blocks (`check_horizon_qp.py`). Avoidable cost subtracts the unavoidable shortfall of the exact rule. Parameters are selected on 15 blocks and evaluated on the other 15, in both directions. The same script draws ten random protected sets of sizes 4, 8 and 16 (seed 20261100 + 10 s + k for size s and draw k), repeats the comparison in kW, counts transitions that satisfy the recursive-feasibility condition, and checks the avoidable-cost bound in every block.

Bootstrap intervals resample paired blocks (5,000 resamples) and are descriptive, conditional on the selected traces.

## Outputs

- `data/source_manifest.json`, `data/processing_manifest.json`: source URLs, licenses, checksums, training scales, selected columns and missing-time conventions.
- `results/*.csv`, `results/*.npz`: main grid, robustness, runtime, QP checks, certificates and recovery.
- `results/diagnostics_*`: diagnostic tables, summary and provenance.
- `results/extensions/`: weighted, hierarchical, ramp, horizon and recovery outputs with raw trajectories and `verification.json`.
- `results/comparisons/`: comparison rules, clairvoyant optima, robustness, reachability, bound checks, `summary.json` and `table_external.tex`.
- `results/paper_primary.tex`, `results/paper_numbers.tex`: main-grid table and timing values quoted in the paper.
- `results/environment_*.json`, `results/comparisons/environment_*.json`, `results/extensions/environment_*.json`: software versions, platforms and elapsed times.
- `MANIFEST.json`: SHA-256 checksums of every tracked file.
- `figures/`: generated figures (not tracked). `render_figures.py` runs last and overwrites the same-named figures written by `analyze.py` with the versions shown in the paper.

## Licenses

Code is released under the MIT license (`LICENSE`). Original results and documentation are released under CC BY 4.0 (`LICENSE-DATA.md`). The UCI datasets keep their own CC BY 4.0 licenses. See `DATA_SOURCES.md`. They are downloaded by the supplied script, not redistributed.

## Citation

See `CITATION.cff`. All versions: [10.5281/zenodo.23457661](https://doi.org/10.5281/zenodo.23457661).

## Distribution boundary

This package contains experiment code, dataset manifests and original numerical results. Editable manuscript sources, bibliographies, journal templates, editorial documents, figure PDFs and precomputed LaTeX table fragments are excluded. Analysis scripts generate the table fragments locally.
