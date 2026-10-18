# Priority-preserving resource allocation study

A reproducible computational study of sampled projected feedback, exact compiled allocation, delay correction and strict aggregate service priorities. This is a research implementation, not a certified physical controller.

The full local workspace contains a manuscript and editorial materials. Those materials are private and excluded from the experiment-code archive. Code, manifests and numerical results are published separately; see Releases and citation below.

## Reproduce the computational study

Use Python 3.12. On the original workstation, the system Python 3.10 SciPy extension could not load; a clean Python 3.12 environment resolved it. `requirements-lock.txt` records the actual working package versions. With `uv`:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/download_data.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/run_experiments.py
.venv/bin/python scripts/robustness.py
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/verify_outputs.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/review_analysis.py
```

The downloader needs network access and retrieves about 250 MB of original UCI archives. It streams the larger archive in chunks without extracting a full uncompressed copy. Allow at least 1 GB of free space for dependencies, archives and outputs. A recent CPU suffices; no GPU or paid service is needed. Downloads dominate elapsed time on a slow connection. Run timing experiments separately from other computational jobs for clean comparisons:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/run_experiments.py --part runtime
```

The full runner's `all` mode records `environment_all.json`; individual modes record `environment_replay.json`, etc. For verification after `all`, the verifier accepts the corresponding full-run metadata.

## Model and comparisons

Current nonnegative requests and capacity define an instantaneous divisible-resource allocation. The first tier is served before the second. The final current-set projection enforces hard feasibility and aggregate priority at each decision epoch. A gain below one adds an exact quadratic movement regularizer. Only the partially served tier has smoothing freedom.

The exact baseline uses the same sorted-breakpoint projection, so comparisons do not rely on a slow generic solver. Delays in replay are explicitly imposed ages of nominal commands; the correction still sees current requests and capacity. Generic SLSQP timings are supplementary. The two-class bike trace is a singleton-face negative control. Neither dataset is a deployment trial, and the normalized electricity allocations are not physical kW allocations.

The experiment grid includes 54 disjoint historical blocks, 3 capacity levels, 4 command ages and 10 configurations (6,480 records), plus 2,240 synthetic records, gain sensitivity, bounded-error experiments, ramp-feasibility probes, static recovery, random certificates and runtime scaling. Repeated rows for delay-independent methods are not independent experimental replicates. Bootstrap intervals resample paired historical blocks and are descriptive, conditional on the selected traces.

## Outputs

- `data/source_manifest.json`: source URLs, licenses, retrieval times and archive checksums.
- `data/processing_manifest.json`: training scales, selected columns and missing-time conventions.
- `results/replay.csv`, `results/synthetic.csv`: full scenario summaries.
- `results/blocks.csv`: exact historical selections.
- `results/paired_intervals.csv`: paired block differences and percentile intervals.
- `results/robustness.csv`, `results/ramp_diagnostics.csv`: limits of the certified model.
- `results/runtime.csv`, `results/qp_checks.csv`: measured timing and independent QP checks.
- `results/certificates.csv`, `results/recovery.csv`: numerical certificate checks.
- `results/environment_*.json`: runtime versions, seed, elapsed time and source digests.
- `results/verification.json`: final automated output verification.
- `figures/`: deterministic PDF/PNG result plots. The analysis also emits local table fragments.

Post-review exploratory analyses are generated separately by `scripts/review_analysis.py`; they add marginal-tier service-ratio fairness, both relative-effect conventions, tier-specific robustness costs, paired timing diagnostics and the data-only tracking certificate. The `results/review_*.csv` files contain their observations and `review_provenance.json` records sources and inputs. Re-running timing changes measured values.

The local manuscript builds with `latexmk -pdf main.tex`. It requires `biber` and `biblatex-apa`, and depends on the local-only bibliography and generated figure/table outputs. Manuscript files are intentionally absent from the code distribution.

## Privacy and distribution

Run `.venv/bin/python scripts/build_code_archive.py` to build and audit a **local code-only** ZIP from an explicit file allowlist. It includes no paper sources, bibliography, templates, Word documents, PDFs, data, results, or Git history. The archive does not upload itself. The author selected MIT for code and CC BY 4.0 for original results and documentation; see LICENSE and LICENSE-DATA.md. No manuscript-source license is granted.

Never add manuscript sources to any public Git branch, tag, release, archive or attachment. `.gitignore` is a convenience, not a history-sanitization mechanism. Any future publication needs an explicit archive-content audit and authorization. Deleting a source in a later commit would not remove earlier public history.

Run `.venv/bin/python scripts/build_repro_archive.py` to create the separate DOI-deposit package including explicitly listed manifests and numerical outputs. Its complete allowlist and checksum inventory are audited locally. This script does not upload or publish.

## Releases and citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23457662.svg)](https://doi.org/10.5281/zenodo.23457662)

- **v0.2.0** — [GitHub release](https://github.com/MadBezoui/priority-preserving-allocation/releases/tag/v0.2.0) · [Zenodo archival record](https://doi.org/10.5281/zenodo.23457662)

Bezoui, M. (2026). *Priority-preserving projected resource allocation: reproducibility package* (Version 0.2.0). Zenodo. https://doi.org/10.5281/zenodo.23457662

The archived release contains the experimental software, manifests and numerical outputs. All 50 archived files were checked against the released Git tree and the package checksum manifest. Manuscript sources and the paper PDF are excluded. Code is MIT; original results and documentation are CC BY 4.0.

The current main branch includes citation metadata added after DOI assignment. The v0.2.0 tag and Zenodo archive retain the immutable release snapshot. `.zenodo.json` supplies metadata for future connected releases; update version metadata before a new release.
