# Priority-preserving resource allocation study

A reproducible computational study of sampled projected feedback, exact compiled allocation, delay correction and strict aggregate service priorities. This is a research implementation, not a certified physical controller.

The full local workspace contains a manuscript and editorial materials. Those materials are private and excluded from the experiment-code archive. The baseline code, manifests and results have a separate archived release. Version v0.3.0 adds the extension experiments. See Releases and citation below.

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

Current nonnegative requests and capacity define an instantaneous divisible-resource allocation. The first tier is served before the second. The final current-set projection enforces hard feasibility and aggregate priority at each decision epoch. A gain below one adds an exact quadratic movement regularizer. For the single-resource model, only the partially served tier has smoothing freedom. A tree of local capacities can leave several partially served tiers and requires retaining the local constraints.

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

## Extension experiments, version 0.3.0

The extension protocol and its input checksum are in `results/extensions/protocol.json`. It was specified before these outcomes were generated. The experiments reuse all 30 electricity blocks for weighted feedback and for closed-loop ramp decisions, with 180 trajectories for each experiment. Ten synthetic hierarchy seeds compare two gains over 128 observations. Ten electricity windows compare two online rules to the clairvoyant quadratic optimum over 96 observations. A fixed-data shock separates return to hard feasibility from restoration of unrestricted priority totals.

After preparing the original datasets, run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/run_extensions.py --part all
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/verify_extensions.py
mkdir -p figures
.venv/bin/python scripts/render_extensions.py
```

The recorded full extension run took 548.5 seconds on the original workstation, excluding independent verification. OSQP 1.1.3 handles the hierarchy projections. Raw arrays, epoch-level records and block summaries remain in `results/extensions`. Source and output hashes are bound in its `environment_all.json`. The verifier checks saved outcomes without regenerating the experiment. Run it before replacing any archived outputs if checking this snapshot. Re-running experiments changes environment timing metadata and can change solver-level rounding. See `EXTENSIONS.md` for interpretation and numerical limits.

## Privacy and distribution

Run `.venv/bin/python scripts/build_extension_archive.py` to build and inspect the **v0.3.0 reproducibility ZIP**. An explicit allowlist selects code, attribution, manifests and numerical outputs. Every entry is hashed and checked, including the numeric arrays. This script does not upload or publish. For v0.3.0, use this builder rather than the older builders retained in v0.2.0.

The author selected MIT for code and CC BY 4.0 for original results and documentation. See LICENSE and LICENSE-DATA.md. No manuscript-source license is granted. Never add manuscript sources to any public Git branch, tag, release, archive or attachment. Deleting a source in a later commit would not remove earlier public history. The code ZIP excludes manuscript sources, bibliography, templates, editorial documents, paper and figure PDFs, editable schematics, original dataset archives and Git history.

## Releases and citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23457661.svg)](https://doi.org/10.5281/zenodo.23457661)

- **v0.3.0**: [GitHub release](https://github.com/MadBezoui/priority-preserving-allocation/releases/tag/v0.3.0), with weighted, hierarchy, closed-loop ramp and horizon experiments.

- **v0.2.0** — [GitHub release](https://github.com/MadBezoui/priority-preserving-allocation/releases/tag/v0.2.0) · [Zenodo archival record](https://doi.org/10.5281/zenodo.23457662)

Bezoui, M. (2026). *Priority-preserving projected resource allocation: reproducibility package* (Version 0.2.0). Zenodo. https://doi.org/10.5281/zenodo.23457662

The archived v0.2.0 release contains the experimental software, manifests and numerical outputs. All 50 archived files were checked against the released Git tree and the package checksum manifest. Manuscript sources and the paper PDF are excluded. Code is MIT; original results and documentation are CC BY 4.0.

The version-specific v0.2.0 DOI identifies the baseline results. Version v0.3.0 includes all extensions and is released through the linked GitHub and Zenodo integration. Its version-specific DOI will be linked from the release once Zenodo has completed archiving. The concept DOI identifies the complete version family.
