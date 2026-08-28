# AGENTS.md

Guidance for AI agents (and cold-starting humans) working in this repository.
This is a research repository, not a product: the stakes are *numbers in a
dissertation that must be traceable to a script*, and the conventions below
exist because the failure modes they prevent have all actually happened here.
The 2026-08-01/02 reproducibility pass is documented in `WORKINGDOC.md` — six
defects that each produced plausible output or exited zero.

## What this is

PhD research and coursework centred on fuzzy systems and scalable clustering
(remote: `fundthmcalculus/grad-school`, GPLv3, Python ≥ 3.11). The centre of
gravity is:

1. **The dissertation proposal** — `research/proposal-defense/` (defense:
   December 2026). Chapter prose in `prose/`, action list in `CHECKLIST.md`
   (item IDs like B13/C9 are load-bearing and cited by the chapters; never
   renumber), PDF via `build_pdf.py`.
2. **The reproduction harness** — `reproduce/`, which regenerates every
   numbered table in the proposal and archives it as evidence.

Around it: experiment code, datasets (`data/`), three pinned research
submodules, and four courses of coursework.

**Read in this order before doing anything load-bearing:**

1. `README.md` — layout
2. `WORKINGDOC.md` — what was broken, what it surfaced, the traps (§7)
3. `reproduce/PROVENANCE_MAP.md` — every proposal table → generator → output → drift status.
   Its dated sections (e.g. the 2026-08-22 whole-document status) state the world as of
   that run — read the latest section, not the oldest
4. `reproduce/README.md` and `reproduce/tables/README.md` — harness usage and knobs;
   `reproduce/preflight.py` is the upstream-contract guard `run_all_tables.sh` runs before
   every numeric phase — a failure does not abort the run, it stamps the archive
   **NOT CITABLE**
5. Submodule docs when touching them: `tribble-cluster/CLAUDE.md`,
   `tribble-fis/README.md` (+ its `*_GUIDE.md` files), `tribble-opt/README.md`

## Repository map

| Path                                                    | What it is                                                                                                                                                         |
|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `reproduce/`                                            | Table generators, provenance map, labelled run archives, WIP orchestrator (`manifest.py` + `run.py`)                                                               |
| `reproduce/tables/`                                     | One script per proposal table; writes `reproduce/outputs/<table>.{md,csv}`                                                                                         |
| `reproduce/outputs/`                                    | Loose `.md/.csv/.log` outputs are **gitignored**; labelled archives under `outputs/<label>/` are **tracked** — they are the evidence later diffs are taken against |
| `reproduce/preflight.py`                                 | Upstream-contract guard (W1-SCALE, VAT-MATRIXFREE, SCALER-ALIAS, MODEL-NAMES, DATASETS, INSTALL-FRESH): checks the properties the harness silently assumes from upstream. `INSTALL-FRESH` fails when an environment's installed build no longer matches the submodule source — a pin bump that didn't take effect |
| `reproduce/dataset_specs.yaml`                           | Dataset dimensions injected into the proposal at build time; an unreadable spec is FATAL for `build_pdf.py` (the reader is a declared root dependency precisely so the documented install works) |
| `fis-tsp-strategy/`                                      | FIS-vs-TSP strategy work; `test_invariants.py` reads `.tsp` instances from `ClusteringExperiments/tsplib/` (fetched by `download.py`, not vendored) |
| `tools/` `reviews/`                                       | `ccshim` — the C toolchain bootstrap for MSVC-less Windows hosts (CHECKLIST B15); dated review passes |
| `research/proposal-defense/`                            | Proposal chapters, `CHECKLIST.md`, `references.bib`, `build_pdf.py`                                                                                                |
| `tribble-fis/`                                          | **Pinned submodule** — the fuzzy-systems library (`tribblefis`; MoG FIS, ANFIS, fuzzy tree/HME, optional Cython + GPU kernels)                                     |
| `tribble-cluster/`                                      | **Pinned submodule** — VAT/pVAT + FCM + Lin-Kernighan (`tribbleclustering`; numba + optional Cython/OpenMP)                                                        |
| `tribble-opt/`                                          | **Pinned submodule** — optimizers (ACO, GA, …)                                                                                                                     |
| `ClusteringExperiments/`                                | VAT/pVAT TSP experiments (moved out of `tribble-cluster` by grad-school #26; now sibling modules)                                                                  |
| `FuzzySystemsExperiments/`                              | Per-dataset FIS scripts (Concrete, PhiUSIIL, turbine, WEC, BETH, CMAPSS RUL, …)                                                                                    |
| `gated-minimax-selection/`                              | Chapter 5: iVAT/minimax membership functions, NERFCM beta-spread, multi-scale selection; `run_all.py` regenerates `outputs/results.json` + figures deterministically from seed. The JSON is **gitignored** — a regenerated artifact of record, not a tracked one, and it is written *after* every figure |
| `experiments/`                                          | `fis-acceleration/`, `fis-to-neural-net/` (TSK↔ReLU equivalence), `nn-cmapss/` — each has its own README + results                                                 |
| `ode_kernels/`                                          | Cython embedded Runge–Kutta integrators (`ode12`…`ode78`, `odeexp`)                                                                                                |
| `data/`                                                 | Datasets. `data/.gitignore` records what is fetched instead of vendored (e.g. PhiUSIIL 57 MB, recoverable from `tribble-fis` git history)                          |
| `AEEM6022/` `AEEM6097/` `CS6101/` `AnalyticalDynamics/` | Coursework — graded past submissions, not active code                                                                                                              |
| `papers/` `presentations/` `notes/`                     | Supporting material, slide decks, administrative records                                                                                                           |
| `WORKINGDOC.md`                                         | Working doc of the reproducibility pass                                                                                                                            |

## Environment

- [uv](https://docs.astral.sh/uv/) is the assumed tool. The **root project is
  not an importable package** (`pyproject.toml` sets `bypass-selection`);
  `uv sync` only installs the root `[project.dependencies]`.
- Research code lives in the submodules, each an independent uv project with
  its own `pyproject.toml` + `uv.lock`. Run table generators in the right
  submodule environment:

  ```bash
  # from the repo root
  uv run --project tribble-fis     python reproduce/tables/table_4_1_mog_baselines.py
  uv run --project tribble-fis     python reproduce/tables/table_6_1_model_family.py
  uv run --project tribble-cluster python reproduce/tables/table_3_1_pvat_scaling.py
  uv run --project tribble-cluster --with scipy \
      python reproduce/experiments/run_cluster_experiment.py --all   # Chapter 3 figures
  ```

- The Chapter 5 driver is self-contained — `cd gated-minimax-selection &&
  python run_all.py` (it writes `./outputs/results.json` + figures, CWD-relative)
  regenerates everything deterministically from seed. Note that `results.json`
  is written *after* every figure, so a crash in the figure phase discards the
  whole numeric phase — check the JSON was actually (re)written, not just the exit code.
- Proposal PDF: `.venv/bin/python research/proposal-defense/build_pdf.py`
  (pandoc + a real TeX engine; WeasyPrint/MathML fallback documented in the script).
  The environment must be current — a stale `.venv` that predates a declared
  dependency dies with `[fatal] could not read reproduce/dataset_specs.yaml`
  (`uv sync` fixes it). CI's `build_pdf.py --validate-only` runs the
  assembly-time checks without a TeX engine and is **not read-only**: it
  copies figures into `research/proposal-defense/prose/fig/` and can dirty
  tracked files — check `git status` after running it.
- Style: `make format` (black + flake8 + mypy; the pinned submodules are
  **excluded** — they carry their own CI and formatting state, and a newer
  local black rewriting them would violate non-negotiable 1. CI pins
  `black==26.5.1` deliberately (an unpinned formatter disagreed with a local
  one on 2026-08-22 — see the workflow comment); match that pin when
  formatting locally.

## Non-negotiables

These are the rules the harness's history was built on. Breaking any of them
recreates a documented past defect.

1. **Never write inside a submodule.** Reproducing a proposal result must keep
   `git -C tribble-* status` clean. Regenerated figures are redirected to
   `reproduce/outputs/figures/` (override: `REPRO_FIG_DIR`); tables to
   `reproduce/outputs/` (override: `REPRO_OUTPUT_DIR`, which exists precisely so
   repeated runs of one generator don't overwrite the canonical output).
2. **Ten seeds is the protocol, not a knob** (`common.SEEDS`, default `0..9`).
   Every reported number is mean ± std across the full set. `REPRO_SEEDS` is
   for smoke runs only — the chapters were once transcribed from 3-seed runs,
   and moving to ten retracted a published conclusion, refuted a hypothesis,
   and exposed a model that fails one seed in ten. `--fast` archives are
   stamped **NOT CITABLE** in their `PROVENANCE.txt`, with the seed set
   recorded *per table*, so a thin cell can never be mistaken for a full one.
   Do not quote a narrowed run in the proposal. Known documented exception:
   the Chapter 5 driver (`gated-minimax-selection/run_all.py`) runs seeds
   `0..4`; its tables carry footers that report the driver's *actual* seed set
   (the generator prints the seeds the driver recorded, never
   `common.SEEDS` — a ten-seed stamp on a five-seed run was a past defect).
   Whether the driver moves to ten seeds or the exception gets its own
   justification is an open methods decision — see
   `reviews/2026-08-28-agents-and-provenance-review.md`.
3. **Read the provenance, not the exit status.** Before quoting any number,
   check `reproduce/PROVENANCE_MAP.md` (status vocabulary: reproduced /
   drifted / stale / cited / traced / ungenerated). Update it when a generator
   changes. `reproduce/check_prose.py` diffs prose against an archived run;
   `reproduce/compare_runs.py` diffs two archives cell-by-cell (it reports
   every table, including the ones that didn't move — "unchanged" is a
   reported result, not an absence of one).
4. **Datasets resolve under `data/`** (`GRAD_SCHOOL_DATA` overrides the root).
   Never silently substitute a dataset — two datasets were once quietly
   swapped (a rounded UCI fetch, a different-feature-set fallback) and it took
   a whole investigation to find. Any fallback must *announce* that its
   results are not comparable. (Concrete can be rebuilt from
   `AEEM6097/project-data/*.xls`; PhiUSIIL is recoverable from
   `tribble-fis` git history — see `data/.gitignore`.)
5. **Submodule pin bumps are an event, not a git command.** After moving a
   pin, reinstall the submodule environment, then run the affected table
   generator *before and after* and diff the CSVs. The established rule: a
   Gaussian accuracy number that shifts means the bump reached the Type-1
   path and needs its own explanation; wall-clock differences alone are
   expected. Diff **every column** of the affected tables, not the columns
   that are easiest to eyeball — the B13 bump check verified three R² values
   and concluded "byte-identical" while the two classification-accuracy
   columns beside them had already collapsed inside the same bump window
   (CHECKLIST B14; the lesson: *repetition is not the same thing as
   coverage*). Reinstall and re-verify *every environment the harness imports
   from*, not just the submodule's own: `tribble-fis/uv.lock` pins
   `tribble-clustering` at a git SHA, so the `tribble-fis` environment can run
   a `tribbleclustering` that is older than the grad-school `tribble-cluster`
   submodule pin (found 2026-08-28: lock at `635ed6e`, pin at `1dcf331` — the
   FCM zero-distance crisp-membership fix was missing from the environment the
   Chapter 3 FCM rows run in, while the archive would have recorded the new
   SHA). `reproduce/preflight.py`'s `INSTALL-FRESH` check exists to catch
   exactly this; keep the lock and the pin in step. Record the check (see the
   pin-bump pattern in `reproduce/README.md` and CHECKLIST items B13/B14).
6. **Hardware-gated tables skip, they don't fail.** Table 3.4's GPU rows need
   a CUDA host; without one they are marked and skipped cleanly.
7. **Generated figures are gitignored — regenerate, don't commit.** Only run
   archives and things genuinely worth committing are tracked (force-add with
   `git add -f` when an output is worth keeping, as `outputs/nn-cmapss/` is).
8. **Never commit data files over 5 MB.** Datasets that size are fetched,
   or recoverable from a tracked source — `data/.gitignore` records how to
   restore each one (e.g. the 57 MB PhiUSIIL file is recovered from
   `tribble-fis` git history, with the command written in the ignore file).
   The largest tracked dataset is `data/WEC_Sydney_100.csv` at ~4.6 MB. If a
   dataset belongs under `data/` but is too big to commit, add it to
   `data/.gitignore` with a recovery note instead of `git add`ing it.

## Environment-variable quick reference (table knobs)

Table knobs are documented in `reproduce/tables/README.md` (`REPRO_SEEDS`,
`REPRO_N_GRID`, `REPRO_NAIVE_CAP`, `REPRO_THETA_SWEEP`, `REPRO_NORM_FAMILIES`,
`REPRO_PHIUSIIL_N`); `REPRO_FAST_SEEDS` / `REPRO_OUTPUT_DIR` / `REPRO_FIG_DIR` in
`reproduce/README.md`; `GRAD_SCHOOL_DATA` in the proposal's `DATASETS.md`. Summary:

| Variable | Meaning |
|---|---|
| `REPRO_SEEDS` | Narrow the seed set — smoke runs only, never cite |
| `REPRO_FAST_SEEDS` | Seed set used by `run_all_tables.sh --fast` (default `0,1,2`) |
| `REPRO_THETA_SWEEP` | θ values for the open-set operating curve. **A list, not a flag.** `=1` is a valid one-row sweep where every cell is legitimately zero — a mis-set knob that looks like a null result |
| `REPRO_N_GRID` | N values for Table 3.1 (default `256..4096`) |
| `REPRO_NAIVE_CAP` | Largest N at which cubic classical VAT is timed |
| `REPRO_NORM_FAMILIES` | Which fuzzy operator families the norm/conorm matrix sweeps |
| `REPRO_PHIUSIIL_N` | Sample cap for PhiUSIIL in the norm/conorm matrix |
| `REPRO_OUTPUT_DIR` | Redirect a generator's output (protects the canonical path) |
| `REPRO_FIG_DIR` | Redirect regenerated figures (default `reproduce/outputs/figures/`) |
| `GRAD_SCHOOL_DATA` | Override the dataset root (default `data/`) |

Running everything:

```bash
reproduce/run_all_tables.sh my-label            # all tables; ~30–40 min on the run-of-record host (i9-14900HX), slower elsewhere
reproduce/run_all_tables.sh --fast smoke-check  # minutes; stamped NOT CITABLE
reproduce/run_all_tables.sh --archive-only my-label   # snapshot current outputs, run nothing
```

Do not edit `run_all_tables.sh` while it is running (bash reads it
incrementally; one mid-run edit killed the archive step of a 13-green run —
that is what `--archive-only` recovers).

## Traps (from `WORKINGDOC.md` §7 — read before debugging)

- **Silence is not success.** Every past defect produced plausible output or
  exited zero: a submodule on the wrong commit, a driver that crashed *after*
  computing but *before* writing results (fixed in `gated-minimax-selection`
  — `results.json` is written after every figure), imports broken by a
  directory move, a provenance file misstating its own seed count. Check
  provenance, not exit codes.
- **A conclusion can be reproducible and still wrong.** Determinism is not
  evidence; the sample (seed count) is.
- **Attribute changes to one variable at a time.** One past sweep changed a
  solver, a dataset, and a feature set simultaneously; the tell was that
  sklearn models — which no solver can touch — had moved. When a number
  moves, find something that *should not* have moved and check it.
- **Don't size an optimisation from a profiler.** cProfile charges per-call
  overhead and once inflated a measured speedup from 9.8% to a published 19%.
  Profiles find hotspots; wall clocks size them.
- **Check which repository you are in.** A `git commit --amend` was once run
  in the wrong repo and rewrote an unrelated commit's message.
- **Control experiments need a manipulation check.** `reproduce/experiments/run_note12_threading.py`
  sets the pattern: register expected pass/fail outcomes in the script's
  docstring *before* running, and print evidence that the variable you claim
  to have varied actually changed something — an invariance result is
  worthless if the knob never bit.
- **`fis-tsp-strategy`'s test suite failing locally is a missing dataset, not
  a regression.** `test_invariants.py` reads `.tsp` instance files out of
  `ClusteringExperiments/tsplib/`, which are fetched on demand (see that
  directory's `download.py`), not vendored. A `FileNotFoundError` on a
  `berlin52.tsp`-style path means fetch the instances first, not that the
  solver code broke.
- **The `# TODO`/`# HACK` markers in `FuzzySystemsExperiments/{iot,turbine,wec-p1,wec}.py`
  are live open research questions, not dead code.** ("Handle 3rd-order
  optimization?", "top-n=3!") Resolving them is a methods decision, not a
  cleanup — a refactoring pass should leave them alone rather than guess at
  the answer or delete the marker.

## Submodule conventions

Pins move over time — check `git submodule status` for the current ones (the
2026-08-28 state: `tribble-fis` `353162c`, `tribble-cluster` `1dcf331`,
`tribble-opt` `4d81121`, advanced 2026-08-25). The pins below are a
*snapshot*, not a guarantee, and so is everything that cites them:
`WORKINGDOC.md`'s state table is the 2026-08-01/02 state, and the run of
record cited by PROVENANCE_MAP's latest whole-document section
(`full-2026-08-22`) ran at the *previous* `tribble-fis` pin (`141596e`) — its
numbers are comparable only to that pin (CHECKLIST B14/D8 record what moved
since, and that both named causes are now inside the current pin).

- **`tribble-fis`** (import `tribblefis`): optional Cython forward-pass kernel
  and optional GPU backend are *not built by default* and fall back silently
  (bit-identical either way). No proposal table uses the
  `trapz_method="fast"` path — all table generators run
  `member_function="gaussian"` — so trapz-fitter changes don't move any
  archived table (two `darwin_*.py` scripts do use it; they are not wired
  into `reproduce/`).
- **`tribble-cluster`** (import `tribbleclustering`): Cython extensions are
  optional at runtime with pure-Python/numba fallbacks; compiled and pure
  paths must stay behaviorally equivalent, and f32/f64 fused variants must be
  mirrored on any kernel change. Tests guard on `CYTHON_AVAILABLE`. Full
  guidance in `tribble-cluster/CLAUDE.md` (numba serial-dependency caveats,
  `inplace=` memory semantics, CI-fast mode, release via `v*` tags).
- **`tribble-opt`**: optimizers; performance reports at the submodule root.
- Each submodule has its own CI and its own `uv sync`/quality gates — run
  them *inside* the submodule, and keep its working tree clean when working
  from this repo.

## CI and style

- PRs to `main` (`.github/workflows/style-check.yml`): **black is enforced and
  pinned to `26.5.1`** (the pin is deliberate — see the workflow comment);
  **flake8 and mypy are informational** (`continue-on-error`). The `test` job
  does `pip install -e ".[dev]"` and runs pytest on `research/least_action`
  (skipped if absent), **and** runs the proposal's assembly-time checks
  (`build_pdf.py --validate-only` — dataset-spec substitution, cross-references,
  the section registry, the bibliography; no TeX engine needed), **and** runs
  the `FuzzySystemsExperiments` / `experiments` / `gated-minimax-selection`
  pytest suites inside a `uv sync --project tribble-fis` environment — a plain
  `pip install ./tribble-fis` would fetch the SIGILL-prone PyPI
  tribble-clustering wheel instead of the git-sourced override recorded in
  the lock (tribble-fis#124). Python 3.11.
- Known CI gaps (issue #105): `reproduce/test_dataset_loaders.py` is a
  standalone script (calls `sys.exit()` at import time), not a pytest suite;
  `ode_kernels/tests` needs a Cython build step not yet wired into CI.
- `ode_kernels` build: `python ode_kernels/setup.py build_ext --inplace`
  (artifacts are gitignored).

## Coursework

`AEEM6022/` (UAV/RL/TSP), `AEEM6097/` (fuzzy systems), `CS6101/` (ML, graded
as notebooks), `AnalyticalDynamics/` (ODEs/chaos) are graded past
submissions. Don't refactor them as if they were production code, and don't
cite their numbers in the proposal without a generator. (`AEEM6097/project-data/`
is an exception: it feeds the Concrete dataset fallback.)
