# grad-school

Research and coursework for a PhD program centred on fuzzy systems, scalable
clustering, and their applications. The centre of gravity of this repository is
the PhD proposal and the harness that makes every one of its tables reproducible;
around it sit the experiment code, the datasets, and four courses of coursework.

## The research

The proposal (defense: December 2026) spans four chapters of research, all of it
reproducible from this repo:

- **Chapter 3 — pVAT**: scalable structure discovery. A compiled, two-stage
  reorder that drops classical VAT from cubic to quadratic scaling, moving the
  feasible problem size from ~5,000 to >130,000 points.
- **Chapter 4 — MoG FIS**: fast fuzzy-inference-system synthesis — mixture-of-
  Gaussians membership functions with TSK consequents — plus open-set anomaly
  detection and feature scoring.
- **Chapter 5 — topological membership generation**: iVAT/minimax-based
  membership functions (NERFCM), where the proposal's method stays in the
  minimax geometry instead of falling back to Euclidean representations.
- **Chapter 6 — hierarchical / refined FIS**: refined consequent solvers built
  on the Chapter 4 base.

Published work travels alongside it: the NAFIPS papers on VAT + ACO for the TSP
(local PDF copies at the repo root; `papers/NAFIPS/` for supporting material),
the TSK ↔ ReLU neural-network equivalence line (`papers/nn-fis-equivalence/`,
`experiments/fis-to-neural-net/`), and the ODE kernel work (`ode_kernels/`).

## Repository layout

| Path | What it is |
|---|---|
| `research/proposal-defense/` | The proposal: chapter prose, references, PDF build, and the action `CHECKLIST.md` |
| `reproduce/` | The reproduction harness: table generators, provenance map, tracked run archives |
| `WORKINGDOC.md` | Working doc of the reproducibility pass — what was broken, what it surfaced |
| `tribble-fis/`, `tribble-opt/`, `tribble-cluster/` | **Pinned submodules** — the research code (fuzzy systems, optimizers, clustering) |
| `ClusteringExperiments/` | VAT/pVAT TSP experiments, iVAT minimax, GPU VAT, DC-VAT |
| `FuzzySystemsExperiments/` | Per-dataset FIS scripts: Concrete, PhiUSIIL, turbine, WEC, IoT, BETH, CMAPSS RUL and CMAPSS failure-mode diagnosis, … |
| `gated-minimax-selection/` | The Chapter 5 driver (`run_all.py` + seeded `results.json`), NERFCM beta-spread, iVAT/multi-scale membership functions |
| `fis-tsp-strategy/` | FIS strategies for TSP |
| `experiments/` | `fis-acceleration`, `fis-to-neural-net`, `nn-cmapss` (each has its own README and results) |
| `ode_kernels/` | Cython-accelerated embedded Runge–Kutta ODE integrators (`ode12`…`ode78`, `odeexp`) |
| `data/` | Datasets: Concrete, Glass, shuttle, WEC, bikeshare, … (`data/.gitignore` says what is fetched instead) |
| `AEEM6022/`, `AEEM6097/`, `CS6101/`, `AnalyticalDynamics/` | Coursework (see below) |
| `papers/`, `presentations/`, `notes/` | Literature notes, slide decks, administrative records |

## Setup

Python ≥ 3.11. [uv](https://docs.astral.sh/uv/) is the assumed tool.

```bash
git clone --recurse-submodules <this repo>
cd grad-school
uv sync        # installs the root dependencies only
```

The root project is not an importable package (see the `bypass-selection` note
in `pyproject.toml`); `uv sync` just installs `[project.dependencies]`. The
research code lives in the submodules and each is run in its own environment via
`uv run --project <submodule>` — the table generators below follow that pattern.

Style checks (also run by CI on PRs to `main`):

```bash
make format    # black + flake8 + mypy
```

See `.github/STYLE_GUIDE.md` for conventions.

## Reproducing the proposal tables

Read these before quoting any number from the proposal:

- **`reproduce/PROVENANCE_MAP.md`** — every numbered table → generator script →
  output file → drift status (reproduced / drifted / stale / cited / …), with
  the notes that explain every exception.
- **`reproduce/README.md`** — how the harness works, including how to check a
  submodule pin bump without moving any table.
- **`WORKINGDOC.md`** — the 2026-08-01/02 pass that made the whole thing
  reproducible: six defects that had all been producing plausible output, nine
  upstream fixes, and two conclusions that had to be retracted.

Run everything, archived under a label with the submodule SHAs and per-table
seed sets recorded alongside it:

```bash
reproduce/run_all_tables.sh my-label            # all tables, ~30–40 min
reproduce/run_all_tables.sh --fast smoke-check  # minutes; reduced seeds, stamped NOT CITABLE
```

Run a single table — fuzzy tables under `tribble-fis`, pVAT under
`tribble-cluster`:

```bash
uv run --project tribble-fis     python reproduce/tables/table_4_1_mog_baselines.py
uv run --project tribble-cluster python reproduce/tables/table_3_1_pvat_scaling.py
```

Conventions that matter:

- **Ten seeds is the protocol, not a knob** (`common.SEEDS`, override
  `REPRO_SEEDS` only for smoke runs). The chapters were once transcribed from
  3-seed runs; re-quoting at ten retracted a crossover, refuted a skew
  hypothesis, and exposed a model that fails one seed in ten.
- Outputs land in `reproduce/outputs/` as `.md` + `.csv`. Labelled run archives
  under `outputs/<label>/` are tracked — they are the evidence a later diff is
  taken against.
- Nothing is ever written inside a submodule; regenerated figures are
  redirected to `reproduce/outputs/figures/`.
- Datasets resolve under `data/` (`GRAD_SCHOOL_DATA` overrides the root).
- Hardware-gated tables (Table 3.4's GPU rows) need a CUDA host; without one
  they are marked and skipped cleanly rather than failing.

Useful knobs, documented in `reproduce/tables/README.md`: `REPRO_FAST_SEEDS`,
`REPRO_THETA_SWEEP` (a comma-separated θ **list**, not a flag — `=1` is a valid
one-row sweep where every cell is legitimately zero), `REPRO_N_GRID`,
`REPRO_NAIVE_CAP`, `REPRO_OUTPUT_DIR`, `REPRO_FIG_DIR`.

## Coursework

- **AEEM6022** — UAV / RL / TSP: projects and homework on convergence and routing.
- **AEEM6097** — fuzzy systems: ACO and GA solvers, iVAT-TSP, VAT,
  midterm and final projects.
- **CS6101** — machine learning, graded as Jupyter notebooks.
- **AnalyticalDynamics** — ODEs and chaos; double-pendulum and Atwood-machine
  simulations, including memory-augmented trajectory prediction.

## License

GPLv3 — see `LICENSE`.
