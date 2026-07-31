# Appendix

## A.1 Supplementary figures

The figures below support the main text but are not needed to follow the argument; they are collected here to keep the chapters readable. *(To be produced at publication quality; see the figure placeholders in the prose chapters and `ACTION_ITEMS.md`.)*

- **A.1.1 VAT / iVAT reordered-dissimilarity galleries** — the reordered images for the NASA shuttle set (58K), the psychiatric-evaluation set (135K), and the synthetic circular-cities construction; the Prim/MST block diagram (`vat_prim_mst_block_diagram_v2.svg`).
- **A.1.2 Selection and multi-scale figures** — the figures `fig1`–`fig11` from `gated-minimax-selection/outputs/`: the synthetic datasets, the minimax-transform heatmaps, the persistence curves, the membership-function plots, the ConiVAT bridge repair, the multi-scale hierarchy, and the selection-method comparison.
- **A.1.3 Fuzzy-model visualizations** — per-feature Gaussian-mixture membership plots; the rendered fuzzy trees (`plot_fuzzy_tree`); the hierarchical-mixture structure; the exported Ruspini triangular partitions.

## A.2 Extended results tables

The main text carries the summary tables (Tables 3.1–3.4, 4.1–4.2, 5.1–5.2, 6.1–6.3); their full, multi-seed versions live here once the repeatability protocol (Goal G4) has been run. All of them regenerate from the harness in `reproduce/tables/`, which emits Markdown and CSV side by side, so the appendix version is the same data at full width rather than a re-transcription.

- **A.2.1** Full adversarial-evaluation ARI grids and the complete stitch-ablation grid (all partitions × sizes).
- **A.2.2** The full selection-method bake-off across all synthetic datasets, and the relational-data results.
- **A.2.3** The broadened fuzzy-model benchmark suite (Concrete, PhiUSIIL, turbine, wave-energy, wine, and the IoT sets) with the baseline methods.

## A.3 The optimization engine (`tribble-opt`)

Per the design decision recorded in Chapter 2, the optimization library is supporting infrastructure rather than a dissertation contribution, and its details live here. It provides the optional *local-polish* stage that sits at the end of the pipeline; the point of *structure before search* is precisely that this engine is not on the critical path. It is nonetheless a substantial piece of software, and several parts are strong enough to stand as their own papers (flagged at the end).

**Scope.** The library covers both continuous and combinatorial optimization behind one interface. The metaheuristics are ant colony optimization, genetic algorithms, particle swarm optimization, and gradient descent, in continuous and combinatorial variants. For the Traveling Salesman Problem it implements nearest-neighbor and convex-hull constructions, 2-opt / 3-opt / Or-opt local search, and a candidate-restricted **Lin–Kernighan** search, with two interchangeable backends — a Numba `@njit` path and a Cython `nogil`/OpenMP path — that produce bit-identical tours. On top of the scalar solvers sits a **quality-diversity** layer: MAP-Elites and CVT-MAP-Elites with an automatic random-projection descriptor and the Iso+LineDD directional variation operator, plus Pareto reporting (NSGA-II / SPEA2 / MOEA/D indicators) as a reporting layer rather than a search driver. The quality-diversity archive shares the same `SolutionDeck` interface as the legacy solvers, so it drops in without rewriting them.

**Performance engineering.** A systematic profiling-and-rewrite pass produced a ranked set of findings; the largest are worth recording because they are the reason the engine is usable at scale.

- **Truncated-normal sampling, ~177× on the primitive.** SciPy was rebuilding a distribution docstring on every sample, consuming roughly 4.8 s of a 7.9 s ant-colony run; replacing it with an inverse-CDF (`ndtri`) sampler removed that cost, giving a ~5–8× end-to-end speedup on the affected runs.
- **Ship the fixed data once, up to ~7.5×.** Re-shipping the problem data to each parallel worker every generation is a per-generation cost, so the speedup grows with run length — about 0.96× at 12 generations, 2.3× at 30, 4.6× at 60, and 7.5× at 100. (The benefit is largest for non-array Python payloads; large NumPy arrays are already memory-mapped by the parallel backend.)
- **JIT the local search, ~370×.** A full 2-opt scan at N = 400 dropped from 479 ms to 1.3 ms under `@njit`, and a full 3-opt pass at N = 500 went from timing out (> 120 s) to 0.63 s. Fixing the hot loop also surfaced and fixed a latent 3-opt out-of-bounds bug.

**Lin–Kernighan quality.** Lin–Kernighan produced the shortest tour at every problem size tested, roughly 6–7% under 2-opt and 9–16% under 3-opt at comparable runtime. The Cython and Numba backends are within ~1.1–1.4× of each other on a single warm tour (the search is branchy), so Cython's real advantages are the absence of JIT warm-up and a batched OpenMP path (64 tours at N = 300: 115.8 ms → 79.7 ms, ~1.45×). The simpler 2-opt/3-opt kernels see a larger ~2.7–3× Cython gain.

**Handoff to the clustering package.** Two report items — replacing the Fuzzy C-Means BFGS step with closed-form alternating updates, and JIT-compiling the iVAT path-max loop — were deliberately deferred here, because the clustering code (FCM, VAT/iVAT) is being split into its own package. That seam is exactly where this engine hands off to the `tribble-cluster` work of Chapter 3.

**Standalone-paper opportunities** (for Dr. Cohen's consideration, not part of the core dissertation): the performance-engineering study on its own; the quality-diversity-over-legacy-solvers layer (CVT-MAP-Elites + Iso+LineDD); and the exact GPU/parallel VAT engine as a systems paper.

## A.4 Reproducibility

- **Code.** Four repositories, submoduled into the `grad-school` working repo: `tribble-cluster` (VAT/iVAT/FCM), `tribble-fis` (the fuzzy models and `tribble-tree`), `tribble-opt` (the optimization engine), and `gated-minimax-selection` (the Chapter 5 membership-generation experiments).
- **The reproduction harness.** `reproduce/` is the single entry point, and the goal is that reproducing a result takes one command rather than archaeology. Each table in the proposal has a generator under `reproduce/tables/` that runs the models over a fixed seed set and writes both Markdown and CSV into `reproduce/outputs/`, reporting mean ± standard deviation. Anything it cannot run — a missing optional baseline, an absent dataset, hardware it does not have — is reported as unavailable and printed with the reason, never silently replaced by an estimate. `reproduce/manifest.py` enumerates every experiment across the four repositories with its command, environment, datasets, and hardware tier; a full orchestrator that walks that manifest is the next step.
- **Drivers.** Beyond the table generators, each original result has a named script — the `gaussian_mixture/*` benchmarks for Chapter 4, `tribble-tree/demo_*.py` for Chapter 6, `gated-minimax-selection/run_all.py` for Chapter 5, and the `experiments/` harnesses for Chapter 3.
- **Environments.** The submodules carry their own locked environments, so the generators are invoked through them (for example, `uv run --project tribble-fis python reproduce/tables/table_6_1_model_family.py`). Dataset preparation is automatic where licensing permits — the Concrete set is built from the spreadsheet in the repo if the CSV is absent.
- **Hardware.** Development ran on a 32-core Intel workstation with 64 GB RAM and a laptop-class RTX 4080 (12 GB, reduced double-precision throughput). Final performance numbers are to be re-taken under the fixed protocol of Goal G4 — pinned clocks and thermals, multiple seeds, error bars, and a datacenter GPU with full double-precision throughput.
- **Commit pins.** The exact commit hashes behind each reported result will be pinned in this section at submission time.

---

*Draft — Appendix prose. A.3 (optimization engine) is written out; A.1/A.2/A.4 are structured to be completed alongside the figures, the repeatability runs, and final commit pins. Source outline in `../chapters/appendix.md`; open items in `../ACTION_ITEMS.md`.*
