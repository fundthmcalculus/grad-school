# Appendix

## A.1 Supplementary figures

A distinction first, because it governs what belongs here. Sixteen figures are called out in the main text and stay there — they carry an argument at the point it is made, and several are load-bearing: Figure 1.2 (the pipeline roadmap) orients the whole document, and Figure 5.2 (band discovery on the log-birth spectrum) is the single figure Chapter 5's contribution rests on. What lands in this appendix is the *supplementary* material: the galleries, the per-fold and per-dataset repetitions, and the diagnostic plots that a reader may want to check but should not have to walk through. Of the sixteen, exactly one is produced today — Figure 3.2, the complexity fit, which the reproduction harness generates in PNG for the Markdown and EPS for the LaTeX build. The rest, and everything listed below, are not yet drawn; that is tracked in `ACTION_ITEMS.md`.

- **A.1.1 VAT / iVAT reordered-dissimilarity galleries** — the reordered images for the NASA shuttle set (58K), the psychiatric-evaluation set (135K), and the synthetic circular-cities construction; the Prim/MST block diagram (`vat_prim_mst_block_diagram_v2.svg`).
- **A.1.2 Selection and multi-scale figures** — the figures `fig1`–`fig11` from `gated-minimax-selection/outputs/`: the synthetic datasets, the minimax-transform heatmaps, the persistence curves, the membership-function plots, the ConiVAT bridge repair, the multi-scale hierarchy, and the selection-method comparison.
- **A.1.3 Fuzzy-model visualizations** — per-feature Gaussian-mixture membership plots; the rendered fuzzy trees (`plot_fuzzy_tree`); the hierarchical-mixture structure; the exported Ruspini triangular partitions.

## A.2 Extended results tables

The main text carries twenty-two summary tables (3.1–3.7, 4.1–4.7, 5.1–5.3, 6.1–6.4, 7.1); their full, multi-seed versions live here once the repeatability protocol (Goal G4) has been run. All of them regenerate from the harness in `reproduce/tables/`, which emits Markdown and CSV side by side, so the appendix version is the same data at full width rather than a re-transcription.

- **A.2.1** Full adversarial-evaluation ARI grids and the complete stitch-ablation grid (all partitions × sizes).
- **A.2.2** The full selection-method bake-off across all synthetic datasets, and the relational-data results.
- **A.2.3** The broadened fuzzy-model benchmark suite (Concrete, PhiUSIIL, turbine, wave-energy, wine, and the IoT sets) with the baseline methods.
- **A.2.4** The three-arm reorder study behind Chapter 3 §3.3.1 — classical cubic, stage-one priority queue, stage-two compact active set — across the full grid of $N$ and both precisions, in absolute seconds with per-seed spreads. The main text reports this normalized, because wall-clock is not portable between machines and ratios are; the appendix version is where the seconds live. It also carries the per-$N$ detail behind Table 3.2's exponent fit, including the stage-two plateau above $N \approx 750$ that the fitted exponent averages over. This is also the evidence base for the possible complexity note discussed in §9.3.
- **A.2.5** The output-partitioning study of Goal G5 (uniform vs. quantile vs. pinned-extreme hybrid), including the per-decile and tail-error breakdowns and the bucket-starvation counts that aggregate error hides.

## A.3 The optimization engine (`tribble-opt`)

Per the design decision recorded in Chapter 2, the optimization library is supporting infrastructure rather than a dissertation contribution, and its details live here. It provides the optional *local-polish* stage that sits at the end of the pipeline; the point of *structure before search* is precisely that this engine is not on the critical path. It is nonetheless a substantial piece of software, and several parts are strong enough to stand as their own papers (flagged at the end).

**Scope.** The library covers both continuous and combinatorial optimization behind one interface. The metaheuristics are ant colony optimization, genetic algorithms, particle swarm optimization, and gradient descent, in continuous and combinatorial variants. For the Traveling Salesman Problem it implements nearest-neighbor and convex-hull constructions, 2-opt / 3-opt / Or-opt local search, and a candidate-restricted **Lin–Kernighan** search, with two interchangeable backends — a Numba `@njit` path and a Cython `nogil`/OpenMP path — that produce bit-identical tours. On top of the scalar solvers sits a **quality-diversity** layer: MAP-Elites and CVT-MAP-Elites with an automatic random-projection descriptor and the Iso+LineDD directional variation operator, plus Pareto reporting (NSGA-II / SPEA2 / MOEA/D indicators) as a reporting layer rather than a search driver. The quality-diversity archive shares the same `SolutionDeck` interface as the legacy solvers, so it drops in without rewriting them.

**Performance engineering.** A systematic profiling-and-rewrite pass produced a ranked set of findings; the largest are worth recording because they are the reason the engine is usable at scale.

- **Truncated-normal sampling, ~177× on the primitive.** SciPy was rebuilding a distribution docstring on every sample, consuming roughly 4.8 s of a 7.9 s ant-colony run; replacing it with an inverse-CDF (`ndtri`) sampler removed that cost, giving a ~5–8× end-to-end speedup on the affected runs.
- **Ship the fixed data once, up to ~7.5×.** Re-shipping the problem data to each parallel worker every generation is a per-generation cost, so the speedup grows with run length — about 0.96× at 12 generations, 2.3× at 30, 4.6× at 60, and 7.5× at 100. (The benefit is largest for non-array Python payloads; large NumPy arrays are already memory-mapped by the parallel backend.)
- **JIT the local search, ~370×.** A full 2-opt scan at N = 400 dropped from 479 ms to 1.3 ms under `@njit`, and a full 3-opt pass at N = 500 went from timing out (> 120 s) to 0.63 s. Fixing the hot loop also surfaced and fixed a latent 3-opt out-of-bounds bug.

**Lin–Kernighan quality.** Lin–Kernighan produced the shortest tour at every problem size tested, roughly 6–7% under 2-opt and 9–16% under 3-opt at comparable runtime. The Cython and Numba backends are within ~1.1–1.4× of each other on a single warm tour (the search is branchy), so Cython's real advantages are the absence of JIT warm-up and a batched OpenMP path (64 tours at N = 300: 115.8 ms → 79.7 ms, ~1.45×). The simpler 2-opt/3-opt kernels see a larger ~2.7–3× Cython gain.

**Handoff to the clustering package.** Two report items — replacing the Fuzzy C-Means BFGS step with closed-form alternating updates, and JIT-compiling the iVAT path-max loop — were deliberately deferred here, because the clustering code (FCM, VAT/iVAT) is being split into its own package. That seam is exactly where this engine hands off to the `tribble-cluster` work of Chapter 3.

**Standalone-paper opportunities** (for Dr. Cohen's consideration, not part of the core dissertation): the performance-engineering study on its own; the quality-diversity-over-legacy-solvers layer (CVT-MAP-Elites + Iso+LineDD); and the exact GPU/parallel VAT engine as a systems paper. That last one is distinct from the possible complexity note of §9.3: the note would be about the sequencing's cost and memory bound and the heap-versus-dense measurement, whereas a systems paper would be about the parallel and GPU engineering envelope. They should not be merged, and neither should absorb the other's claim.

## A.4 Feature scoring: why the composite metric earned its keep

This section exists because a refactor deleted something whose value nobody had
measured, and measuring it afterwards produced the clearest single argument in
the proposal for consensus over a single statistic.

The Mixture-of-Gaussians construction of Chapter 4 begins by scoring each feature
for how well it separates the output classes, keeping the best few, and building
membership functions only over those. Everything downstream — rule count, clause
count, readability, training time — follows from that ranking. It is the least
glamorous step in the pipeline and, it turns out, the one with the most leverage.

The scorer originally combined four measures of distributional separation:
Bhattacharyya divergence, Jensen–Shannon distance, histogram overlap, and
histogram correlation. It reduced them not by averaging but by taking the mean of
their **arithmetic and geometric** means. That detail is the whole design. A
geometric mean collapses toward zero if *any* of its terms is near zero, so a
feature scores well only when **every** measure agrees it separates the classes.
The composite is a consensus rule, and consensus is insurance against any one
metric's assumptions failing.

An upstream refactor replaced the composite with a choice of one metric,
defaulting to Bhattacharyya. On the PhiUSIIL phishing set the effect was severe,
and it is instructive precisely because the accuracy number understates it.

**Table A.1 — Feature ranking depends on the scorer.** Top five features on
PhiUSIIL under each rule, with normalized scores.

| Rank | Wasserstein | Bhattacharyya | Composite |
|---|---|---|---|
| 1 | **URLSimilarityIndex** (1.000) | HasSocialNet (1.000) | HasSocialNet (1.000) |
| 2 | HasSocialNet (0.867) | HasTitle (0.855) | **URLSimilarityIndex** (0.947) |
| 3 | HasCopyrightInfo (0.743) | NoOfSelfRef (0.784) | HasTitle (0.848) |
| 4 | HasDescription (0.629) | NoOfCSS (0.777) | NoOfCSS (0.820) |
| 5 | DomainTitleMatchScore (0.471) | NoOfImage (0.762) | NoOfSelfRef (0.815) |

One feature, `URLSimilarityIndex`, carries almost the entire signal. Wasserstein
ranks it first; the composite ranks it second; Bhattacharyya does not place it in
the top twenty at all.

**Table A.2 — What the ranking costs.** Test accuracy and fit time against the
number of features retained; 20,000-row sample, ten seeds.

| Features kept | Wasserstein | Bhattacharyya | Composite |
|---:|---:|---:|---:|
| 1 | **0.9967** (0.41 s) | 0.4267 (0.16 s) | 0.4267 (0.41 s) |
| 2 | 0.9967 | 0.4527 | **0.9967** (0.48 s) |
| 3 | 0.9967 | 0.8457 | 0.9967 |
| 5 | 0.9965 | 0.9131 | 0.9966 |
| 7 | **0.9997** (0.77 s) | 0.9183 | 0.9966 |
| 10 | 0.9996 | 0.9274 | 0.9980 |
| 15 | 0.9957 | 0.9477 | **0.9999** (1.90 s) |
| 20 | 0.9989 | 0.9477 (2.40 s) | 0.9995 |

Wasserstein reaches 99.7% on a **single** feature in 0.41 s. The composite needs
two, because it ranks `URLSimilarityIndex` second, and thereafter matches or
leads — it takes the highest score in the table, 0.9999 at fifteen features.
Bhattacharyya never reaches what the other two achieve on one feature: at twenty
features and six times the training cost of Wasserstein-at-one it is still five
points short.

**A note on which composite this is.** The scorer measured here is the one
restored as `method="composite"`, which is not identical to the four-metric blend
that was the original default. That earlier blend ranked `URLSimilarityIndex`
*first* and so reached 0.9969 at a single feature, flat through fifteen. The
restored composite ranks it second and is fractionally stronger at the top end.
The distinction matters only for the one-feature row; the argument below is
unchanged by it.

**Why the parametric metric fails here.** Of the three rules, Bhattacharyya as
implemented is the only one carrying a distributional assumption: it fits a
Gaussian to each class and integrates their overlap. `URLSimilarityIndex` is a
bounded similarity score, and such quantities are typically spiky or bimodal
rather than Gaussian, so a Gaussian-fit divergence mismeasures its separation
badly. Wasserstein makes no such assumption and finds it immediately. The
composite finds it because the measures that do see it outvote the one that does
not — its geometric-mean term requires broad agreement before a feature scores
well, which is what makes it robust to any single metric's blind spot.

**What I take from this, for the thesis rather than the library.** Two things,
and the second is the one that generalizes.

The narrow conclusion is that Wasserstein is the better default, and it is now
the shipped one. It is the only rule that reaches full accuracy on a *single*
feature, which is the regime this construction is built for, and it makes no
distributional assumption to get there. The composite edges it out at fifteen
features (0.9999 against 0.9957), but a fifteen-feature rule base is not the
thing Chapter 4 is arguing for.

The broader conclusion concerns *interpretability*, which is this dissertation's
central claim and not merely its accuracy. Chapter 4 argues that the construction
yields a readable rule base — a handful of clauses over named features. Table A.2
shows that claim is not a property of the model architecture at all; it is a
property of the **feature ranking**. With a ranking that works, the model is
readable *and* accurate at one feature. With a ranking that does not, accuracy is
not recovered at any size tested — Bhattacharyya is still five points short at
twenty features, and reaching even 0.957 takes forty-one. A forty-feature rule
base is not readable by anyone, so the failure costs the readability outright
while the accuracy column merely looks disappointing. The interpretability argument therefore rests on a step the
pipeline treats as preprocessing, and a change to that step damaged
interpretability considerably more than it damaged accuracy — which is exactly
the kind of degradation an accuracy-only evaluation would never surface.

That is the case for keeping the composite available even though a single
well-chosen metric beats it here. Its value is not the ranking it produces on any
one dataset; it is that it cannot be silently wrong in the way a single metric
can, because it requires agreement. On a new dataset, where nobody yet knows
which assumption fails, that is worth having.

*Reproduced by `reproduce/tables/table_a1_feature_scoring.py`. Upstream:
tribble-fis issues #49 (the regression) and #50 (`top_p` semantics).*

---

## A.5 Reproducibility

- **Code.** Four repositories, submoduled into the `grad-school` working repo: `tribble-cluster` (VAT/iVAT/FCM), `tribble-fis` (the fuzzy models and `tribble-tree`), `tribble-opt` (the optimization engine), and `gated-minimax-selection` (the Chapter 5 membership-generation experiments).
- **The reproduction harness.** `reproduce/` is the single entry point, and the goal is that reproducing a result takes one command rather than archaeology. Each table in the proposal has a generator under `reproduce/tables/` that runs the models over a fixed seed set and writes both Markdown and CSV into `reproduce/outputs/`, reporting mean ± standard deviation. Anything it cannot run — a missing optional baseline, an absent dataset, hardware it does not have — is reported as unavailable and printed with the reason, never silently replaced by an estimate. `reproduce/manifest.py` enumerates every experiment across the four repositories with its command, environment, datasets, and hardware tier; a full orchestrator that walks that manifest is the next step.
- **Drivers.** Beyond the table generators, each original result has a named script — the `FuzzySystemsExperiments/*` benchmarks for Chapter 4, `tribble-fis/tribble-tree/demo_*.py` for Chapter 6, `gated-minimax-selection/run_all.py` for Chapter 5, and the `ClusteringExperiments/` harnesses for Chapter 3.
- **Environments.** The submodules carry their own locked environments, so the generators are invoked through them (for example, `uv run --project tribble-fis python reproduce/tables/table_6_1_model_family.py`). Dataset preparation is automatic where licensing permits — the Concrete set is built from the spreadsheet in the repo if the CSV is absent.
- **Hardware.** Development ran on a 32-core 14th-generation Intel i9 workstation with **96 GB** of RAM and a laptop-class RTX 4080 (12 GB, reduced double-precision throughput). Routine runs are held to a self-imposed **64 GB working cap**, so that a large reorder cannot start paging and turn a memory measurement into a disk measurement; Chapter 3's Table 3.3 reports the ceiling under both the cap and the physical limit, and the one result that deliberately exceeds the cap (the 135,000-point reorder, 72.9 GB at float32) is labeled as such. Final performance numbers are to be re-taken under the fixed protocol of Goal G4 — pinned clocks and thermals, multiple seeds, error bars, and a datacenter GPU with full double-precision throughput.
- **Estimates versus demonstrations.** These are held to different standards, deliberately, and Chapter 7's Goal G4 states the rule. Every accuracy figure and every comparative timing ratio is an *estimate*: ten seeds minimum, reported with a spread. The large-scale reorders are *demonstrations* — they establish that a problem of that size can be processed at all, which is a question without a sampling distribution — and are recorded with their hardware, precision, and memory footprint instead of an error bar. Capability and accuracy are both established on the small and mid-size sets, where ground truth exists to score against; the large runs exist to show the same code reaches that scale.
- **Datasets, and what a third party can actually obtain.** This matters more than it usually does, because the datasets are not uniformly available. Concrete, PhiUSIIL, RT-IOT2022, BETH, and the shuttle set are public and a reader can reproduce those results directly. The 135,000-row psychiatric-evaluation set used for the memory results in Chapter 3 is **not** public and cannot be redistributed; its feature names were anonymized before I ever saw them, which is why Chapter 3 treats it purely as a scaling exercise and draws no conclusion from any individual feature. The consequence for reproducibility should be stated plainly: that specific memory measurement is not independently reproducible, and the fix is to re-take it on a public dataset of comparable size rather than to ask anyone to take it on trust.
- **Two implementations, cross-validating.** The reorder exists in two forms — the stage-one priority-queue path (`pvat.py`) and the stage-two compact-active-set Cython kernel (`pcvat.pyx`) — and they are required to produce bit-identical orderings. That equality is itself a test: each path validates the other, and the test suite asserts it against the serial reference rather than against permutation-invariant summaries. Chapter 3 §3.3.2 records why that distinction matters, since an earlier bug survived precisely because the tests only checked invariant quantities.
- **Commit pins.** The exact commit hashes behind each reported result will be pinned in this section at submission time. The permalinks already in Chapter 3 §3.4 are pinned to a specific commit for exactly this reason.

---

*Draft — Appendix prose. A.3 (optimization engine), A.4 (feature scoring) and A.5 (reproducibility) are written out; A.1/A.2 are inventories to be filled as the figures and repeatability runs land. Source outline in `../chapters/appendix.md`; open items in `../ACTION_ITEMS.md`.*
