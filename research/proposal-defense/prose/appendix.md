# Appendix

## A.1 Supplementary figures

A distinction first, because it governs what belongs here. Sixteen figures are called out in the main text and stay there — they carry an argument at the point it is made, and several are load-bearing: Figure 1.2 (the pipeline roadmap) orients the whole document, and Figure 5.2 (band discovery on the log-birth spectrum) is the single figure Chapter 5's contribution rests on. What lands in this appendix is the *supplementary* material: the galleries, the per-fold and per-dataset repetitions, and the diagnostic plots that a reader may want to check but should not have to walk through. Of the sixteen, fourteen are produced today, all through `reproduce/figures/` in PNG for the Markdown and EPS for the LaTeX build, all against one style module so they read as one set. Nine of the fourteen are computed rather than drawn — they run the same code and read the same table CSVs as the chapters they sit in. Two are not drawn, both deliberately and both recorded with their reason in `reproduce/figures/registry.py`: **Figure 4.3**, because the correction-pass experiment it would report has not been run, and **Figure 6.3**, because the *library* method the figure is written against — `MimoGaussianPredictorMemory.predict_trajectory` — returns its input window unchanged (§6.3.6 gives the detail). That is a defect in one exported method, not an absence of any working rollout: `AnalyticalDynamics/test_double_pendulum.py` carries its own iterative rollout with divergence detection (`run_iterative_prediction`), which is what the figure would be drawn from if the section's claim were about that script rather than about the library path. Everything listed below in this appendix is separate supplementary material and is not yet drawn; that is tracked in `ACTION_ITEMS.md`.

- **A.1.1 VAT / iVAT reordered-dissimilarity galleries** — the reordered images for the NASA shuttle set (58K), the psychiatric-evaluation set (135K), and the synthetic circular-cities construction; the Prim/MST block diagram (`vat_prim_mst_block_diagram_v2.svg`).
- **A.1.2 Selection and multi-scale figures** — the figures `fig1`–`fig11` from `gated-minimax-selection/outputs/`: the synthetic datasets, the minimax-transform heatmaps, the persistence curves, the membership-function plots, the ConiVAT bridge repair, the multi-scale hierarchy, and the selection-method comparison.
- **A.1.3 Fuzzy-model visualizations** — per-feature Gaussian-mixture membership plots; the rendered fuzzy trees (`plot_fuzzy_tree`); the hierarchical-mixture structure; the exported Ruspini triangular partitions.

## A.2 Extended results tables

The main text carries twenty-two summary tables (3.1–3.7, 4.1–4.7, 5.1–5.3, 6.1–6.4, 7.1). An earlier version of this paragraph promised their "full, multi-seed versions" here once the repeatability protocol had run, which is now the wrong promise: the seed half of Goal G4 has run, and every numbered table in the main text is already quoted at the ten-seed floor with a spread. What the appendix actually owes is the **per-seed detail** those summary rows aggregate away — the individual splits behind each mean, which is how a reader finds the one seed in ten that carries a failure, as Chapter 6's mixture-of-experts divergence did. What remains outstanding from G4 is the hardware half, and that is a re-run rather than a wider table. Most of these regenerate from the harness in `reproduce/tables/`, which emits Markdown and CSV side by side, so the appendix version is the same data at full width rather than a re-transcription; the tables that do not come from there are named in A.5.

- **A.2.1** Full adversarial-evaluation ARI grids and the complete stitch-ablation grid (all partitions × sizes).
- **A.2.2** The full selection-method bake-off across all synthetic datasets, and the relational-data results.
- **A.2.3** The broadened fuzzy-model benchmark suite (Concrete, PhiUSIIL, turbine, wave-energy, wine, and the IoT sets) with the baseline methods.
- **A.2.4** The three-arm reorder study behind Chapter 3 §3.3.1 — classical cubic, stage-one priority queue, stage-two compact active set — across the full grid of $N$ and both precisions, in absolute seconds with per-seed spreads. The main text reports this normalized, because wall-clock is not portable between machines and ratios are far more so; the appendix version is where the seconds live. It also carries the per-$N$ detail behind Table 3.2's exponent fit — and one thing it carries must be labelled, not folded in. An earlier version of this line promised the **stage-two plateau above $N \approx 750$** as underlying detail that the fitted exponent averages over. That plateau is **withdrawn**. It was measured repeatably on the four-core development laptop and it does not reproduce on the host of record: across five independent runs here stage two is monotone in $N$ and beats stage one at every size in the grid, including inside the band said to collapse to parity, and the exponent fit is *cleaner* without it rather than being distorted by its absence (Chapter 3 §3.4, Chapter 7 §7.2, `PROVENANCE_MAP.md` note 11). So the appendix carries it as a separately labelled laptop-only artifact and as nothing else — kept visible because it is the document's cleanest case of a measurement that was repeatable, seeded, error-barred and still a property of the machine rather than of the code, which is the entire argument for the hardware half of Goal G4. This subsection is also the evidence base for the possible complexity note discussed in §9.3.
- **A.2.5** The output-partitioning study of Goal G5 (uniform vs. quantile vs. pinned-extreme hybrid), including the per-decile and tail-error breakdowns and the bucket-starvation counts that aggregate error hides.

## A.3 The optimization engine (`tribble-opt`)

Per the design decision recorded in Chapter 2, the optimization library is supporting infrastructure rather than a dissertation contribution, and its details live here. It provides the optional *local-polish* stage that sits at the end of the pipeline; the point of *structure before search* is precisely that this engine is not on the critical path. It is nonetheless a substantial piece of software, and several parts are strong enough to stand as their own papers (flagged at the end).

**What the numbers in this section are, before any of them are read.** Every ratio below is a single-run microbenchmark taken during the profiling pass that produced it. There are no seeds, no spreads, no recorded host, and no generator — `reproduce/PROVENANCE_MAP.md` carries no row for A.3 at all, because nothing in `reproduce/` emits it. They are therefore held to a *lower* standard than anything in A.5 two sections below, and I would rather say so here than have a reader discover the inconsistency. They exist to record which rewrites mattered, not to establish by how much, and every one of them should be read as an order of magnitude. The reason to insist on that is this project's own precedent, recorded in `WORKINGDOC.md` §7: cProfile charges per-call overhead, so a deep-call-chain pandas `__getitem__` appeared to account for 57% of runtime, and a speedup I published off that profile at 19% was really 9.8% against a wall clock. Profiles find hotspots; wall clocks size them. Nothing below has been re-taken under the ten-seed protocol, and none of it is load-bearing for any claim in the dissertation — which is why it sits in an appendix for supporting infrastructure rather than in a chapter.

**Scope.** The library covers both continuous and combinatorial optimization behind one interface. The metaheuristics are ant colony optimization, genetic algorithms, particle swarm optimization, and gradient descent, in continuous and combinatorial variants. For the Traveling Salesman Problem it implements nearest-neighbor and convex-hull constructions, 2-opt / 3-opt / Or-opt local search, and a candidate-restricted **Lin–Kernighan** search, with two interchangeable backends — a Numba `@njit` path and a Cython `nogil`/OpenMP path — that produce bit-identical tours. On top of the scalar solvers sits a **quality-diversity** layer: MAP-Elites and CVT-MAP-Elites with an automatic random-projection descriptor and the Iso+LineDD directional variation operator, plus Pareto reporting (NSGA-II / SPEA2 / MOEA/D indicators) as a reporting layer rather than a search driver. The quality-diversity archive shares the same `SolutionDeck` interface as the legacy solvers, so it drops in without rewriting them.

**Performance engineering.** A systematic profiling-and-rewrite pass produced a ranked set of findings; the largest are worth recording because they are the reason the engine is usable at scale.

- **Truncated-normal sampling, ~177× on the primitive.** SciPy was rebuilding a distribution docstring on every sample, consuming roughly 4.8 s of a 7.9 s ant-colony run; replacing it with an inverse-CDF (`ndtri`) sampler removed that cost, giving a ~5–8× end-to-end speedup on the affected runs.
- **Ship the fixed data once, up to ~7.5×.** Re-shipping the problem data to each parallel worker every generation is a per-generation cost, so the speedup grows with run length — about 0.96× at 12 generations, 2.3× at 30, 4.6× at 60, and 7.5× at 100. (The benefit is largest for non-array Python payloads; large NumPy arrays are already memory-mapped by the parallel backend.)
- **JIT the local search, ~370×.** A full 2-opt scan at N = 400 dropped from 479 ms to 1.3 ms under `@njit`, and a full 3-opt pass at N = 500 went from timing out (> 120 s) to 0.63 s. Fixing the hot loop also surfaced and fixed a latent 3-opt out-of-bounds bug.

**Lin–Kernighan quality.** Lin–Kernighan produced the shortest tour at every problem size tested, roughly 6–7% under 2-opt and 9–16% under 3-opt at comparable runtime. The Cython and Numba backends are within ~1.1–1.4× of each other on a single warm tour (the search is branchy), so Cython's real advantages are the absence of JIT warm-up and a batched OpenMP path (64 tours at N = 300: 115.8 ms → 79.7 ms, ~1.45×). The simpler 2-opt/3-opt kernels see a larger ~2.7–3× Cython gain.

**Handoff to the clustering package.** Two report items — replacing the Fuzzy C-Means BFGS step with closed-form alternating updates, and JIT-compiling the iVAT path-max loop — were deliberately deferred here, because the clustering code (FCM, VAT/iVAT) is being split into its own package. That seam is exactly where this engine hands off to the `tribble-cluster` work of Chapter 3.

**Standalone-paper opportunities** (for Dr. Cohen's consideration, not part of the core dissertation): the performance-engineering study on its own; the quality-diversity-over-legacy-solvers layer (CVT-MAP-Elites + Iso+LineDD); and the exact GPU/parallel VAT engine as a systems paper.

That third one is **on hold until Table 3.4 is re-quoted**, and the reason is worth stating rather than leaving the proposal to look better supported than it is. A systems paper of that kind would be written around a speed envelope, and the envelope is currently being re-measured: `PROVENANCE_MAP.md` note 15 marks Table 3.4 **drifted**, and its Fuzzy C-Means row overstates the GPU by roughly an order of magnitude, because the quoted ratio compares a NumPy broadcasting implementation against one using the gram identity and two GEMMs — a difference of *formulation*, not of device. Run the GPU's own formulation on the CPU and most of the ratio goes away. The result the paper would actually rest on, exactness against the serial reference, does reproduce everywhere it is claimed; the speed figures do not read as quoted. So the item stays on the list with its envelope marked as under re-measurement, because proposing a systems paper on a drifted table is precisely how a drifted number becomes a published one.

It is also distinct from the possible complexity note of §9.3: the note would be about the sequencing's cost and memory bound and the heap-versus-dense measurement, whereas a systems paper would be about the parallel and GPU engineering envelope. They should not be merged, and neither should absorb the other's claim.

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
Bhattacharyya does not reach, at any feature count in this table, what the other
two achieve on one — and it pays for the shortfall in training time as well, since
that cost grows with the number of features retained. Exactly *how* far short it
falls turns out to depend on the machine, which is the subject of the next
paragraph and the reason this section rests its argument on the one-feature row.

**A portability caveat on one column — and it is the column carrying the weakest
part of this section's argument.** The bhattacharyya accuracies as printed are the
`reproduce/outputs/main-d0efefc/` archive's, and they are not host-portable. The
difference has been chased rather than assumed (`PROVENANCE_MAP.md` note 12; the
full write-up is `reproduce/outputs/NOTE12_THREADING.md`). Within one environment
the column is bit-identical across two independent full sweeps, four thread counts
and four BLAS kernel families. Between hosts, at identical code, identical seeds
and a byte-identical Table A.1 ranking, it moves by up to 0.043 — and it moves
**only from four features on**: at one and two features the two hosts agree
exactly, and at three they differ by 0.0002, so whatever causes it acts on the fit
and not on the ranking or the data. The divergence appears once the model has
enough features to fit something. The standing explanation was a threading or BLAS
difference and neither half of it survived measurement. **Thread count is
refuted**: from one thread to thirty-two the generator's wall clock moves 2.4×
while the reported accuracy moves exactly 0.000000 in all twenty-seven cells.
**The kernel-family sweep is inconclusive rather than a second refutation, because
its manipulation check failed** — dropping OpenBLAS from AVX2 to SSE-only changed
runtime by 1.6%, so the variable demonstrably loaded and then did nothing this
workload can feel, and an unchanged accuracy says nothing either way. That failed
check is the more useful finding, because it undercuts the framing rather than one
branch of it: a computation this indifferent to which vector instruction path
executes it is not spending its time in the BLAS, so "a BLAS difference" is an
unlikely explanation for a 0.043 swing in it. What is left is the part of the stack
this workload does use and that does differ — **library versions are the leading
untested candidate**, numpy, scipy and scikit-learn being recorded on this host and
unrecorded in the archive that disagrees — and re-running against pinned older
versions here is the next experiment, needing no second machine. Until it is run,
the harness's own instruction stands and this section adopts it: **do not quote
this column to four decimals across machines.**

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
property of the **feature ranking**.

The evidence for that is the **one-feature row**, and I want the argument to rest
there rather than further down the column, for two reasons. It is the largest
effect in the table by a wide margin — 0.9967 against 0.4267, a gap of 0.57, more
than ten times the largest environment effect measured anywhere in that column and
against a thread-count effect of exactly zero. And it sits in the region where the
two hosts agree to the digit: both return 0.4267 at one feature and 0.4527 at two.
So the
claim reads: with a ranking that works, the model is readable *and* accurate at a
single clause; with a ranking that does not, a single clause scores 0.4267, and
the rule base has to grow before the accuracy comes back at all. The readability
was spent by the ranking, silently, before the model was ever built. A change to
a step the pipeline treats as preprocessing damaged interpretability more than it
damaged accuracy — which is exactly the kind of degradation an accuracy-only
evaluation would never surface. An earlier version of this paragraph put a number
on how many features bhattacharyya needs to recover a given accuracy; I have
removed it rather than re-quoted it, because that number is read off precisely the
cells note 12 says are not portable, and on the host of record it does not hold.

**One dataset carries all of this, and it is not a neutral one.** Both tables
above are PhiUSIIL and nothing else. Chapter 6 says twice that PhiUSIIL is
saturated — every method it tests lands within a fraction of a perfect score
there — and that it should therefore carry no weight in a comparison between
methods. That verdict does not void the argument here, because this section is not
comparing methods: it holds the model fixed and varies the *ranking*, and
PhiUSIIL's concentration is what makes the mechanism visible at all, since a
dataset with one dominant feature is the cleanest available test of whether a
scorer finds it. But the mechanism is demonstrated on one unusually concentrated
dataset, and its generality is not. Whether a scorer choice costs this much where
the signal is spread across many features is unmeasured; A.2.3's broadened
benchmark suite is where it would be measured, and until then this is a
demonstration that the failure mode exists rather than a characterization of how
often it bites.

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
- **The reproduction harness.** `reproduce/` is the single entry point, and the goal is that reproducing a result takes one command rather than archaeology. **Most** tables in the proposal have a generator under `reproduce/tables/` that runs the models over a fixed seed set and writes both Markdown and CSV into `reproduce/outputs/`, reporting mean ± standard deviation. Anything a generator cannot run — a missing optional baseline, an absent dataset, hardware it does not have — is reported as unavailable and printed with the reason, never silently replaced by an estimate. `reproduce/manifest.py` enumerates every experiment across the four repositories with its command, environment, datasets, and hardware tier; a full orchestrator that walks that manifest is the next step.
- **Where the "most" bites, since a blanket claim here would be the wrong kind of reassurance.** Three tables have no generator at all: **7.1** is a goals-and-status matrix and structural by design; **6.3** is structural until Goal G6 measures the interpretability counts its pending row wants; and **6.4** has an entry point but not a generator — the experiment lives at `AnalyticalDynamics/test_double_pendulum.py` and `test_atwood_machine.py` in this repository, runs at one fixed seed, does not go through `reproduce/common.py`, and emits no CSV, which is why Chapter 6 declines to quote its 38% figure to two digits. Three more do have generators, but not under `reproduce/tables/`: **3.5, 3.6 and 3.7** come from the `ClusteringExperiments/` scripts, driven through `reproduce/experiments/run_cluster_experiment.py`, which puts their directory on `sys.path` and redirects their figure output into `reproduce/outputs/`. Per-table provenance for all twenty-two is in `reproduce/PROVENANCE_MAP.md`, which is the file to check rather than this bullet.
- **And "one command" is not yet true from a clean start, which is worth conceding in the section that promises it.** `WORKINGDOC.md` §5 records that `origin/main`'s recorded submodule SHAs do not exist on their remotes: PR #28 added `branch = main` to `.gitmodules`, but that only affects `git submodule update --remote`, while an ordinary clone or a CI checkout uses the recorded gitlink — so a fresh clone of the default branch fails at submodule update. No document in this repository claims the pins have since been re-pointed, so I state it as still open rather than as fixed. The working branch pins commits that resolve, and the fix on `main` is one `git submodule update --remote` plus a commit. Until that lands, "reproducing a result takes one command" is true on a checkout that resolves and not yet true for a third party starting from `main`, and that distinction is exactly the kind a reproducibility section should not blur.
- **Drivers.** Beyond the table generators, each original result has a named script — the `FuzzySystemsExperiments/*` benchmarks for Chapter 4, `tribble-fis/tribble-tree/demo_*.py` for Chapter 6, `gated-minimax-selection/run_all.py` for Chapter 5, and the `ClusteringExperiments/` harnesses for Chapter 3.
- **Environments.** The submodules carry their own locked environments, so the generators are invoked through them (for example, `uv run --project tribble-fis python reproduce/tables/table_6_1_model_family.py`). Dataset preparation is automatic where licensing permits — the Concrete set is built from the spreadsheet in the repo if the CSV is absent.
- **Hardware.** Development ran on a 32-core 14th-generation Intel i9 workstation with **96 GB** of RAM and a laptop-class RTX 4080 (12 GB, reduced double-precision throughput). Routine runs are held to a self-imposed **64 GB working cap**, so that a large reorder cannot start paging and turn a memory measurement into a disk measurement; Chapter 3's Table 3.3 reports the ceiling under both the cap and the physical limit, and the one result that deliberately exceeds the cap (the 135,000-point reorder, 72.9 GB at float32) is labeled as such. Chapter 3's CPU timings all come from this host as of the run of record `reproduce/outputs/full-14900hx-r2/` (ten seeds, thirteen generators green in one pass); an earlier draft's swept timing grid came from a four-core i7-1185G7 development laptop, and consolidating onto one host retracted two claims rather than confirming them — see §3.4 and Goal G4. What remains outstanding from the G4 protocol is the hardware half: clocks and thermals are not pinned, and the GPU rows still come from a consumer card with reduced double-precision throughput. Every archive **generated since the machine block was added** records host, CPU, cores, RAM, GPU, governor and the numeric stack (numpy, scipy, scikit-learn, BLAS build), because a measurement that survives a change in all of those is the only kind attributable to the code. The qualifier is load-bearing and the gap it names is not hypothetical. `reproduce/outputs/main-d0efefc/` — an archive several chapters quote — predates the block and has **no machine record at all**: no host, no CPU, no core count, no numeric stack. Its `logs/` carries no `table_a1_feature_scoring.log`, and that generator does not appear in its status list, so the Appendix A.4 figures taken from it came from a hand run outside the orchestrator; only its seed list is pinned, in the Markdown footer. That single omission is why note 12's cross-host difference cannot be *attributed* to anything rather than merely being unexplained: code, commit, seeds and data are all ruled out by direct comparison, and the one thing that differs between the two runs is the one thing that archive did not record. It is the clearest argument in this document for capturing the environment before you need it.
- **Estimates versus demonstrations.** These are held to different standards, deliberately, and Chapter 7's Goal G4 states the rule. Every accuracy figure and every comparative timing ratio is an *estimate*: ten seeds minimum, reported with a spread. The large-scale reorders are *demonstrations* — they establish that a problem of that size can be processed at all, which is a question without a sampling distribution — and are recorded with their hardware, precision, and memory footprint instead of an error bar. Capability and accuracy are both established on the small and mid-size sets, where ground truth exists to score against; the large runs exist to show the same code reaches that scale.
- **Datasets, and what a third party can actually obtain.** This matters more than it usually does, because the datasets are not uniformly available, and "public" and "reproducible from this repository" are two different claims that an earlier version of this bullet ran together. **Concrete, PhiUSIIL and the shuttle set are public and present**, and a reader can reproduce those results directly. **RT-IOT2022 and BETH are public but are not in this repository**, so their rows cannot be reproduced directly by anyone, myself included: that is why Table 4.4's RT-IOT2022 accuracy cell is empty and labelled rather than estimated, and why the open-set study of §4.3.5 ran leave-one-class-out on Glass — 214 samples, a stress test rather than a demonstration — instead of on BETH as designed. Obtaining BETH or a set of comparable scale is what would move the complement-rule result from parity to a comparison; it is a research decision before a coding one, since leave-one-class-out needs at least three classes and BETH is binary. The 135,000-row psychiatric-evaluation set used for the memory results in Chapter 3 is **not** public and cannot be redistributed; its feature names were anonymized before I ever saw them, which is why Chapter 3 treats it purely as a scaling exercise and draws no conclusion from any individual feature. The consequence for reproducibility should be stated plainly: that specific memory measurement is not independently reproducible, and the fix is to re-take it on a public dataset of comparable size rather than to ask anyone to take it on trust.
- **Two implementations, cross-validating.** The reorder exists in two forms — the stage-one priority-queue path (`pvat.py`) and the stage-two compact-active-set Cython kernel (`pcvat.pyx`) — and they are required to produce bit-identical orderings. That equality is itself a test: each path validates the other, and the test suite asserts it against the serial reference rather than against permutation-invariant summaries. Chapter 3 §3.3.2 records why that distinction matters, since an earlier bug survived precisely because the tests only checked invariant quantities.
- **Commit pins.** The exact commit hashes behind each reported result will be pinned in this section at submission time. The permalinks already in Chapter 3 §3.4 are pinned to a specific commit for exactly this reason.

---

*Draft — Appendix prose. A.3 (optimization engine), A.4 (feature scoring) and A.5 (reproducibility) are written out; A.1/A.2 are inventories to be filled as the figures and the per-seed detail land. Open items in `../ACTION_ITEMS.md`.*
