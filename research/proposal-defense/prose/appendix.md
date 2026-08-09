# Appendix

## A.1 Supplementary figures

Fifteen figures are called out in the main text and stay there. Several are load-bearing: Figure 1.2 (the pipeline roadmap) orients the document, and Figure 5.2 (band discovery on the log-birth spectrum) carries Chapter 5's contribution. This appendix takes the *supplementary* material: galleries, per-fold and per-dataset repetitions, diagnostic plots a reader may want to check but need not walk through.

All fifteen are produced today, through `reproduce/figures/`, in PNG for the Markdown and EPS for the LaTeX build, against one style module; ten of those fifteen are computed, running the same code and reading the same table CSVs as their chapters. **Figure 4.3** was the exception until this pass: it waited on a correction-pass experiment that had not been run. That experiment ran, on Glass rather than the originally scoped RT-IOT2022 (RT-IOT2022 still is not among the datasets the harness can load), and the figure was retargeted to it rather than left waiting on a dataset that was never going to arrive — the reasoning is recorded in `reproduce/figures/registry.py`. Everything below is separate supplementary material, undrawn, tracked in `CHECKLIST.md`.

- **A.1.1 VAT / iVAT reordered-dissimilarity galleries.** Reordered images for the NASA shuttle set (58K), the psychiatric-evaluation set (135K), and the synthetic circular-cities construction; the Prim/MST block diagram (`vat_prim_mst_block_diagram_v2.svg`).
- **A.1.2 Selection and multi-scale figures.** Figures `fig1`–`fig11` from `gated-minimax-selection/outputs/`: the synthetic datasets, the minimax-transform heatmaps, the persistence curves, the membership-function plots, the ConiVAT bridge repair, the multi-scale hierarchy, the selection-method comparison.
- **A.1.3 Fuzzy-model visualizations.** Per-feature Gaussian-mixture membership plots; the rendered fuzzy trees (`plot_fuzzy_tree`); the hierarchical-mixture structure; the exported Ruspini triangular partitions.

## A.2 Extended results tables

The main text carries twenty-one summary tables (3.1–3.7, 4.1–4.7, 5.1–5.3, 6.1–6.3, 7.1). The seed half of Goal G4 has run, and every numbered table is already quoted at the ten-seed floor with a spread, so what the appendix owes is not a multi-seed version of each one. It is the **per-seed detail** those rows aggregate away, the splits behind each mean, which is how a reader finds the one seed in ten that carries a failure, as Chapter 6's mixture-of-experts divergence did. G4's outstanding half is hardware, which a re-run fixes and a wider table cannot. Most of these tables regenerate from `reproduce/tables/`, so the appendix version is the same data at full width; the rest are named in A.5.

- **A.2.1** Full adversarial-evaluation ARI grids and the complete stitch-ablation grid (all partitions × sizes).
- **A.2.2** The full selection-method bake-off across all synthetic datasets, and the relational-data results.
- **A.2.3** The broadened fuzzy-model benchmark suite (Concrete, PhiUSIIL, turbine, wave-energy, wine, and the IoT sets) with the baseline methods.
- **A.2.4** The three-arm reorder study behind Chapter 3 §3.3.1 (classical cubic, stage-one priority queue, stage-two compact active set) over the full grid of $N$ and both precisions, in absolute seconds with per-seed spreads. The main text normalizes, wall-clock being unportable between machines and ratios far more so; the seconds live here, with the per-$N$ detail behind Table 3.2's exponent fit. Across five independent runs on the host of record stage two is monotone in $N$ and beats stage one at every size in the grid (Chapter 3 §3.4, `PROVENANCE_MAP.md` note 11). This subsection is also the evidence base for the possible complexity note of §9.3.

  *The fitted exponents, per arm.* Classical fits 3.20 against a theoretical 3 and stage two 1.97 against 2, both stable across five independent runs (3.14–3.21 and 1.93–1.97), and both confirmed. Stage one fits 1.86, stable across 1.84–1.87, short of the $\approx 2.1$ that $O(N^2 \log N)$ predicts and below the pure quadratic reference of 2.00. The shortfall is attributed to an unresolvable log factor, asserted rather than shown: Table 3.2 cannot separate a log factor invisible over a decade and a half from an arm doing less work than the bound allows (Goal G4a: bounded, not confirmed). The prescription is one line in the generator, a constrained fit beside the free one: fix $t = c \cdot N^2 \log N$ with $c$ the only free parameter and report its residual against the free fit's. Within noise the log factor is present; worse than a free exponent of 1.86, and the story changes.
- **A.2.5** The output-partitioning study of Goal G5 (uniform vs. quantile vs. pinned-extreme hybrid) at all three consequent orders, including the per-decile and tail-error breakdowns and the bucket-starvation counts that aggregate error hides.

## A.3 The optimization engine (`tribble-opt`)

Per Chapter 2's recorded design decision, the optimization library is supporting infrastructure, never a dissertation contribution, so its details live here. It provides the optional *local-polish* stage at the end of the pipeline; the point of *structure before search* is that this engine is not on the critical path. It is substantial software all the same, and several parts could stand as their own papers (flagged at the end).

**What the numbers in this section are, before any of them are read.** Every ratio below is a single-run microbenchmark from the profiling pass that produced it: no seeds, no spreads, no recorded host, no generator, and no `reproduce/PROVENANCE_MAP.md` row for A.3 at all, because nothing in `reproduce/` emits it. Hold them to a *lower* standard than anything in A.5. They record which rewrites mattered, not by how much, and every one should be read as an order of magnitude. The precedent is this project's own, in `WORKINGDOC.md` §7: cProfile charges per-call overhead, so a deep-call-chain pandas `__getitem__` appeared to account for 57% of runtime, and a speedup I published off that profile at 19% was really 9.8% against a wall clock. Profiles find hotspots; wall clocks size them. Nothing below has been re-taken under the ten-seed protocol, and none of it is load-bearing for any claim in the dissertation.

**Scope.** One interface covers continuous and combinatorial optimization. Metaheuristics: ant colony optimization, genetic algorithms, particle swarm optimization, gradient descent, each in continuous and combinatorial variants. For the Traveling Salesman Problem: nearest-neighbor and convex-hull constructions, 2-opt / 3-opt / Or-opt local search, and a candidate-restricted **Lin–Kernighan** search over two interchangeable backends producing bit-identical tours (Numba `@njit`, Cython `nogil`/OpenMP). Above the scalar solvers, a **quality-diversity** layer: MAP-Elites and CVT-MAP-Elites with an automatic random-projection descriptor and the Iso+LineDD directional variation operator, plus Pareto reporting (NSGA-II / SPEA2 / MOEA/D indicators) as a reporting layer only, never a search driver. The archive shares the legacy solvers' `SolutionDeck` interface, so it drops in without rewriting them.

**Performance engineering.** A profiling-and-rewrite pass produced a ranked set of findings; the largest are why the engine is usable at scale.

- **Truncated-normal sampling, ~177× on the primitive.** SciPy was rebuilding a distribution docstring on every sample, consuming roughly 4.8 s of a 7.9 s ant-colony run; replacing it with an inverse-CDF (`ndtri`) sampler removed that cost, giving a ~5–8× end-to-end speedup on the affected runs.
- **Ship the fixed data once, up to ~7.5×.** Re-shipping the problem data to each parallel worker every generation is a per-generation cost, so the speedup grows with run length: about 0.96× at 12 generations, 2.3× at 30, 4.6× at 60, and 7.5× at 100. (The benefit is largest for non-array Python payloads; large NumPy arrays are already memory-mapped by the parallel backend.)
- **JIT the local search, ~370×.** A full 2-opt scan at N = 400 dropped from 479 ms to 1.3 ms under `@njit`, and a full 3-opt pass at N = 500 went from timing out (> 120 s) to 0.63 s. Fixing the hot loop also surfaced and fixed a latent 3-opt out-of-bounds bug.

**Lin–Kernighan quality.** Lin–Kernighan produced the shortest tour at every size tested, roughly 6–7% under 2-opt and 9–16% under 3-opt at comparable runtime. The Cython and Numba backends are within ~1.1–1.4× of each other on a single warm tour (the search is branchy), so Cython's real advantages are no JIT warm-up and a batched OpenMP path (64 tours at N = 300: 115.8 ms → 79.7 ms, ~1.45×). The simpler 2-opt/3-opt kernels see a larger ~2.7–3× Cython gain.

**Handoff to the clustering package.** Two report items were deferred here: replacing the Fuzzy C-Means BFGS step with closed-form alternating updates, and JIT-compiling the iVAT path-max loop. The clustering code (FCM, VAT/iVAT) is being split into its own package, and that seam is where this engine hands off to the `tribble-cluster` work of Chapter 3.

**Standalone-paper opportunities** (for Dr. Cohen's consideration, not part of the core dissertation): the performance-engineering study on its own; the quality-diversity-over-legacy-solvers layer (CVT-MAP-Elites + Iso+LineDD); and the exact GPU/parallel VAT engine as a systems paper.

That third one is **on hold until Table 3.4 is re-quoted**. Such a paper would be written around a speed envelope, and this envelope is being re-measured. `PROVENANCE_MAP.md` note 15 marks Table 3.4 **drifted**: its Fuzzy C-Means row overstates the GPU by roughly an order of magnitude, because the quoted ratio compares a NumPy broadcasting implementation against one using the gram identity and two GEMMs. That is a difference of *formulation*, not of device. Run the GPU's own formulation on the CPU and most of the ratio goes away. What the paper would rest on, exactness against the serial reference, does reproduce everywhere it is claimed; the speed figures do not read as quoted. So the item stays, envelope marked under re-measurement: proposing a systems paper on a drifted table is how a drifted number gets published.

It is also distinct from the possible complexity note of §9.3, which would cover the sequencing's cost and memory bound and the heap-versus-dense measurement, while a systems paper would cover the parallel and GPU engineering envelope. Neither should absorb the other's claim.

## A.4 Feature scoring: why the composite metric earned its keep

A refactor deleted something whose value nobody had measured. Measuring it afterwards produced the proposal's clearest single argument for consensus over a single statistic.

The Mixture-of-Gaussians construction of Chapter 4 begins by scoring each feature for how well it separates the output classes, keeping the best few and building membership functions only over those. Everything downstream (rule count, clause count, readability, training time) follows from that ranking. It is the least glamorous step in the pipeline and, it turns out, the one with the most leverage.

The scorer originally combined four measures of distributional separation: Bhattacharyya divergence, Jensen–Shannon distance, histogram overlap, and histogram correlation. It reduced them not by averaging but by taking the mean of their **arithmetic and geometric** means. That detail is the whole design. A geometric mean collapses toward zero if *any* term is near zero, so a feature scores well only when **every** measure agrees it separates the classes. The composite is a consensus rule, and consensus insures against any one metric's assumptions failing.

An upstream refactor replaced it with a choice of one metric, defaulting to Bhattacharyya. On the PhiUSIIL phishing set the effect was severe, and instructive precisely because the accuracy number understates it.

**Table A.1 — Feature ranking depends on the scorer.** Top five features on
PhiUSIIL under each rule, with normalized scores.

| Rank | Wasserstein | Bhattacharyya | Composite |
|---|---|---|---|
| 1 | **URLSimilarityIndex** (1.000) | HasSocialNet (1.000) | HasSocialNet (1.000) |
| 2 | HasSocialNet (0.867) | HasTitle (0.855) | **URLSimilarityIndex** (0.947) |
| 3 | HasCopyrightInfo (0.743) | NoOfSelfRef (0.784) | HasTitle (0.848) |
| 4 | HasDescription (0.629) | NoOfCSS (0.777) | NoOfCSS (0.820) |
| 5 | DomainTitleMatchScore (0.471) | NoOfImage (0.762) | NoOfSelfRef (0.815) |

One feature, `URLSimilarityIndex`, carries almost the entire signal. Wasserstein ranks it first; the composite ranks it second; Bhattacharyya does not place it in the top twenty at all.

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

Wasserstein reaches 99.7% on a **single** feature in 0.41 s. The composite needs two, ranking `URLSimilarityIndex` second, and thereafter matches or leads, taking the table's highest score. At no feature count here does Bhattacharyya reach what the other two achieve on one, and it pays for the shortfall in training time too, since that cost grows with the features retained. Exactly *how* far short it falls depends on the machine.

**A portability caveat on the weakest column in this argument.** The bhattacharyya accuracies as printed come from `reproduce/outputs/main-d0efefc/` and are not host-portable; the difference was chased, not assumed (`PROVENANCE_MAP.md` note 12; write-up `reproduce/outputs/NOTE12_THREADING.md`). Within one environment the column is bit-identical across two full sweeps, four thread counts and four BLAS kernel families. Between hosts, at identical code, identical seeds and a byte-identical Table A.1 ranking, it moves up to 0.043, and only from four features on (one and two agree exactly, three differ by 0.0002), so the cause acts on the fit, not the ranking or the data. Neither half of the threading-or-BLAS explanation stands: thread count is refuted outright, and the kernel-family sweep is inconclusive because its manipulation check failed. A.6 carries both sweeps. **Library versions are the leading untested candidate**: numpy, scipy and scikit-learn are recorded on this host, unrecorded in the disagreeing archive, and pinning older versions here is the next experiment, needing no second machine. Until then the harness's instruction stands: **do not quote this column to four decimals across machines.**

**Which composite this is.** The scorer measured here is the restored `method="composite"`, not identical to the four-metric blend that was the original default. That blend ranked `URLSimilarityIndex` *first* and reached 0.9969 at a single feature, flat through fifteen; the restored one ranks it second and is fractionally stronger at the top end. The distinction matters only for the one-feature row. The argument below is unchanged.

**Why the parametric metric fails here.** Of the three rules, Bhattacharyya as implemented is the only one carrying a distributional assumption: it fits a Gaussian to each class and integrates their overlap. `URLSimilarityIndex` is a bounded similarity score, and such quantities are typically spiky or bimodal, not Gaussian, so a Gaussian-fit divergence mismeasures its separation badly. Wasserstein assumes nothing and finds it at once. The composite finds it because the measures that do see it outvote the one that does not.

Two conclusions follow, and the second generalizes. Narrowly, Wasserstein is the better default and is now the shipped one: the only rule reaching full accuracy on a *single* feature, the regime this construction is built for. The composite edges it out at fifteen features (0.9999 against 0.9957), but a fifteen-feature rule base is not what Chapter 4 argues for.

The broader conclusion concerns *interpretability*, this dissertation's central claim, which accuracy alone does not establish. Chapter 4 argues the construction yields a readable rule base, a handful of clauses over named features. Table A.2 shows that claim is no property of the model architecture. It is a property of the **feature ranking**.

The evidence is the **one-feature row**, not further down the column, for two reasons. It is the largest effect in the table by a wide margin: 0.9967 against 0.4267, a gap of 0.57, more than ten times the largest environment effect in that column and against a thread-count effect of exactly zero. And the two hosts agree there to the digit, both returning 0.4267 at one feature and 0.4527 at two. The claim, then: with a ranking that works, the model is readable *and* accurate at a single clause; with a ranking that does not, a single clause scores 0.4267 and the rule base has to grow before the accuracy comes back. The readability was spent by the ranking, silently, before the model was ever built. A change to a step the pipeline treats as preprocessing damaged interpretability more than accuracy, exactly the degradation an accuracy-only evaluation never surfaces. No figure is quoted for how many features bhattacharyya needs to reach a given accuracy: that count would be read off precisely the cells note 12 says are not portable, and it does not hold on the host of record.

**One dataset carries all of this, and it is not a neutral one.** Both tables above are PhiUSIIL and nothing else. Chapter 6 says twice that PhiUSIIL is saturated, every method it tests landing within a fraction of a perfect score, and that it should therefore carry no weight in a comparison between methods. That verdict does not void the argument here, because this section compares no methods: it holds the model fixed and varies the *ranking*, and PhiUSIIL's concentration is what makes the mechanism visible, a dataset with one dominant feature being the cleanest test of whether a scorer finds it. But the mechanism is demonstrated on one unusually concentrated dataset and its generality is not. Whether a scorer choice costs this much where the signal is spread across many features is unmeasured; A.2.3's broadened suite is where it would be measured. Until then this shows the failure mode exists and does not characterize how often it bites.

So the composite stays available even though a single well-chosen metric beats it here. Its value is not the ranking it produces on one dataset but that it cannot be silently wrong the way a single metric can, because it requires agreement. On a new dataset, where nobody yet knows which assumption fails, that is worth having.

*Reproduced by `reproduce/tables/table_a1_feature_scoring.py`. Upstream:
tribble-fis issues #49 (the regression) and #50 (`top_p` semantics).*

---

## A.5 Reproducibility

- **Code.** Four repositories, submoduled into the `grad-school` working repo: `tribble-cluster` (VAT/iVAT/FCM), `tribble-fis` (the fuzzy models and `tribble-tree`), `tribble-opt` (the optimization engine), and `gated-minimax-selection` (the Chapter 5 membership-generation experiments).
- **Drivers.** Beyond the table generators, each original result has a named script: `FuzzySystemsExperiments/*` for Chapter 4, `tribble-fis/tribble-tree/demo_*.py` for Chapter 6, `gated-minimax-selection/run_all.py` for Chapter 5, the `ClusteringExperiments/` harnesses for Chapter 3.
- **Environments.** The submodules carry locked environments, so generators run through them (`uv run --project tribble-fis python reproduce/tables/table_6_1_model_family.py`). Dataset preparation is automatic where licensing permits: the Concrete set builds from the repo's spreadsheet if the CSV is absent.
- **Commit pins.** Exact commit hashes for each reported result will be pinned here at submission time; Chapter 3 §3.4's permalinks are already pinned to a specific commit for this reason.

#### The reproduction harness

`reproduce/` is the single entry point, and the goal is that reproducing a result takes one command instead of archaeology. **Most** tables have a generator under `reproduce/tables/` that runs the models over a fixed seed set and writes Markdown and CSV into `reproduce/outputs/` with mean ± standard deviation. Anything a generator cannot run (a missing optional baseline, an absent dataset, hardware it does not have) is reported unavailable with the reason, never silently replaced by an estimate. `reproduce/manifest.py` enumerates every experiment across the four repositories with its command, environment, datasets and hardware tier; a full orchestrator walking it is the next step.

#### Where the "most" bites

Two tables have no generator at all. **7.1** is a goals-and-status matrix, structural by design. **6.3** is structural until Goal G6 measures the interpretability counts its pending row wants. Three more have generators outside `reproduce/tables/`: **3.5, 3.6 and 3.7** come from the `ClusteringExperiments/` scripts through `reproduce/experiments/run_cluster_experiment.py`, which puts their directory on `sys.path` and redirects figure output into `reproduce/outputs/`. Per-table provenance for all twenty-one is in `reproduce/PROVENANCE_MAP.md`, the file to check.

#### "One command" is not yet true from a clean start

`WORKINGDOC.md` §5 records that `origin/main`'s recorded submodule SHAs do not exist on their remotes. PR #28 added `branch = main` to `.gitmodules`, but that only affects `git submodule update --remote`; an ordinary clone or CI checkout uses the recorded gitlink, so a fresh clone of the default branch fails at submodule update. No document here claims the pins have been re-pointed since, so I state it as still open. The working branch pins commits that resolve; the fix on `main` is one `git submodule update --remote` plus a commit. Until that lands, "one command" holds on a checkout that resolves and not for a third party starting from `main`.

#### Hardware

Development ran on a 32-core 14th-generation Intel i9 workstation, **96 GB** of RAM, laptop-class RTX 4080 (12 GB, reduced double-precision throughput). Routine runs are held to a self-imposed **64 GB working cap**, so a large reorder cannot start paging and turn a memory measurement into a disk measurement; Chapter 3's Table 3.3 reports the ceiling under both the cap and the physical limit, and the one result that deliberately exceeds the cap (the 135,000-point reorder, 72.9 GB at float32) is labeled as such. Chapter 3's CPU timings all come from this host as of the run of record `reproduce/outputs/full-14900hx-r2/` (ten seeds, thirteen generators green in one pass), the swept timing grid included; §3.4 measures how far a speedup ratio moves when the host changes. G4's outstanding half is hardware: clocks and thermals unpinned, GPU rows still from that consumer card.

#### The machine record, and the archive that has none

Every archive **generated since the machine block was added** records host, CPU, cores, RAM, GPU, governor and the numeric stack (numpy, scipy, scikit-learn, BLAS build): a measurement that survives a change in all of those is the only kind attributable to the code. The qualifier is load-bearing. `reproduce/outputs/main-d0efefc/`, an archive several chapters quote, predates the block and has **no machine record at all**: no host, no CPU, no core count, no numeric stack. Its `logs/` carries no `table_a1_feature_scoring.log`, and that generator is absent from its status list, so the Appendix A.4 figures taken from it came from a hand run outside the orchestrator; only its seed list is pinned, in the Markdown footer. That omission is why note 12's cross-host difference can only be left unexplained and never *attributed*: code, commit, seeds and data are ruled out by direct comparison, and the one thing differing between the two runs is the one thing that archive did not record. Capture the environment before you need it.

#### Estimates versus demonstrations

Deliberately different standards, by the rule Chapter 7's Goal G4 states. Every accuracy figure and every comparative timing ratio is an *estimate*: ten seeds minimum, reported with a spread. The large-scale reorders are *demonstrations*: they establish that a problem of that size can be processed at all, a question without a sampling distribution, and are recorded with hardware, precision and memory footprint instead of an error bar. Capability and accuracy are both established on the small and mid-size sets, where ground truth exists to score against.

#### The three measurements that break the seed floor

Goal G4a's ten-seed floor has three known violations, named here because a board-wide rule with unnamed exceptions is worse than a narrower rule stated plainly. None is yet repaired.

- **The §6.3.5 optimizer study**: ten seeds, so the seed half is met, but the archive `reproduce/outputs/opt-hotcold-2026-08-02/` carries a machine block reading a four-logical-core Xeon virtual machine under Linux, not the run of record's thirty-two-core i9. A host in the archive and undeclared in the prose is what the single-host discipline exists to catch. The identification sweep beside it runs at three seeds. Name the host.
- **Appendix A.3**, the `tribble-opt` section: every speedup with no seeds, no spreads, no host and no generator, taken from an engineering profiling log and labelled as such there. A.3's opening carries the `WORKINGDOC.md` §7 precedent, where a 19% speedup sized off a cProfile pass measured 9.8% on a wall clock.

#### Datasets, and what a third party can actually obtain

The datasets are not uniformly available, and "public" and "reproducible from this repository" are two different claims. **Concrete, PhiUSIIL and the shuttle set are public and present**; a reader can reproduce those results directly. **RT-IOT2022 and BETH are public but are not in this repository**, so their rows cannot be reproduced directly by anyone, myself included: hence Table 4.4's RT-IOT2022 accuracy cell being empty and labelled instead of estimated, and hence §4.3.5's open-set study running leave-one-class-out on Glass (214 samples, a stress test and not a demonstration) instead of on BETH as designed. Obtaining BETH or a set of comparable scale would move the complement-rule result from parity to a comparison, a research decision before a coding one: leave-one-class-out needs at least three classes and BETH is binary. The 135,000-row psychiatric-evaluation set behind Chapter 3's memory results is **not** public and cannot be redistributed; its feature names were anonymized before I ever saw them, so Chapter 3 treats it purely as a scaling exercise and draws no conclusion from any individual feature. That measurement is therefore not independently reproducible, and the fix is to re-take it on a public dataset of comparable size, not to ask anyone to take it on trust.

**What Goal G2 adds, and why it arrives without baselines.** G2's non-coordinate data is identified and verified here: UCR/UEA sets under dynamic time warping through `aeon` (Crop, at 24,000 series and 24 classes, ElectricDevices, StarLightCurves, ECG5000, FordA); TUDataset graphs under a graph kernel; and the Duin–Pękalska collection, distributed *as* distance matrices and so matching Chapter 3's claim most literally. No natural competitor can run on them: Deshpande and Kumar's kd-tree and bounding-box methods need Euclidean coordinates by construction, clusiVAT samples a coordinate space, eVAT's GPU front end computes distances from points, and warped series have no fixed vector embedding, the premise of DTW. That is the seam §3.2 claims, and why "beat the baselines" is unavailable to G2: in the regime the experiment exists to demonstrate, there are no baselines to beat. Chapter 7 gives the four criteria that replace it.

#### Two implementations, cross-validating

The reorder exists in two forms, the stage-one priority-queue path (`pvat.py`) and the stage-two compact-active-set Cython kernel (`pcvat.pyx`), required to produce bit-identical orderings. That equality is itself a test: each path validates the other, asserted against the serial reference, never against permutation-invariant summaries. Chapter 3 §3.3.2 records why that matters: an earlier bug survived because the tests only checked invariant quantities.

---

## A.6 A selection of side quests

These are the investigations that did not work: things built or measured to answer a question, where what came back was negative. Each entry is the question, what was done, what came back, and what it cost or changed. Two of them changed the shipped code. Following the side quests are two failures predicted in advance, with the evidence that makes each likely, and three narrowings of scope with their reasons. Those are not side quests; they sit here because Chapter 7's statuses point here for their evidence.

#### The matrix-free reorder (Chapter 3 §3.3.2, Goal G4d)

*Can the reorder run without ever materializing the distance matrix, computing each $D_{i,j}$ on demand?* The package contained exactly one matrix-free implementation, `vat_prim_mst_seq`. Checked elementwise against the serial reference, it returns the seed vertex followed by every other vertex in ascending index order: chance-level agreement, and nothing in the package ever called it. It is removed from the public API at the commit Chapter 3 pins (`tribble-cluster e3c27e6`), and Table 3.3 keeps its row, so the negative result stays measured rather than silently deleted. The cost is the regime past about 155,000 points, which the in-place scheme does not reach and which no result in this document occupies. Goal G4d is the rebuild, and it is fifth in the cut order.

#### The hybrid bucket arm, and the axis the study forgot to vary (Chapter 4 §4.3.2)

*Is there a third option that dominates both, equal-frequency boundaries with the two extreme bucket centroids pinned to the observed min and max?* `partition_output` shipped exactly that for the life of this work, and the sweep ran all three arms to find out. There is a third option and it is the worst of the three, which took four passes to establish because each of the first three measured it in the one regime where it cannot be seen.

The sweep ran first and second order only. Across those, 126 cells over three schemes and six configurations, the largest separation between the hybrid and pure quantile is 0.004 against seed deviations of ±0.018 to ±0.027, and 0.004 is also the bound across every archive under `reproduce/outputs/`, at five seeds and ten alike. The two arms are never *identical*: they differ in all six configurations on at least one of four metrics. What they never differ by is more than noise. So three studies concluded no scheme could be recommended, and §4.3.2 said so.

The missing axis was **zeroth order**, and it was one line of the generator away the whole time. `solve_tsk_consequents` holds the first and last rules' constant terms at the centroids it is handed, exactly, as an equality constraint. At zeroth order that constant is a rule's entire output, so the three arms span 0.828: uniform 0.394 ± 0.065, pure quantile 0.242 ± 0.070, pinned quantile −0.434 ± 0.241, with the ordering holding at three, four and six buckets. Reading the solved coefficients shows the mechanism directly. Handed `[0.0, 0.4038, 1.0]`, the solve returns `[0.0, 0.411, 1.0]`, ends untouched, so the bottom rule emits the target's global minimum for a bucket of 344 points whose mean is 0.195. At first and second order the same ends stay pinned while the free interior constant runs to $-0.38$ and $-1.19$, outside the target's own range: the solve spending intercepts as free parameters and paying the bias back through the linear terms. That compensation is the whole reason the pin looked inert.

Two lessons, and the second is the expensive one. A defect can be real, load-bearing and invisible to every aggregate metric in the study designed to find it. And an experiment can be run three times, at ten seeds, across 126 cells, and still not vary the axis its own question depends on. The failure was not too few seeds or too few configurations but a regime never entered. What made it visible in the end was reading the coefficients instead of the scores.

An earlier belief about this arm is withdrawn: that the pinned values were discarded before reaching inference, because the closed-form solve re-derives its own bucket means. It does re-derive the *free* ones. `pin_extremes` defaults to true and no generator here overrides it, so the two ends survive the solve, which the coefficients above show and which the hybrid's differing scores already implied. `CHECKLIST.md` §F carries the working record of this saga (labelled H5 there), including the earlier corrected bound. It first recorded the two arms as bit-identical in all eighteen configurations, a claim that holds in no archive.

#### The z-score collapse, and the third explanation nobody tested (Chapter 4 §4.1)

*Why does centring the features, rather than bounding them, break the headline model so badly?* Under real z-scoring the first-order flat MoG-TSK measured 0.014 ± 0.195, below its own raw-feature score of 0.666, with RMSE going from 7.5 to 16.1 MPa. That is not a degradation, it is a failure, and §4.1 built an argument on it: boundedness as an assumption the construction rests on rather than a convenience.

Two innocent explanations were ruled out properly. It was not the ridge scale: sweeping `l2_reg` over 1e-2 through 0 moved the first-order gap by 0.001. It was not the BIC component count, which is genuinely not scale-invariant and does pick different rule bases in the two arms; pinning `n_gaussians` so both arms got an identical rule base left the collapse intact at −0.407, −0.524 and −0.634 for two, three and four components. It underfit the *training* set as well, MSE 0.030 against 0.009 at seed 0, so it was a fitting failure rather than an extrapolation failure. Each of those was the right check to run.

The third explanation was not a candidate because it was not a variable: **the output partition.** Switching from pinned-quantile to uniform output cuts, changing nothing about the feature transform, takes that cell from 0.014 ± 0.195 to 0.713 ± 0.035 and flips Δ z-score − raw from −0.651 to +0.018. The interaction is the two pinned extreme rules, their constants held at the target's global minimum and maximum, against features whose range z-scoring leaves unbounded, so the linear terms that had to make up the difference had a much wider dynamic range to do it over. Two effects, each modest alone, multiplying.

What survives is smaller and more specific: bounding the inputs is worth 0.083 at first order and 0.015 at second, and z-scoring's real cost is *variance*: ±0.115 against min-max's ±0.026 at full second order. The figures quoted from the pinned arm are withdrawn rather than re-taken, since the effect they characterised is gone. The methodological point is worth keeping: two careful eliminations plus a confident conclusion is still a conclusion drawn from a two-variable search over a three-variable space, and nothing in the eliminating protects against a variable that was never in it.

#### Thread count and the BLAS kernel family (Appendix A.4)

*Why do the bhattacharyya accuracies in Table A.2 move by up to 0.043 between two hosts running identical code, identical seeds and a byte-identical ranking?* The standing explanation was threading or the BLAS build, and both halves were swept (`reproduce/outputs/NOTE12_THREADING.md`).

**Thread count is refuted outright.** One thread to thirty-two moves the wall clock 2.4× and the accuracy by exactly 0.000000, in all twenty-seven cells.

**The kernel-family sweep is inconclusive, and its failed manipulation check is the more useful finding.** Taking OpenBLAS from AVX2 to SSE-only changed runtime by 1.6%, so the variable loaded and then did nothing this workload can feel, and an unchanged accuracy under a manipulation that did not manipulate anything says nothing either way. What does survive is indirect and more useful than the sweep it came from: a workload that indifferent to the vector instruction path is not spending its time in the BLAS, which makes a BLAS difference an unlikely explanation for a 0.043 swing. It also set the standard Chapter 5 §5.4 then applied to its own scale-invariance sweep, which has no manipulation check at all and is reported inconclusive for that reason.

Library versions are the leading untested candidate, and A.4 carries the caveat the two sweeps leave standing.

#### Phase four's soft bands (Chapter 5, Goal G1)

*Will soft kernel-weighted band memberships fix small-sample over-segmentation?* Phase four of `MEMBERSHIP_ROADMAP.md` proposed them as the fix, and the one attempt, on `feat/mf-phase4-bands`, added a single-block-band drop and a containment-aware merge: `single_scale` cleaned up, genuine hierarchies were unregressed, and the case it was built for still failed (Goal G1 has the adjusted Rand indices). The cause is not sampling density. Single-linkage chains through the diffuse cluster, so one cluster produces about eighteen nested significant blocks spanning birth heights 25 to 180, and birth-height banding shreds that and is then fooled by its own containment test. What that cost is the expectation itself: birth height is a clean band coordinate only where each cluster occupies a narrow birth range, the consequence Goals G1 and G7 both carry, and the likely deliverable is now a single-versus-multi-level gate rather than a kernel.

#### `m5py` against a current scikit-learn (Chapter 6 §6.4)

*Can the M5 model-tree row be filled from the package that already exists?* No. `m5py` does not import against scikit-learn 1.9.0, and pinning an older scikit-learn to rescue that row would move every other number in the chapter. Chapter 6 §6.4 states the fault and the three branches out of it, and Chapter 7 dates the decision to 31 March 2027 so the discovery does not land mid-suite. This one cost a table row rather than a claim: it is a dependency fault, not an experiment left unrun.

### Two failures predicted in advance, and three narrowings

#### The capstone's likelier failure, already on record (§7.1)

A mechanism for the failure §7.1 treats as the likelier of its two is on record: phase-6 validation in `gated-minimax-selection/notes/MF_PROGRESS_LOG.md` scores the ultrametric step memberships worse than crisp zero-one labels against true Gaussian posteriors.

#### The mixture's EM, and the answers the document had pre-absorbed (Chapter 6, Goal G3)

G3's prediction is registered in advance because the document had already absorbed every outcome. §6.2 concedes that a single-layer TSK system is functionally equivalent to a mixture of experts; §6.3.5 and §6.4 both measure additional search buying less as consequent capacity grows, refinement worth 0.914 at zeroth order, 0.072 at first and 0.037 at second, a factor of twenty-five; and §6.3.3 says that if the EM slips, the one-shot mixture stands as a completed contribution. Left alone, EM ≈ one-shot reads as confirmation, EM > one-shot as the deliverable, EM-never-runs as a de-scope: three doors and no experiment.

The asymmetry runs the uncomfortable way too, and the outcome that would embarrass the thesis is the positive one: if joint EM re-estimation beats the greedy one-shot fit by more than G3's band, a global search over the gates found what the structure-first construction left on the table, and §6.3.5's "once structure is recovered, search has little left to find" needs weakening in print.

#### Three narrowings, and what each one costs (Goals G3, G6, G8)

Three goals are narrowed rather than dropped. Each narrowing's reason is here; Chapter 7 states its price.

- **G3's baselines.** Fumanal-Idocin et al. [@fumanal2025fast] and the deep TSK fuzzy classifier are complete published architectures with no implementation the harness can reach, and neither is load-bearing: ANFIS and GA-FIS answer "faster than conventional fuzzy training", CART, Random Forest, M5 and flat TSK answer "competitive with standard regressors". Both stay cited in §6.2.
- **G6's expert-audience study.** No protocol, no computed sample size, no institutional-review timeline, and "six domain experts preferred the hierarchy" would not be defensible in a dissertation whose interpretability bibliography is one XAI entry deep.
- **G8's construction.** It had held one quarter, 2028 Q1, which Chapter 10's Gantt omitted and which already carries the capstone, G6, G7, the write-up and the defense. It spends interpretability, the dissertation's own thesis, making it the one goal whose success would weaken the argument around it, a tension §6.2 flags. And the §5.3.5 disjunct counter that decides whether it is worth having has never returned a value other than one, in either mode on any recorded run.

---

## A.7 Benchmark dataset inventory, by category

A.5 above answers *is this dataset public, and is it present in this repository.* This section answers a different question: for each task category the document measures something on, does a **small, fast** dataset and a **large, at-scale** dataset both exist — the pairing that lets a method be iterated on cheaply and then shown not to fall over at size. Availability and role-completeness are not the same claim, and keeping them separate matters here: a dataset can be public and present and still leave a category with no large partner, and a category can name a large dataset that has never once been measured. Compiled by reading every prose chapter, `CHECKLIST.md`, `NEXT_STEPS.md`, and the loader code under `reproduce/tables/`, `reproduce/optimizers/`, and `FuzzySystemsExperiments/`.

**Status vocabulary**, reusing A.5's own distinction rather than inventing a second one: *measured* (a seeded, repeatable generator produced the number cited), *demonstrated* (a single-shot run at scale, recorded with hardware and footprint instead of a spread, per §7.2's rule), *named* (appears in the prose or in a loader script, but no run of any kind exists), and one state A.5 does not need but this table does — *unwired* (a real dataset, verified loadable, that no generator or manifest entry uses yet).

### A.7.1 Regression

| Dataset | Size | Status | Role |
|---|---|---|---|
| UCI Concrete Compressive Strength | 1,030 × 8 | measured (Ch1, Ch2, Ch4 §4.3.2–4.4, Ch6, every table generator that touches regression) | small / fast — the *only* regression benchmark in the document, at every consequent order |
| Diabetes (sklearn) | 442 × 10 | measured (Table 4.8, dedup sweep only) | small / fast — chosen for the tolerance sweep, not a modeling flagship |

**Gap.** No large regression dataset appears anywhere. Concrete carries the entire regression story, including Chapter 6's model-family comparison and the optimizer study's hot-start problem (A.7.6). There is no regression counterpart to classification's PhiUSIIL/RT-IOT2022 pairing below.

*A pilot investigation of this gap exists*, `reproduce/regression_scale/RESULTS_2026-08-05.md`: single-seed, not canonically sourced, and not yet a decision about which dataset or model family to carry forward — see CHECKLIST **C13**. Filed as a start, not a close.

### A.7.2 Classification

| Dataset | Size | Status | Role |
|---|---|---|---|
| Glass (UCI) | 214 × 9, 6 classes | measured — also the anomaly substitute (A.7.3) and the Table 4.8/4.9 dedup and correction-pass testbed | small / fast |
| Wine, Breast Cancer, Digits (sklearn) | 178×13 / 569×30 / 1,797×64 | measured (Table 4.8, dedup sweep only) | small / fast |
| PhiUSIIL phishing URL | 235,000 × 54, binary | measured historically (Table 4.1: 0.997 ± 0.001 acc, 0.28–0.64 s) — **but no longer reproducible from a clean checkout.** The repo loader's bundled copy lived at `tribble-fis/gaussian_mixture/phishing_data/`, and `gaussian_mixture/` was deleted upstream (commit `8484fd6`, per `_fuzzy_models.py`'s own comment); `data/` in this repository holds only `Concrete_Data.csv`. A fresh run falls through to a `ucimlrepo` fetch that returns a *different* feature set, which the loader's own comment flags as producing results "not comparable" to every number quoted from it | large / scale — the one role currently filled, on a fragile path |
| RT-IOT2022 | 123,000 × 83, 12 classes | named — `FuzzySystemsExperiments/iot.py` exists but its own comment states the `rt-iot2022/` data directory "is not in this repo"; Table 4.4 marks the row unrun | large / scale — the *intended* flagship, never measured |

**Gap, of a different shape than A.7.1's.** Both roles are nominally filled, but the large role has no solid representative today: PhiUSIIL's measured numbers are real but sit on a reproduction path this pass found to be broken, and RT-IOT2022, the dataset actually named as the chapter's scale target, has never been run at all. A.5 states "Concrete, PhiUSIIL and the shuttle set are public and present, a reader can reproduce those results directly" — that sentence is no longer accurate for PhiUSIIL and is worth revisiting there.

### A.7.3 Anomaly / open-set detection

| Dataset | Size | Status | Role |
|---|---|---|---|
| Glass, leave-one-class-out | 214 × 9, 6 classes | measured (Tables 4.6–4.7, Fig 4.2) — explicitly called "a stress test, not a demonstration," i.e. a substitute standing in for the missing large set | small / fast |
| BETH (host telemetry) | binary | named — loaders exist but no local data; also blocked by design constraint: leave-one-class-out requires ≥3 classes. See Ch 7 §7.3 for the fallback (use Glass as stress test) and 2027 Q2 decision point. | large — never measured |

**Gap.** No large anomaly dataset exists in any form; A.5 already says this plainly. Worth restating here because it is the cleanest single-category case of a missing large partner: the small side is not a stopgap awaiting data, it is standing in for an experiment design that has not been decided yet (Ch 7 §7.3).

### A.7.4 Clustering / structure discovery (Ch3)

| Dataset | Size | Status | Role |
|---|---|---|---|
| Synthetic batteries (circular-cities, two_moons, circles, aniso, bridged) | 120–1,500 pts | measured (Fig 2.2; Tables 3.5–3.7) | small / fast |
| NASA/UCI Statlog Shuttle | ~58,000 × 7, 7 classes | demonstrated — an exact reorder, "in about a minute," recorded with hardware and precision per §7.2's rule; fetched over the network via `ucimlrepo` (`FuzzySystemsExperiments/nasa.py`), not wired into `reproduce/manifest.py` as a repeatable table cell | large / scale — also the flagship for the Chapter 7 capstone, which notes it *has coordinates* and so does not by itself close Goal G2 |
| Psychiatric-evaluation set (private) | 135,000 × 165 | demonstrated — same single-shot standard, but **not public and not redistributable**: feature names were anonymized before the author saw them, so no conclusion is drawn from any individual feature, and the measurement is not independently reproducible by anyone else | large / scale |

**Gap, again a different shape.** Two large representatives exist, but both are demonstrations rather than measurements — single-shot, no seed spread, by design (§7.2) — and one of the two cannot be handed to anyone else at all. This category has a large *role* filled twice over and a large *measurement* filled zero times; the small/fast side is the only one with the seeded, repeatable evidence the document's own G4a standard asks for.

### A.7.5 Topological membership generation (Ch5, all synthetic)

| Dataset family | Size | Status | Role |
|---|---|---|---|
| two_gaussians, bridged_gaussians, concentric_rings, varying_density, uniform_noise | 120–160 pts | measured (Table 5.1) | small / fast |
| nested_gaussians, three_level_hierarchy, density_hierarchy | n = 96–120, single fixed realization | measured, but with no seed spread — "singly-realized" (Table 5.2, Fig 5.2) | small |
| three_clusters_tree, chain_then_ring, multi_scale_hierarchy | n = 30, 40, 39 | measured — the chapter's only coordinate-free experiment, and it scores NERFCM rather than the chapter's own selector | small |
| scalable_single_scale, scalable_many_scale, scalable_log_separated | n = 100…5,000, generator-swept | named only. §5.4's own text is explicit: "no recorded run of the [8, 4, 2] recovery exists at any size other than 96... the 'unchanged from 100 up to 5,000' sentence describes a table never written" | large / scale — the generators exist and run (`battery_hierarchical.SCALABLE`), but no output has ever been produced or registered in `reproduce/manifest.py` |

**Gap.** This is the one category where the chapter has already caught and stated its own gap in the prose: every dataset actually scored is small (≤160 points, several at a single fixed size with no seed spread), and the large/scaling regime this chapter needs to support its own invariance claim exists only as an unrun generator.

### A.7.6 Optimizer / identification benchmarks — a role reuse, not a separate pool

`reproduce/optimizers/` and Chapter 6 §6.3.5 do not introduce new datasets; they put Concrete and PhiUSIIL through a different task (antecedent-refinement search, classical-vs-construction identification at scale) and inherit both datasets' status from A.7.1 and A.7.2 above — Concrete filling the small/fast rung, PhiUSIIL the large/scale one, on the same fragile reproduction path noted there. Appendix A.3's TSP timings reuse the two_moons/circles synthetics from A.7.4 for the same reason.

### A.7.7 Non-coordinate / relational family for Goal G2 — verified loadable, not yet wired in

| Dataset | Size | Status |
|---|---|---|
| ECG5000 | 5,000 series × 140 | unwired — verified loadable via `aeon.datasets.load_classification` (network fetch), no `reproduce/` generator uses it |
| FordA | 4,921 × 500 | unwired, same verification path |
| ElectricDevices | 16,637 × 96 | unwired, same verification path |
| StarLightCurves | 9,236 × 1,024 | unwired, same verification path |
| Crop | 24,000 × 46, 24 classes | unwired — named in `NEXT_STEPS.md` as "the scale target," ≈4.6 GB as a float64 dissimilarity matrix |
| TUDataset graphs (MUTAG, PROTEINS, ENZYMES, NCI1); Duin–Pękalska dissimilarity collection | not stated | unwired, and one step earlier than the row above: `NEXT_STEPS.md` calls this family "to confirm," with verification still in progress |

**Not a gap in the same sense as the others — a different state entirely, worth distinguishing.** Both a small/mid representative (ECG5000, FordA) and a large one (Crop) have already been identified and confirmed loadable; what has not happened is a single generator or manifest entry that uses either. This is a category where the small/large pairing is *planned*, not missing — closer to A.7.5's gap than to A.7.1's or A.7.3's, but one step earlier: nothing has been run yet, including the small side.

### A.7.8 Named in the prose or in a script, with no working path today

| Dataset | Where named | What exists |
|---|---|---|
| Gas turbine set | Ch6 §6.4 | loader `FuzzySystemsExperiments/turbine.py` exists; its data directory does not |
| Wine quality (UCI) | Ch6 §6.4 | loader `FuzzySystemsExperiments/wine_red.py` exists — despite the filename, it reads `winequality-white.csv`, per its own in-code comment; that file is not in this repository |
| IoT-botnet set | Ch6 §6.4 | loader `FuzzySystemsExperiments/iot-botnet.py` exists; per CHECKLIST **B9** it "has no recoverable explicit list" for which columns need log-scaling, and no local data either |
| Wave-energy set | Ch6 §6.4 | no loader, no data, no other mention anywhere in the repository |

These four are not placed into A.7.1–A.7.5's categories because none has ever been assigned a small-or-large role in the first place; they are named as a "broadened suite" and left there. Table 7.1 already marks this tier "not started; no loaders wired," which this row-by-row pass confirms rather than contradicts.

### Summary — where the small/large pairing is actually missing

Sorted by how complete the gap is, not by chapter:

1. **Regression (A.7.1) has no large dataset in any form.** Not named, not attempted, not demonstrated.
2. **Anomaly detection (A.7.3) has no large dataset in any form**, and the reason is partly a research decision (a one-class protocol) rather than only a missing file.
3. **Topological membership generation (A.7.5) has no large *measurement*.** The generators exist; the run does not, by the chapter's own admission.
4. **Classification (A.7.2) has a named large dataset that was never measured** (RT-IOT2022) **and a measured large dataset whose reproduction path is now broken** (PhiUSIIL) — a category that looks complete until either row is checked.
5. **Clustering (A.7.4) has two large representatives, and zero large *measurements*** — both are single-shot demonstrations by design, one of them permanently non-reproducible by a third party.
6. **The non-coordinate/relational family (A.7.7) has a full small-and-large pairing already identified and confirmed loadable, run zero times.** This is the one row where "wire it in" is closer to true than "find a dataset."

None of the above is filled in here. That is the point of the exercise: A.5 already says which datasets are public and present; this section says, by category, which small/large pairing is real and which is a name.

---

*Draft — Appendix prose. A.3 (optimization engine), A.4 (feature scoring), A.5 (reproducibility), A.6 (side quests) and A.7 (dataset inventory) are written out; A.1/A.2 are inventories to be filled as the figures and the per-seed detail land. Open items in `../CHECKLIST.md`.*
