# Appendix

## A.1 Supplementary figures

Thirty-five figures are called out in the main text and stay there (sixteen before the 2026-09 expansion recorded in CHECKLIST D9, which added derivation-led figures to Chapters 2–6). Several are load-bearing: Figure 1.2 (the pipeline roadmap) orients the document, and Figure 5.2 (band discovery on the log-birth spectrum) carries Chapter 5's contribution. This appendix takes the *supplementary* material: galleries, per-fold and per-dataset repetitions, diagnostic plots a reader may want to check but need not walk through.

All are produced today, through `reproduce/figures/`, in PNG for the Markdown and EPS for the LaTeX build, against one style module; most are computed, running the same code and reading the same table CSVs as their chapters, and every figure drawn from a table names the archive it read in its own corner. **Figure 4.8** (numbered 4.3 before the expansion) was the exception until this pass: it waited on a correction-pass experiment that had not been run. That experiment ran, on Glass rather than the originally scoped RT-IOT2022 (RT-IOT2022 still is not among the datasets the harness can load), and the figure was retargeted to it rather than left waiting on a dataset that was never going to arrive — the reasoning is recorded in `reproduce/figures/registry.py`. Everything below is separate supplementary material, undrawn, tracked in `CHECKLIST.md`.

- **A.1.1 VAT / iVAT reordered-dissimilarity galleries.** Reordered images for the NASA shuttle set (58K), the psychiatric-evaluation set (135K), and the synthetic circular-cities construction; the Prim/MST block diagram (`vat_prim_mst_block_diagram_v2.svg`).
- **A.1.2 Selection and multi-scale figures.** Figures `fig1`–`fig11` from `gated-minimax-selection/outputs/`: the synthetic datasets, the minimax-transform heatmaps, the persistence curves, the membership-function plots, the ConiVAT bridge repair, the multi-scale hierarchy, the selection-method comparison.
- **A.1.3 Fuzzy-model visualizations.** Per-feature Gaussian-mixture membership plots; the rendered fuzzy trees (`plot_fuzzy_tree`); the hierarchical-mixture structure; the exported Ruspini triangular partitions.

## A.2 Extended results tables

The main text carries twenty summary tables (3.1–3.3 and 3.5–3.7 — 3.4 is retired, see A.9 — plus 4.1–4.7, 5.1–5.3, 6.1–6.3, 7.1), and this appendix carries five of its own (A.1–A.5). The seed half of Goal G4 has run, and every numbered table is already quoted at the ten-seed floor with a spread, so what the appendix owes is not a multi-seed version of each one. It is the **per-seed detail** those rows aggregate away, the splits behind each mean, which is how a reader finds the one seed in ten that carries a failure, as Chapter 6's mixture-of-experts divergence did. G4's outstanding half is hardware, which a re-run fixes and a wider table cannot. Most of these tables regenerate from `reproduce/tables/`, so the appendix version is the same data at full width; the rest are named in A.5.

- **A.2.1** Full adversarial-evaluation ARI grids and the complete stitch-ablation grid (all partitions × sizes).
- **A.2.2** The full selection-method bake-off across all synthetic datasets, and the relational-data results. The HDBSCAN\* head-to-head that belongs with them is written out in **A.8**, because its outcome decides what Chapter 5 claims rather than merely widening a table.
- **A.2.3** The broadened fuzzy-model benchmark suite (Concrete, PhiUSIIL, turbine, wave-energy [WEC_Perth and WEC_Sydney], wine, and the IoT sets) with the baseline methods. WEC_Perth: optimal Tribble R²=0.6475 (RMSE=58,688) with rank-Gaussian preprocessing, quantile bucketing, and top_n=10 feature selection, achieved in 3.93s; Random Forest baseline R²=0.80, confirming the model-data mismatch and the necessity of aggressive feature selection for Tribble on high-dimensional noisy datasets.
- **A.2.4** The three-arm reorder study behind Chapter 3 §3.3.1 (classical cubic, stage-one priority queue, stage-two compact active set) over the full grid of $N$ and both precisions, in absolute seconds with per-seed spreads. The main text normalizes, wall-clock being unportable between machines and ratios far more so; the seconds live here, with the per-$N$ detail behind Table 3.2's exponent fit. Across five independent runs on the host of record stage two is monotone in $N$ and beats stage one at every size in the grid (Chapter 3 §3.4, `PROVENANCE_MAP.md` note 11). This subsection is also the evidence base for the mergeVAT methods paper's complexity audit (§3.2).

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

That third one is **withdrawn, not merely on hold** — see A.9. Its speed envelope was Table 3.4, that table is removed from the document, and the device back ends it ran on no longer exist upstream. What follows is the record of why, kept in the appendix so the work is not simply deleted.

*Superseded text, retained because it explains the shape of the problem:* the paper was previously described as on hold until Table 3.4 is re-quoted. Such a paper would be written around a speed envelope, and this envelope is being re-measured. `PROVENANCE_MAP.md` note 15 marks Table 3.4 **drifted**: its Fuzzy C-Means row overstates the GPU by roughly an order of magnitude, because the quoted ratio compares a NumPy broadcasting implementation against one using the gram identity and two GEMMs. That is a difference of *formulation*, not of device. Run the GPU's own formulation on the CPU and most of the ratio goes away. What the paper would rest on, exactness against the serial reference, does reproduce everywhere it is claimed; the speed figures do not read as quoted. So the item stays, envelope marked under re-measurement: proposing a systems paper on a drifted table is how a drifted number gets published.

It is also distinct from the mergeVAT methods paper's complexity audit — the sequencing's cost and memory bound and the heap-versus-dense measurement — while a systems paper would cover the parallel and GPU engineering envelope. Neither should absorb the other's claim.

## A.4 Feature scoring: why the composite metric earned its keep

A refactor deleted something whose value nobody had measured. Measuring it afterwards produced the proposal's clearest single argument for consensus over a single statistic.

The Mixture-of-Gaussians construction of Chapter 4 begins by scoring each feature for how well it separates the output classes, keeping the best few and building membership functions only over those. Everything downstream (rule count, clause count, readability, training time) follows from that ranking. It is the least glamorous step in the pipeline and, it turns out, the one with the most leverage.

The scorer originally combined four measures of distributional separation: Bhattacharyya divergence, Jensen–Shannon distance, histogram overlap, and histogram correlation. It reduced them not by averaging but by taking the mean of their **arithmetic and geometric** means. That detail is the whole design. A geometric mean collapses toward zero if *any* term is near zero, so a feature scores well only when **every** measure agrees it separates the classes. The composite is a consensus rule, and consensus insures against any one metric's assumptions failing.

An upstream refactor replaced it with a choice of one metric, defaulting to Bhattacharyya. On the PhiUSIIL phishing set the effect was severe, and instructive precisely because the accuracy number understates it.

**Table A.1 — Feature ranking depends on the scorer.** Top five features on
PhiUSIIL under each rule, with normalized scores.

| Rank | Wasserstein | Bhattacharyya | Composite |
|---|---|---|---|
| 1 | HasSocialNet (1.000) | IsHTTPS (1.000) | IsHTTPS (1.000) |
| 2 | HasCopyrightInfo (0.858) | HasSocialNet (0.848) | HasSocialNet (0.975) |
| 3 | HasDescription (0.726) | NoOfQMarkInURL (0.810) | HasTitle (0.827) |
| 4 | DomainTitleMatchScore (0.544) | IsDomainIP (0.761) | NoOfCSS (0.800) |
| 5 | HasSubmitButton (0.539) | NoOfEqualsInURL (0.744) | NoOfSelfRef (0.795) |

**These tables were re-derived leak-free on 2026-08-30 and both of them moved.** Until then the ranked set included `URLSimilarityIndex` — a URL's similarity to a whitelist of *known-legitimate* URLs, the label in disguise, and the single most separating feature in the file at AUC 0.996 — together with `TLDLegitimateProb` and `URLCharProb`, two further probabilities fitted on this corpus's own labels. `repro_data.load_phiusiil` now drops all three on load (grad-school #215; `PROVENANCE_MAP.md` note 31), so nothing below trains on them. The previous version of this section rested on that feature almost entirely, and what replaced it is a different and more sober result; the argument is rewritten rather than re-quoted.

No single feature carries the signal any more. The three scorers now disagree at rank 1 — Wasserstein takes `HasSocialNet` (does the page link to a social network), the other two take `IsHTTPS` — and the disagreement is real rather than a scale artifact: `IsHTTPS` is one of nine features that are *exactly constant* across the legitimate class of this corpus, so a scorer that ranks it first is ranking a property of how PhiUSIIL was collected, not of phishing.

**Table A.2 — What the ranking costs.** Test accuracy and fit time against the
number of features retained; 20,000-row sample, ten seeds.

| Features kept | Wasserstein | Bhattacharyya | Composite |
|---:|---:|---:|---:|
| 1 | 0.5733 (0.26 s) | **0.7924** (0.09 s) | **0.7924** (0.10 s) |
| 2 | 0.5733 | 0.7924 | 0.7924 |
| 3 | 0.5733 | 0.8106 | 0.7924 |
| 4 | 0.4384 | 0.8175 | 0.7624 |
| 5 | 0.4401 | 0.8181 | 0.9186 |
| 7 | 0.6372 | 0.8191 | 0.9496 |
| 10 | 0.7204 | 0.9575 | 0.9761 |
| 15 | 0.9571 | 0.9755 | 0.9766 |
| 20 | 0.9700 (0.71 s) | 0.9799 (0.76 s) | **0.9827** (0.77 s) |

**Nothing here is readable at a single clause.** The best one-feature cell is 0.7924, and the column that used to reach 0.9967 there now reaches 0.5733. The 0.9967 *was* `URLSimilarityIndex`: one clause, thresholded on a number computed from the answer. Removing it does not degrade the result so much as reveal that there was never a one-clause result on this dataset.

**The ordering of the scorers inverts.** Wasserstein was the strongest at one feature (0.9967) and is now the weakest (0.5733), behind both others at *every* budget below fifteen; Bhattacharyya was the weakest (0.4267) and is now the strongest of the three up to four features. Wasserstein is also non-monotone leak-free — 0.5733 at three features, 0.4384 at four, 0.6372 at seven — which is what a ranking that has run out of dominant signal looks like: adding a mediocre feature can cost more than it brings. The composite is the best of the three from five features on and takes the table's highest score at twenty (0.9827).

**A portability caveat, now attached to numbers that no longer exist.** The bhattacharyya accuracies this paragraph was written about came from `reproduce/outputs/main-d0efefc/` and were shown not to be host-portable — the difference was chased, not assumed (`PROVENANCE_MAP.md` note 12; write-up `reproduce/outputs/NOTE12_THREADING.md`). Within one environment the column was bit-identical across two full sweeps, four thread counts and four BLAS kernel families; between hosts, at identical code, identical seeds and a byte-identical Table A.1 ranking, it moved up to 0.043, and only from four features on, so the cause acts on the fit rather than on the ranking or the data. Neither half of the threading-or-BLAS explanation stood: thread count was refuted outright, and the kernel-family sweep was inconclusive because its manipulation check failed. A.6 carries both sweeps, and library versions remain the leading untested candidate — numpy, scipy and scikit-learn are recorded on this host and unrecorded in the disagreeing archive. **All of that concerns the leak-era column.** The tables above are leak-free and were taken on the host of record only, so the cross-host comparison has not been repeated and the specific deltas are withdrawn. The instruction stands and is if anything more binding: **do not quote this column to four decimals across machines.**

**Which composite this is.** The scorer measured here is the restored `method="composite"`, not identical to the four-metric blend that was the original default. The distinction used to be described in terms of where each ranked `URLSimilarityIndex`; with that feature dropped, the two blends have not been compared leak-free and this row is **unmeasured**, not merely re-quoted. Nothing below depends on it.

**The mechanism that was on offer here has been withdrawn.** This paragraph used to explain Bhattacharyya's failure by its distributional assumption: it fits a Gaussian to each class and integrates the overlap, and `URLSimilarityIndex` is a bounded, spiky similarity score that a Gaussian-fit divergence mismeasures, while Wasserstein assumes nothing and finds it at once. That explanation was correct about the mechanism and wrong about which way it points, because the feature it explained was the leak. Leak-free, Bhattacharyya is the *better* scorer at small budgets and Wasserstein the worse, so a parametric assumption is not the liability this section claimed. **No mechanism is offered in its place.** Why Wasserstein's leak-free ranking underperforms is unexplained and worth measuring; asserting a story for it now would repeat the mistake this rewrite is correcting.

**The narrow conclusion is withdrawn, and it reached the library.** This section previously concluded that Wasserstein is the better default *"and is now the shipped one: the only rule reaching full accuracy on a single feature."* That recommendation was made on the strength of the one-feature row, and the one-feature row was the leak. Leak-free, Wasserstein is behind both other scorers at every budget below fifteen. **A library default was chosen on a leaked result** — the shipped `tribble-fis` scorer default is the concrete artifact of it — and re-examining that choice against this table is owed work, not a formality. The composite, not Wasserstein, is the best of the three here from five features on.

The broader conclusion concerns *interpretability*, this dissertation's central claim, which accuracy alone does not establish. Chapter 4 argues the construction yields a readable rule base, a handful of clauses over named features. Table A.2 still shows that claim is no property of the model architecture — but what it now shows is stronger and less comfortable: on this dataset, **with the leak removed, no ranking delivers a readable *and* accurate rule base at all.** Reaching 0.96 takes fifteen features under every scorer.

The evidence is still the **one-feature row**, and it now says the opposite of what it used to. It used to read 0.9967 against 0.4267 — a gap of 0.57 — and support the claim that *with a ranking that works, the model is readable and accurate at a single clause*. Leak-free it reads **0.7924 against 0.5733**, a gap of 0.22, and the columns have swapped places. What survives is that the ranking matters at a fixed budget, which is a real and useful finding. What does not survive is the single clause: there is no scorer under which one feature is enough here, so the readability this section claimed for the good ranking was, in fact, the leak's. That is the sharper version of the same warning — a change to a step the pipeline treats as preprocessing damaged interpretability more than accuracy, and an accuracy-only evaluation never surfaced it. It was the *dataset*, not a scorer, that was doing the damage, and it took two years of green sweeps to see it.

**Note 12's portability caveat has to be re-established, not carried over.** The bhattacharyya-column figures it was written about (0.4267 at one feature, 0.4527 at two, agreeing to the digit across two hosts) no longer exist. The leak-free numbers above were taken on the host of record only, so the cross-host comparison behind note 12 is void until it is repeated. Treat the instruction as still standing and the specific figures as withdrawn.

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

#### Write results before figures

For about two weeks the Chapter 5 driver crashed in a figure routine (`fig_membership`, a stale `axes[row, 0]` index left behind when a commit changed that figure's grid from `subplots(2, 2)` to `subplots(2, 1)`) *after* the numeric phase but *before* `results.json` was written, so every run discarded its numbers and left the on-disk JSON frozen at its last good date. The figures never changed and the numbers were never wrong; they were simply unreproducible until the crash was fixed, whereupon they regenerated byte-identical. A driver that writes its results artifact only after rendering every figure makes a cosmetic failure indistinguishable from a numeric one — write the results first.

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

The datasets are not uniformly available, and "public" and "reproducible from this repository" are two different claims. **Concrete, PhiUSIIL, the shuttle set, RT-IOT2022 and BETH are all public and present as of 2026-08-12**; a reader can reproduce Concrete, PhiUSIIL, the shuttle set and RT-IOT2022's results directly. BETH remains the one exception, and the reason is no longer file presence: its data load ({{dataset.beth.rows_approx}} rows), but leave-one-class-out needs at least three classes and BETH is binary, so §4.3.5's open-set study still runs on Glass ({{dataset.glass.rows}} samples, a stress test) and on RT-IOT2022 (Table 4.7b, a real demonstration) rather than on BETH as originally designed. Obtaining a working one-class protocol, not obtaining the data, would move that comparison the rest of the way — a research decision before a coding one. The {{dataset.psychiatric.rows}}-row psychiatric-evaluation set behind Chapter 3's memory results is **not** public and cannot be redistributed; its feature names were anonymized before I ever saw them, so Chapter 3 treats it purely as a scaling exercise and draws no conclusion from any individual feature. That measurement is therefore not independently reproducible, and the fix is to re-take it on a public dataset of comparable size, not to ask anyone to take it on trust.

**What Goal G2 adds, and why it arrives without baselines.** G2's non-coordinate data is identified and verified here: UCR/UEA sets under dynamic time warping through `aeon` (Crop, at 24,000 series and 24 classes, ElectricDevices, StarLightCurves, ECG5000, FordA); TUDataset graphs under a graph kernel; and the Duin–Pękalska collection, distributed *as* distance matrices and so matching Chapter 3's claim most literally. No natural competitor can run on them: Deshpande and Kumar's kd-tree and bounding-box methods need Euclidean coordinates by construction, clusiVAT samples a coordinate space, eVAT's GPU front end computes distances from points, and warped series have no fixed vector embedding, the premise of DTW. That is the seam §3.2 claims, and why "beat the baselines" is unavailable to G2: in the regime the experiment exists to demonstrate, there are no baselines to beat. Chapter 7 gives the four criteria that replace it.

#### Two implementations, cross-validating

The reorder exists in two forms, the stage-one priority-queue path (`pvat.py`) and the stage-two compact-active-set Cython kernel (`pcvat.pyx`), required to produce bit-identical orderings. That equality is itself a test: each path validates the other, asserted against the serial reference, never against permutation-invariant summaries. Chapter 3 §3.3.2 records why that matters: an earlier bug survived because the tests only checked invariant quantities.

---

## A.6 A selection of side quests

These are the investigations that did not work: things built or measured to answer a question, where what came back was negative. Each entry is the question, what was done, what came back, and what it cost or changed. Two of them changed the shipped code. Following the side quests are two failures predicted in advance, with the evidence that makes each likely, and three narrowings of scope with their reasons. Those are not side quests; they sit here because Chapter 7's statuses point here for their evidence.

#### The matrix-free reorder (Chapter 3 §3.3.2, Goal G4d)

*Can the reorder run without ever materializing the distance matrix, computing each $D_{i,j}$ on demand?* Yes, and this entry is kept because the answer reversed.

The package contained exactly one matrix-free implementation, `vat_prim_mst_seq`, and it was wrong. Checked elementwise against the serial reference it returned the seed vertex followed by every other vertex in ascending index order — chance-level agreement — and nothing in the package called it. The cause was a distance helper typed for a scalar index and handed an array of candidates, so the reduction collapsed to one scalar, every candidate received the same key, and the heap popped in index order. It was removed from the public API at `tribble-cluster e3c27e6`, which is the commit §3.4's verification permalinks still point at, and this appendix and Table 3.3 recorded the negative result rather than deleting it.

**Upstream repaired it at `c9be437` (2026-08-10), with a regression test, and the pin this harness runs — `635ed6e` — contains that repair.** The two facts that matter and are easy to conflate: `e3c27e6` genuinely does not have the fix, and it is genuinely not the pinned commit any more.

Re-measured against Chapter 7's own decision rule for **G4d**, which asked for exactly this. The ordering is elementwise identical to the serial reference at $N = 1{,}000$, $2{,}000$ and $5{,}000$, ten seeds each: $1.000 \pm 0.000$ against chance levels of $0.001$, $0.0005$ and $0.0002$, and not one run shows the old ascending-index signature. Peak working set, measured per arm in a fresh process because the compiled kernel allocates outside Python's allocator, is flat — 64.7, 64.9, 65.0 and 64.8 MB at $N = 2{,}000$, $4{,}000$, $8{,}000$ and $12{,}000$ — while the materialising arm's peak climbs 193.6 MB → 4.67 GB over the same ladder. Across it the implied matrix grows thirty-six-fold, the materialising peak twenty-four-fold, and the matrix-free peak by a factor of 1.00. The wall clock goes the same way: the matrix-free arm runs at 0.14–0.22× the materialising arm's time, both starting from samples, so G4d's second threshold — *fails if more than an order of magnitude slower* — is passed in the opposite direction from the one it anticipated.

Two honest limits. At float32 the ordering is $0.9996 \pm 0.0012$ rather than exact, the tie-breaking §3.2 describes rather than an error. And the 155,000-point claim is an **extrapolation from a ratio stable to 1.62× across an 8× change in $N$**, not a measurement there: the in-place arm cannot be run at that size on this host, its matrix being 96 GB at float32. Regenerate with `reproduce/experiments/check_matrix_free_reorder.py`, whose pass/fail outcomes are registered in its docstring before it runs.

#### The hybrid bucket arm, and the axis the study forgot to vary (Chapter 4 §4.3.2)

*Is there a third option that dominates both, equal-frequency boundaries with the two extreme bucket centroids pinned to the observed min and max?* `partition_output` shipped exactly that for the life of this work, and the sweep ran all three arms to find out. There is a third option and it is the worst of the three, which took four passes to establish because each of the first three measured it in the one regime where it cannot be seen.

The sweep ran first and second order only. Across those, 126 cells over three schemes and six configurations, the largest separation between the hybrid and pure quantile is 0.004 against seed deviations of ±0.018 to ±0.027, and 0.004 is also the bound across every archive under `reproduce/outputs/`, at five seeds and ten alike. The two arms are never *identical*: they differ in all six configurations on at least one of four metrics. What they never differ by is more than noise. So three studies concluded no scheme could be recommended, and §4.3.2 said so.

The missing axis was **zeroth order**, and it was one line of the generator away the whole time. `solve_tsk_consequents` holds the first and last rules' constant terms at the centroids it is handed, exactly, as an equality constraint. At zeroth order that constant is a rule's entire output, so the three arms span 0.828: uniform 0.394 ± 0.065, pure quantile 0.242 ± 0.070, pinned quantile −0.434 ± 0.241, with the ordering holding at three, four and six buckets. Reading the solved coefficients shows the mechanism directly. Handed `[0.0, 0.4038, 1.0]`, the solve returns `[0.0, 0.411, 1.0]`, ends untouched, so the bottom rule emits the target's global minimum for a bucket of 344 points whose mean is 0.195. At first and second order the same ends stay pinned while the free interior constant runs to $-0.38$ and $-1.19$, outside the target's own range: the solve spending intercepts as free parameters and paying the bias back through the linear terms. That compensation is the whole reason the pin looked inert.

Two lessons, and the second is the expensive one. A defect can be real, load-bearing and invisible to every aggregate metric in the study designed to find it. And an experiment can be run three times, at ten seeds, across 126 cells, and still not vary the axis its own question depends on. The failure was not too few seeds or too few configurations but a regime never entered. What made it visible in the end was reading the coefficients instead of the scores.

`pin_extremes` defaults to true and no generator here overrides it, so the two pinned ends survive the closed-form solve — it re-derives only the *free* bucket means. The coefficients above show this, and the hybrid's differing scores already implied it. `CHECKLIST.md` §F (H5) carries the working record of this saga.

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
- **G8's construction.** It spends interpretability, the dissertation's own thesis, making it the one goal whose success would weaken the argument around it, a tension §6.2 flags. And the §5.3.5 disjunct counter that decides whether it is worth having has never returned a value other than one, in either mode on any recorded run.

---

## A.7 Benchmark dataset inventory, by category

A.5 above answers *is this dataset public, and is it present in this repository.* This section answers a different question: for each task category the document measures something on, does a **small, fast** dataset and a **large, at-scale** dataset both exist — the pairing that lets a method be iterated on cheaply and then shown not to fall over at size. Availability and role-completeness are not the same claim, and keeping them separate matters here: a dataset can be public and present and still leave a category with no large partner, and a category can name a large dataset that has never once been measured. Compiled by reading every prose chapter, `CHECKLIST.md`, `NEXT_STEPS.md`, and the loader code under `reproduce/tables/`, `reproduce/optimizers/`, and `FuzzySystemsExperiments/`.

**Status vocabulary**, reusing A.5's own distinction rather than inventing a second one: *measured* (a seeded, repeatable generator produced the number cited), *demonstrated* (a single-shot run at scale, recorded with hardware and footprint instead of a spread, per §7.2's rule), *named* (appears in the prose or in a loader script, but no run of any kind exists), and one state A.5 does not need but this table does — *unwired* (a real dataset, verified loadable, that no generator or manifest entry uses yet).

### A.7.1 Regression

| Dataset | Size | Status | Role |
|---|---|---|---|
| UCI Concrete Compressive Strength | {{dataset.concrete.shape}} | measured (Ch1, Ch2, Ch4 §4.3.2–4.4, Ch6, every table generator that touches regression) | small / fast — the *only* regression benchmark in the document, at every consequent order |
| Diabetes (sklearn) | {{dataset.diabetes.shape}} | measured (Table 4.8, dedup sweep only) | small / fast — chosen for the tolerance sweep, not a modeling flagship |
| California Housing (sklearn, canonical) | {{dataset.california_housing.shape}} | measured, ten seeds (`table_a7_regression_scale.py`, 2026-08-12): RF $R^2 = 0.809 \pm 0.008$; flat MoG $0.631 \pm 0.020$ | large / scale — first large regression partner |
| Superconductivity (UCI id 464) | {{dataset.superconductivity.shape}}, decorrelated | measured, ten seeds, same generator: RF $R^2 = 0.923 \pm 0.004$; flat MoG and HME **unstable** ($-0.261 \pm 1.431$, $-0.766 \pm 2.411$) | large / scale — same generator, second dataset |
| N-CMAPSS DS02 turbofan RUL | 6.5M raw rows → aggregated | **demonstrated** (one run on the dataset's own fixed split, `FuzzySystemsExperiments/cmapss_all_datasets.py`, not the harness): DS02 per-sample RMSE **7.23** on real sensors only, in line with public-file CNN 7.22 and beating MLP 8.34 (both on a 20-channel input this pipeline deliberately excludes); real-sensors-only does *not* match the virtual-channel (T40/P30) result, which reaches ~6.5 — an earlier claim that it did is corrected (**C18**) | large / scale — physics/prognostics domain, §4.4.1 / Table 4.10; fixed-split benchmark, so the variance study reseeds the train subsample not the split (**C14**, future PR); data not redistributable (~28 GB) |

**Gap, narrowed but not closed.** Concrete still carries the entire *small* regression story. A large partner now exists at this document's own ten-seed floor: `reproduce/tables/table_a7_regression_scale.py` (2026-08-12) measures California Housing (`sklearn.fetch_california_housing()`, canonical, {{dataset.california_housing.shape}}) and Superconductivity (UCI id 464, direct download, {{dataset.superconductivity.shape}}, decorrelated via `sklearn.cluster.FeatureAgglomeration` before every model — raw features break the flat MoG's closed-form solve, per the pilot's own finding) across the flat/fuzzy-tree/HME/CART/Random Forest family. Random Forest wins both cleanly (California Housing $R^2 = 0.809 \pm 0.008$; Superconductivity $R^2 = 0.923 \pm 0.004$). The more interesting finding is a caveat, not a win: the flat MoG and HME mixture are **wildly unstable on Superconductivity even after decorrelation** — $R^2 = -0.261 \pm 1.431$ and $-0.766 \pm 2.411$ respectively, occasionally catastrophically negative — echoing the seed-9 HME divergence Chapter 6's `table_concrete_reconciliation` already documents on Concrete. So the *pairing* exists now; what it demonstrates is that the flat/HME instability generalizes to a second, larger, unrelated dataset, which is itself worth a sentence in Chapter 6, not that this construction scales cleanly to large regression data. Superseded: `reproduce/regression_scale/RESULTS_2026-08-05.md`'s single-seed pilot (CHECKLIST **C13**), which this generator formalizes at ten seeds and canonical sourcing for both datasets.

The one row that reads *well* for the construction is also the one not yet under the harness: **N-CMAPSS turbofan RUL** (§4.4.1, Table 4.10), a large-scale physics/prognostics regression on which the answer-first construction beats the published deep-learning baselines on their own input set in about a second. That is exactly the kind of result the *measured* rows do not supply — so it is precisely the one to be most careful with, and it is recorded as *demonstrated*, one run on the dataset's own fixed split, on non-redistributable data, not *measured*. Note the reproducibility axis differs from every other row here: N-CMAPSS ships a fixed train/test split (the split the baselines are scored on), so the variance study that would promote it re-seeds the **training subsample**, not the split (CHECKLIST **C14**, a future PR). Doing so would give the large-regression category a genuine at-scale win rather than a pair of RF-loses-narrowly rows plus an instability caveat; until then it is a promising single shot and labelled as one.

### A.7.2 Classification

| Dataset | Size | Status | Role |
|---|---|---|---|
| Glass (UCI) | {{dataset.glass.shape_full}} | measured — also the anomaly substitute (A.7.3) and the Table 4.8/4.9 dedup and correction-pass testbed | small / fast |
| Wine, Breast Cancer, Digits (sklearn) | {{dataset.wine.shape}} / {{dataset.breast_cancer.shape}} / {{dataset.digits.shape}} | measured (Table 4.8, dedup sweep only) | small / fast |
| PhiUSIIL phishing URL | {{dataset.phiusiil.shape}}, binary | re-derived leak-free 2026-08-30 (Table 4.5: **0.440 ± 0.181** acc, 0.13 ± 0.02 s; it read 0.997 ± 0.001 while `URLSimilarityIndex` was in the feature set) — **but no longer reproducible from a clean checkout.** The repo loader's bundled copy lived at `tribble-fis/gaussian_mixture/phishing_data/`, and `gaussian_mixture/` was deleted upstream (commit `8484fd6`, per `_fuzzy_models.py`'s own comment); `data/` in this repository holds only `Concrete_Data.csv`. A fresh run falls through to a `ucimlrepo` fetch that returns a *different* feature set, which the loader's own comment flags as producing results "not comparable" to every number quoted from it | large / scale — the one role currently filled, on a fragile path |
| RT-IOT2022 | {{dataset.rt_iot2022.shape_full}} | **in the repository as of 2026-08-12, both roles now measured.** *Open-set* (Table 4.7b, ten seeds, re-measured 2026-08-27 after a leaky feature was removed): the complement rule loses to Isolation Forest at this scale (+0.366 vs +0.535 Youden's $J$). *Classification/timing* (Table 4.4, ten seeds, `table_4_1_mog_baselines.py`): MoG trains in $4.24 \pm 0.68$ s at $0.927 \pm 0.002$ accuracy against Random Forest's $0.998 \pm 0.000$ — the same speed-not-superiority shape as the PhiUSIIL row | large / scale — both roles filled and measured; both favor the reference baseline on accuracy |

**Gap closed, in the "both roles measured" sense — not in the "this work wins" sense.** RT-IOT2022 is present and both of its named claims now have real ten-seed-or-better numbers behind them, and neither favors the construction on accuracy: Random Forest and Isolation Forest both beat the MoG-based arm on their respective tasks. What the numbers do support is the *speed and structural* half of each claim — twelve rules in about four seconds, no second model needed for the open-set rule — which is what Chapters 1, 4 and 8 actually claim; "no fuzzy baseline exists to be faster than" (Goal **C1**) is still the open half of that argument. PhiUSIIL's reproduction path is unchanged — its measured numbers are real but sit on a path this pass found to be broken. A.5 states "Concrete, PhiUSIIL and the shuttle set are public and present, a reader can reproduce those results directly" — that sentence is still not accurate for PhiUSIIL and is worth revisiting there.

### A.7.3 Anomaly / open-set detection

| Dataset | Size | Status | Role |
|---|---|---|---|
| Glass, leave-one-class-out | {{dataset.glass.shape_full}} | measured (Tables 4.6–4.7, Fig 4.6) — explicitly called "a stress test, not a demonstration," i.e. a substitute standing in for the missing large set | small / fast |
| RT-IOT2022, leave-one-class-out | {{dataset.rt_iot2022.shape_full}} | **measured, re-measured 2026-08-27** (Table 4.7b, ten seeds): the complement rule loses to Isolation Forest at this scale (+0.366 vs +0.535 Youden's $J$) | large / scale — the missing partner, now filled, unfavorably |
| BETH (host telemetry) | binary, {{dataset.beth.rows_approx}} rows | data present locally since 2026-08-12 (`load_beth()` loads train/val/test splits) but blocked by a design constraint, not a missing file: leave-one-class-out requires ≥3 classes and BETH is binary. See Ch 7 §7.3 for the 2027 Q2 decision point. | large — data available, never measured; the *intended* large partner, superseded for now by RT-IOT2022 |

**Gap, closed by RT-IOT2022 rather than by BETH.** This category no longer lacks a large anomaly measurement — Table 4.7b fills exactly the role this section used to say nothing filled — but the result is not favorable, and BETH, the dataset Chapter 4 originally designed this experiment around, is still blocked on the same one-class-protocol decision as before. The small side (Glass) is no longer standing in for a missing large set; it is now one of two, alongside RT-IOT2022, and BETH remains the one genuinely open item in this category.

### A.7.4 Clustering / structure discovery (Ch3)

| Dataset | Size | Status | Role |
|---|---|---|---|
| Synthetic batteries (circular-cities, two_moons, circles, aniso, bridged) | 120–1,500 pts | measured (Fig 2.2; Tables 3.5–3.7) | small / fast |
| NASA/UCI Statlog Shuttle | {{dataset.shuttle.shape_full}} | demonstrated — an exact reorder, "in about a minute," recorded with hardware and precision per §7.2's rule; fetched over the network via `ucimlrepo` (`FuzzySystemsExperiments/nasa.py`), not wired into `reproduce/manifest.py` as a repeatable table cell | large / scale — also the flagship for the Chapter 7 capstone, which notes it *has coordinates* and so does not by itself close Goal G2 |
| Psychiatric-evaluation set (private) | {{dataset.psychiatric.shape}} | demonstrated — same single-shot standard, but **not public and not redistributable**: feature names were anonymized before the author saw them, so no conclusion is drawn from any individual feature, and the measurement is not independently reproducible by anyone else | large / scale |

**Gap, again a different shape.** Two large representatives exist, but both are demonstrations rather than measurements — single-shot, no seed spread, by design (§7.2) — and one of the two cannot be handed to anyone else at all. This category has a large *role* filled twice over and a large *measurement* filled zero times; the small/fast side is the only one with the seeded, repeatable evidence the document's own G4a standard asks for.

### A.7.5 Topological membership generation (Ch5, all synthetic)

| Dataset family | Size | Status | Role |
|---|---|---|---|
| two_gaussians, bridged_gaussians, concentric_rings, varying_density, uniform_noise | 120–160 pts | measured (Table 5.1) | small / fast |
| nested_gaussians, three_level_hierarchy, density_hierarchy | n = 96–120, single fixed realization | measured, but with no seed spread — "singly-realized" (Table 5.2, Fig 5.2) | small |
| three_clusters_tree, chain_then_ring, multi_scale_hierarchy | n = 30, 40, 45 | measured — the chapter's only coordinate-free experiment; NERFCM plus, since the #160 ground-truth fix, the flat and multi-scale selectors (`run_nonmetric.py` E4) | small |
| scalable_single_scale, scalable_many_scale, scalable_log_separated | n = 100…5,000, generator-swept | **measured, ten seeds, 2026-08-12** (`table_5_4_ch5_g1_scaling.py`, registered in `reproduce/manifest.py`). `many_scale`: [8,4,2] at ARI 1.00, every seed, every $n$. `single_scale`: granularity mode agrees only 5–7/10 seeds — less stable than the single n=96 run implied. `log_separated`: gradual ARI climb 0.73→0.99 from $n=100$ to $n\ge2{,}000$, not a sharp threshold | large / scale — measured against a flat set-cover baseline, not yet against the one-pass generator (phase five, still unbuilt) |

**Gap, narrowed.** The scaling regime this chapter needed to support its own invariance claim is no longer an unrun generator — it is measured, at ten seeds, with a genuine mixed result: `many_scale` confirms cleanly, `single_scale` turns out less stable than the single-seed study suggested, and `log_separated` shows a gradual rather than sharp size-dependence. What is still missing is timing for the full pipeline at these sizes and the one-pass construction itself (Goal G1's phase five), so this closes the *measurement* gap the chapter's own prose flagged, not the *construction* gap Goal G1 is ultimately about.

### A.7.6 Optimizer / identification benchmarks — a role reuse, not a separate pool

`reproduce/optimizers/` and Chapter 6 §6.3.5 do not introduce new datasets; they put Concrete and PhiUSIIL through a different task (antecedent-refinement search, classical-vs-construction identification at scale) and inherit both datasets' status from A.7.1 and A.7.2 above — Concrete filling the small/fast rung, PhiUSIIL the large/scale one, on the same fragile reproduction path noted there. Appendix A.3's TSP timings reuse the two_moons/circles synthetics from A.7.4 for the same reason.

### A.7.7 Non-coordinate / relational family for Goal G2 — three of six now measured

| Dataset | Size | Status |
|---|---|---|
| ECG5000 | {{dataset.ecg5000.rows}} series × {{dataset.ecg5000.features}} | **measured, 2026-08-12** (`table_3_7_g2_dtw_nonmetric.py`, registered in `reproduce/manifest.py`): exactness 1.000 (N≤1024, 10 seeds), triangle-inequality violations 20.9%. Downstream comparison also run: set-cover beats NERFCM-given-$k$ by 0.122 ARI (0.715 vs 0.593) |
| FordA | {{dataset.forda.shape}} | **measured, same generator**: exactness 1.000, violations 0.4% (below the synthetic proxy). Downstream: every method scores ≈0 ARI (k_true=2 not recoverable from DTW dissimilarities by NERFCM, the set-cover, single-linkage, or beta-plateau) |
| Crop | {{dataset.crop.shape_full}} | **measured, same generator, the scale target**: exactness 1.000, violations 23.6%, matrix build 1,597s + reorder 4.7s. Downstream: NERFCM 0.029 ARI, set-cover 0.064 — both weak in absolute terms, technically within 0.05 of each other |
| ElectricDevices | {{dataset.electric_devices.shape}} | unwired — verified loadable via `aeon.datasets.load_classification`, no run attempted this pass |
| StarLightCurves | {{dataset.starlight_curves.shape}} | unwired, same status |
| TUDataset graphs (MUTAG, PROTEINS, ENZYMES, NCI1); Duin–Pękalska dissimilarity collection | not stated | unwired, and one step earlier: verification still in progress |

**No longer a gap of the "nothing has been run" kind — it is now a partial-evidence gap, and an honest one.** Exactness holds at 1.000 on every real DTW dataset tested, closing that half of Goal G2's decision rule. The downstream-usefulness half is not closed: the decision rule needs the set-cover within 0.05 ARI of NERFCM-given-$k$ on at least three of the five DTW sets, and while three sets are now measured, only two show the criterion literally met, and both of those passes are low-information — Crop because both methods are weak, FordA because every method tested is at chance level. ECG5000, the one dataset with real recoverable structure, fails the criterion because the set-cover *outperforms* NERFCM by more than the tolerance, not because it underperforms. See §3.4's Table 3.7 and §7.2's Goal G2 entry for the full reading.

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

1. **Regression (A.7.1) now has a large dataset, measured at ten seeds (2026-08-12)** — California Housing and Superconductivity, Random Forest winning both, flat MoG/HME unstable on the second. Closed as a *pairing*; the instability finding is a new small caveat, not a gap.
2. **Anomaly detection (A.7.3) has no large dataset in any form**, and the reason is partly a research decision (a one-class protocol) rather than only a missing file. Unchanged.
3. **Topological membership generation (A.7.5) now has a large measurement (2026-08-12)**, ten seeds across three fixed-structure families, with a genuine mixed result (`many_scale` solid, `single_scale` less stable than believed). What is still missing is the one-pass construction itself, not the scaling measurement.
4. **Classification (A.7.2)'s named large dataset is now measured on both roles, and neither favors this work on accuracy** (RT-IOT2022: Table 4.7b's open-set complement rule loses to Isolation Forest; Table 4.4's classification/timing row loses to Random Forest by seven points, winning on speed and structure instead). **The other measured large dataset's reproduction path is still broken** (PhiUSIIL).
5. **Clustering (A.7.4) has two large representatives, and zero large *measurements*** — both are single-shot demonstrations by design, one of them permanently non-reproducible by a third party. Unchanged.
6. **The non-coordinate/relational family (A.7.7) now has three of its identified datasets measured (2026-08-12), exactness 1.000 on every one, up to the named 24,000-point scale target.** The downstream-usefulness half of Goal G2's decision rule is not yet met on the evidence gathered — a real, disclosed partial result, not the "run zero times" state this row described before.

Two categories (A.7.3, A.7.4) are unchanged from the earlier pass. Four (A.7.1, A.7.2, A.7.5, A.7.7) moved, three of them substantially, over one session (2026-08-11/12) — see `reproduce/outputs/SESSION_FINDINGS_2026-08-12.md` for the full run log, every real number, and what was deliberately not attempted. That is still the point of this section: A.5 says which datasets are public and present; this section says, by category, which small/large pairing is real, which is a name, and — now, for four of six categories — what the real pairing actually showed.

## A.8 The HDBSCAN\* head-to-head (Chapter 5)

Chapter 5's selection machinery — the persistence gate of §5.3.2 that makes $k$ an
output, and the band discovery of §5.3.3 that returns a stack of partitions — is
described there as machinery and is not claimed as a contribution. This section is
why. It also discharges the "required before claiming" item raised by the Chapter 5
prior-art review (`PRIOR_ART_CH5.md` §3), which asked specifically for HDBSCAN
`leaf` extraction and a `dbscan_clustering(eps)` sweep on the nested [8, 4, 2]
synthetic.

Generator: `gated-minimax-selection/run_hdbscan_baselines.py`, registered in
`reproduce/manifest.py` as `table-5-5-7-ch5-hdbscan-baselines`. Seventeen
dissimilarity matrices (the flat battery, the nested battery, five non-metric
families, three shortest-path matrices), twelve HDBSCAN\* configurations each
(`min_samples` ∈ {1, 5} × extraction ∈ {eom, leaf} × `min_cluster_size` ∈
{3, 5, 10}), ten seeds. Findings and caveats:
`gated-minimax-selection/notes/HDBSCAN_BASELINES.md`. It needs the `hdbscan`
contrib package, which is in no repository environment — scikit-learn's `HDBSCAN`
offers eom and leaf but no cut-distance accessor, and the eps sweep is half the
point.

The comparison is run at $m_{\mathrm{pts}} = 1$, where mutual reachability equals
the input dissimilarity and HDBSCAN\* *is* single-linkage on it
[@campello2015hdbscan, Corollary 3.5]. Both sides therefore consume the **identical
hierarchy**, and the only thing under test is the extraction rule.

Two design points matter for reading the tables. HDBSCAN\*'s quality on this
battery is a strong function of `min_cluster_size` — on concentric rings
excess-of-mass scores 0.061 at `min_cluster_size` = 3 and 0.863 at 10 — so the
baseline is swept over that parameter rather than run at a default; a mechanism
inferred from a gap at any single setting would be an artifact of the setting. And
because the gate runs at one fixed `gap_sigma` everywhere, the baseline is scored at
one fixed configuration too, with the per-dataset choice reported separately as the
generous case.

**Table A.3 — one fixed setting per method, ten seeds.** Twelve of the seventeen
datasets have a flat ground truth. Each replicate gets its own battery mean and
those are then averaged, so the deviation is across whole batteries rather than
across datasets.

| method | setting | mean ARI | sd | min | max |
|---|---|---:|---:|---:|---:|
| gated set-cover (Ch. 5) | `gap_sigma` = 2.0, identical everywhere | 0.835 | 0.037 | 0.794 | 0.887 |
| HDBSCAN\* excess-of-mass | $m_{\mathrm{pts}}$ = 5, `min_cluster_size` = 5 | 0.817 | 0.019 | 0.790 | 0.855 |
| HDBSCAN\* excess-of-mass | $m_{\mathrm{pts}}$ = 1, `min_cluster_size` = 10 | 0.805 | 0.037 | 0.772 | 0.879 |
| HDBSCAN\* excess-of-mass | $m_{\mathrm{pts}}$ = 1, `min_cluster_size` = 5 | 0.798 | 0.014 | 0.771 | 0.816 |
| HDBSCAN\* leaf | $m_{\mathrm{pts}}$ = 5, `min_cluster_size` = 10 | 0.713 | 0.024 | 0.674 | 0.747 |

**+0.018 at roughly half a standard deviation, with the ranges overlapping almost
entirely. There is no accuracy claim here.** A single seed would have suggested
otherwise: at the generators' default seeds the gate scores 0.887 against the best
baseline configuration's 0.820, a gap of +0.067 — and 0.887 is the gate's *best of
ten*. This is what the ten-seed floor (A.5) exists to catch, and it is why Chapter 5
reports the machinery without advancing it.

Allowing the baseline a per-dataset choice of both parameters makes it worse than a
wash. Over ten replicates and twelve datasets, scored against the best of
HDBSCAN\*'s twelve configurations on each dataset (0.02 ARI to count as a verdict),
the flat gate wins 7, loses 29 and ties 84; the band selector wins 8, loses 19 and
ties 93.

**Table A.4 — per dataset, against a baseline tuned on that dataset.** Mean ± sd
over ten seeds.

| Dataset | gated set-cover | HDBSCAN\* tuned | delta |
|---|---|---|---:|
| multi_scale_hierarchy | 0.552 ± 0.015 | 1.000 ± 0.000 | −0.448 |
| cosine_topics | 0.423 ± 0.387 | 0.682 ± 0.183 | −0.259 |
| graph_communities | 0.095 ± 0.156 | 0.202 ± 0.090 | −0.107 |
| bridged_gaussians | 0.978 ± 0.040 | 0.987 ± 0.017 | −0.009 |
| varying_density | 0.976 ± 0.042 | 0.983 ± 0.027 | −0.007 |
| two_gaussians, dtw_traces, edit_strings, hamming_categorical, three_clusters_tree, chain_then_ring | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.000 |
| **concentric_rings** | **1.000 ± 0.000** | 0.873 ± 0.175 | **+0.127** |

Three things read off that table. **The one clear win is `concentric_rings`** — the
non-convex case Chapter 5's motivation rests on — at 1.000 with *zero* variance
across ten seeds against a tuned baseline's 0.873 ± 0.175, exactly stable where the
baseline is strongly seed-sensitive. **`multi_scale_hierarchy`'s −0.448 is a
flat-versus-multiscale artifact rather than a defeat**, since that dataset's flat
truth is its fine level of six sub-clusters and the band selector recovers bands
[6, 3] at ARI 1.000. **And `cosine_topics` at 0.423 ± 0.387 is a real weakness**,
carried forward into Chapter 5 §5.6 rather than averaged away: the gate is unstable
on cosine dissimilarities, and a single seed puts it at 0.803, near the top of that
range.

What the comparison does establish is a parameter-sensitivity contrast, and the
reason it is a cost rather than a knob is that parameter choice dominates seed
noise. Across configurations the battery mean ranges 0.713–0.817; across seeds at
any fixed configuration the standard deviation is 0.014–0.037. Choosing
`min_cluster_size` badly costs an order of magnitude more than the seed does, and on
individual datasets the swing reaches 0.802 ARI on concentric rings and 1.000 on
`three_clusters_tree`, where $n$ = 30 makes `min_cluster_size` = 10 a tenth of the
data.

**Table A.5 — nested structure: whether a baseline can return the whole hierarchy.**
Per-level ARI at the default seeds, finest level first, with band-recovery stability
over ten seeds in the last two columns.

| Dataset | truth granularities | bands found | per level | HDBSCAN\* leaf per level | eps-sweep oracle | exact recovery, 10 seeds | distinct vectors |
|---|---|---|---|---|---|---:|---:|
| nested_gaussians | [6, 2] | [6, 2] | 1.000, 1.000 | 1.000, 0.324 | 1.000, 1.000 | 100% | 1 |
| three_level_hierarchy | [8, 4, 2] | [8, 4, 2] | 1.000, 1.000, 1.000 | 1.000, 0.581, 0.236 | 1.000, 1.000, 1.000 | 90% | 2 |
| density_hierarchy | [4, 2] | [4, 2] | 1.000, 1.000 | 1.000, 0.492 | 1.000, 1.000 | 70% | 4 |
| relational_nested_hierarchy | [6, 3] | [6, 3] | 1.000, 1.000 | 1.000, 1.000 | 1.000, 1.000 | 100% | 1 |

Two findings, pointing opposite ways. Against a **single-output** extractor the
structural argument is clean: leaf and excess-of-mass both lock onto the finest
level and then score 0.24–0.58 on the coarser ones, because one partition cannot be
two granularities at once. Against a **cut-distance sweep it does not hold.**
Sweeping `dbscan_clustering(eps)` over 400 cut heights on `three_level_hierarchy`
yields only **seven** distinct partitions, and three of them are exactly $k$ = 8,
$k$ = 4 and $k$ = 2 at ARI 1.000 each:

| eps | k | fine | medium | coarse |
|---:|---:|---:|---:|---:|
| 0.032 | 0 | 0.000 | 0.000 | 0.000 |
| 0.553 | 8 | 0.948 | 0.542 | 0.216 |
| 1.074 | 8 | **1.000** | 0.581 | 0.236 |
| 3.680 | 7 | 0.862 | 0.702 | 0.300 |
| 4.201 | 4 | 0.581 | **1.000** | 0.492 |
| 27.129 | 2 | 0.236 | 0.492 | **1.000** |
| 193.883 | 1 | 0.000 | 0.000 | 0.000 |

The candidate set is small enough to read the three real scales off by eye, which is
why §5.3.3 describes band discovery as automatic selection among those candidates
and not as recovery of the hierarchy.

The last two columns of Table A.5 are the qualification a single-seed run cannot
supply. The [8, 4, 2] result holds on nine seeds of ten; `density_hierarchy` holds
on only seven, producing four distinct granularity vectors — consistent with the
ten-seed scaling study in Chapter 5 §5.4, where `single_scale`'s modal granularity
agreed in only 5–7 of 10 seeds. Band recovery is not the deterministic property
Table 5.2 makes it look, and Chapter 5 §5.6 says so.

One further result closes off an argument rather than supporting a claim.
$m_{\mathrm{pts}} = 5$ engages the $k$NN core distance and mutual-reachability
machinery, and it **ran without error on all seventeen matrices**, every non-metric
family included, so a density estimate is computable on a non-metric $D^*$; the
framing "no density estimator is available here" is false and appears nowhere in
Chapter 5. It is merely unreliable there: switching it on costs `graph_communities`
0.253 → 0.011 and `cosine_topics` 0.903 → 0.458, while *helping* on the coordinate
sets (concentric rings 0.863 → 1.000), and $m_{\mathrm{pts}} = 5$ is in fact the
best single configuration across the battery as a whole. A narrow empirical
observation on a few datasets, not a methodological reason.

**Caveats.** `min_cluster_size` is swept over only {3, 5, 10} and `min_samples` over
only {1, 5}. All seventeen matrices are small, $n$ = 30–160. HDBSCAN's noise flag is
scored as a label of its own, with the kinder `ari_noise_excluded` also recorded in
the JSON. Contrib `hdbscan` only, not cross-checked against scikit-learn's
implementation.

---

*Draft — Appendix prose. A.3 (optimization engine), A.4 (feature scoring), A.5 (reproducibility), A.6 (side quests), A.7 (dataset inventory) and A.8 (the Chapter 5 HDBSCAN\* head-to-head) are written out; A.1/A.2 are inventories to be filled as the figures and the per-seed detail land. Open items in `../CHECKLIST.md`.*

## A.9 The GPU / Borůvka path, and why it is not in the body

Removed from Chapter 3 on 2026-08-30. This section is a **record of work done, not a
result**, and nothing in the body cites a number from it.

**What was built.** A device-resident Borůvka front end for mergeVAT: pairwise
distances, MST construction and the VAT ordering all executed on the card, so the
data does not shuttle back to the host between stages. It rests on the same property
§3.3.3 opens with — the ordering depends only on the MST, so any MST builder yields
the same answer — and it did: the device ordering matched serial VAT elementwise at
double precision, at every size and seed tested, with a VAT order match of 1.000. The
one arm that was not 1.000 was a 48,000-point float32 demonstration at 0.99992, four
positions in forty-eight thousand, with the Prim totals agreeing to every digit
printed. That is minimum-spanning-tree tie-breaking, not error.

**Why it left the body.** Three reasons, and only the third is fatal on its own.

*The headline number measured the wrong thing.* Table 3.4 had claimed Fuzzy C-Means
ran thirty to fifty times faster on the device than on the 32-core CPU. The device
kernel used the gram identity and two GEMMs; the CPU arm it was compared against
computed distances by NumPy broadcasting. Held to a matched formulation the device
win was **1.2–3.7×**. The same split, smaller, applied to the VAT front end:
2.28×–5.02× at matched ordering-only work against 5.52×–12.13× when the two arms were
allowed to do different amounts of it. A table whose largest entry is a property of
the baseline's implementation is not a speed table, and correcting it left an
envelope too modest to carry a chapter section.

*The one genuinely interesting result was uninterpretable without hardware I do not
have.* Pairwise distances **lose** to the CPU — 0.29–0.64× — at low dimension or in
double precision, because a consumer card's double-precision throughput is a small
fraction of its single-precision throughput. Whether that is a fact about this card
or about the algorithm is precisely what a full-rate-FP64 datacenter card would
decide, and no such card was available. A negative result nobody can attribute is not
publishable and is barely quotable.

*The backend was removed upstream.* `tribble-clustering` deleted its CuPy back ends
and its `[gpu]` extra in `1ec9667` (2026-08-30). `tribbleclustering.gpu` no longer
exists, so `reproduce/tables/table_3_4_gpu_speedups.py` had nothing left to import
and was deleted with the table. The measurement could not be repeated today even
with the card that would have settled the second point.

**What removing it costs, and what it buys.** It costs the exactness demonstration —
a device path reproducing serial VAT bit-for-bit is a real and slightly surprising
result, and it now survives only as this paragraph. It buys three things. The chapter
no longer contains a speed table whose provenance map entry read *drifted*. The
prior-art collision with Parveen and Sreevalsan-Nair's pVAT — a GPU VAT that also
swaps Prim for Borůvka, and the one place this work overlapped theirs directly —
disappears, because the overlapping component is gone. And Chapter 3's argument is
unaffected: the priority-queue reorder, the in-place memory scheme, the
divide-and-conquer stitch and the non-metric extension are all CPU results and never
depended on the device path.

**Reviving it.** Goal **G4c** carries this. It was previously gated on hardware
alone; it now has a software precondition first — the device kernels would have to
return to `tribble-clustering`, or be rebuilt — and only then the datacenter card
that decides the double-precision question. The research scripts remain in
`ClusteringExperiments/` (`boruvka_gpu.py`, `gpu_vat.py`, `boruvka_dgx_spark.py` and
the VAT-TSP device studies beside them) and are untouched; `reproduce/manifest.py`
records the `ch3-boruvka-gpu` entry as descoped rather than deleting it. Nothing
under `research/proposal-defense/` cites any of it.

## A.10 Analytical derivations

The chapters state results and cite the code that produces them; this appendix derives them. Every derivation below was checked against the shipped implementation it describes, and where the two differ — or where a chapter's sentence claims more than the algebra supports — the derivation says so rather than smoothing it over. The numbering is by chapter, so a reader can go from a claim to its proof and back. Notation follows Chapter 2: $N$ samples, $M$ features, $K$ output classes or buckets, $D$ an $N \times N$ dissimilarity matrix; $T$ and $S$ are a t-norm and its dual t-conorm.

| Derivation | Claim it supports | Where it is used |
|---|---|---|
| A.10.1 | TSK output is linear in the consequent coefficients | §2.1, §4.3.2, §6.3.1 |
| A.10.2 | FCM's alternating updates are closed-form | §2.4 |
| A.10.3 | The minimax transform is an ultrametric; $u(D^2) = u(D)^2$; the beta-spread is inert on it | §2.2, §5.2, §5.3.1 |
| A.10.4 | The three reorder arms cost $O(N^3)$, $O(N^2 \log N)$ and $O(N^2)$ | §3.3.1 |
| A.10.5 | The memory ceilings of Table 3.3 and their $\sqrt{3}$, $\sqrt{2}$ factors | §3.3.2, §3.4 |
| A.10.6 | The VAT ordering depends only on the MST | §2.2, §3.2, §3.3.3 |
| A.10.7 | The MST tour bound, and the two preconditions §3.3.6 elides | §3.3.6 |
| A.10.8 | The classifier is Gaussian naive Bayes with the normalising constants dropped | §4.2, §4.3.1 |
| A.10.9 | Parameter and screening-cost counts | §4.3.1, §4.3.4 |
| A.10.10 | The anomaly rule: saturation, the one-class threshold, monotonicity in $\theta$, and why the complement score loses resolution | §4.3.5, §4.4 |
| A.10.11 | Why pinning the extreme consequents is fatal at zeroth order and harmless above it | §4.3.2 |
| A.10.12 | Rank invariance of axis-aligned trees, and why Gaussian memberships have none | §4.3, §6.3.2 |
| A.10.13 | The persistence ramp is crisp on an ultrametric | §5.3.3 |
| A.10.14 | The gated greedy cover returns a disjoint antichain; the MAD gate's constant | §5.3.4 |
| A.10.15 | Birth height as an inverse density proxy, and why the axis is logarithmic | §2.3, §5.3.2 |
| A.10.16 | The ridge-TSK solve, its conditioning, the pinned columns and the $\sqrt{h}$ row weighting | §6.3.1, §6.3.3, §6.4 |
| A.10.17 | Soft-tree routing weights form a partition of unity and keep the output linear | §6.3.2, §6.3.3 |
| A.10.18 | The EM refinement of the mixture, step by step | §6.3.3 |
| A.10.19 | A triangular Ruspini partition is the linear B-spline basis | §6.3.4 |

### A.10.1 TSK inference, and the linearity that everything else uses

A first-order Takagi–Sugeno–Kang system with $R$ rules over inputs $x \in \mathbb{R}^M$ has, for rule $r$, an antecedent firing strength and a consequent:

$$ w_r(x) = \mathop{T}_{j=1}^{M} \mu_{rj}(x_j), \qquad f_r(x) = m_r + \phi(x)^{\top} c_r, $$

where $\mu_{rj}$ is the membership function of rule $r$ on feature $j$, $\phi(x) \in \mathbb{R}^{q}$ is the consequent basis (the raw inputs at first order, their products at second, orthogonal polynomials in §6.3.1), $m_r$ is the rule's constant and $c_r \in \mathbb{R}^q$ its coefficients. Under the product t-norm the antecedent is $w_r(x) = \prod_j \mu_{rj}(x_j)$; the code applies the within-feature t-conorm first when a feature carries several membership functions (§4.3.1), which does not change what follows. Weighted-average defuzzification gives

$$ \hat y(x) = \sum_{r=1}^{R} \bar w_r(x)\, f_r(x), \qquad \bar w_r(x) = \frac{w_r(x)}{\sum_{s=1}^{R} w_s(x)}, $$

with the convention that a row whose total firing is below a floor is left at zero rather than blended uniformly (`_normalize_firing_strengths`). Substituting $f_r$ and collecting the coefficients into one vector $\beta = (m_1, c_1, \ldots, m_R, c_R) \in \mathbb{R}^{R(1+q)}$,

$$ \hat y(x) = \sum_{r} \bar w_r(x)\,\big[\,1 \;\; \phi(x)^{\top}\big] \begin{bmatrix} m_r \\ c_r \end{bmatrix} = \Phi(x)^{\top} \beta, \qquad \Phi(x) = \big(\bar w_1(x)\,[1\;\phi(x)^\top],\; \ldots,\; \bar w_R(x)\,[1\;\phi(x)^\top]\big)^{\top}. $$

For fixed antecedents the map $\beta \mapsto \hat y(x)$ is therefore linear, and over a training set the predictions are $\hat{\mathbf y} = \Phi \beta$ with $\Phi \in \mathbb{R}^{N \times R(1+q)}$ built exactly as `solve_tsk_consequents_from_firing` builds it: the per-rule block $[1 \mid \phi(X)]$ is the same for every rule and only the firing weights differ, so the design is the outer product `norm_fs[:, :, None] * phi[:, None, :]` reshaped. Fitting the consequents is then a linear least-squares problem, which A.10.16 solves. This single fact is why Chapters 4 and 6 need no search for the THEN side of any rule.

Two things the linearity does *not* give. It does not make the model linear in the antecedent parameters — $\bar w_r$ depends on every $\mu$ and every $\sigma$ non-linearly, which is why antecedent refinement (§6.3.5) is a search and the consequent solve is not. And it is linear only because the firing strengths are treated as fixed during the solve; a joint fit of antecedents and consequents is not a least-squares problem, which is the alternation ANFIS uses and the EM of A.10.18 formalizes.

### A.10.2 Fuzzy c-means: the closed-form alternating updates

FCM minimizes $J(W, C) = \sum_{i=1}^{N} \sum_{j=1}^{c} w_{ij}^{m} \lVert x_i - c_j \rVert^2$ subject to $\sum_j w_{ij} = 1$ for every $i$ and $w_{ij} \ge 0$, with $m > 1$. Both updates §2.4 calls "closed form" follow from setting partial derivatives to zero.

*Centroids, memberships fixed.* $J$ is a sum of convex quadratics in each $c_j$ separately; $\partial J / \partial c_j = -2 \sum_i w_{ij}^m (x_i - c_j) = 0$ gives

$$ c_j = \frac{\sum_{i} w_{ij}^{m}\, x_i}{\sum_{i} w_{ij}^{m}}, $$

a weighted mean, which is the only place FCM uses coordinates — and the reason the relational variant NERFCM has to work differently (§2.4, §5.2).

*Memberships, centroids fixed.* Each row $i$ separates. With $d_{ij} = \lVert x_i - c_j \rVert^2$ and a Lagrange multiplier $\lambda_i$ for the row constraint, $\partial/\partial w_{ij}$ of $\sum_j w_{ij}^m d_{ij} - \lambda_i (\sum_j w_{ij} - 1)$ gives $m\, w_{ij}^{m-1} d_{ij} = \lambda_i$, so $w_{ij} = (\lambda_i / m d_{ij})^{1/(m-1)}$. Imposing $\sum_j w_{ij} = 1$ eliminates $\lambda_i$:

$$ w_{ij} = \left( \sum_{k=1}^{c} \left( \frac{d_{ij}}{d_{ik}} \right)^{\frac{1}{m-1}} \right)^{-1}. $$

Because the minimizer of each sub-problem is unique and each step cannot increase $J$, alternating the two is a coordinate-descent that converges to a stationary point — a local optimum, and which one depends on the start. That is the initialization sensitivity §2.4 records and that the VAT front end is meant to remove. At $m \to 1^+$ the memberships harden to nearest-centroid assignment and the scheme is k-means; at $m \to \infty$ every $w_{ij} \to 1/c$, which is why $m$ is kept in $[2, 4]$.

### A.10.3 The minimax transform: ultrametric, MST bottleneck, and two consequences

Let $D$ be any symmetric non-negative dissimilarity on $N$ points (no metric assumption), viewed as a complete weighted graph. Define the *minimax* or *bottleneck path* distance

$$ D^*_{ij} = \min_{p \,:\, i \to j} \; \max_{(u,v) \in p} D_{uv}, $$

the minimum over paths of the largest edge on the path. This is what `minimax_transform` computes with the Prim-style recurrence $D^*_{jk} = \max(D_{ij}, D^*_{ik})$ as vertex $j$ joins the tree through vertex $i$: the recurrence is correct because on a tree the path between two vertices is unique, and the largest edge on the path from $j$ to $k$ is either the attaching edge or the largest edge on the path from $i$ to $k$.

**Claim 1: $D^*$ equals the largest edge on the MST path.** Let $P_T(i,j)$ be the unique path between $i$ and $j$ in a minimum spanning tree $T$ and $e$ its heaviest edge, weight $b$. Removing $e$ from $T$ splits the vertices into two sides with $i$ and $j$ apart; by the cut property every edge crossing that cut weighs at least $b$ (else swapping it for $e$ would give a lighter spanning tree). Any path from $i$ to $j$ must cross the cut, so its maximum edge is at least $b$. The tree path achieves exactly $b$. Hence $D^*_{ij} = b$. This is also the single-linkage cophenetic distance, since single linkage merges $i$'s and $j$'s clusters at exactly the threshold $b$ — the identity §5.2 credits to Johnson [@johnson1967hierarchical] and that makes the iVAT image and the dendrogram two views of one object.

**Claim 2: $D^*$ is an ultrametric.** Concatenate a bottleneck path from $i$ to $k$ with one from $k$ to $j$; the result is a path from $i$ to $j$ whose largest edge is $\max(D^*_{ik}, D^*_{kj})$. Minimizing over paths can only lower it, so

$$ D^*_{ij} \le \max\!\big(D^*_{ik},\, D^*_{kj}\big) \qquad \text{for all } i, j, k, $$

the strong triangle inequality. Symmetry and $D^*_{ii} = 0$ are immediate. Note what was *not* assumed: nothing about $D$ itself. That is why Chapter 3's engine and Chapter 5's selectors run unchanged on fractional-Minkowski, cosine and DTW input (Table 3.7) — the transform manufactures a metric, indeed an ultrametric, out of any dissimilarity at all.

**Consequence (a): $u(D^2) = u(D)^2$.** Write $u(\cdot)$ for the transform. Both $\max$ and $\min$ commute with any monotone non-decreasing map $g$: $g(\max(a,b)) = \max(g(a), g(b))$ and likewise for $\min$. Squaring is monotone on $[0, \infty)$, so $u(D^2)_{ij} = \min_p \max_{e \in p} D_e^2 = \big(\min_p \max_{e \in p} D_e\big)^2 = u(D)^2_{ij}$. This is the remark in §5.2 about iRFCM: squaring before or after the transform is the same operation, and the pipelines differ only in what is fitted around it.

**Consequence (b): the beta-spread is inert on $D^*$.** NERFCM's beta-spread adds $\beta$ to every off-diagonal entry of $D$ until the relational objective's implied squared distances are non-negative; it is needed exactly when $D$ is not the matrix of squared Euclidean distances of any point configuration. A finite ultrametric embeds isometrically in Euclidean space [@lemin1985isometric] — equivalently, an ultrametric is of strict negative type — so $D^*$ needs no spread: $\beta = 0$ already satisfies the admissibility condition, which is the justification §5.3.1 gives for dropping the safeguard on transformed input. On raw non-metric $D$ this does not hold and the spread is doing work, which is why the comparison in Table 5.1 keeps NERFCM on $D$ as a separate column.

### A.10.4 The cost of the three reorder arms

All three arms (§3.3.1, Figure 3.1) share the outer loop: $N - 1$ rounds, in round $r$ one vertex leaves the unplaced set $U_r$ with $|U_r| = N - r$. They differ only in how the next vertex — the unplaced vertex nearest the placed set — is found.

*Classical.* The reference implementations recompute, for every $u \in U_r$, its distance to every placed vertex and take the minimum: $(N - r) \cdot r$ lookups in round $r$. Summed,

$$ \sum_{r=1}^{N-1} r\,(N - r) = \frac{N(N-1)(N+1)}{6} = \frac{N^3 - N}{6} \;\sim\; \frac{N^3}{6}, $$

hence $O(N^3)$. The waste is that the minimum over placed vertices for each $u$ changes by at most one candidate per round and is recomputed from scratch anyway.

*Stage one: priority queue with lazy deletion.* Maintain for each unplaced $u$ its current best key $\kappa(u) = \min_{v \text{ placed}} D_{vu}$. When $v$ joins the tree, relax its row: for each unplaced $u$ with $D_{vu} < \kappa(u)$, set $\kappa(u) \leftarrow D_{vu}$ and push $(\kappa(u), u)$. The next vertex is popped; stale entries (an $u$ already placed, or a key no longer current) are discarded on the way out. Each round pushes at most $N - r$ entries, so the total number of pushes is at most $\sum_r (N - r) = N(N-1)/2$, and so is the number of pops. The heap therefore holds $O(N^2)$ entries, each push or pop costs $O(\log N^2) = O(2 \log N)$, and the total is $O(N^2 \log N)$. The relaxation itself is $O(N)$ per round, $O(N^2)$ in all, so the heap is the only source of the log factor — which is the observation stage two acts on.

*Stage two: compact active set, relaxation fused with selection.* The reorder never needs the queue's full order, only its minimum once per round. Keep the unplaced vertices packed in the first $N - r$ slots of an array with their keys; in one pass over the active slots, update $\kappa(u) \leftarrow \min(\kappa(u), D_{vu})$ and track the running minimum. Removing the selected vertex is a swap with the last active slot, $O(1)$. Round $r$ costs $N - r$ comparisons and no allocation:

$$ \sum_{r=1}^{N-1} (N - r) = \frac{N(N-1)}{2} \;\sim\; \frac{N^2}{2}, $$

with $O(N)$ workspace beyond the matrix. Two sequencing facts make the fusion legal. The argmin for round $r+1$ is over keys as they stand *after* round $r$'s relaxation, and the running minimum is taken over exactly those updated keys; and the swap-removal does not disturb any key. Table 3.2's fitted exponents (3.20, 1.86, 1.97) are the empirical counterparts of the three sums, with the $\log N$ factor of stage one too small to resolve over a decade and a half of $N$.

### A.10.5 Memory ceilings and the $\sqrt{3}$, $\sqrt{2}$ factors

A scheme that holds $k$ dense $N \times N$ matrices at $s$ bytes per entry needs $F(N) = k\,s\,N^2$ bytes. Under a budget $B$ the largest feasible size is

$$ N_{\max} = \left\lfloor \sqrt{\frac{B}{k\,s}} \right\rfloor. $$

The classical VAT keeps $D$, the reordered copy and a work matrix, $k = 3$; the in-place scheme of §3.3.2 keeps $D$ alone, $k = 1$; and the matrix-free path has no $N^2$ term at all. Hence the ratio of reachable sizes between the in-place and classical schemes is $\sqrt{3/1} = \sqrt{3} \approx 1.73$ at any precision and any budget, and halving the precision ($s = 8 \to 4$) buys a further $\sqrt{2}$. These are the two factors §3.3.2 quotes; they are properties of the formula, not of any measurement.

Table 3.3 uses decimal gigabytes, $B = 64 \times 10^9$ and $96 \times 10^9$ bytes: $\lfloor\sqrt{64 \times 10^9 / 24}\rfloor = 51{,}639$ and $\lfloor\sqrt{96 \times 10^9 / 24}\rfloor = 63{,}245$ for the classical scheme at float64, matching the table; $\sqrt{96 \times 10^9 / 4} = 154{,}919$ for the in-place float32 ceiling, the 155,000 §3.4 rounds to. The 135,000-point run sits at $F = 4 \times 135{,}000^2 = 72.9$ GB, inside the machine and outside the cap, as the table says. What the formula does *not* cover is the $O(N)$ workspace of the reorder, the Python process, and the BLAS scratch, which is why the reachable sizes are stated as arithmetic ceilings and the two large runs as demonstrations.

### A.10.6 Why the VAT ordering depends only on the minimum spanning tree

VAT's reorder is Prim's algorithm from a fixed seed, and the ordering is the sequence in which vertices join the tree. The claim §2.2 and §3.2 rest on is that any MST algorithm — Borůvka, Kruskal, a device kernel — yields the same ordering, so the reorder can be composed with whatever MST front end is cheapest.

*Proof, distinct weights.* When all pairwise dissimilarities are distinct the MST $T$ is unique. At each Prim step the placed set $P$ defines a cut $(P, V \setminus P)$; Prim adds the endpoint of the lightest edge crossing it. By the cut property the lightest crossing edge belongs to every MST, hence to $T$. So the next vertex is determined by $T$ and $P$ alone: it is the endpoint of the lightest *tree* edge leaving $P$. Given the seed, induction over the steps shows the whole sequence is a function of $T$. Any algorithm returning $T$ therefore yields the same VAT ordering once the tree is walked from the seed by "lightest tree edge leaving the placed set" — an $O(N \log N)$ post-pass on $N - 1$ edges.

*Ties.* With equal weights there may be several MSTs, and two builders may return different ones; the cut property then only guarantees *some* lightest crossing edge is in each tree. Two constructions can thus produce different but equally valid orderings, and this is exactly what §3.2 records at the largest single-precision size, where float32 rounding turns near-ties into ties (Table 3.3's $0.9996 \pm 0.0012$ row). The reduction "same MST $\Rightarrow$ same ordering" is therefore exact when the MST is unique and holds up to tie-breaking otherwise, which is how §3.2 states it.

*Corollary: the VAT cut is single-linkage.* Cutting the ordered image at a threshold $t$ — declaring consecutive vertices in the same block when the edge that joined them weighs at most $t$ — partitions the vertices exactly as removing all tree edges heavier than $t$ does, and that is the single-linkage partition at level $t$ (the MST is the single-linkage hierarchy [@gower1969mst]). This is the identity `IVATMeans` reads its partition off (§3.3.5), and the one Chapter 5 turns into memberships.

### A.10.7 The MST tour bound, and what §3.3.6 actually has

The standard argument. Let $T$ be an MST of weight $w(T)$ and $\mathrm{OPT}$ the optimal tour. Deleting any edge from $\mathrm{OPT}$ leaves a Hamiltonian path, which is a spanning tree, so $w(T) \le w(\mathrm{OPT})$. A depth-first walk of $T$ traverses every edge twice, a closed walk of length exactly $2 w(T)$; visiting the vertices in first-visit order and skipping repeats gives a Hamiltonian tour, and each skip replaces a sub-walk by the direct edge, which is no longer **if the triangle inequality holds**. So

$$ w(\text{DFS tour}) \;\le\; 2\, w(T) \;\le\; 2\, w(\mathrm{OPT}) \qquad \text{(metric } D \text{ only)}. $$

Two things §3.3.6 states more strongly than this supports, and both are worth naming rather than leaving for a reviewer.

1. **The bound needs a metric.** The whole point of Chapter 3 is exactness on non-metric $D$, and there the shortcutting step fails: a direct edge can be longer than the walk it replaces, so no factor is guaranteed. The 2× statement holds on Euclidean and DTW-with-triangle-inequality inputs and is unsupported on the fractional-Minkowski and cosine rows of Table 3.7.

2. **The bound is for a depth-first walk, not for a Prim order.** The VAT ordering is Prim's visitation order, and a Prim order is not a DFS preorder: consecutive VAT vertices need not be adjacent in the tree, since the next vertex attaches to *whichever* placed vertex is nearest, not to the last one added. The tour that visits the VAT-ordered points in sequence is therefore not the tour the argument bounds. A short random search over small Euclidean instances (a scratch check, not a harness result) finds Prim-order tours longer than $2 w(T)$, so the argument cannot be repaired by appeal to the tree weight alone; whether the ratio to $\mathrm{OPT}$ stays under two is an open question this document does not settle.

The safe statement is that the depth-first walk of the same MST — available at no extra cost, since the tree is already built — is a provable 2-approximation on metric input, while the VAT order is an *empirically* reasonable warm start whose bound is not established. §3.3.6 is amended to say so; nothing else in the chapter depends on the tour bound.

### A.10.8 The classifier is Gaussian naive Bayes with the normalising constants dropped

§4.2 concedes that the classification construction is "closely related to" Gaussian naive Bayes. The relationship is exact and worth having in one line, because it also predicts a systematic difference.

Take the simplest configuration: one Gaussian per retained feature $j$ and class $k$, with centre $\mu_{jk}$ and width $\sigma_{jk}$, product t-norm across features. The rule firing is

$$ w_k(x) = \prod_{j=1}^{M} \exp\!\left( -\tfrac{1}{2} \Big( \tfrac{x_j - \mu_{jk}}{\sigma_{jk}} \Big)^2 \right). $$

Gaussian naive Bayes assigns $\arg\max_k \pi_k\, p_k(x)$ with $p_k(x) = \prod_j \mathcal{N}(x_j; \mu_{jk}, \sigma_{jk}^2) = \prod_j \frac{1}{\sqrt{2\pi}\,\sigma_{jk}} \exp(\cdot)$. Comparing,

$$ \log w_k(x) = \log p_k(x) + \sum_{j=1}^{M} \log\!\big(\sqrt{2\pi}\,\sigma_{jk}\big). $$

So $\arg\max_k w_k(x)$ is the naive-Bayes decision with **uniform priors** and with the per-class term $\sum_j \log \sigma_{jk}$ **added back**: relative to naive Bayes, the construction favours the class whose retained features are *wider*, by exactly that sum. On a class-balanced problem with equal widths the two coincide; on Glass, where the automatic component count produces near-zero widths on some classes (§4.3.1, Figure 4.1), they do not, and the direction of the difference is predictable — a degenerate spike is penalised heavily by naive Bayes's $-\log \sigma$ and not at all here. This is the concrete content of "the missing row a committee will ask for" in Table 4.5: the two models differ by a known additive term, and the row would measure what that term costs.

With several components per feature the correspondence loosens: the within-feature t-conorm $S$ of unnormalised Gaussians is not the mixture density $\sum_c \pi_c \mathcal{N}_c$ — under the probabilistic sum, $S(a, b) = a + b - ab$, it is the mixture's sum minus a product term, and under max it is the dominant component alone. The factorisation across features is the same, so the "naive" half of naive Bayes is retained exactly and the "Gaussian" half approximately.

### A.10.9 Parameter and screening-cost counts

*Grid rule base.* Partitioning input $j$ into $N_{\mu_j}$ sets and forming a rule per combination gives $N_{\text{rules}} = \prod_j N_{\mu_j}$; at a common $c$ sets per input that is $c^M$, and each rule carries its own consequent, so the parameter count is at least $c^M (1 + q)$.

*This construction.* One rule per class or bucket, $K$ rules. Rule $k$ carries, on each retained feature $j$, a mixture of $p_{jk}$ Gaussians with two parameters each, so

$$ N_{\text{params}} = \sum_{k=1}^{K} \sum_{j \in \mathcal{F}_k} 2\, p_{jk} \;\le\; 2\,K\,M\,p_{\max}, $$

linear in $K$, $M$ and the component cap. The consequents add $K(1 + q)$, where $q$ is the basis size: $M$ at first order, $2M$ at second (squares), $M + \binom{M}{2} + M$ for the full second-order basis with cross terms — the one place a quadratic in $M$ enters the *model*, and only when that basis is chosen.

*The screen.* `calculate_gaussian_correlation` compares, per feature, the class-conditional distributions of every pair of classes: $\binom{K}{2} = K(K-1)/2$ comparisons per feature, $M K (K-1)/2$ in all, $O(MK^2)$. For RT-IOT2022 that is $83 \times 66 = 5{,}478$ comparisons, cheap in absolute terms but the only super-linear factor in the fit, which is why §4.3.4 refuses to call the cost "linear in everything".

### A.10.10 The anomaly rule: four facts from the algebra

The rule (§4.3.5) is $\mu_{\text{anom}}(x) = 1 - S(c_1, \ldots, c_K)$ with $c_k = \min(\max(\mu_k(x) + \theta, 0), 1)$, and the decision is $\arg\max$ over the $K$ class firings and this one. `tsk_firing_strengths` and `_anomaly_argmax` implement exactly this.

**(a) One clipped input saturates any t-conorm.** A t-conorm satisfies $S(a, 0) = a$ and is monotone in each argument, so $S(1, y) \ge S(1, 0) = 1$, and $S(1, y) \le 1$ by range; hence $S(1, y) = 1$, and by associativity one argument equal to 1 makes the whole aggregate 1. For the Hamacher conorm the code ships, $S(x, y) = (x + y - 2xy)/(1 - xy)$, at $x = 1$ numerator and denominator are both $1 - y$ (the code returns 1 where the denominator vanishes, at $x = y = 1$). Therefore

$$ \mu_{\text{anom}}(x) > 0 \iff \mu_k(x) < 1 - \theta \;\text{ for every } k. $$

At the inherited $\theta = 0.99$ the bar is a firing of $0.01$; wherever any class clears it the anomaly rule is identically zero and the conorm plays no part in the decision. That is the "total degeneracy at the default" of §4.3.5.

**(b) The multi-class decision.** The anomaly label wins iff $1 - S(c) > \max_k \mu_k(x)$. When no class fires above $1 - \theta$ the clip is inactive, $c_k = \mu_k + \theta$, and the decision genuinely depends on the conorm's aggregation of several boosted firings — the regime Table 4.6's sweep over $\theta \in [0.5, 0.8]$ operates in, where firings of $0.2$–$0.5$ are below the bar.

**(c) The one-class reduction.** With a single known class the column-wise conorm has nothing to aggregate and `t_conorm(x, None, …)` returns $x$ itself, so $\mu_{\text{anom}} = 1 - \min(\mu + \theta, 1)$. The anomaly wins iff $1 - \mu - \theta > \mu$ (the clipped case gives $0 > \mu$, never), i.e.

$$ \text{anomaly} \iff \mu(x) < \frac{1 - \theta}{2}. $$

So on a one-class fit $\theta$ is a threshold on the firing strength and nothing more, the norm family cannot change a prediction, and Table 4.11(e)'s measured $0.0012$ advantage of sweeping $\theta$ over thresholding the score directly is the noise it should be.

**(d) Monotonicity in $\theta$.** Each $c_k$ is non-decreasing in $\theta$, $S$ is non-decreasing in each argument, so $\mu_{\text{anom}}$ is non-increasing in $\theta$ at every $x$. The set of points labelled anomalous therefore shrinks as $\theta$ grows, which makes both the detection rate and the false-alarm rate non-increasing functions of $\theta$ — the shape Table 4.6 and Table 4.7c show, saturating to zero once $\theta \ge 1$ makes every $c_k = 1$.

**(e) Why the complement score loses resolution and the surprisal does not.** For the one-class detector under the product t-norm, the firing is $\prod_j \mu_j(x)$, the *complement* score is $s_c = 1 - \prod_j \mu_j$ and the *surprisal* score is $s_u = \sum_j -\log \mu_j = -\log(1 - s_c)$. The map $s_c \mapsto -\log(1 - s_c)$ is strictly increasing on $[0, 1)$, so the two rankings — and hence the two ROC curves — are identical in exact arithmetic, as §4.4 says. In floating point they are not: $1 - \prod_j \mu_j$ is computed as $1$ minus a small number, and rounds to exactly $1.0$ once $\prod_j \mu_j < 2^{-53} \approx 1.1 \times 10^{-16}$. With Gaussian memberships $\prod_j \mu_j = \exp(-\tfrac{1}{2} \sum_j z_j^2)$, so the collapse begins when $\sum_j z_j^2 > 106 \ln 2 \approx 73.5$ — a point about $8.6\sigma$ out in a single feature, or just over $3\sigma$ in each of eight, which heavy-tailed process identifiers reach routinely. Every such point ties at $s_c = 1$, exactly the 1,508-of-4,002 distinct-score collapse Table 4.11 reports, while $s_u$ keeps summing in the log domain and never rounds. The onset depends on the tails, not the feature count, which is why BETH hits it at eight features when the library's own note expected sixty.

### A.10.11 Pinning the extreme consequents: fatal at zeroth order, absorbed above it

From A.10.1, $\hat y(x) = \sum_r \bar w_r(x) \big(m_r + \phi(x)^\top c_r\big)$. At **zeroth order** $c_r = 0$ and $\hat y(x) = \sum_r \bar w_r(x)\, m_r$: the prediction is a convex combination of the rule constants, so it lies in $[\min_r m_r, \max_r m_r]$ and, for a sample whose firing is concentrated on rule $r$, equals $m_r$ up to the leakage into neighbouring rules. The squared error contributed by the samples of bucket $r$ is then approximately

$$ \sum_{i \in r} (y_i - m_r)^2 = \sum_{i \in r} (y_i - \bar y_r)^2 + n_r\,(m_r - \bar y_r)^2, $$

variance within the bucket plus a bias term that is zero only when $m_r = \bar y_r$, the bucket mean. Pinning $m_1 = y_{\min}$ and $m_R = y_{\max}$ (the `pin_extremes` equality constraint) sets the bias of the two outer rules to $(\bar y_1 - y_{\min})^2$ and $(y_{\max} - \bar y_R)^2$ per sample with nothing to offset it. On Concrete §4.3.2 records the outer buckets' means at $0.195$ and $0.653$ on the unit interval against pinned values of $0$ and $1$, so the bias is $0.038$ and $0.120$ per sample across $687$ of the $1{,}030$ rows — the mechanism behind $R^2 = -0.434$ against $0.394$ unpinned. The unconstrained least-squares solution puts $m_r$ at the firing-weighted mean of the bucket's targets (A.10.16 with $\Phi$ reduced to the weight columns), which is why uniform's "held at their buckets' own means" is the right pin and the extreme observation the wrong one.

At **first order and above** the constraint still fixes $m_1$ and $m_R$, but the consequent $m_r + \phi(x)^\top c_r$ can reproduce any constant offset over the bucket's support as long as $\phi(x)$ is not orthogonal to the constant direction there — and with min-max-scaled inputs on $[0, 1]$ it never is. The solve therefore spends the free $c_r$ to cancel the imposed bias: the fitted intercepts §4.3.2 reports at $-0.38$ and $-1.19$, outside the target's own range, are exactly this compensation. The penalty falls from $0.676$ to $0.005$ in $R^2$ because the bias is absorbed, not because it is absent.

### A.10.12 Rank invariance: why the transform is free for trees and not for Gaussians

Let $\varphi_j : \mathbb{R} \to \mathbb{R}$ be strictly increasing for each feature $j$ (log, min-max, z-score, or their compositions), and $\tilde x_j = \varphi_j(x_j)$.

*Axis-aligned trees.* Every split is a test $x_j \le t$. Since $\varphi_j$ is strictly increasing, $x_j \le t \iff \varphi_j(x_j) \le \varphi_j(t)$: for every split on the original scale there is a split on the transformed scale inducing the same partition of the training set, and conversely. CART's impurity criteria (variance reduction, Gini, entropy) are functions of the *partition* alone, so the greedy criterion takes the same value on corresponding splits, the same split is chosen at every node (up to ties among splits inducing identical partitions), and the induced trees are identical as functions of the training rows. Predictions on test rows agree wherever the chosen threshold falls between the same two training values, which is the usual midpoint convention. A random forest is a fixed set of such trees over bootstrap samples with the same seeds, hence identical too. This is the control Table 4.1 measures at $+0.001$ and $+0.000$.

*Gaussian memberships.* A Gaussian $\mu(x; m, \sigma) = \exp(-\tfrac{1}{2}((x - m)/\sigma)^2)$ satisfies $\mu(\varphi(x); m', \sigma') = \mu(x; m, \sigma)$ for all $x$ only if $\varphi$ is affine: the level sets of the two sides are intervals symmetric about $m'$ and $m$ respectively, and a strictly increasing $\varphi$ maps the symmetric interval $[m - a, m + a]$ onto a symmetric interval for every $a$ only when it is affine. A log transform is not affine, so no re-fit of $(m, \sigma)$ reproduces the original membership on a skewed feature; the fitted Gaussian on the raw scale describes the skew where the one on the log scale describes the structure, and the $+0.10$ in $R^2$ of Table 4.1 is the difference. The soft fuzzy tree inherits the same sensitivity through its sigmoidal splits and its Gaussian-antecedent leaves, which is §6.3.2's point that its raw-feature readability is bought rather than free.

### A.10.13 The persistence ramp is crisp on an ultrametric

Take a block $B$ of the single-linkage hierarchy on $D^*$: a dendrogram node, born at height $h_b$ (the merge that creates it) and dying at $h_d > h_b$ (the merge that absorbs it into a larger node). Define $d_B(x) = \min_{y \in B} D^*_{xy}$, the bottleneck height at which $x$ would join $B$, and the ramp $\mu_B(x) = \mathrm{clip}\big((h_d - d_B(x)) / (h_d - h_b),\, 0,\, 1\big)$.

*Members.* For $x \in B$, $x$ is connected to every other member at threshold $h_b$ (that is what being a node born at $h_b$ means), so $d_B(x) \le h_b$ and $\mu_B(x) = 1$. (Strictly $d_B(x) = 0$ for the trivial self-distance; the code takes the minimum over members including $x$.)

*Non-members.* Suppose $x \notin B$ had $d_B(x) < h_d$. Then at threshold $d_B(x) < h_d$ the point $x$ is connected to some $y \in B$, so the single-linkage component containing $B$ at that threshold also contains $x$ — but $B$ is by definition the maximal component containing its members for all thresholds in $[h_b, h_d)$, and it does not contain $x$. Contradiction; hence $d_B(x) \ge h_d$ and $\mu_B(x) = 0$.

So no point has $h_b < d_B(x) < h_d$: the interval the ramp slopes across is empty, and $\mu_B$ takes only the values $0$ and $1$. This is a property of the ultrametric, not an implementation defect, and it is why `block_membership` defaults to $\mu_B(x) = 2^{-(d_B(x)/h_d)^2} = \exp\!\big(-\ln 2\,(d_B(x)/h_d)^2\big)$, a Gaussian in minimax distance with its half-maximum at the death height: members read $1$, a non-member joining exactly at $h_d$ reads $\tfrac{1}{2}$, and the skirt is graded by how far past the block's dissolution height the point attaches. Both curves are parameterised by merge heights alone; $\arg\max$ over the Gaussian memberships of a band still returns the crisp labels because for any two blocks $B, B'$ in a disjoint antichain and $x \in B$, $d_B(x) = 0 < d_{B'}(x)$. Figure 5.3 draws both.

### A.10.14 The gated greedy cover returns a disjoint antichain

*Setup.* Candidate blocks are dendrogram nodes. Nodes of a hierarchy form a **laminar** family: any two are either nested or disjoint. The gate admits a block iff its persistence $h_d - h_b$ exceeds $\mathrm{med} + \gamma \cdot 1.4826 \cdot \mathrm{MAD}$ over all blocks (`select_coverage_cover`, $\gamma$ = `gap_sigma`), and its size lies in $[3, 0.6\,N]$. Among eligible blocks the cover then repeatedly takes the block with the largest number of still-uncovered members, ties broken toward larger persistence, and stops when the best gain is zero.

*Claim.* The selected blocks are pairwise disjoint.

*Proof.* Suppose two selected blocks intersect; by laminarity one contains the other, say $C \subsetneq A$. Case 1, $A$ selected first: at $C$'s turn every member of $C$ is already covered, so $C$'s gain is zero and the stopping rule excludes it. Case 2, $C$ selected first: at that step $A$ was eligible with $\mathrm{gain}(A) \ge \mathrm{gain}(C)$, because the uncovered members of $C$ are uncovered members of $A$. If the inequality were strict $A$ would have been chosen. If equal, $A \setminus C$ was already covered by earlier selections; each earlier block is nested with or disjoint from $A$, and none contains $A$ (else $C \subset A$ would already be covered and have zero gain), so they are proper subsets of $A$ and together with $C$ cover all of $A$ — giving $A$ zero gain at every later step, so it is never selected. Either way the assumption fails. $\square$

The selected set is therefore an antichain of the hierarchy, i.e. a *local cut* in Campello *et al.*'s sense [@campello2013fosc], which is the framing §5.3.4 adopts; the cover's only freedom is *which* antichain, decided by the stability measure (persistence, here) rather than the excess of mass FOSC uses. The regression test `test_selection_antichain.py` checks the conclusion; the proof says why it can never fail.

*The constant $1.4826$.* For a normal sample the median absolute deviation converges to $\sigma\,\Phi^{-1}(3/4) \approx 0.6745\,\sigma$, so $\mathrm{MAD}/0.6745 = 1.4826\,\mathrm{MAD}$ is a consistent estimate of $\sigma$. The gate is thus "persistence more than $\gamma$ robust standard deviations above the median", with the median and MAD immune to the very outliers it is looking for — which is the reason to prefer them to the mean and standard deviation, where a few real clusters would inflate the threshold that is meant to detect them.

### A.10.15 Birth height as an inverse density proxy, and the logarithmic axis

Consider points drawn from a homogeneous Poisson process of intensity $\rho$ in $\mathbb{R}^d$. The distance $r$ from a point to its nearest neighbour satisfies $\Pr(r > t) = \exp(-\rho V_d t^d)$, with $V_d$ the volume of the unit $d$-ball, so its median is

$$ r_{1/2} = \left( \frac{\ln 2}{\rho\, V_d} \right)^{1/d} \;\propto\; \rho^{-1/d}. $$

Single-linkage merge heights inside a homogeneous region are nearest-neighbour-type distances — a component grows by absorbing the point nearest to it — so a cluster of density $\rho$ is *born* (its members become connected) at a height of order $\rho^{-1/d}$, and two clusters of densities $\rho_1 > \rho_2$ have births separated by

$$ \log h_2 - \log h_1 \approx \frac{1}{d} \log \frac{\rho_1}{\rho_2}. $$

Two consequences. Density ratios appear as *differences on a log axis*, which is why `discover_band_edges` looks for gaps in $\log(\text{birth})$ rather than in birth: a factor of $5.6\times$ or $7.1\times$ in scale (Table 5.2's construction) is the same gap wherever it sits along the axis. And the gap shrinks with dimension as $1/d$, so the well-separated-scales assumption §5.3.2 states gets harder to satisfy as $d$ grows — a density contrast of $10\times$ is a gap of $\log 10 \approx 2.3$ in one dimension and $0.23$ in ten, below the `min_log_gap` of $0.5$ the code requires. That is a limit of the method worth stating alongside Hartigan's result that single linkage is not density-consistent for $d > 1$: the birth-height reading is a proxy, sharpest in low dimension, not an estimator.

### A.10.16 The ridge-TSK solve: normal equations, conditioning, pinned columns, row weights

From A.10.1, $\hat{\mathbf y} = \Phi \beta$ with $\Phi \in \mathbb{R}^{N \times P}$. The solver minimises

$$ J(\beta) = \lVert \mathbf y - \Phi \beta \rVert_2^2 + \lambda\, \beta^{\top} \mathbf D \beta, \qquad \mathbf D = \mathrm{diag}(d_1, \ldots, d_P),\; d_p \in \{0, 1\}, $$

with $d_p = 0$ on each rule's constant column (the bucket mean) and $1$ elsewhere: intercepts are never shrunk, so the rule constants stay at the scale of the target. $J$ is convex and quadratic; $\nabla J = -2\Phi^\top(\mathbf y - \Phi\beta) + 2\lambda \mathbf D \beta = 0$ gives the ridge normal equations

$$ \big(\Phi^{\top}\Phi + \lambda \mathbf D\big)\, \beta = \Phi^{\top} \mathbf y. $$

*Why the code does not form them.* With the singular values $\sigma_1 \ge \ldots \ge \sigma_P$ of $\Phi$, the Gram matrix $\Phi^\top \Phi$ has singular values $\sigma_p^2$, so $\kappa(\Phi^\top\Phi) = \kappa(\Phi)^2$: forming the normal equations squares the condition number. Two rules with nearly collinear firing columns make $\Phi$ ill-conditioned; squaring made it singular to working precision, and `numpy.linalg.solve` returned finite coefficients of order $10^{24}$ — the $10{,}536$ MPa divergence §6.4 records. The fix is to observe that $J(\beta) = \big\lVert \begin{bmatrix} \mathbf y \\ \mathbf 0 \end{bmatrix} - \begin{bmatrix} \Phi \\ \sqrt{\lambda}\, \mathbf D^{1/2} \end{bmatrix} \beta \big\rVert_2^2$, an ordinary least-squares problem on an *augmented* design whose condition number is that of $\Phi$ (improved by the ridge rows) rather than its square. `lstsq` on it, via the SVD, truncates negligible singular values instead of dividing by them. That is what `solve_tsk_consequents_from_firing` does, and why a non-zero default $\lambda$ was the second half of the fix: with $\lambda > 0$ the augmented matrix has full column rank on the penalised columns whatever $\Phi$ does.

*Pinned columns.* `pin_extremes` fixes the constant of the first and last rules at prescribed values $v$. Partition $\beta = (\beta_F, \beta_P)$ into free and pinned coefficients and $\Phi = [\Phi_F \;\; \Phi_P]$ accordingly. With $\beta_P = v$ fixed, $J$ becomes $\lVert (\mathbf y - \Phi_P v) - \Phi_F \beta_F \rVert^2 + \lambda \beta_F^\top \mathbf D_F \beta_F$, the same problem on the residual target $\mathbf y - \Phi_P v$. Its minimiser is the exact solution of the equality-constrained problem (the KKT conditions reduce to it, since the constraint is on coordinates), so pinning costs no accuracy in the *solve* — only, at zeroth order, in the *model* (A.10.11).

*Weighted rows.* Minimising $\sum_i h_i (y_i - \Phi_i \beta)^2$ for non-negative weights $h_i$ equals minimising $\lVert H^{1/2}(\mathbf y - \Phi\beta) \rVert^2$ with $H = \mathrm{diag}(h)$, i.e. the unweighted problem on rows scaled by $\sqrt{h_i}$. This is why the M-step of A.10.18 can reuse the same solver by scaling each row by $\sqrt{h_{i\ell}}$: the responsibility weighting is implemented exactly, not approximated.

### A.10.17 Soft-tree routing weights form a partition of unity

Let each internal node $g$ of a binary tree carry a gate $s_g(x) \in [0, 1]$, the membership of $x$ in the left branch, with $1 - s_g(x)$ for the right (a sigmoid of $x_j - t$ for the soft tree of §6.3.2, a Gaussian partition-of-unity gate for the mixture of §6.3.3). Define the weight of leaf $\ell$ as the product along its root-to-leaf path,

$$ w_\ell(x) = \prod_{g \in \mathrm{path}(\ell)} \big[ s_g(x) \big]^{[\ell \text{ left of } g]} \big[ 1 - s_g(x) \big]^{[\ell \text{ right of } g]}. $$

*Claim.* $\sum_\ell w_\ell(x) = 1$ for every $x$. *Proof by induction on depth.* A single leaf has weight $1$. For a tree rooted at $g$ with subtrees $L$ and $R$, $\sum_\ell w_\ell = s_g \sum_{\ell \in L} w'_\ell + (1 - s_g) \sum_{\ell \in R} w''_\ell = s_g + (1 - s_g) = 1$ by the inductive hypothesis on the subtrees. $\square$

So the tree output $\hat y(x) = \sum_\ell w_\ell(x) f_\ell(x)$ is a convex combination of the leaf models, with the same structure as A.10.1's weighted average but with the normalisation *built in* rather than computed — there is no division and no zero-firing case. Substituting $f_\ell(x) = m_\ell + \phi(x)^\top c_\ell$ shows $\hat y$ is linear in the leaf coefficients for fixed gates, so every leaf (fuzzy tree) or expert (mixture) is fitted by A.10.16 with rows weighted by $w_\ell(x_i)$ — the single shared primitive §6.1 describes. A crisp CART tree is the special case $s_g \in \{0, 1\}$, where exactly one $w_\ell$ is non-zero and the model reduces to one leaf model per region. Because the gates test only original inputs $x_j$, every intermediate quantity is a product of memberships of named variables, which is how the Magdalena condition of §6.2 is met by construction.

### A.10.18 The EM refinement of the hierarchical mixture

Model the mixture of §6.3.3 as a conditional density: with $L$ leaves, gate weights $w_\ell(x)$ from A.10.17 (parameterised by the gate parameters $\gamma$), and Gaussian experts $y \mid x, \ell \sim \mathcal{N}\big(f_\ell(x; \beta_\ell),\, \sigma_\ell^2\big)$,

$$ p(y \mid x) = \sum_{\ell=1}^{L} w_\ell(x; \gamma)\; \mathcal{N}\big(y;\, f_\ell(x; \beta_\ell),\, \sigma_\ell^2\big). $$

The log-likelihood $\sum_i \log p(y_i \mid x_i)$ has a sum inside the log, so it is not separable; EM introduces the latent leaf assignment and maximises the expected complete-data log-likelihood instead.

*E-step.* The responsibility of leaf $\ell$ for sample $i$ is the posterior

$$ h_{i\ell} = \frac{w_\ell(x_i; \gamma)\, \mathcal{N}(y_i; f_\ell(x_i), \sigma_\ell^2)}{\sum_{m=1}^{L} w_m(x_i; \gamma)\, \mathcal{N}(y_i; f_m(x_i), \sigma_m^2)}, $$

computed as $\exp(a_{i\ell} - \mathrm{logsumexp}_m\, a_{im})$ with $a_{i\ell} = \log w_\ell(x_i) - \tfrac{1}{2}\log(2\pi\sigma_\ell^2) - (y_i - f_\ell(x_i))^2 / 2\sigma_\ell^2$, so that a leaf whose density underflows does not divide by zero — the "log-sum-exp accumulation" guard §6.3.3 lists.

*M-step, experts.* The expected complete-data log-likelihood separates over leaves; for leaf $\ell$ the $\beta_\ell$-dependent part is $-\tfrac{1}{2\sigma_\ell^2} \sum_i h_{i\ell} (y_i - f_\ell(x_i; \beta_\ell))^2$. Maximising it is the weighted least-squares problem of A.10.16 with weights $h_{i\ell}$, so each expert is re-solved by the same ridge primitive on rows scaled by $\sqrt{h_{i\ell}}$; the variance update is $\sigma_\ell^2 = \sum_i h_{i\ell} r_{i\ell}^2 / \sum_i h_{i\ell}$ with $r_{i\ell}$ the new residuals, floored below to keep a leaf that captures a handful of points from collapsing its variance and its responsibilities to a spike (the "variance floor").

*M-step, gates.* The $\gamma$-dependent part is $\sum_i \sum_\ell h_{i\ell} \log w_\ell(x_i; \gamma)$, a weighted multinomial log-likelihood. For a Gaussian partition-of-unity gate over one named input, the responsibilities aggregated to each internal node give closed-form weighted mean and variance updates — the standard mixture-of-Gaussians M-step applied at each gate — which is why §6.3.3 can say the gate update is closed form; for sigmoidal gates it is a weighted logistic regression, one Newton step of which is the IRLS update. A leaf whose total responsibility $\sum_i h_{i\ell}$ falls below a floor is pruned (the "starved component" guard).

*Monotonicity.* Each iteration does not decrease the observed log-likelihood, by the usual Jensen argument; the one-shot greedy fit of §6.3.3 is a valid initialisation, so EM can only improve on it in likelihood — which is the reason to run it, and the reason the one-shot result stands as a completed contribution if the EM slips. Nothing in this estimator is new (Jordan and Jacobs [@jordan1994hierarchical]); what is particular is that the expert M-step *is* A.10.16.

### A.10.19 A triangular Ruspini partition is the linear B-spline basis

Given sorted apex knots $c_0 < c_1 < \cdots < c_{k-1}$, `build_triangular_partition` defines term $i$ as the triangle rising from $c_{i-1}$ to $1$ at $c_i$ and falling to $0$ at $c_{i+1}$, with the first term a left shoulder ($1$ for $x \le c_0$) and the last a right shoulder. On the interior interval $[c_i, c_{i+1}]$ exactly two terms are non-zero,

$$ \mu_i(x) = \frac{c_{i+1} - x}{c_{i+1} - c_i}, \qquad \mu_{i+1}(x) = \frac{x - c_i}{c_{i+1} - c_i}, \qquad \mu_i(x) + \mu_{i+1}(x) = 1, $$

and on the two shoulders the single active term equals $1$. So $\sum_i \mu_i(x) = 1$ for every $x \in \mathbb{R}$: a Ruspini partition, or strong fuzzy partition of unity. These hat functions are precisely the linear (order-2) B-splines on the knot vector $(c_i)$ with the end knots repeated, and the sum-to-one identity is the B-spline partition-of-unity property [@deboor2001splines], intrinsic to the basis rather than a constraint imposed on it — which is the point §6.3.4 makes about the export.

Two consequences follow for the refinement. First, moving any apex knot (keeping the order) rebuilds the two adjacent hat functions and preserves the identity automatically, so *apex-only* refinement searches a monotone knot vector and never leaves the space of partitions of unity: it is free-knot linear-spline fitting under a different name. Second, with constant consequents $m_i$ the exported FIS output $\hat y(x) = \sum_i \mu_i(x)\, m_i$ is the piecewise-linear interpolant through the points $(c_i, m_i)$, flat beyond the outer knots — a function a reader can draw by hand from the rule table, which is the interpretability-by-construction claim in its most literal form. `verify_partition_of_unity` checks the identity numerically; this derivation is why it holds to machine precision, as the Chapter 5 hierarchy's partition-of-unity error ($\approx 10^{-16}$, §5.4) also does for the same reason.
