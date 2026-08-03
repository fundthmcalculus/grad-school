# Chapter 7 — Goals for Completion

This chapter states what remains to turn this proposal into a dissertation. Chapters 3 and 4 are done and published or nearly so; Chapters 5 and 6 have working code and preliminary results. What is left is finishing, hardening, and connecting the pieces into one system, and the plan uses all fifteen months of runway from the proposal defense at the end of 2026 to the final defense in March 2028.

Every goal below carries a named experiment, a metric, a threshold, and the outcome that counts as refutation, because a committee cannot approve what it cannot check. Where the likely outcome is a null result, it is predicted in advance. The statuses here are decisions; Appendix A.6 collects the evidence behind them, including the four claims this project withdrew.

## 7.1 The capstone: the integrated pipeline

The integration deliverable is the pipeline running as one system: Chapter 3's structure discovery, Chapter 5's membership generation, model synthesis from Chapters 4 and 6, refinement from Appendix A, in one reproducible driver over one flagship case study. Two gaps make that more than packaging. Chapter 5's membership functions have never reached Chapter 6's inference machinery, so every Chapter 5 result is a clustering score standing in for the fuzzy-model quality that chapter exists to produce. And the Concrete benchmark is run three ways in three places, so Chapters 4 and 6 cannot be read against each other.

It is not first in the logical order. It is an *integration* result, that the stages compose, and two deliverables precede it: the baseline adapters of checklist **C1**, since the speed argument in Chapters 1, 4 and 8 has no conventional fuzzy method measured beside it, and Goal **G2**, since the coordinate-free regime is so far shown only on synthetic non-metric matrices built from coordinate data. All three are must-haves, and C1 and G2 make the two headline claims measurable.

*Decision rule.* One driver on the UCI shuttle set (Chapter 3's reorder, Chapter 5's memberships, Chapter 6's inference and Ruspini export) at ten seeds on shared splits, plus the *same driver* on one of Goal G2's dynamic-time-warping sets. Metrics: end-to-end accuracy of the FIS built from Chapter 5's memberships, against the Chapter 4 Gaussian construction on identical splits and against CART and a random forest, plus rule and clause counts. Threshold: within 0.02 accuracy of Chapter 4's construction on shuttle, a rule base no more than twice the size, *and* completion on a DTW matrix where Chapter 4's construction cannot run at all.

Two failures, each a result rather than a delay. More than 0.02 behind Chapter 4 means §5.4's adjusted-Rand proxy was optimistic and topological memberships are a *worse* antecedent generator than a per-feature Gaussian mixture where both can run; Chapter 5's contribution narrows to the coordinate-free case and §5.4 has to say so. A driver that will not run on a DTW matrix leaves integration shown only inside the regime Chapter 4 already covers, making §7.3's coordinate caveat fatal. Appendix A.6 has the mechanism already on record for the first, which makes it a design change rather than a schedule slip.

## 7.2 Proposed studies

Two notes on labels. Goal G4 was one row here and is now five tracked items, **G4a–G4e**: four kinds of work with four statuses, where approving one row meant approving all of them. The TODO notes in Chapters 3 to 6 that say "see Goal G4" mean the family, and the one they point at is **G4a**. Two checklist items that were not goals are promoted: **C1**, the fuzzy baseline adapters, and **C7**, the memory-augmented result. Chapter 10 was scheduling neither.

### G4a — The measurement protocol (protocol; substantially in place)

One protocol governs every performance and scaling number, for scalability and stability alike: pinned clocks and thermals, one identified host, ten seeds, error bars, the numeric stack recorded. **Most of it is in place, and Table 7.1 says so.** Done: the ten-seed floor, enforced in `reproduce/common.py` and in every provenance file (B1); the machine block on every emitted table (B2); ratios with seconds kept in the companion CSV (B3); Chapter 3's timing grid re-taken on one host as the run of record `reproduce/outputs/full-14900hx-r2/`, with §3.4 and Table 3.2 re-quoted from it (B5, B5b). Outstanding: clocks and thermals, the submodule-SHA guard (B4), and Table 3.3's cross-precision ordering check, taken at one small size and needing a repeat near the claimed ceiling.

**The complexity item is discharged, with one arm of three qualified.** Classical and stage two land where they should and are confirmed. Stage one does not: its exponent falls short of what $O(N^2 \log N)$ predicts and below the pure quadratic reference, and the log factor invoked to explain that is asserted rather than shown, so stage one is **bounded rather than confirmed**, not cubic, which is what §3.3.1's progression needs. Appendix A.2.4 has the per-$N$ exponents and the constrained fit that would settle it.

The protocol covers *accuracy* claims too, because an accuracy failure set the floor: three seeds to ten refuted the central hypothesis of §4.3.2 and exposed a split diverging far outside the target's range (Appendix A.6). Ten seeds is the floor for any *estimate*; a mean without a spread, over a sample too small to contain the failure modes, is not evidence. The one deliberate exception, single-shot scale demonstrations recorded with hardware, precision and footprint instead, is stated in Appendix A.5.

Three measurements break that floor: **Table 6.4**'s one-seed row, which is checklist **C7** below, the **§6.3.5 optimizer study**'s undeclared host, and **Appendix A.3**'s unseeded speedups. Appendix A.6 names them in full, because a board-wide rule with unnamed violations is worse than a narrower rule stated plainly. One boundary sits inside the rule rather than outside it: Appendix A.4's Table A.2, bit-for-bit reproducible within one host and not portable off it. That is the protocol working.

The hardware half is not a formality either. Re-taking Chapter 3's grid on the workstation withdrew a stage-two timing plateau and a parity band that five runs on the development laptop had measured cleanly (Appendix A.2.4, and A.6 for the figures and the question that survives).

That episode is the strongest argument I have for the protocol, and it is worth stating as a general lesson because the shape recurs. The plateau was measured repeatedly, deterministically, with error bars, at ten seeds — everything the seed-count half of this goal asks for — and it was still an artifact, because every one of those runs was on the same machine. **Repeatability establishes that a measurement is not noise; it says nothing about whether it is a property of the code or of the host.** Only a second host distinguishes those, which is why the harness now records the numeric stack (numpy, scipy, scikit-learn and the BLAS build) alongside the CPU and RAM.

*Decision rule.* Discharged when every numbered table either carries a machine block naming one host at ten seeds, or is labeled a demonstration with hardware, precision and footprint, or is named as an exception in A.6. A table fitting none of the three comes out.

### G4b — The eVAT and clusiVAT head-to-head (experiment; not started)

Chapter 3 owes a direct comparison against eVAT (Meng and Yuan 2018) and clusiVAT on identical datasets. Checklist **C5**, not started in a specific sense: neither implementation is in hand, so this is obtain-or-write before it is measurement, and Chapter 10 gives it its own bar.

*Decision rule.* Exact VAT ordering on shared datasets across the swept grid, ten seeds, run-of-record host. Metrics: wall clock, peak memory, ordering agreement against the serial reference. Two thresholds, since the competitors fail differently. clusiVAT samples and is approximate: agreement 1.000 where it does not reach it, within one order of magnitude on wall clock. eVAT is already exact on a GPU (§3.2 concedes I am not claiming the first), so the claim is in-place footprint at matched $N$. Refuted if eVAT matches the in-place ceiling, collapsing Chapter 3's memory contribution to a constant factor.

### G4c — The datacenter-GPU re-run (experiment; blocked on hardware)

The pairwise-distance kernel loses to the CPU below 1× at low dimension and double precision on a consumer card, and §3.3.3 predicts that full-rate FP64 flips it. Untested, and labeled as such in the chapter and in Table 3.4 (checklist **C8**). Blocked on hardware access rather than effort, so Chapter 10 lists it with its gate instead of a quarter.

*Decision rule.* One kernel, one grid, two cards: the RTX 4080 already measured and one datacenter card, ten seeds, float64 and float32, across the dimensions where the loss is worst. Confirmed if the float64 low-dimensional row reaches ≥ 1× on the datacenter card. Refuted if it stays below 1×, making the loss a property of the *algorithm's* arithmetic intensity rather than the card's throughput ratio, so §3.3.3's caveat becomes a limitation and not a hardware artifact. With no card, §7.4's fallback applies.

### G4d — The matrix-free reorder (build; not started)

§3.3.2 had described this as built. It is not: the package's only matrix-free implementation does not produce the VAT ordering at all, and Table 3.3 reports that as a measured negative result. Nothing in Chapter 3 depends on it, since the in-place scheme at single precision reaches about 155,000 points, covering every result reported. G4d does *not* include a dense-Prim baseline: per §3.2 the compiled kernel already is one.

*Decision rule.* Compute each $D_{i,j}$ on demand, verify the ordering elementwise against the serial reference at $N \in \{1{,}000, 2{,}000, 5{,}000\}$ across ten seeds, then report the reachable $N$ under the 64 GB working cap. Succeeds if the ordering is exact at every size and the ceiling moves past Table 3.3's 154,919-point single-precision figure. Second threshold, a wall clock: if the matrix-free path at 155,000 points is more than an order of magnitude slower than the in-place path there, the memory wall was the wrong wall to attack next and this becomes future work, which is why it is a cut candidate.

### G4e — The general merge operator (open question; partially measured)

The method is named after the merge, and **the merge itself is unfinished.** §3.3.4's divide-and-conquer stitch works and is measured: Table 3.6 has the principled version recovering an adjusted Rand index of 1.00 across every partition tested, where naive concatenation collapses to 0.47. But it is a two-way stitch over blocks chosen by farthest-point sampling, not a general operator. Three unknowns bear on a distributed implementation: whether the stitch composes, how the reconstruction-error bound grows under repeated application, and how to choose block boundaries when the data does not partition cleanly, farthest-point sampling being a heuristic the ablation shows necessary and not sufficient. G4b, G4c and G4d are engineering; this one is a question. Checklist **C10**.

*Decision rule.* Composition is cheap to settle and goes first. Order four blocks, merge them pairwise in both orders and all at once, and compare the three orderings elementwise on the two-moons and circles constructions of Table 3.5 across ten seeds and a grid of partition counts. The operator composes if all three agree exactly and does not if any pair disagrees. If it does not, the half-million-point distributed target is withdrawn, the error-growth question is moot, and Chapter 3 says the merge is a single-level operator, which is what Table 3.6 measures. The error-growth bound and the block-boundary question stay open either way, named as future work.

### C7 — The memory-augmented result, fixed and measured (build + experiment; not started)

§7.3 promises the double pendulum and its relatives to the committee with nothing scheduled behind it, and as things are the promise cannot be kept: the row is the first exception in Appendix A.6, its $R^2$/RMSE pairs are mutually inconsistent about the target's scale, and the rollout the trajectory claim rests on returns its input unchanged. Either the work happens or the promise comes out, and I am scheduling the work, since the defect is diagnosed to a one-line slice.

*Decision rule.* Two parts. Fix `MimoGaussianPredictorMemory.predict_trajectory` to slice `window_size + memory_size` rows of history, and verify that the rollout advances at the four `(window_size, memory_size)` pairs where it is known to fail. Then wire both scripts, the double pendulum and the Atwood machine, through `reproduce/common.py` at ten seeds and re-measure $R^2$ and RMSE so the target's scale is consistent. The one-step claim survives if the memory-augmented arm still leads at ten seeds with the gap larger than the pooled spread. The *iterated* claim survives only if the fixed rollout beats the memoryless model over a horizon, which nothing here supports. If the re-measure erases the gap, the section's headline error reduction goes with it and Chapter 6 loses what it calls its clearest single result: a real risk on a one-seed number, and why this is scheduled first.

### G1 — Direct one-pass membership generation (build + experiment; partly built)

Collapse Chapter 5's two-stage select-then-fit pipeline into a single pass: each block emits its native membership function, the disjunction recombines them, the surviving envelope is the model. The differentiator, and it feeds the capstone.

Phases one to three of `MEMBERSHIP_ROADMAP.md` are built: memberships come straight from the persistence structure at no accuracy cost, the argmax of the generated partition reproduces the hard labels at every scale, each scale is a valid Ruspini partition of unity. Phase five, the one-pass refactor, is plumbing and unattempted. Phase four, the soft kernel-weighted band membership meant to fix small-sample over-segmentation, is the research-interesting piece, and **it has been attempted once, on `feat/mf-phase4-bands`, and it did not fix it**: `log_separated` at small $n$ moves from an adjusted Rand index of 0 to roughly 0.57, against the flat set-cover's 1.00. Appendix A.6 has the mechanism and the finding that the cause is not sampling density. The expectation has been tested and it is losing.

*Decision rule.* Three fixed-structure families from `battery_hierarchical.SCALABLE` — `single_scale`, `many_scale`, `log_separated` — at $n$ = 100, 250, 500, 1,000, 2,000 and 5,000, the one-pass generator against both the two-stage selector and the flat set-cover on identical data. Metrics: recovered granularity vector, per-level adjusted Rand index of the defuzzified partition against each ground-truth level, partition-of-unity error under Ruspini normalization. Thresholds: `many_scale` must not regress from granularities [8, 4, 2] at ARI 1.00 at any $n$; partition-of-unity error must stay at machine precision; `log_separated` at $n \le 500$ must reach ARI ≥ 0.95 with a single band of three clusters, the flat cover's answer and the number phase four failed to reach. Below 0.95 is *not* fixed, leaving no room for partial success.

If soft bands land where phase four landed, the conclusion is not that the kernel needs tuning. It is that birth height is a clean band coordinate only where each cluster occupies a narrow birth range, that the flat set-cover is simply correct for single-level widely-varying-spread data, and that the deliverable is a **single-versus-multi-level gate** detecting one antichain and deferring to the flat cover. G1 then ships as phases one to three plus five, the fix relabelled a gate; Chapter 5 §5.5 needs the same correction.

### G2 — Real non-coordinate benchmarks (experiment; not started, datasets verified)

Everything topological is so far demonstrated on synthetic data with known ground truth. The core niche, working where there are no coordinates, has to be shown on genuinely non-metric domains: time series under dynamic time warping, sequences under edit distance, graphs under a kernel dissimilarity. The single most important credibility gap to close, serving Chapters 3 and 5 both.

**"Beat the baselines" is not available as a success criterion here**, because in the regime the experiment exists to demonstrate there are no baselines to beat: the natural competitors all require coordinates the data does not have. Appendix A.5 lists the verified datasets and derives that unavailability. A criterion that cannot fail is not a criterion, so the criterion is four other things.

*Decision rule.* Four thresholds on the DTW, graph-kernel and Duin–Pękalska families of A.5.

1. **Exactness under real non-metricity.** The ordering on each real dissimilarity matrix must be elementwise identical to the serial reference at every size and seed, which Table 3.7's synthetic rows establish and its last row cannot. Pass-or-fail, and it carries Chapter 3's claim. Refuted by any agreement other than 1.000 anywhere: the engine would then assume a metric internally, and §3.2's regime claim would be false rather than unproven.
2. **Harder non-metricity than the proxy it replaces.** Report the triangle-inequality violation rate. Two sets checked so far violate on 29.3% and 16.3% of sampled triples, against 14% for the fractional-Minkowski stand-in of Table 3.7. Threshold: at least as often as that proxy, or the new experiment is *easier* than the old.
3. **Downstream usefulness against the baselines that can run.** NERFCM given $k$, ConiVAT, single-linkage, beta-plateau and bottleneck-bootstrap all run on a dissimilarity matrix. The gated set-cover, discovering $k$, must land within 0.05 adjusted Rand index of NERFCM-given-$k$ on at least three of the five DTW sets, and `select_coverage_cover` and `select_multiscale` must run on those matrices at all, which per §5.4 they never have, Chapter 5's relational block evaluating only NERFCM. Refuted if the set-cover misses 0.05 on every real non-coordinate set: the coordinate-free property stays true, being provable from the code, and becomes *useless*, narrowing Chapter 5's niche claim to "cannot use coordinates," a proof obligation and not a result.
4. **Reachable size on real relational data.** Crop at 24,000 objects is about 4.6 GB as a float64 dissimilarity matrix, and the natural place to exercise on-demand distance computation, since materializing 288 million DTW pairs is what one wants to avoid. A demonstration in G4a's sense, with hardware, precision and footprint.

### G3 — The hierarchical mixture, finished and compared (build + experiment; one-shot built)

Implement the EM refinement of the mixture of experts, and benchmark the family against the baselines a reviewer will demand on identical splits.

**The baseline list is narrowed deliberately, and Chapter 6 §6.4's version should narrow to match.** To be built: ANFIS and a genetic-algorithm-tuned FIS (checklist **C1**, first in the schedule), CART, Random Forest, flat TSK, and M5 model trees *if* the M5 decision goes that way. Not to be built: Fumanal-Idocin et al. (2025) or the deep TSK fuzzy classifier, for the reasons Appendix A.6 sets out. Reimplementing one, the closest being Fumanal-Idocin et al., displaces G3b: the price, named rather than absorbed.

**M5 is a dependency fault, not an unrun experiment, so it needs a decision date.** Chapter 6 §6.4 records the fault itself. Three branches: patch or upgrade `m5py`; write an M5′ implementation, potentially a month rather than an afternoon; or drop the row and say in Table 6.2 that M5 is unavailable here. Chapter 10 carries **31 March 2027** as the date, before G3's suite is built, because deciding later means discovering mid-suite that one of four baselines is a build.

*Decision rule for the EM, stated in advance because otherwise every outcome is a win.* The document has pre-absorbed all three answers, and Appendix A.6 shows how, along with the outcome here that would embarrass the thesis. So the prediction goes first. **I expect EM ≈ one-shot at second order on Concrete, and I expect what it buys to be stability rather than level.** Metric: $R^2$ and its ten-seed standard deviation at matched capacity on shared splits, plus the divergence rate, the fraction of seeds whose predictions leave the target's observed range, the failure the seed-9 episode exposed. Threshold: a positive result moves the mean by at least 0.02 $R^2$ over the one-shot fit *or* cuts the ten-seed spread by at least a third, with zero divergent seeds. Anything inside that band is the predicted null result, reported as a confirmed prediction: a third instance of *structure before search*, beside §6.3.5's population-method finding and §6.4's refinement decay.

### G5 — Output partitioning: reopened (decision; three studies run)

I recorded this goal as settled, quantile boundaries the recommended default, on a three-seed sweep showing quantile's advantage growing monotonically with target skew. Re-running at ten seeds refutes that, and the recommendation is withdrawn. Appendix A.6 carries the refutation, the three findings that survive it, and the instability figures. What is owed is a decision, and it is owed to Chapter 4 §4.3.2, which currently recommends a scheme on the strength of the withdrawn sweep.

**Estimated effort: about three weeks, correcting a padded figure of one quarter.** The skew sweep is one of the cheapest generators in the harness, well under a minute a run, so re-running it under a guard is minutes of compute, and nothing left needs new data: a framing decision, a guard, one confirming sweep. Chapter 10's bar is sized to match, in place of the sixty days it carried.

*Decision rule.* Re-run the skew sweep with a candidate guard on the quantile path, the simplest being a minimum-occupancy floor on the consequent solve that refuses a bucket whose conditioning is worse than a stated threshold and falls back to the neighbouring bucket's consequent. Metric: the ten-seed standard deviation of $R^2$ at skews above 5, where the instability begins. The guard succeeds if it brings quantile's spread inside uniform's at every skew without costing more than 0.02 in mean $R^2$ at low skew. If it does not, the recommendation is the other branch: a heavily skewed target needs a target transform, the partition scheme being the wrong place to fix it, and §4.3.2 says that instead of recommending a scheme. Either answer closes the goal; leaving it open does not.

### G6 — Interpretability, measured (experiment; not started, and now scoped down)

The interpretability claim should be measured, not asserted, and the earlier disjunction ("an established interpretability metric or a small expert-audience study") named no metric anywhere in the document.

**What G6 is.** Rule counts, clause counts and root-to-leaf path lengths at matched accuracy, filling Table 6.3's pending row; those are read off a fitted model and owed regardless. Plus a *named* metric family, both already cited here: Valente de Oliveira's semantic constraints on membership functions, and the interpretability criteria of Guillaume and Charnomordic's partition-generation and FisPro work, evaluated on the Ruspini export of §6.3.4. Concretely coverage, distinguishability, normality and partition-of-unity error, per model, alongside the counts. The construction satisfies two by design and the phase-2 work already measures one at machine precision, which makes the report a check on a claim rather than a new experiment.

**The expert-audience study is dropped**, and stays post-defense work; Appendix A.6 sets out why it would not be defensible as things stand. The consequence is stated rather than absorbed. §6.2 lists "whether a rule base mixing one- and two-dimensional antecedents reads coherently to a domain expert" as a question only a person can answer and points at G6 as its home; that question is now unanswered here, and §6.2's bound has to hold for the hierarchy as built. §2.6's position on post-hoc explanation is reframed instead of tested, since "post-hoc answers a different question" needs no experiment.

*Decision rule.* Counts and the four semantic criteria on the flat FIS, the fuzzy tree, the mixture and the Ruspini export, on Concrete and one G3b dataset, at matched accuracy. Threshold, for the claim that the hierarchy buys a readable path at comparable accuracy: at accuracies within one pooled standard deviation of each other, the hierarchy's mean root-to-leaf path must mention strictly fewer variables than the flat model's rules do, and the Ruspini export must satisfy coverage and partition-of-unity exactly. Refuted if the paths are no shorter at matched accuracy, in which case the interpretability-for-accuracy trade is a loss and not a trade, and §6.5 has to say so.

### G7 — Adaptive multi-scale (open question; stretch, first cut)

Replace Chapter 5's gap heuristic for band discovery with a model-based criterion, a change-point or barcode-stability test, so that overlapping density scales become tractable. Explicitly a stretch goal and the first thing I cut. "Tractable" needs a criterion, and the evidence to beat exists.

*Decision rule.* The same three scaling families as G1, plus one purpose-built family with deliberately overlapping log-birth ranges. Metrics: recovered scale count and per-level adjusted Rand index. Threshold: the correct scale count at every $n$ from 100 to 5,000 — one band for `log_separated`, three for `many_scale`, one for `single_scale`, where the gap heuristic reports a spurious extra coarse band — and per-level ARI ≥ 0.95, against the gap heuristic's granularities of [1, 1, 1] and ARI 0.00 at $n$ = 100 and 500, and 0.57 at $n$ = 250, which is the phase-four kernel band's ceiling too.

Refuted if a change-point or stability criterion cannot beat the flat set-cover's 1.00 on single-level widely-varying-spread data. The phase-four diagnosis of Appendix A.6 would then be the right one, birth height unusable as a band coordinate, and the deliverable is again the single-versus-multi-level gate. A smaller result, still worth reporting, and why this is the designated first cut: its likeliest outcome is a negative result G1 would also produce.

### G8 — Joint memberships where the structure requires them (retargeted post-defense)

Every membership function here is one-dimensional, which keeps the rule count linear and the clauses readable, and is a hard expressive limit: a ring is not the intersection of per-axis intervals. The proposal extends to joint two-feature memberships *only* for clusters with no faithful axis-aligned description, using the topological disjunct count of §5.3.5 as the detector.

**The construction is retargeted post-defense, to a journal extension**, for three reasons Appendix A.6 sets out: the quarter it had held is already oversubscribed, the construction spends interpretability, and the §5.3.5 detector that would justify building it has never fired.

**What stays in scope is the measurement, not the construction.** *Decision rule.* Run the disjunct counter over G2's real datasets and report how often a class has no faithful axis-aligned description; it folds into G2's sweep at almost no cost and decides G8's fate. Common such clusters mean this construction is the wrong tool for that data, a limit on scope; rare ones make G8 worth building afterwards. Either way §6.2's claim should hold *without* G8, so the hole is bounded rather than pending.

## 7.3 Application showcases

The flagship case study is the **UCI shuttle set**, the same 58,000-point NASA reentry telemetry Chapter 3 uses to demonstrate scale: one dataset carried from the reordering through membership generation and model synthesis to the final rule base, public where the psychiatric set of §3.3.2 is not, and imbalanced the way the complement rule of §4.3.5 was built for (seven classes, roughly 80% of records in one flight condition).

One limitation belongs here. The shuttle set **has coordinates**, seven sensor channels, so a capstone built on it does *not* exercise the coordinate-free regime Chapter 5's premise rests on. The capstone and Goal G2 answer different questions, which is why §7.1 requires the *same driver* to complete on one of G2's DTW matrices.

I will also carry the memory-augmented dynamical-systems result, the double pendulum and its relatives, as a deliberately aerospace-flavored demonstration. **That promise is conditional on C7 above**, whose defects §7.2 and Appendix A.6 list, and which Chapter 10 schedules in the first quarter. If C7's ten-seed re-measure erases the gap between the memoryless and the memory-augmented arm, this showcase comes out of the defense.

The **BETH** host-telemetry set needs a sentence, because Chapter 4 describes it as the testbed for the open-set claim while every open-set number there is measured on Glass. The obstacle is a research decision before a coding one: leave-one-class-out needs at least three classes and BETH is binary, so it needs its own one-class path, and the comparison against a one-class SVM and an isolation forest has to be scored on a footing that path defines. Chapter 10 gives the decision a short slot in 2027 Q2 with a fallback: if the one-class path is not settled by the end of that quarter, the open-set claim stands on Glass as a *mechanism* validation and not a performance claim, and Chapter 4 §4.4 says so in those words. Glass at 214 samples across six classes is a stress test, not a demonstration.

## 7.4 Risks and the de-scoping order

One risk has no goal above it: the prior-art overlap in Chapter 5 with Bonis and Oudot. If a reviewer collapses the three axes of daylight already stated, the integration and the one-pass membership generation still stand as novel. The EM's fallback is in G3, the GPU's in G4c.

**The baseline tables in Chapters 4 and 6 are the first experiments I owe, which means they are now first in the schedule.** Chapter 10 had buried those adapters inside Goal G3 in the final quarters with no bar of their own, the largest single inconsistency between the two chapters. Eleven cells across Tables 4.5 and 6.2 (eight and three) read `N/A` until the adapters exist. Chapter 1 §1.1 meanwhile states the speed claim as an absolute, seconds measured with no ratio quoted, and §7.1's "orders of magnitude" phrasing is brought into line with it.

Two exposures a committee will find. My selection gate in Chapter 5 loses outright to more aggressive selectors on the bridge case, and Table 5.3's coverage column weakens even the conservatism defence I recorded: bottleneck-bootstrap repairs the bridge while still declining three-quarters of the noise. And Chapter 3's non-metric claim rests on synthetic matrices built from coordinate data until G2 runs.

The runway is oversubscribed, not merely tight, so the de-scoping plan is an ordered list rather than one designated victim: **G7** first, then **G3b** narrowed to three datasets, then **G6's** semantic-criteria half beyond the counts, then the §9.3 **VAT complexity note**, then **G4d**, then **G4e** narrowed to the composition test alone. Chapter 10 §10.6 carries that list with the reason for each cut, and the two must not drift apart. What I will not cut, and would rather extend the timeline than lose: the **C1** adapters, **G2**, **G4a**, **C7**, and the capstone. Those five carry the two headline claims and the pipeline argument. Chapters 3 and 4 remain the floor: done, and defensible on their own.

## 7.5 Goals, mapped

**Table 7.1 — Goals for completion, mapped.** Read the **Kind** column first: it says what "done" means. A *protocol* is discharged when every table complies or is named as an exception; an *experiment* by a measurement against §7.2's threshold, and can be refuted; a *build* by working, verified code; an *open question* may end in a negative result, and two of them probably will. Quarters are relative to the confirmed December 2026 proposal, with the final defense in March 2028 (Chapter 10); rows are ordered by target quarter and the cut order is §10.6's. Items are also tracked in `CHECKLIST.md` and `ACTION_ITEMS.md`, with the identifier in each status cell.

| Goal | Kind | Feeds | Current status | Priority | Target |
|---|---|---|---|---|---|
| **C1** ANFIS + GA-FIS baseline adapters | experiment | Ch 1, 4, 6, 8 | not started; eleven cells across Tables 4.5 and 6.2 read `N/A` (C1, D4) | **must — first** | 2027 Q1 |
| **G4a** measurement protocol | protocol | Ch 3, 4, 5, 6 | seed floor, single host, machine block, ratio reporting and estimates-vs-demonstrations in place (B1, B2, B3, B5, B5b); clocks/thermals, the SHA guard (B4) and three named exceptions outstanding | must | 2027 Q1 |
| **C7** rollout fix + Table 6.4 in the harness | build + experiment | Ch 6, §7.3 | defect diagnosed to a one-line slice; row is one-seed, outside the harness (C7) | must | 2027 Q1 |
| **M5** decide: patch `m5py`, write M5′, or drop the row | decision | Ch 6 | blocked on a dependency fault, not an unrun experiment (D4) | must | decide by 2027-03-31 |
| **G4b** eVAT + clusiVAT head-to-head | experiment | Ch 3 | not started; neither implementation in hand (C5) | must | 2027 Q1–Q2 |
| **G1** one-pass membership generation | build + experiment | Ch 5, capstone | phases 1–3 built; phase 4 attempted once and did **not** fix the target failure; phase 5 not attempted | differentiator | 2027 Q2 |
| **G5** output partitioning (**reopened** — see §7.2) | decision | Ch 4 | three studies run; recommendation withdrawn; ~3 weeks of work remains | should | 2027 Q2 |
| **G2** real non-coordinate benchmarks (DTW/edit/graph) | experiment | Ch 3, 5 | not started; datasets identified and verified in this environment | **must — top credibility item** | 2027 Q2–Q3 |
| **BETH** one-class path, or stand on Glass | decision + experiment | Ch 4 | not started; needs a one-class evaluation path before any code | should | 2027 Q2 |
| **C3** Ch 5 → Ch 6 minimal end-to-end | experiment | Ch 5, 6, capstone | not started; pulled forward out of the 2028 Q1 capstone (C3) | must | 2027 Q3 |
| **G3** HME EM + narrowed baseline suite | build + experiment | Ch 6 | one-shot built; EM designed, not implemented; baseline list narrowed in §7.2 | must | 2027 Q3–Q4 |
| **G4e** general merge operator — composition test | open question | Ch 3 | two-way stitch measured (Table 3.6); composition, error growth and block choice untested (C10) | should | 2027 Q4 |
| **G3b** broadened dataset suite | experiment | Ch 6, App A.2.3 | not started; no loaders wired | should — **cut 2** | 2027 Q4 |
| **capstone** integrated end-to-end pipeline | integration | Ch 3→5→6 | not started; depends on G1 and C3 | must | 2028 Q1 |
| **G6** interpretability: counts + named criteria | experiment | Ch 6 | not started; counts computable from a fitted model today; expert study **dropped**, see §7.2 (C9) | should — **cut 3** | 2028 Q1 |
| **G7** adaptive multi-scale (overlapping scales) | open question | Ch 5 | not started; phase-4 evidence says birth height is the wrong band coordinate | stretch — **cut 1** | 2028 Q1 |
| **G4d** matrix-free reorder | build | Ch 3 | not started; the one existing implementation is a measured negative result (Table 3.3) | could — **cut 5** | unscheduled |
| **G4c** datacenter-GPU re-run | experiment | Ch 3 | **blocked on hardware access**, not on effort (C8) | should | gated, not scheduled |
| **G8** joint memberships for non-axis-aligned clusters | open question | Ch 5, 6 | detector exists but has never returned a value other than 1; **retargeted post-defense**, see §7.2 (E3, E4) | post-defense | not in the runway |

---

*Draft — Chapter 7 prose, in the author's voice. One table. Goals tracked in `../ACTION_ITEMS.md` and `../CHECKLIST.md`, and mapped to the Chapter 10 timeline.*
