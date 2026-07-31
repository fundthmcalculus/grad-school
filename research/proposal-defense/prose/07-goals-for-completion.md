# Chapter 7 — Goals for Completion

This chapter states what remains to turn this proposal into a dissertation. I have a real head start — Chapters 3 and 4 are done and published or nearly so, and Chapters 5 and 6 have working code and preliminary results — so the goals here are about finishing, hardening, and, above all, connecting the pieces into one system. The plan is deliberately aggressive: the proposal defense is at the end of 2026 and the final defense is in March 2028, which is a little over a year of research runway, and I intend to use it to produce a substantial and convincing body of results rather than a minimal one.

## 7.1 The capstone: the integrated pipeline

The single most important deliverable is the end-to-end pipeline running as one system. Each chapter proves a stage in isolation — structure discovery in Chapter 3, membership generation in Chapter 5, model synthesis in Chapters 4 and 6, optional refinement in Appendix A — but the central claim of the dissertation, that a structure-first approach builds interpretable fuzzy models orders of magnitude faster and at far greater scale, is only fully demonstrated when the stages are chained: a bare dissimilarity matrix in, a readable fuzzy inference system out, with the speed and scale numbers measured across the whole path. The concrete deliverable is one reproducible driver and one flagship case study carried through every stage from start to finish.

## 7.2 Proposed studies

I organize the remaining work as six goals.

**G1 — Direct one-pass membership generation.** Collapse Chapter 5's two-stage select-then-fit pipeline into a single pass, in which each block emits its native membership function, the disjunction recombines them, and the surviving envelope is the model. The research-interesting piece is a soft, kernel-weighted band membership, which I expect to fix the small-sample over-segmentation. This is the differentiator, and it feeds the capstone.

**G2 — Real non-coordinate benchmarks.** Everything topological is so far demonstrated on synthetic data with known ground truth. The core niche — that this works where there are no coordinates — has to be shown on genuine non-metric domains: time series under dynamic time warping, sequences under edit distance, graphs under a kernel dissimilarity. This is the single most important credibility gap to close, and it serves both Chapter 3 and Chapter 5.

**G3 — The hierarchical mixture, finished and compared.** Implement the EM refinement of the mixture of experts, and benchmark the whole model family against the baselines a reviewer will demand — ANFIS, CART/C4.5, M5 model trees, flat TSK, and the recent Fumanal-Idocin and D-TSK-FC methods — on identical splits.

**G4 — Scale and hardware credibility.** This is the consolidation point for the board-wide repeatable-performance standard that recurs throughout the document. Every performance and scaling number, for both scalability and stability, is to be re-run under one fixed protocol: pinned clocks and thermals, multiple seeds, reported error bars, and a datacenter GPU with full double-precision throughput. It also includes the head-to-head against eVAT and clusiVAT that Chapter 3 owes, and a push toward a distributed pVAT at half a million points.

**G5 — Interpretability, measured.** The interpretability claim should be measured, not asserted: rule counts, path lengths, and either an established interpretability metric or a small expert-audience study, and an empirical demonstration of the Magdalena condition (hierarchies over named inputs).

**G6 — Adaptive multi-scale (stretch).** Replace Chapter 5's gap heuristic for band discovery with a model-based criterion — a change-point or barcode-stability test — so that overlapping density scales, which the gap heuristic cannot handle, become tractable. I mark this explicitly as a stretch goal and the first thing I will cut if time runs short.

## 7.3 Application showcases

The flagship case study is still flexible, and I will settle it with my committee, but the two strong candidates are a large cybersecurity/IoT dataset (RT-IOT2022 or an IoT-botnet set), where speed, scale, and readable rules all matter at once, and the UCI-58 shuttle set already used in Chapter 3, which gives a clean single thread from structure discovery through to the final model. Alongside the primary showcase I will carry the memory-augmented dynamical-systems result — the double pendulum and its relatives — as a deliberately aerospace-flavored demonstration for the committee.

## 7.4 Risks and an honest de-scoping plan

I would rather name the risks than have them named for me.

The nearest one is the prior-art overlap in Chapter 5 with Bonis and Oudot; my mitigation is the three axes of daylight I already stated, and if a reviewer collapses them, the fallback is that the integration and the one-pass membership generation still stand as novel. The EM in Chapter 6 is designed but not built, and if the implementation slips, the one-shot mixture and the fuzzy trees are already complete contributions. The GPU story depends on getting time on a datacenter card; without it I will report the CPU-parallel and single-precision results clearly scoped, rather than overclaim. And the baseline tables in Chapters 4 and 6 are the first experiments I owe, because the speed and accuracy claims are only as strong as the methods they are measured against.

The floor under all of this is that Chapters 3 and 4 — the accelerated exact VAT engine and the fast Mixture-of-Gaussians synthesis — are done and defensible on their own. The proposed work extends and connects them; it is not load-bearing for the degree in the way a single make-or-break experiment would be. That is by design.

## 7.5 Goals, mapped

**Table 7.1 — Goals for completion, mapped.** Quarters are relative to the ~Dec 2026 proposal; final defense March 2028 (see Chapter 10). Status and items also tracked in `ACTION_ITEMS.md`.

| Goal | Feeds | Current status | Priority | Target |
|---|---|---|---|---|
| **G4** repeatable perf + eVAT/clusiVAT head-to-head | Ch 3, 5, 6 | not started | must | 2027 Q1 |
| **G1** one-pass membership generation | Ch 5, capstone | preliminary done | differentiator | 2027 Q2 |
| **G2** real non-coordinate benchmarks (DTW/edit/graph) | Ch 3, 5 | not started | must | 2027 Q3 |
| **G3** HME EM + full baseline suite | Ch 6 | one-shot done; EM design-only | must | 2027 Q3–Q4 |
| **capstone** integrated end-to-end pipeline | Ch 3→5→6 | not started | must | 2028 Q1 |
| **G5** interpretability, measured | Ch 6 | not started | should | 2028 Q1 |
| **G6** adaptive multi-scale (overlapping scales) | Ch 5 | not started | stretch (first cut) | 2028 Q1 |

---

*Draft — Chapter 7 prose, in the author's voice. One table placeholder. Source outline in `../chapters/07-goals-for-completion.md`; goals G1–G6 tracked in `../ACTION_ITEMS.md` and mapped to the Chapter 10 timeline.*
