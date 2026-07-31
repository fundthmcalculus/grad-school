# PhD Dissertation Proposal — Master Outline

**Author:** Scott Phillips
**Committee:** Dr. Kelly Cohen (chair) · Dr. Vladik Kreinovich · Dr. Manish Kumar · Dr. Ali Minai · Dr. Justin Zhan
**Program:** Department of Aerospace Engineering & Engineering Mechanics, College of Engineering and Applied Sciences, University of Cincinnati
**Status:** Scaffold / annotated outline — prose to follow after outline approval
**Target proposal defense:** ~December 2026 (≈4 months out)
**Target final (terminal) defense:** March 2028 (~15-month research runway)

---

## Title

> **Reproducing Like Tribbles: Scaling Fuzzy Inference Systems from Hundreds to Hundreds of Thousands**

Spine per author decision: primary framing = *scalable full-pipeline*, mechanism = *structure-driven* ("structure before search"), property = *interpretable*. Overarching goal: **orders-of-magnitude faster and larger FIS training and inference.** The "tribble" motif ties the three code repositories (tribble-cluster, tribble-fis, tribble-opt) to the prolific-scaling theme.

---

## Central thesis statement (one paragraph — to be written)

> Classical Fuzzy Inference Systems are prized for interpretability but are slow and hard to train at scale: rule-base explosion is exponential in inputs, and the dominant training methods (genetic algorithms, gradient descent, ANFIS) are stochastic, initialization-sensitive, and do not scale. This dissertation argues that the *structure already latent in the data* — recovered through fast relational/topological analysis (VAT/iVAT, single-linkage, persistence) — can replace most of that stochastic search, yielding FIS that train orders of magnitude faster, scale to hundreds of thousands of samples, and remain interpretable by construction. We develop and validate a pipeline: (1) scalable exact structure discovery, (2) automatic, density-free membership-function and rule generation from that structure, (3) fast closed-form/one-pass model synthesis, with (4) an optimization engine for optional refinement.

---

## The four research pillars → chapter mapping

| Pillar | Repo | Status | Home |
|---|---|---|---|
| 1. Scalable structure discovery (pVAT — priority-queue/Borůvka VAT, VAT/iVAT, VAT↔TSP) | `tribble-cluster` | **Done** (NAFIPS papers 1 & 2) | Part II, Ch 3 |
| 2. Automatic selection + membership generation (persistence-gated set-cover, multi-scale, relational MF) | `gated-minimax-selection` | **Proposed / active** (strong preliminary results) | Part III, Ch 5 |
| 3. Fast interpretable FIS synthesis (MoG antecedents, ridge-TSK, hierarchical trees/HME, Ruspini, MIMO) | `tribble-fis` | **Split:** MoG done; ridge-TSK/hFIS partial | Part II Ch 4 (MoG) + Part III Ch 6 (ridge-TSK/hFIS) |
| 4. Optimization / refinement engine (metaheuristics, Lin-Kernighan, quality-diversity, perf kernels) | `tribble-opt` | **Done** (supporting infrastructure — NOT core dissertation) | **Appendix A.3**; brief motivating mention only in Ch 2; standalone-paper flags below |

---

## Document structure

### Part I — Introduction & Preliminaries
- **Ch 1 — Introduction** (`01-introduction.md`): Prelude/motivation, unique contributions, dissertation outline.
- **Ch 2 — Background & Preliminaries** (`02-background.md`): Fuzzy logic & FIS; VAT/iVAT & single-linkage; persistence/TDA; the optimization/initialization bottleneck (brief — motivates "structure before search"; tribble-opt library details live in Appendix A.3); interpretability & the accuracy–interpretability trade.

### Part II — Completed Work
- **Ch 3 — Scalable Structure Discovery: pVAT** (`03-scalable-structure-discovery-pvat.md`): accelerated exact VAT/iVAT via priority-queue argmin (CPU) and Borůvka MST (parallel/GPU), in-place memory, divide-&-conquer with principled stitch, arbitrary/non-metric dissimilarity; VAT↔TSP hot-start. Name honors Dr. Kreinovich's priority-queue observation (was "mergeVAT").
- **Ch 4 — Fast Interpretable FIS Synthesis via Mixture-of-Gaussians** (`04-fast-fis-synthesis-mog.md`): MoG antecedent/rule generation with no GA/GD, closed-form ridge-TSK consequents (done portion), interpretability by construction.

### Part III — Proposed Work & Goals for Completion
- **Ch 5 — Topological Membership Generation** (`05-topological-membership-generation.md`) *(proposed, with preliminary results)*: persistence-gated set-cover selection, multi-scale persistence (Option D), relational fuzzy MF from the minimax hierarchy — the bridge from clustering to FIS.
- **Ch 6 — Hierarchical & Refined Fuzzy Models** (`06-hierarchical-refined-fis.md`) *(partially done)*: ridge-TSK completion, fuzzy trees / hierarchical mixture of experts (HME), EM refinement, antecedent refinement, MIMO temporal memory for dynamical systems.
- **Ch 7 — Goals for Completion** (`07-goals-for-completion.md`): integrated end-to-end pipeline + aggressive planned experiments and milestones.
- **Ch 8 — Conclusion** (`08-conclusion.md`).
- **Ch 9 — Publications** (`09-publications.md`).
- **Ch 10 — Timeline** (`10-timeline.md`).
- **Bibliography** (`bibliography.md`) · **Appendix** (`appendix.md`).

---

## Standalone-paper opportunities (future — not core dissertation)

Per author decision, tribble-opt is supporting infrastructure parked in **Appendix A.3**, not a dissertation chapter. These pieces are still strong enough to be their own papers if pursued later:
- **Performance-engineering study** (`tribble-opt/PERFORMANCE_REPORT.md`): 15-item ranked findings, truncnorm 177× fix, ship-once 7.5×, njit local search ~370×. A methods/systems paper.
- **Quality-Diversity over legacy solvers** (`tribble-opt/QD_PARETO_PLAN.md`): CVT-MAP-Elites + Iso+LineDD as a drop-in archive layer. An optimization-venue paper.
- **Lin-Kernighan dual-backend + VAT-blocked TSP** (`tribble-cluster` VAT-TSP thread + `tribble-opt/LK_PERFORMANCE_REPORT.md`): could be spun from Ch 3's VAT↔TSP material.
- **Exact GPU/parallel VAT engine** as its own systems paper (differentiate vs eVAT, Fast-VAT, clusiVAT).

## Cross-cutting honesty / defensibility notes (carry into every chapter)

The repo docs are candid adversarial self-reviews; preserve that rigor:
- **Novelty = composition + regime, not primitives.** VAT, iVAT, FCM, MST-cut, minimax, Lin-Kernighan, persistence gating are all faithfully re-implemented prior art. State this plainly; claim the integrated architecture and the unoccupied regimes.
- **Concede nearest precedents** explicitly (Bonis & Oudot 2014/2018 for persistence-based fuzzy membership; Medina-Chico 2001 for soft trees w/ linear leaves; Wu 2020 for TSK≡MoE; eVAT/Fast-VAT/clusiVAT for fast VAT; Magdalena 2018 for hierarchy≠interpretability).
- **Report negatives** (VAT is a poor closed-tour TSP start; GPU FP64 pairwise loses at low-d; population methods overfit CV in antecedent refinement; tree/HME trades accuracy for interpretability).
- **Fixes to make before submission:** retire the "priority-queue MST speedup" O-notation framing for dense graphs; drop the ungrounded "pVAT six-orders-of-magnitude" web claim; fix Zhang-2023 attribution; re-verify DOIs.
- **BOARD-WIDE TODO — repeatable performance results (scalability AND stability):** every performance/scaling number in the dissertation (Ch 3 pVAT, Ch 5 selection scaling, Ch 6 benchmarks) must be reproduced under one fixed protocol before it is cited — pinned clocks/thermals, multiple seeds, reported error bars, and a datacenter GPU with full-rate FP64. Current numbers are single-machine point estimates, some thermally throttled. Consolidated as Goal G4 (Ch 7); a `TODO — repeatable performance` note is mirrored in each chapter that reports numbers.

---

## Reproduction harness

`reproduce/` at the repo root is the single entry point for regenerating results. Each proposal table has a generator in `reproduce/tables/` that runs over a fixed seed set and emits Markdown + CSV to `reproduce/outputs/` with mean ± std; unavailable methods/datasets are reported as such, never estimated. `reproduce/manifest.py` registers every experiment across the four submodules (command, env, datasets, hardware tier) for a future `run.py` orchestrator. Goal: one-command reproduction, with hardware-gated runs skipped cleanly.

## Source-document map (where the material lives)

- **Ch 3:** `tribble-cluster/` — `docs/performance-novelty.md`, `docs/novel-niche.md`, `docs/novelty-review.md`, `docs/vat-tsp-prior-art.md`, `experiments/findings/{SUMMARY_REPORT,white-paper,ADVERSARIAL_EVAL,STITCHED_VAT,HARDENING,DC_VAT_SCALING,VAT_TSP_*}.md`; `presentations/quals/slides/{paper1,paper2}.md`.
- **Ch 4:** `tribble-fis/` — `presentations/quals/slides/draft-paper3.md`, `gaussian_mixture/*`, `src/tribblefis/{gaussian_classifier,gaussian_regressor,gauss_math}.py`, `consequent-plan.md` (ridge solver).
- **Ch 5:** `gated-minimax-selection/` — `SUMMARY.md`, `notes/{FINDINGS,OPTION_D_MULTISCALE,SELECTION_METHODS_COMPARISON,MEMBERSHIP_ROADMAP,RELATIONDATA,SCALING_STUDY}.md`; memory: `project_option_d_multiscale.md`, `project_selection_methods_comparison.md`.
- **Ch 6:** `tribble-fis/tribble-tree/` — `README.md`, `HFIS_NOVELTY_REVIEW.md`, `EM_REFINEMENT.md`, `hfis_review.bib`; `MIMO_MEMORY_GUIDE.md`; `consequent-plan.md`.
- **Ch 2 / infra:** `tribble-opt/` — `PERFORMANCE_REPORT.md`, `QD_PARETO_PLAN.md`, `LK_PERFORMANCE_REPORT.md`, `PERF_PLAN.md`, `CYTHON_ANALYSIS.md`.
