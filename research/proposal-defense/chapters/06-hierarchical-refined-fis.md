# Chapter 6 — Hierarchical & Refined Fuzzy Models (Partially Complete → Proposed)

**Status:** Outline · Part III (PARTIALLY DONE: ridge-TSK solver + trees/HME built; EM refinement design-only; antecedent refinement in progress)
**Repo:** `tribble-fis/tribble-tree/` (`fuzzytree`), `tribble-fis/consequent-plan.md`, `tribble-fis/MIMO_MEMORY_GUIDE.md`; refinement via `tribble-opt`.
**One-line claim:** a single closed-form firing-weighted ridge least-squares consequent solver serves as one shared primitive across flat FIS, soft fuzzy trees, and hierarchical mixture-of-experts — trading a little accuracy for an explicit, readable variable hierarchy — plus optional structure-driven refinement and temporal-memory extensions.

---

## 6.1 Introduction

- Motivation: Ch 4 gives fast flat models; some problems want an explicit *hierarchy* of named-variable decisions (interpretability) and/or higher accuracy via local experts. And any generated FIS may want optional refinement.
- **The unifying technical idea (thesis-level for tribble-fis):** for fixed firing strengths the TSK output is *linear in the consequent coefficients* → optimal consequents have a closed-form regularized (ridge) least-squares solution. Use that one primitive everywhere.
- Contributions: (1) shared ridge-TSK consequent solver (done); (2) soft fuzzy trees w/ exact per-leaf ridge leaves (done); (3) hierarchical mixture of fuzzy experts (HME) (one-shot done, EM proposed); (4) antecedent refinement via optimizers (in progress); (5) Ruspini export (done); (6) MIMO temporal-memory FIS for dynamical systems (done).

## 6.2 Background & Prior Art (concede nearest competitors)

- **Medina-Chico et al. 2001** — soft CART with *linear* leaves fit by backprop: the single most important competitor. Our scoped daylight: *exact per-leaf firing-weighted ridge* leaf (closed form, not backprop).
- **Wu et al. 2020** — single-layer TSK ≡ mixture-of-experts: narrows the HME nesting claim; engage directly.
- **Jordan–Jacobs 1994** (HME + EM); Raju et al. 1991 (hierarchical fuzzy, linear rule growth); Janikow 1998 / Yuan–Shaw 1995 (fuzzy trees, ambiguity); Olaru–Wehenkel 2003 (soft > crisp); Fumanal-Idocin 2025 (closest recent).
- **Magdalena 2018** ("Do hierarchical fuzzy systems really improve interpretability?") — the mandatory rebuttal: our hierarchy splits only on **named original inputs**, never synthetic intermediates.

## 6.3 Methodology

### 6.3.1 Shared ridge-TSK consequent solver (DONE — `consequent-plan.md`)
- Closed-form regularized weighted least squares (ridge normal equations; intercept/bucket-mean columns unpenalized); pluggable basis (raw monomials vs **Legendre/Chebyshev** for conditioning); CV-selected (order, λ); sparse interaction selection (Lasso/ElasticNet).
- Replaces per-bucket pinv/lstsq + L-BFGS. **Caution (verified):** do NOT motivate with "ANFIS LSE overfits" — that premise was refuted; motivate by conditioning + a single reusable primitive.

### 6.3.2 Soft fuzzy trees (DONE — `fuzzytree`)
- CART-style recursive partitioning; split criteria = firing-weighted variance reduction (regression), Yuan–Shaw ambiguity (classification), Janikow info-gain, differentiation prefilter; soft multi-path membership; **exact ridge-TSK leaves.**
- Rendered as short IF–THEN rules (one per root→leaf path, only path variables, importance-ordered) — `render_tree_text` / `plot_fuzzy_tree`.

### 6.3.3 Hierarchical mixture of fuzzy experts — HME (one-shot DONE; EM PROPOSED)
- Tree of fuzzy partition-of-unity gates; leaves = full multi-rule TSK sub-FIS; soft-inclusion overlapping training sets.
- **EM refinement is design-only** (`EM_REFINEMENT.md`): E-step gate×expert responsibilities; M-step √h-weighted ridge updates, per-expert σ²; closed-form Gaussian gate updates; log-sum-exp guards, variance floors, starved-component pruning. **Novelty must not rest on the estimator.**

### 6.3.4 Declarative structure & Ruspini export (DONE)
- `VariablePlan`: JSON-serializable structure spec, precedence lattice (path-pin > level-order > auto), expert-in-the-loop steering.
- Ruspini export: implicit per-class Gaussian mixture → explicit shared triangular strong partition; refine **apex knots only** (partition-preserving) via line-search/GA.

### 6.3.5 Antecedent refinement via optimizers (IN PROGRESS — links to tribble-opt)
- Optimize never-tuned (μ, σ) against a held-out-fold objective with the ridge solver as inner fitness: Differential Evolution → real-valued GA (tournament, SBX/BLX-α, Gaussian mutation, elitism) → optional ADAM polish.
- **Honest finding to report:** local L-BFGS-B often beats GA/DE here — population methods overfit the CV estimate. (Consistent with the dissertation thesis that structure beats blind search.)

### 6.3.6 MIMO temporal-memory FIS (DONE — `MIMO_MEMORY_GUIDE.md`)
- Augment each feature with current / short-term-avg / long-term-avg windows (+ optional time index) → LSTM-like temporal modeling without recurrence; single-step + iterative trajectory rollout.
- Application/validation domain: dynamical systems (double pendulum, Atwood machine) — an aerospace-flavored showcase for the committee.

## 6.4 Results (current) & Proposed Experiments

*Current (from `tribble-tree/README.md`, `consequent-plan.md`, `HFIS_NOVELTY_REVIEW.md`):*
- **UCI Concrete (regression):** flat R²=0.658 → 1st-order tree R²=0.746 → **HME R²=0.791** (RMSE 9.38→8.09→7.34 MPa). Auto-tree recovers domain knowledge (splits on Cement, then Age at the 28-day mark).
- **PhiUSIIL phishing:** flat acc 0.998 → HME 0.996; trees split on interpretable signals (HasSocialNet, HasCopyrightInfo, URLSimilarityIndex).
- **Antecedent refinement:** Concrete R² ~0.88 → 0.92 (repo numbers, to re-verify).
- **MIMO memory:** double pendulum R² 0.92→0.96 / RMSE 0.045→0.028 (~33% better).
- **Honest scope:** tree/HME is an interpretability-for-accuracy *trade*; it does not beat the flat base on accuracy nor shrink rule count — payoff is explicit hierarchy + readability at shallow depth.

*Proposed (to complete the chapter):*
- Implement + evaluate **EM refinement of HME** (structural EM, multi-input gates, deterministic-annealing EM).
- Add baselines reviewers will demand: **ANFIS, CART/C4.5, M5 model trees, flat TSK, Fumanal-Idocin 2025, D-TSK-FC.**
- Broaden benchmarks already scaffolded: turbine, wave-energy-converter, wine, DARWIN, BETH, IoT-botnet, power consumption.
- Close open literature searches (knot/breakpoint optimization precedent; dedicated fuzzy-MoE search); fix Zhang-2023 attribution.

## 6.5 Discussion & Contributions

- The shared-primitive architecture is the defensible novelty (every block is prior art; the integration is not).
- Interpretability payoff quantified against the accuracy trade; Magdalena-2018 rebuttal front and center.
- Ties the pipeline together: Ch 5 supplies antecedents when there's no Gaussian assumption; Ch 6 supplies the consequents, hierarchy, and refinement; Ch 2 supplies the optimizers.

---

### Open items
- EM implementation is the biggest single proposed deliverable here — size it in the timeline.
- Decide whether MIMO/dynamical-systems becomes its own short application chapter or stays a section (recommend: section, but it's a nice aerospace hook for the committee).
- De-duplicate the six caller scripts' predict loops (engineering debt).
