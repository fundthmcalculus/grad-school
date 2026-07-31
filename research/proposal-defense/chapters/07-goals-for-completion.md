# Chapter 7 — Goals for Completion

**Status:** Outline · Part III
**Mirrors:** Pickering Ch 8 "Goals for Completion"; Arnett Ch 5 "Goals for completion".
**Purpose:** state precisely what remains to turn the proposal into a dissertation, at aggressive/impressive scope (author directive: ~1.5 years, "a lot of results").

---

## 7.1 The Integrated Pipeline (the capstone deliverable)

- **Goal:** a single end-to-end system — Data → mergeVAT structure (Ch 3) → persistence/multi-scale selection + MF generation (Ch 5) → FIS synthesis (Ch 4/Ch 6) → optional structure-driven refinement (Ch 2/Ch 6) → interpretable FIS — demonstrated on real, large, non-metric data.
- **Why it's the capstone:** each chapter proves a stage; the dissertation's central claim (orders-of-magnitude faster & larger, interpretable) is only fully demonstrated end-to-end.
- Deliverable: one reproducible driver + one flagship case study carried through every stage.

## 7.2 Proposed Studies (aggressive scope)

### G1 — Direct one-pass membership generation (Ch 5)
- Implement `MEMBERSHIP_ROADMAP.md` phases 1–6; soft/kernel-weighted band membership (fixes small-n over-segmentation); one-pass MF → FIS antecedents.

### G2 — Real non-metric / non-coordinate benchmarks (Ch 3 + Ch 5)
- DTW time-series, edit-distance sequences, graph/kernel dissimilarities — the core niche, currently only synthetic. This is the single most important credibility gap to close.

### G3 — HME EM refinement + full baseline suite (Ch 6)
- Implement EM (E/M steps per `EM_REFINEMENT.md`); benchmark against ANFIS, CART/C4.5, M5, flat TSK, Fumanal-Idocin 2025, D-TSK-FC.

### G4 — Scale & hardware credibility (Ch 3)
- Thermally-stable re-timing with fixed clocks + error bars (multi-seed); datacenter GPU with full-rate FP64; head-to-head vs eVAT & clusiVAT on identical datasets; distributed mergeVAT toward 500K elements.

### G5 — Interpretability evaluation (Ch 6 + Ch 2.6)
- Quantify interpretability (rule count, path length, expert-audience study or established metric); Magdalena-2018 rebuttal demonstrated empirically; optional SHAP contrast (interpretable-by-construction vs post-hoc).

### G6 — Adaptive multi-scale for overlapping scales (Ch 5, stretch)
- Change-point / barcode-stability band discovery beyond the gap heuristic; density-normalized persistence for a global scale-free test.

## 7.3 Application Showcase(s)

- Primary flagship carried end-to-end (author-preferred: a large **cybersecurity/IoT** dataset — RT-IOT2022 / IoT-botnet — where speed, scale, and interpretable rules all matter; alternative **UCI-58 Shuttle (statlog)**, already used in Ch 3, giving a clean thread from structure discovery through to the final FIS). Kept flexible for now.
- Aerospace-flavored showcase for the committee: MIMO temporal-memory FIS on a dynamical system (double pendulum / Atwood; extend toward a flight-dynamics or turbine dataset — `turbine-data.csv` is in-repo).

## 7.4 Risk register & honest de-scoping plan

- **Prior-art risk (Ch 5):** Bonis–Oudot overlap — mitigation = the three daylight axes + FIS target; if a reviewer collapses the gap, fall back to the *integration* + one-pass-MF novelty.
- **EM risk (Ch 6):** design-only; if implementation slips, HME one-shot + trees still stand as completed contributions.
- **Hardware risk (Ch 3):** consumer-GPU FP64 penalty — mitigation = datacenter run; if unavailable, report CPU-parallel + float32 GPU only, clearly scoped.
- **Baseline risk (Ch 4/Ch 6):** the speed/accuracy claims need ANFIS/GA-FIS head-to-heads; these are the first must-do experiments.

## 7.5 Summary table (goal × chapter × status × milestone)

*Build a table: Goal | Feeds chapter | Current status | Target milestone (quarter) — cross-reference Ch 10 Timeline.*

---

### Open items
- Flagship dataset: author-preferred IoT (RT-IOT2022 / IoT-botnet) or UCI-58 Shuttle — left flexible; confirm final choice later.
- Prioritize G1–G6 for the timeline (recommend G2 + G3 + G4 as the "must", G1 as the differentiator, G5/G6 as stretch).
