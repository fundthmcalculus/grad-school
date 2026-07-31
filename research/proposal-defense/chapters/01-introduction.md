# Chapter 1 — Introduction

**Status:** Outline · Part I
**Mirrors:** Pickering Ch 1 (Prelude / Unique Contributions / Dissertation Outline); Arnett Ch 1.
**Length target:** 4–6 pages.

---

## 1.1 Prelude / Motivation

*Purpose: set the stakes and the gap in ~2 pages.*

- **The promise of FIS.** Fuzzy Inference Systems give human-readable IF–THEN rules over linguistic terms; universal approximators (Wang 1998; Joo–Lee 2002); interpretable-by-construction — the property regulators and domain experts increasingly demand.
- **The problem.** FIS are slow and don't scale:
  - *Rule-base explosion*: number of rules is the product over inputs of per-input MF counts (state the formula). Exponential in input dimension.
  - *Training pathology*: dominant methods — genetic algorithms (Cordón 2001), gradient descent, ANFIS (Jang 1993) — are stochastic, slow to converge, and acutely initialization-sensitive. GD needs a very good initial guess or lands in poor local minima.
  - *Scale wall*: classic structure-analysis tools (e.g., VAT, O(N³)) become infeasible past a few thousand points; real datasets are 10⁴–10⁶ rows.
- **The thesis (one sentence).** The structure latent in the data — recovered by fast relational/topological analysis — can replace most stochastic search, giving FIS that train orders of magnitude faster, scale to 10⁵+ samples, and stay interpretable.
- **Motivating vignettes / target datasets** (concrete, from the work): NASA Space-Shuttle reentry (58K×7); obfuscated psych-eval patient data (135K×165, 73 GB dense); PhiUSIIL phishing (235K), RT-IOT2022 (123K×83×12 classes); UCI Concrete regression; dynamical systems (double pendulum, Atwood machine).
- **Framing note:** lead with scale/speed (primary), carry interpretability as an enabling property; nod to XAI/regulation as secondary context (lighter than Pickering).

## 1.2 Unique Contributions

*Purpose: crisp enumerated list the committee can hold in their heads. Each maps to a chapter. State novelty as composition + regime, concede primitives.*

1. **mergeVAT — exact VAT/iVAT at scale** (Ch 3, done). Reduced VAT/iVAT from O(N³) to O(N²log N) via heap/priority-queue argmin; in-place memory (3 matrices → 1); GPU-Borůvka MST bit-identical to serial VAT; divide-&-conquer with a *principled bounded stitch*; correctness on *arbitrary, non-metric* dissimilarity. Key result: feasible size 5K → 130K+; the "58K×58K in 60s" headline.
2. **Fast MoG FIS synthesis** (Ch 4, done). Mixture-of-Gaussians antecedent + rule generation that avoids rule-base explosion (K rules for K classes) and needs *no* post-hoc GA/GD; closed-form ridge-TSK consequents. Key result: PhiUSIIL 97–99% in ~6 s; RT-IOT2022 (123K×83, 12 classes) < 60 s.
3. **Topological membership generation** (Ch 5, proposed w/ preliminary results). Persistence-gated set-cover selection with *k as an output*; multi-scale persistence (Option D) recovering a *hierarchy* of partitions; density-free relational membership functions straight from the minimax/iVAT hierarchy — the bridge from clustering to FIS antecedents. Key result: nested/hierarchical ARI 0.58–0.75 → **1.00** across all levels; minimax transform lifts NERFCM on rings 0.02 → 1.00.
4. **Hierarchical & refined fuzzy models** (Ch 6, partial). Shared closed-form ridge-TSK consequent solver as one primitive across flat FIS, soft fuzzy trees, and hierarchical mixture-of-experts; Ruspini partition-of-unity export; MIMO temporal-memory FIS for dynamical systems. Key result: Concrete R² flat 0.66 → 1st-order tree 0.75 → HME 0.79; MIMO improves double-pendulum RMSE ~33%.
5. **Optimization/refinement engine** (supporting infrastructure — **Appendix A.3**, not a core chapter; done). Metaheuristics (ACO/GA/PSO/GD), Lin-Kernighan dual backend, quality-diversity (CVT-MAP-Elites), high-performance kernels. Provides the optional *local polish* stage; the thesis deliberately keeps it off the critical path ("structure before search"). *Standalone-paper candidates flagged.*

> **Contribution statement to write:** the integrated pipeline is the dissertation's novelty; each primitive is prior art, faithfully credited. The claim is that *structure-first* FIS construction occupies regimes (exact non-metric VAT at scale; density-free relational MF; one-pass fast training) that existing methods cannot.

## 1.3 Dissertation Outline

*Purpose: one paragraph per chapter, roadmap figure.*

- Reproduce the Part I/II/III structure from the master outline.
- **Figure 1.1 (to create):** the pipeline schematic — Data → [Ch 3 structure discovery] → [Ch 5 selection + MF generation] → [Ch 4/Ch 6 FIS synthesis] → [Ch 6 refinement via Ch 2 optimizers] → Interpretable FIS. Annotate each stage with the "orders of magnitude" claim and the chapter number. This is the single most important orienting figure — mirrors how Arnett/Pickering anchor the reader.

---

### Open items for this chapter
- Decide how heavily to invoke XAI/regulation framing (secondary per author).
- ~~Committee names~~ → **RESOLVED:** Cohen (chair), Kreinovich, Kumar, Minai, Zhan.
- ~~Working title~~ → **RESOLVED:** "Reproducing Like Tribbles: Scaling Fuzzy Inference Systems from Hundreds to Hundreds of Thousands." Consider opening the Prelude with the tribble metaphor (prolific, exponential scaling) as a hook.
