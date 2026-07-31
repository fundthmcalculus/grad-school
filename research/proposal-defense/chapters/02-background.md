# Chapter 2 — Background & Preliminaries

**Status:** Outline · Part I
**Mirrors:** Pickering Ch 2 + Ch 3 background; Arnett Ch 2 (Fuzzy Logic / Formal Methods / Neural Networks).
**Length target:** 10–15 pages. This is the shared toolbox every later chapter draws on.
**Role note:** the optimization engine (`tribble-opt`) is NOT core dissertation material (author decision) — its library details live in **Appendix A.3**. Ch 2 keeps only a brief, motivating treatment of *why* stochastic search is the bottleneck (this framing — "structure before search" — is central to the thesis).

---

## 2.1 Fuzzy Inference Systems

- **Fuzzy sets & membership functions**; linguistic variables; t-norms / t-conorms (min/max vs product/probabilistic).
- **Mamdani vs Takagi–Sugeno–Kang (TSK)**; TSK orders 0/1/2 and consequents linear in coefficients (this linearity is the hinge for Ch 4/Ch 6's closed-form solver).
- **Ruspini partition / strong partition of unity** (Ruspini 1969): triangular MFs, ∑μ=1 — the interpretable-by-construction export target.
- **Rule-base explosion** and standard mitigations (parameter reduction, hierarchical FIS).
- **Reuse the Arnett constraint framing** where useful (triangular MFs, Ruspini partitioning, product inference, weighted-average defuzzification) — a clean, citable formalism from the same lab.
- Prior art: Takagi–Sugeno 1985; Jang 1993 (ANFIS); Wang–Mendel 1992; Sugeno–Yasukawa 1993; Wu et al. 2020 (TSK optimization taxonomy).

## 2.2 Cluster Tendency: VAT / iVAT & Single-Linkage

- **VAT** (Bezdek & Hathaway 2002): dissimilarity matrix → modified-Prim MST reorder → Reordered Dissimilarity Image.
- **iVAT** (Wang et al. 2010; fast O(N²) recurrence Havens–Bezdek 2012): minimax / path-based (bottleneck) transform.
- **Key equivalence to foreground:** VAT ordering depends *only on the MST*; MST-cut ≡ single-linkage (Gower–Ross 1969; Zahn 1971). Everything in Ch 3 & Ch 5 rests on this.
- **The minimax/ultrametric geometry** and why centroid methods (k-means/FCM) cannot represent it — motivates Ch 5.
- Prior art / competitors to introduce here (differentiated later): clusiVAT (Kumar 2013/2016), eVAT (Meng–Yuan 2018), Fast-VAT (Avinash–Lachheb 2025), ConiVAT (Rathore 2020), Kumar–Bezdek 2020 survey.

## 2.3 Persistence & Topological Data Analysis (light)

- **Persistence** of single-linkage dendrogram blocks: birth/death/persistence; persistence diagram.
- **Birth height ↔ inverse local density** intuition (the load-bearing idea for multi-scale selection in Ch 5).
- Just enough TDA to support Ch 5; concede Bonis–Oudot (2014/2018) and ToMATo (Chazal 2013) as nearest precedents.

## 2.4 Fuzzy C-Means & Relational Fuzzy Clustering

- **FCM** (Dunn 1973; Bezdek 1981): objective, alternating optimization, initialization sensitivity (motivates VAT-seeding).
- **Relational fuzzy clustering**: NERFCM (Hathaway–Bezdek 1994), FANNY — clustering directly on a dissimilarity matrix (used in Ch 5).
- Minimax-linkage medoids (Bien–Tibshirani 2011); Chehreghani minimax embedding.

## 2.5 The Optimization / Initialization Bottleneck (brief — motivates "structure before search")

*Keep tight. Goal: establish the thesis foil, not survey the optimizer library. Full tribble-opt details → Appendix A.3.*

- **Why classic FIS training is slow:** population/stochastic search (Genetic Algorithms, ACO, PSO) is derivative-free and parallelizable but slow to converge and sensitive to nothing-to-go-on initialization; gradient descent / ANFIS is fast *only* with a very good initial guess, else poor local minima.
- **The dissertation's move:** recover structure first (Ch 3/Ch 5) so that model synthesis is largely closed-form/one-pass (Ch 4/Ch 6) and any remaining search is a cheap local polish — not the training engine. Author's own finding reinforces this: population methods often *overfit* the CV estimate in antecedent refinement, and local polish barely improves the MoG models.
- **Pointer:** the optimizer library that provides that optional polish (metaheuristics, Lin-Kernighan dual backend, quality-diversity/CVT-MAP-Elites, high-performance kernels) is documented in **Appendix A.3** and is not claimed as core dissertation novelty.

## 2.6 Interpretability & the Accuracy–Interpretability Trade-off

- Definitions of interpretability / explainability — **the author's own operational view** (readable off the model's structure; short rule base over named variables; hand-editable). NOTE: do NOT attribute this framing to Pickering — the author developed this work independently, without knowledge of her proposal; the reference PDFs are structural templates only. If a multi-lens account is cited, present it as parallel/independent work, not a source.
- **The trade-off, honestly stated:** hierarchical/tree FIS trade accuracy for readability (Olaru–Wehenkel 2003; Alcalá 2007). Engage **Magdalena 2018** ("Do hierarchical fuzzy systems really improve interpretability?") — the rebuttal is that our hierarchies split only on *named original inputs*, never synthetic intermediates.
- Post-hoc XAI (SHAP) as contrast to interpretable-by-construction (optional, if committee wants the XAI bridge).

---

### Open items for this chapter
- Decide depth of TDA/persistence (enough for Ch 5, not a survey).
- Decide whether to include a formal-methods/verification subsection (Arnett had one; likely out of scope here unless committee wants FIS verification). *Note: committee member Dr. Kreinovich's interval/fuzzy-computation work may make a short verification/soundness nod worthwhile.*
- §2.5 stays tight — tribble-opt detail belongs in Appendix A.3, not here.
