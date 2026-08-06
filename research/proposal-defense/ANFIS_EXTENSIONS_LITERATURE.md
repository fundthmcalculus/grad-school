# ANFIS methodology extensions — literature scan

_Compiled 2026-08-06. Scope: what has been proposed beyond Jang's original hybrid-learning
ANFIS, surveyed to inform the ANFIS baseline work tracked in `NEXT_STEPS.md` Tier 1 item 3
(`reproduce/tables/_baseline_anfis.py`) and the rule-base-explosion question central to this
document's own construction method (Ch 4/Ch 5)._

No PDFs are checked into this repo (see `.gitignore` in this directory — reference PDFs are
kept locally, not tracked). Everything below is a link plus enough annotation to decide what's
worth reading in full. Open-access items (arXiv, MDPI) can be fetched directly; paywalled items
(ScienceDirect, Springer, IEEE Xplore) are linked to their landing page only.

## Priority legend

- 🔴 **HIGH** — read before the next ANFIS-baseline or rule-explosion pass
- ⚪ Background / secondary

---

## 🔴 HIGH PRIORITY — rule-base construction & avoiding combinatorial explosion

**This is the section that matters most.** Standard ANFIS builds its rule base by grid-partitioning
each input dimension and taking the Cartesian product, so rule count grows as `prod(K_f)` over
features — the exact failure mode this document's own construction method is built to avoid. The
literature's answers to the same problem are the most direct prior art / comparison points for
that claim, and should be read before the next revision of whatever chapter argues the
construction method's rule-count behavior is a contribution.

1. **Jang, J.-S.R. (1993). "ANFIS: Adaptive-Network-Based Fuzzy Inference System."**
   *IEEE Trans. Systems, Man, and Cybernetics*, 23(3), 665–685. DOI `10.1109/21.256541`.
   [Semantic Scholar entry / PDF](https://www.semanticscholar.org/paper/ANFIS:-adaptive-network-based-fuzzy-inference-Jang/0095b6bb7c92f5deeffa8a311b80f75e680325eb)
   The foundational paper — grid partitioning and the hybrid (LSE + gradient) learning rule
   originate here. Read first if not already cited; everything else in this section is a
   response to the scaling behavior this paper introduces.

2. **Performance Comparison of ANFIS Models by Input Space Partitioning Methods.**
   *Symmetry*, 10(12), 700 (2018). Open access (MDPI).
   [https://doi.org/10.3390/sym10120700](https://doi.org/10.3390/sym10120700) ·
   [ResearchGate mirror](https://www.researchgate.net/publication/329379572_Performance_Comparison_of_ANFIS_Models_by_Input_Space_Partitioning_Methods)
   Direct head-to-head of grid partitioning vs. subtractive clustering vs. FCM clustering as
   the rule-generation step. The most directly comparable existing benchmark for a "how do
   different constructions trade off rule count vs. accuracy" table.

3. **A generalized framework for ANFIS synthesis procedures by clustering techniques.**
   *Applied Soft Computing* (2020), ScienceDirect (paywalled).
   [https://www.sciencedirect.com/science/article/abs/pii/S1568494620305603](https://www.sciencedirect.com/science/article/abs/pii/S1568494620305603)
   Generalizes clustering-based ANFIS synthesis (one rule per cluster in the *joint* input
   space, not per per-feature grid cell) — the standard fix for the exponential rule count.

4. **Clustered ANFIS network using fuzzy c-means, subtractive clustering, and grid
   partitioning for hourly solar radiation forecasting.**
   *Theoretical and Applied Climatology* (2018), Springer (paywalled).
   [https://link.springer.com/article/10.1007/s00704-018-2576-4](https://link.springer.com/article/10.1007/s00704-018-2576-4) ·
   [Academia.edu mirror](https://www.academia.edu/37242304/Clustered_ANFIS_network_using_fuzzy_c_means_subtractive_clustering_and_grid_partitioning_for_hourly_solar_radiation_forecasting)
   Applied three-way comparison (grid / subtractive / FCM) on a real forecasting task — useful
   as a second measured data point alongside item 2.

5. **Adaptive Neuro Fuzzy Networks based on Quantum Subtractive Clustering.**
   arXiv:2102.00820 (2021). Open access.
   [https://arxiv.org/pdf/2102.00820](https://arxiv.org/pdf/2102.00820)
   A more recent subtractive-clustering variant explicitly motivated by high-dimensional
   input spaces — same problem statement as this document's construction chapter.

6. **KANFIS: A Neuro-Symbolic Framework for Interpretable and Uncertainty-Aware Learning.**
   arXiv:2602.03034 (Feb 2026). Open access.
   [https://arxiv.org/pdf/2602.03034](https://arxiv.org/pdf/2602.03034) ·
   [lit-review summary](https://www.themoonlight.io/en/review/kanfis-a-neuro-symbolic-framework-for-interpretable-and-uncertainty-aware-learning)
   **Read this one closely.** Replaces ANFIS's product-based rule combination with an additive
   aggregation, so both parameter count and rule complexity scale **linearly**, not
   exponentially, with input dimensionality — a structural fix to the same problem from a
   different angle than clustering. Also supports Type-1 and Interval Type-2 fuzzy sets in the
   same architecture. Very recent (Feb 2026); worth checking whether it cites or is citable
   against this document's own construction method.

7. **Rule-Based Modeling of Low-Dimensional Data with PCA and Binary Particle Swarm
   Optimization (BPSO) in ANFIS.** arXiv:2502.03895 (2025). Open access.
   [https://arxiv.org/pdf/2502.03895](https://arxiv.org/pdf/2502.03895)
   Attacks rule-base size from the feature-selection side (PCA + BPSO to cut the number of
   antecedent variables before construction) rather than the partitioning side — the same
   philosophy as the interaction/feature-selection work recently landed in `tribble-fis`
   (cross-term detection for the mixture-of-Gaussians FIS). Good comparison point for whether
   feature reduction alone is a sufficient substitute for a smarter construction method.

8. **A Dynamic Fuzzy Rule and Attribute Management Framework for Fuzzy Inference Systems
   in High-Dimensional Data.** arXiv:2504.19148 (2025). Open access.
   [https://arxiv.org/pdf/2504.19148](https://arxiv.org/pdf/2504.19148)
   Runtime add/prune/merge management of both rules and attributes — relevant if the
   construction method is ever extended to an online/streaming setting (see the evolving-FIS
   section below).

9. **Adaptive network based fuzzy inference system (ANFIS) training approaches: a
   comprehensive survey.** *Artificial Intelligence Review* (2017), ACM DL / Springer
   (paywalled). [https://dl.acm.org/doi/10.1007/s10462-017-9610-2](https://dl.acm.org/doi/10.1007/s10462-017-9610-2)
   Not clustering-specific, but the best available survey anchor for the whole
   training-method landscape — useful as a single citation covering the breadth this document
   doesn't need to re-derive.

---

## ⚪ Uncertainty quantification — Type-2 / Interval Type-2 ANFIS

Premise membership functions carry a footprint of uncertainty rather than a crisp value, with
an added type-reduction layer before defuzzification. Relevant if prediction intervals (rather
than point estimates) become a goal.

- **Interval Type 2 Adaptive Neuro-Fuzzy Inference System-Based Artificial Pacemaker Design
  and Stability Analysis.** PubMed. [https://pubmed.ncbi.nlm.nih.gov/37938200/](https://pubmed.ncbi.nlm.nih.gov/37938200/)
- **Interval type-2 ANFIS.** ResearchGate.
  [https://www.researchgate.net/publication/226279249_Interval_type-2_ANFIS](https://www.researchgate.net/publication/226279249_Interval_type-2_ANFIS)
- **Explainable Uncertainty Quantification for Wastewater Treatment Energy Prediction via
  Interval Type-2 Neuro-Fuzzy System.** arXiv:2601.18897 (2026). Open access.
  [https://arxiv.org/html/2601.18897v1](https://arxiv.org/html/2601.18897v1)
- **A Self-organizing Interval Type-2 Fuzzy Neural Network for Multi-Step Time Series
  Prediction.** arXiv:2407.08010. Open access.
  [https://arxiv.org/pdf/2407.08010](https://arxiv.org/pdf/2407.08010)

## ⚪ Metaheuristic-optimized training (alternative to the hybrid LSE+gradient rule)

Population-based search (GA, PSO, ACO, DE, GWO, WOA, BBO, hybrids) tuning premise and/or
consequent parameters instead of Jang's hybrid rule. `tribble-opt` already has GA/DE/PSO
infrastructure that could plug into this without new optimizer code.

- **Meta optimization of an ANFIS with grey wolf optimizer and biogeography-based
  optimization for spatial prediction of landslide susceptibility.** ScienceDirect (paywalled).
  [https://www.sciencedirect.com/science/article/abs/pii/S0341816218305770](https://www.sciencedirect.com/science/article/abs/pii/S0341816218305770)
- **A hybrid Genetic–Grey Wolf Optimization algorithm for optimizing Takagi–Sugeno–Kang
  fuzzy systems.** *Neural Computing and Applications* (2022), Springer (paywalled).
  [https://link.springer.com/article/10.1007/s00521-022-07356-5](https://link.springer.com/article/10.1007/s00521-022-07356-5)

## ⚪ Online / streaming (evolving fuzzy systems)

ANFIS assumes a fixed batch and fixed rule structure. This family relaxes both — rules are
added, merged, or pruned as data streams in.

- **FLEXFIS: A Robust Incremental Learning Approach for Evolving Takagi–Sugeno Fuzzy
  Models.** [ResearchGate](https://www.researchgate.net/publication/224315082_FLEXFIS_A_Robust_Incremental_Learning_Approach_for_Evolving_Takagi-Sugeno_Fuzzy_Models)
- **DENFIS: Dynamic Evolving Neural-Fuzzy Inference System and Its Application for
  Time-Series Prediction.** [Academia.edu](https://www.academia.edu/31783959/DENFIS_Dynamic_Evolving_Neural_Fuzzy_Inference_System_and_Its_Application_for_Time-Series_Prediction)
- **Data-driven evolving fuzzy systems using eTS and FLEXFIS: comparative analysis.**
  *International Journal of General Systems*, 37(1). [Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/03081070701500059)

## ⚪ Recurrent / temporal ANFIS

Natural analogue for the memory-window MIMO work already in `tribble-fis`
(`MIMO_MEMORY_GUIDE.md`).

- **RS-ANFIS: A recurrent-state adaptive neuro-fuzzy model for accurate and interpretable
  short-term load forecasting in volatile energy systems.** ScienceDirect (2026, paywalled).
  [https://www.sciencedirect.com/science/article/pii/S2772941926000438](https://www.sciencedirect.com/science/article/pii/S2772941926000438)
- **Survey on Deep Fuzzy Systems in regression applications: a view on interpretability.**
  arXiv:2209.04230. Open access. [https://arxiv.org/pdf/2209.04230](https://arxiv.org/pdf/2209.04230)
  Orientation map for the deep/recurrent/ensemble fuzzy-regression landscape generally.

## ⚪ Deep / ensemble hybrids

Heavier-weight, and in tension with this document's preference for closed-form/interpretable
solves over black-box hybrids — background only.

- **Deep Neural Fuzzy System Oriented toward High-Dimensional Data and Interpretable AI.**
  *Applied Sciences* (MDPI, open access). [https://www.mdpi.com/2076-3417/11/16/7766](https://www.mdpi.com/2076-3417/11/16/7766)
- **A data-driven implicit deep adaptive neuro-fuzzy inference system capable of manifold
  learning for function approximation.** ScienceDirect (2024, paywalled).
  [https://www.sciencedirect.com/science/article/abs/pii/S1568494624002321](https://www.sciencedirect.com/science/article/abs/pii/S1568494624002321)
- **Deep learning-based novel ensemble method with best score transferred-ANFIS
  (BST-ANFIS) for energy consumption prediction.** PMC (open access).
  [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11888908/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11888908/)

---

## Suggested next step

Read section 🔴 items 1, 2, and 6 (Jang original, the grid/subtractive/FCM comparison, and
KANFIS) before the next pass on the construction method's rule-count argument — those three
are the closest existing framing of "how do you avoid `prod(K_f)` rule growth" and the most
likely committee-question source.
