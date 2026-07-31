# Chapter 5 — Topological Membership Generation (Proposed)

**Status:** Outline · Part III (PROPOSED / active — strong preliminary results)
**Repo:** `gated-minimax-selection`
**Role:** the conceptual **bridge** — turns Ch 3's structure (minimax/iVAT hierarchy) into Ch 4/Ch 6's FIS antecedents *without* coordinates or a Gaussian assumption. This is the dissertation's forward-looking differentiator.
**One-line claim:** from a dissimilarity matrix alone, extract fuzzy membership functions *and* the cluster count (and even the number of scales) from the single-linkage/iVAT minimax hierarchy, via a persistence-gated set-cover — with no pre-specified k and without single-linkage's chaining/scale artifacts.

---

## 5.1 Introduction & Motivation

- Gap: Ch 4's MoG assumes coordinates + roughly Gaussian structure. Many problems give only a dissimilarity matrix (DTW, edit distance, graph/kernel dissimilarity) and non-convex structure (rings). Where do MFs come from then?
- The VAT family (Kumar–Bezdek 2020 survey) uses iVAT only for tendency assessment / cluster counting — **no prior VAT work generates fuzzy membership functions.** That is the gap this chapter fills.
- Contributions (proposed, with preliminary results): (1) the minimax transform D* as the load-bearing preprocessing that makes relational fuzzy clustering work on non-convex data; (2) persistence-gated set-cover with k as an output; (3) multi-scale persistence (Option D) recovering a hierarchy of partitions; (4) native membership functions per dendrogram block (persistence-ramp) → FIS antecedents; (5) topological disjunct-arity detection.

## 5.2 Background & Prior Art (concede the nearest precedent honestly)

- Persistence-based clustering: **Bonis & Oudot 2014/2018** (beta-plateau) and **ToMATo** (Chazal 2013) substantially pre-empt "persistence-based membership from a dissimilarity matrix with gap-based k." **AuToMATo** (arXiv:2408.06958, bottleneck-bootstrap).
- **Defensible daylight (state precisely):**
  1. *Connectivity/ultrametric (density-free)* filtration vs their density mode-seeking.
  2. *Deterministic metric ramp* MF vs their random-walk hitting probability.
  3. *Target = membership functions for a fuzzy inference system* (linguistic antecedents, disjunctive OR-terms) — absent from their work and the entire VAT family.
- Relational fuzzy: NERFCM (Hathaway–Bezdek 1994) + beta-spread admissibility; ConiVAT (Rathore 2020) constraint-based iVAT + Mahalanobis metric learning (Xing et al.).

## 5.3 Methodology

### 5.3.1 The minimax/iVAT transform D* (the result-carrier)
- D* = minimum transitive (bottleneck) dissimilarity from the iVAT recurrence.
- **Honest reckoning (from FINDINGS.md):** it is the *transform*, not the selection machinery, that carries the win — NERFCM on concentric rings goes 0.02 (on D) → **1.00** (on D*). Lead with this; don't over-credit the selector.

### 5.3.2 Persistence-gated set-cover selection (`select_coverage_cover`)
- Admit blocks whose absolute persistence is a MAD statistical outlier (median + gap_sigma·1.4826·MAD); greedily cover the data by uncovered-gain; **k is an output.**
- Rationale: under a fixed t-conorm, over-segmentation is cheap, under-coverage expensive → set-cover, not "pick exactly k."

### 5.3.3 Multi-scale persistence — Option D (the headline)
- Discover density-scale **bands** from gaps in the single-linkage **log-birth-height** spectrum (birth height ↔ inverse local density); run the *same* gated set-cover within each band → a **hierarchy of partitions**, number of scales itself an output.
- Provable generalization: shared gate; reduces to flat selection on single-scale data; zero bands on uniform noise. (`persistence_significance_threshold` is the flat coverage_cover gate factored out — this is what makes it a generalization, not a rival.)
- Honest framing (per memory `project_option_d_multiscale`): the target is **nested-hierarchy recovery**, never "beat coverage_cover on varying-density" (flat is already scale-invariant — see falsification experiment).

### 5.3.4 Membership-function extraction (four validated variants A–D)
- Mapping 2 **persistence-ramp** MF: μ_B(x) = clip((death − d_B(x)) / (death − birth), 0, 1) — built from the block's own birth/death, no medoid/Gaussian fit.
- Option B: Ruspini partition-of-unity (∑μ=1, 0.0 error). Option A: auto-tuned Ruspini (spread-aware support). Option C: feature-space Mahalanobis rules (works on Gaussians, fails on rings — report honestly). Option D: multi-scale framework.
- Disjunctive OR-terms combined via fixed t-conorm (`disjunct.py`).

### 5.3.5 Topological disjunct-arity detection
- Count D*-connected components rather than geometric convex pieces — well-posed where geometric decomposition is ill-posed (rings).

## 5.4 Preliminary Results

*From `FINDINGS.md`, `OPTION_D_MULTISCALE.md`, `SELECTION_METHODS_COMPARISON.md`, `SCALING_STUDY.md`, `RELATIONDATA.md`; figures fig1–fig11.*

- **Master ARI table (5 synthetic sets):** iVAT-cover (k discovered, no constraints) ≈ 0.98–1.00, matching NERFCM-given-k and ConiVAT; declines only on uniform noise (tendency-aware). concentric_rings NERFCM(D)=0.02 → NERFCM(D*)=1.00; bridged_gaussians plain SL=0.00 → ConiVAT=1.00.
- **Multi-scale headline:** mean ARI over *all* ground-truth levels: nested_gaussians 0.66→**1.00**, three_level_hierarchy 0.58→**1.00**, density_hierarchy 0.75→**1.00**. three_level_hierarchy recovers granularities **[8,4,2]**, each band = one level at ARI 1.0, without being told the number of levels.
- **Falsification experiment (avoid a strawman):** flat coverage_cover holds ARI ≈0.983 across a 30× spread ratio — no single-level varying-density win claimed.
- **Scaling:** with exact O(N²) `minimax_transform_fast`, full pipeline to n=5000 in ~5 s; many_scale recovers [8,4,2] at ARI 1.0 from n=100→5000.
- **Selection bake-off:** no universal winner — Persistence-Gap fails the bridge (0.001) but is noise-conservative; Beta-Plateau/Bottleneck-Bootstrap fix the bridge (0.927/0.891) but over-fire on noise (k=7). Bridge vs noise are incompatible for a single fixed threshold.
- **Relational-only data:** multi_scale_hierarchy leaves both D and D* stuck ~0.29 → validates multi-scale as the genuinely hard, open problem.

## 5.5 Proposed Work (what turns this from active → done)

- **Direct one-pass MF generation** (`MEMBERSHIP_ROADMAP.md`, 6 phases): collapse select-then-fit into a single pass — every block emits its native ramp MF, t-conorm recombines, surviving envelope *is* the fuzzy model. Phase 4 (soft/kernel-weighted band membership) is the research-interesting piece (fixes small-n over-segmentation).
- **Adaptive/model-based band discovery** for *overlapping* scales (change-point or barcode-stability), replacing the gap heuristic (which assumes scale-separated levels).
- **Real non-metric datasets** (current ground truth all synthetic): DTW time-series, edit distance, graph/kernel dissimilarities.
- **Formal prior-art search + head-to-head** vs Bonis–Oudot beta-plateau and AuToMATo on identical data.
- **Wire the output MFs into the tribble-fis FIS** (the integration that ties Ch 5 → Ch 6 → the pipeline).
- Soft-metric validation (fuzzy-ARI / cross-entropy vs known Gaussian posteriors).

## 5.6 Discussion & Contributions

- Positioning: density-free, connectivity-based MF generator for FIS — the missing bridge from VAT/iVAT structure to fuzzy antecedents.
- Limits (state up front): band discovery assumes scale-separated levels (overlapping scales ill-posed for the gap heuristic); min_band_coverage can drop small real clusters; prior-art overlap with Bonis–Oudot must be actively managed.

---

### Open items
- Decide how much of Options A–D to present vs consolidate (recommend: lead with D + persistence-ramp; A/B/C as supporting).
- The MEMBERSHIP_ROADMAP phases are the natural spine of the "proposed work" + timeline.
