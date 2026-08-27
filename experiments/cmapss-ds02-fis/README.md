# N-CMAPSS DS02: FIS exploration

Exploratory scripts behind the DS02 investigation into (a) whether the
condition-correction step is doing real work, and (b) how the TRIBBLE fuzzy
system's structure (consequent order, rule count, regularization, antecedent
membership-function shape) affects remaining-useful-life quality.

**Run from the repository root** (paths to `data/nasa-cmapps2/`, `FuzzySystemsExperiments/`,
and `outputs/` are relative to it); figures/CSVs are written to the gitignored
`outputs/hdbscan-ds02/`. Needs the same deps as the CMAPSS drivers plus
matplotlib/seaborn.

## Clustering tendency (does condition correction matter?)
- `hdbscan_percycle.py`   — HDBSCAN on per-cycle features, raw vs corrected.
- `hdbscan_samples.py`    — HDBSCAN on subsampled 1 Hz samples (raw = flight regime; corrected = degradation continuum).
- `ivat_samples.py`       — iVAT reordered-dissimilarity images, raw vs corrected (reuses `gated-minimax-selection/ivat_mf.py`).

## FIS structure / complexity
- `sweep_tsk_order.py`        — 1st vs 2nd vs full-2nd consequents; parameter count vs accuracy.
- `sweep_loworder_l2_topp.py` — l2_reg x top_p sweep for the low-order models.
- `sweep_rules.py` / `plot_rules.py` — number of rules (output buckets) per consequent order.

## Membership-function investigation
- `membership_compare.py` / `plot_membership.py` — gaussian vs fast/EM trapezoid vs triangular.

The membership-function work fed tribble-fis issues #163/#164/#165 and PRs
#167/#168/#169. The EM antecedent-collapse / `width_reg` / plateau-regularizer
probe scripts were removed once the fix shipped and the plateau path was shown to
be a non-lever.

## Reducing the training residual (iterative methods)
- `_ds02_harness.py`            — shared DS02 featurisation (condition correction → memory features → cap → scale), so the scripts below can drop in their own regressor and read the *training* residual directly.
- `iterative_train_residual.py` — DS02: residual boosting (weak/strong base × shrinkage) and staged rule growth. Boosting cuts train but overfits; 3 buckets is the honest sweet spot.
- `iterative_pooled.py`         — the same sweep on the pooled all-datasets model. Boosting again only overfits; **4 output buckets** is the honest per-sample win (15.80 → 14.87), now baked into `cmapss_all_datasets.py`'s `raw_memory` config.

Orthogonal (Legendre) consequents were also tried here and were a no-op at
full-2nd (identical to raw monomials), so that probe script was removed.

## Reducing RMSE (upstream + structural sweeps)
- `sweep_features_target.py`         — #4 memory-feature geometry (stride/window/memory) and #5 the RUL target cap. The cap is the real lever: a constant ceiling Rc≈58–60 on top of the health-onset cap cuts DS02 per-sample test RMSE 6.48 → 6.23 (below ~50 it collapses).
- `sweep_antecedent_consequent.py`   — #6 antecedent granularity (`n_gaussians`) and #7 RBF consequents. n_gaussians=2 is marginal; RBF consequents are a dead-end (test ~12).
- `sweep_ceiling_combo.py`           — refine the RUL ceiling and stack the marginal winners: **window 10 + Rc≈60 → 6.14** (−5%), the best DS02 config found.
- `blend_wc_rm.py`                   — #8 blend the two pooled models. A **70/30 whole_cycle/raw_memory blend** hits per-engine RMSE **8.54** (vs 9.20 / 13.13 alone) and dominates whole_cycle on all three metrics — productionised in `cmapss_all_datasets.py`.

`sweep_features_target.py` / `sweep_ceiling_combo.py` use the harness's
`load_corrected` (corrected frames before featurisation) so they can re-featurise
with their own geometry.
