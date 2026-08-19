# N-CMAPSS DS02: FIS exploration

Exploratory scripts behind the DS02 investigation into (a) whether the
condition-correction step is doing real work, and (b) how the TRIBBLE fuzzy
system's structure (consequent order, rule count, regularization, antecedent
membership-function shape) affects remaining-useful-life quality.

**Run from the repository root** (paths to `NASA-CMAPSS/`, `FuzzySystemsExperiments/`,
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
