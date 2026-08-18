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
- `em_trapezoid_debug.py`  — why EM antecedents collapse (support/plateau, firing coverage).
- `em_widthreg_sweep.py`, `em_widthreg_sweep_extra.py`, `plot_widthreg.py` — the `width_reg` recovery.
- `em_plateau_sweep.py`    — plateau regularizer (a non-lever).

These fed tribble-fis issues #163/#164/#165 and PRs #167/#168/#169.
