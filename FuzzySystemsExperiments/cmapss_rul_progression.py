"""Plot the RMSE progression across this DOE's successive rounds.

Milestone values are hand-recorded from Stage 3's "final confirmation" output
across three separate runs of cmapss_rul.py, in chronological order:

1. Raw features, fixed defaults -- the Stage 1 screen (18 pipelines, no
   per-pipeline hyperparameter tuning yet). A3_raw_memory/B2/C1_raw and
   A3_raw_memory/B1/C3_physical.
2. + construction hyperparameter tuning -- Stage 2's expanded D_GRID
   (norm_conorm='hamacher', l2_reg=0.01 found via one-factor-at-a-time
   probe, confirmed across all 3 pipelines). Same two pipelines, tuned.
3. + condition-corrected features + StandardScaler -- regressing Xs/Xv
   sensor channels against W operating conditions before aggregation (see
   cmapss_rul.py's fit_raw_condition_correction), plus a StandardScaler
   axis added to D_GRID. "overall" here is A3_raw_memory_cc/B3/C3_physical
   -- confirmed via full Stage 2 grid search (288 configs), not just the
   one-off hand-picked config a manual probe first found (6.48, unchanged
   by the grid search -- it was already the right answer).

   "overall" is the *fair* number as of this milestone, not a leakage
   caveat: B3 uses the dataset's W + X_s + exactly 2 of its 14 "virtual
   sensor" channels (T40, P30) -- the published DS02 CNN/MLP baselines'
   *exact* 20-channel input set (Arias Chao et al. 2021 Table 2, cited by
   Custode et al. 2022, confirmed against co-author Hyunho Mo's own
   released code). All 14 virtual sensors (a real leakage-risk sensitivity
   arm) scored 6.54 -- worse, despite 12 more input channels, and 10x the
   fit time (10.5s vs. 1.0s). "real_sensor" stays the stricter 18-channel
   (W+X_s only) A1_whole_cycle_cc/B1/C3_physical.

Re-run this any time a new milestone is confirmed -- append to MILESTONES.
Run from the grad-school repo root: `python FuzzySystemsExperiments/cmapss_rul_progression.py`.
"""

from cmapss_rul_plots import plot_progression

MILESTONES = [
    {"label": "Raw features,\nfixed defaults", "overall": 13.12, "real_sensor": 19.38},
    {"label": "+ construction\ntuning", "overall": 8.83, "real_sensor": 19.12},
    {"label": "+ condition correction\n+ scaling (fair)", "overall": 6.48, "real_sensor": 11.23},
]

if __name__ == "__main__":
    plot_progression(MILESTONES, "FuzzySystemsExperiments/cmapss_rul_progression.png")
    print("wrote FuzzySystemsExperiments/cmapss_rul_progression.png")
