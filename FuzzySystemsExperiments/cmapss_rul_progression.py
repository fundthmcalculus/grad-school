"""Plot the RMSE progression across this DOE's successive rounds.

Milestone values are hand-recorded from Stage 3's "final confirmation" output
across three separate runs of cmapss_rul.py, in chronological order:

1. Raw features, fixed defaults -- the Stage 1 screen (18 pipelines, no
   per-pipeline hyperparameter tuning yet). A3_raw_memory/B2/C1_raw and
   A3_raw_memory/B1/C3_physical.
2. + construction hyperparameter tuning -- Stage 2's expanded D_GRID
   (norm_conorm='hamacher', l2_reg=0.01 found via one-factor-at-a-time
   probe, confirmed across all 3 pipelines). Same two pipelines, tuned.
3. + condition-corrected features + StandardScaler -- this round's addition:
   regressing Xs/Xv sensor channels against W operating conditions before
   aggregation (see cmapss_rul.py's fit_raw_condition_correction), plus a
   StandardScaler axis added to D_GRID. A3_raw_memory_cc/B2/C3_physical and
   A1_whole_cycle_cc/B1/C3_physical.

Re-run this any time a new milestone is confirmed -- append to MILESTONES.
Run from the grad-school repo root: `python FuzzySystemsExperiments/cmapss_rul_progression.py`.
"""

from cmapss_rul_plots import plot_progression

MILESTONES = [
    {"label": "Raw features,\nfixed defaults", "overall": 13.12, "real_sensor": 19.38},
    {"label": "+ construction\ntuning", "overall": 8.83, "real_sensor": 19.12},
    {"label": "+ condition correction\n+ scaling", "overall": 6.54, "real_sensor": 11.23},
]

if __name__ == "__main__":
    plot_progression(MILESTONES, "FuzzySystemsExperiments/cmapss_rul_progression.png")
    print("wrote FuzzySystemsExperiments/cmapss_rul_progression.png")
