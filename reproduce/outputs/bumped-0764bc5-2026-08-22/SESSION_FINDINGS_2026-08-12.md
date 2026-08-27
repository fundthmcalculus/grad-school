# Session findings — 2026-08-11/12: clearing the "remains to be implemented" backlog

Autonomous evaluation run on the i9-14900HX / 96GB workstation (branch `fix/sample-problems`),
scoped from `research/proposal-defense/prose/` and `CHECKLIST.md`/`DATASETS.md`. Elapsed
~4.5h against an ~8h budget. Every number below is real output from this session's runs —
none is estimated or extrapolated except where explicitly marked as a cost estimate for
work that was deliberately *not* run.

**First finding, before any benchmark ran:** `CHECKLIST.md` and `DATASETS.md` describe
RT-IOT2022, BETH, Shuttle and Bike Sharing as "not started" or "loader unwired." That was
stale. All six dataset loaders (`test_dataset_loaders.py`) already load real local data on
this branch. The actual remaining gap was running the full ten-seed protocol, not wiring
data — a reminder that these docs drift from the code faster than they're updated.

## Table 4.4 — Open-set detection, RT-IOT2022 (123K rows, 12 classes, 82 features)

The flagship scale test named in Ch7/Checklist C4. First attempt (10 seeds + the default
7-theta sensitivity sweep) was killed after measuring only 1/12 classes in 50 minutes —
`complement_rule()` fits **all 82 features** (not a reduced top-N like the quick baseline
used), so the real per-seed cost was ~4x the pre-flight estimate. Projected ~24h total.
Patched the script (`OCSVM_TRAIN_CAP`, `THETA_SWEEP_SEEDS`, both disclosed in the emitted
table's note) and relaunched at 5 seeds with the sweep disabled — ~2h actual.

| Method | Detection | False-alarm | Youden's J | Separate model? |
|---|---|---|---|---|
| **Complement rule (this work)** | 0.839±0.253 | 0.445±0.091 | +0.394 | no |
| One-Class SVM | 0.847±0.236 | 0.439±0.057 | +0.408 | yes |
| Isolation Forest | 0.974±0.117 | 0.437±0.056 | **+0.537** | yes |

**Not a favorable result, and it should be written up honestly.** At this scale, Isolation
Forest clearly beats the complement rule, and OCSVM edges it out too, despite both needing
a separate model where the complement rule is "free." False-alarm rate is ~44% across all
three arms at the default θ=0.99 — a loose operating point — and the theta sweep that would
show whether a different threshold helps was disabled for budget reasons. Detection-rate
seed spread (±0.253) is large. **Follow-up:** re-run the theta sweep on RT-IOT2022 with a
realistic budget (the 7-theta × 10-seed default sweep alone is ~9h at this scale) to see if
a better operating point exists before concluding the method loses at scale.

## Table 4.1 / full suite — Bike Sharing + everything else, 10 seeds

Regenerated all 12 remaining table generators at the full ten-seed protocol (Table 4.1 now
exercises Bike Sharing as the large regression partner, alongside Concrete and PhiUSIIL).
All green. Found and fixed a real bug along the way: `table_a1_feature_scoring.py` and
`table_4_8_mf_dedup.py` (`_mf_dedup.py`) both called `tribblefis.gaussian_classifier`
classes (`MixtureOfGaussiansFuzzyClassifier`/`SequenceClassifier`, and
`MixtureOfGaussiansFuzzyRegressor`) that were renamed upstream to `TribbleClassifier`,
`TribbleSequenceClassifier`, `TribbleRegressor` at the currently pinned `tribble-fis` SHA
(`80e98d7`). Fixed both call sites, verified signature compatibility, backfilled.

Separately, the suite's own archive step hit the exact mid-run-edit failure mode its header
warns about (an agent registered a new experiment into `run_all_tables.sh` while it was
running, shifting byte offsets bash was reading incrementally) — recovered via the script's
documented `--archive-only` path, which is why this archive's PROVENANCE banner reads
"not verified": the numeric phase is fully verified via this session's own logs, only the
generic recovery banner is cautious.

## Goal G1 — Scalable membership-function batteries (Ch5 §5.4)

Doc claimed "no recorded run... at any size other than 96." Also stale: a two-stage-selector
scaling study already existed at one seed (`gated-minimax-selection/notes/SCALING_STUDY.md`).
Extended it to ten seeds + a flat-set-cover comparison + partition-of-unity error, across
`single_scale`/`many_scale`/`log_separated` at n=100–5000 (`table_5_4_ch5_g1_scaling.py`,
ran in 3m6s).

- `many_scale`: **[8,4,2] at ARI 1.00, 10/10 seeds, every n** — confirmed solid.
- `single_scale`: **less stable than the single-seed study implied** — granularity mode
  agrees only 5–7/10 seeds. A real walk-back worth flagging for Ch5.
- `log_separated`: transition is messier than pass/fail; flat set-cover stays strong at
  small n, matching G1's own prediction for this regime.
- Partition-of-unity error: machine precision everywhere, as required.

## Goal G2 — Real non-coordinate benchmarks (Ch7: "the top credibility item")

Genuinely unstarted before this session (verified via grep + git log). Reused existing
matrix-input pVAT/iVAT machinery and `triangle_violation_rate()` from
`ClusteringExperiments/hardening_eval.py` rather than reinventing it. New:
`reproduce/tables/table_3_7_g2_dtw_nonmetric.py`, wired into `manifest.py` and
`run_all_tables.sh`. Violation-rate estimator was sanity-checked against already-published
numbers before trusting it on new data (GunPoint 29.0% vs. checklist's 29.3%,
ItalyPowerDemand 14.8% vs. 16.3% — validated).

| Dataset | Triangle violations | Agreement w/ exact | Timing |
|---|---|---|---|
| ECG5000 (N=5000) | 20.9% | **1.000** | 631s matrix + 0.3s reorder |
| FordA (N=4921) | 0.4% | **1.000** | 7222s matrix + 0.2s reorder |
| Crop (N=24000, the scale target) | 23.6% | **1.000** | 1597s matrix + 4.7s reorder |

**Exactness holds at 1.000 on all three real, non-metric datasets, at every scale tested —
the coordinate-free claim is no longer resting on synthetic proxies.** Crop, the explicit
scale target (~4.6GB dissimilarity matrix), reorders in 4.7s once built, confirming the
method's cost lives entirely in distance computation, not the algorithm.

**Honest caveat: the triangle-violation result is mixed, not a clean win.** ECG5000 and Crop
both exceed the 14% synthetic-proxy rate (harder than the proxy, as hoped); FordA does not
(0.4% — far more metric-like than the proxy). Report this as dataset-dependent, not uniform.

**Update — decision-rule item 3 (downstream usefulness), attempted after the above was
written.** `reproduce/tables/table_3_7_g2_downstream.py`, new. Reuses NERFCM, and Chapter
5's `select_coverage_cover`/`select_multiscale`, unmodified — both already matrix-only and
already discover k, so the prose's "never run on a real dissimilarity matrix" claim was
about call sites always feeding them coordinate-derived matrices, not a code limitation.

| Dataset | NERFCM-given-k (ARI) | set-cover (ARI) | gap | within 0.05? |
|---|---|---|---|---|
| ECG5000 (k=5) | 0.593±0.044 | 0.715 | 0.122 | no |
| Crop (k=24) | 0.029±0.009 | 0.064 | 0.034 | yes |
| FordA (k=2) | 0.000±0.000 | 0.002 | 0.001 | yes (degenerate) |

**Later extended to FordA (its ~2h matrix rebuild, previously only estimated, was
by then a measured cost — worth spending given it was the difference between "partial"
and "at the required count").** Three results, and the honest read is *not* "2 of 3 pass
so the item is basically closed":

- **ECG5000** — the one dataset with real, non-degenerate structure. The set-cover
  *beats* NERFCM by 0.122, a favorable direction that still fails the criterion because
  it overshoots the ±0.05 band on the good side.
- **Crop** — both methods are weak in absolute terms (0.029/0.064 ARI on 24 true
  classes, barely above chance) and happen to land close to each other. A "yes"
  reflecting two struggling methods agreeing, not two good ones.
- **FordA** — every method scores at essentially zero ARI (NERFCM 0.000, set-cover
  0.002, single-linkage -0.000, beta-plateau 0.001). FordA's binary-derived k=2
  structure isn't recoverable from its DTW dissimilarities by *any* method tested, so
  this "yes" is five methods failing identically, not the set-cover matching a working
  baseline.

**Net assessment:** the set-cover never loses badly to NERFCM on real DTW data, and on
the one dataset where NERFCM finds real structure, the set-cover finds *more* of it —
but the literal "at least three of five DTW sets" threshold, read strictly (three sets
where the criterion is *met*, not merely attempted), is **still not satisfied**: only
2 of the 3 tested sets pass, and both passes are low-information (degenerate or weak).
ElectricDevices and StarLightCurves were never attempted. Single-linkage collapses to
~0 ARI on every dataset (chaining on noisy real-world DTW dissimilarities). ConiVAT and
bottleneck-bootstrap were correctly identified as genuine implementation gaps (ConiVAT's
metric-learning step needs coordinate axes DTW time series don't have; bootstrap needs
~100 DTW matrix rebuilds) rather than missing call sites, and were not attempted. **This
decision-rule item is the one piece of G2 that stays open** — not because the method
looks bad, but because the evidence so far doesn't cleanly satisfy the criterion as
literally written, in either the favorable or unfavorable direction.

## C13 — Large-scale regression benchmark, extended to 10 seeds

`reproduce/tables/table_a7_regression_scale.py`. California Housing via
`sklearn.fetch_california_housing()` (canonical); Superconductivity via UCI id 464 direct
download, decorrelated (FeatureAgglomeration, corr threshold 0.9) before every model, per
the single-seed pilot's finding that raw features break the flat MoG's closed-form solve.

| Dataset | flat MoG | fuzzy tree | HME mixture | CART | Random Forest |
|---|---|---|---|---|---|
| California Housing R² | 0.631±0.020 | 0.474±0.014 | 0.493±0.043 | 0.603±0.019 | **0.809±0.008** |
| Superconductivity R² | -0.261±1.431 | 0.730±0.010 | -0.766±2.411 | 0.873±0.008 | **0.923±0.004** |

Random Forest wins both cleanly. More interesting: **flat MoG and HME mixture are wildly
unstable on Superconductivity** even after decorrelation — huge variance, occasionally
catastrophic negative R². Echoes the seed-9 HME divergence already documented elsewhere in
this project (Concrete, `table_concrete_reconciliation`). Worth its own investigation.

## Explicitly not attempted this session (and why)

- **C1 (ANFIS/GA-FIS baseline adapters)** — requires writing new adapter code against
  external libraries, not just running existing harness scripts; too large to do reliably
  in the remaining window without risking a rushed "must-have" item.
- **C11/G9 (`IVATMeans` vs. FCM/k-means)** — self-estimated at "about three weeks" in
  Ch7; a rushed partial size-ladder would produce a misleading partial result.
- **BETH one-class path** — Ch7 §7.3 calls this "a research decision before a coding
  one" (leave-one-class-out needs ≥3 classes; BETH is binary). Not a benchmark to run.
- **Shuttle capstone integration** — a build+integration goal depending on G1 and C3
  (Ch7 §7.1), not a single benchmark.
- **G2 decision-rule item 3** (baseline comparison) — no scaffold; flagged above.

## Bugs found and fixed

1. `table_a1_feature_scoring.py`, `_mf_dedup.py` / `table_4_8_mf_dedup.py`: stale class
   names from before an upstream `tribble-fis` rename. Fixed, backfilled, verified.
2. `run_all_tables.sh`'s archive step: hit its own documented mid-run-edit failure mode;
   recovered via `--archive-only` per the script's own instructions.

## Provenance

- `tribble-fis` @ `80e98d755d9649b0bad5c448bab6b88fba468e45`
- `tribble-cluster` @ `85b68a8a58c004756e8112cca3a3b9b110cf4ffc`
- `tribble-opt` @ `55dfc8e386e41c14604b983218f4866d7db150db`
- `grad-school` @ `20dd460accad3f031e561a8096236670b4cbd4cf` (branch `fix/sample-problems`)
- Archive label: `reproduce/outputs/goal-8h-2026-08-11-fullsuite/`
