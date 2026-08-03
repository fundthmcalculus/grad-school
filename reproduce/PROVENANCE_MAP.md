# Provenance map — every proposal table to the script that makes it

One row per numbered table in `research/proposal-defense/prose/`. The point of
this file is that no number in the proposal should be untraceable: each one
either names the script and output file it comes from, or is explicitly marked
as having no generator yet.

**Status vocabulary**

| Status | Means |
|---|---|
| **reproduced** | A generator exists, ran, and the prose cells match its output. |
| **drifted** | A generator exists and ran, but the prose cells do not match it. The prose is quoting an older run. |
| **stale** | The prose is quoting a run made against superseded code. A corrected run exists. |
| **cited** | Deliberately not harness-reproduced; attributed to a published measurement, with the reason recorded. |
| **traced** | No table generator, but the numbers trace to a named script's findings file. |
| **ungenerated** | No script produces this table. Hand-authored or structural. |

---

## Chapter 3 — pVAT

| Table | Generator | Output | Status |
|---|---|---|---|
| 3.1 Reorder time | `reproduce/tables/table_3_1_pvat_scaling.py`, `table_3_1_reorder_three_arm.py` | `outputs/table_3_1.{md,csv}`, `outputs/table_3_1_three_arm.{md,csv}` | **reproduced** for the swept grid; headline row **cited** — note 1; re-taken on one host — note 11 |
| 3.2 Complexity fit | `reproduce/tables/table_3_1_reorder_three_arm.py` | `outputs/table_3_1_complexity_fit.{md,csv}` | **reproduced** — exponents confirm; stage-two plateau does **not** reproduce, note 11 |
| 3.3 Memory footprint | `reproduce/tables/table_3_2_memory_precision.py` | `outputs/table_3_2_memory_precision.{md,csv}` | **reproduced** — all 32 cells identical to `main-d0efefc` |
| 3.4 GPU speedups | `ClusteringExperiments/{boruvka_gpu,gpu_vat}.py` | findings only | **ungenerated** — needs a GPU host |
| 3.5 Adversarial ARI | `ClusteringExperiments/adversarial_eval.py` | `ClusteringExperiments/findings/…` | **reproduced** — two cells corrected, note 10 |
| 3.6 Stitch ablation | `ClusteringExperiments/principled_stitch.py` | `ClusteringExperiments/findings/…` | **reproduced** — re-quoted, note 10 |
| 3.7 Non-metric agreement | `ClusteringExperiments/hardening_eval.py` | `ClusteringExperiments/findings/…` | **reproduced** — cells match |

**These row numbers were one behind the prose.** Table 3.2 (the complexity fit)
was inserted into the chapter and this map never renumbered, so every row from
the second down named the table above it. The visible cost: Table 3.3's memory
ceilings were listed as having *no generator* while
`table_3_2_memory_precision.py` produced them and reproduced all 32 cells
exactly — the file whose one job is to make numbers traceable was pointing a
reader at nothing. The generator's own docstring says "Table 3.3"; the filename
still says 3.2, and is left alone because renaming it would break the archives
that carry `table_3_2_memory_precision.csv`.

**Note 10 — these three have moved, and were unrunnable before that.** They
originally lived in the `tribble-cluster` submodule and imported each other as
`from experiments.foo import ...`, which needed the submodule root on
`sys.path`; invoked by path — the form the manifest used — every one died with
`ModuleNotFoundError: No module named 'experiments'` before doing any work.

grad-school #26 then moved them to `ClusteringExperiments/` **without updating
those imports**, so all 37 affected files were still broken on arrival. The
imports are now rewritten to plain sibling form and the run instructions in
their docstrings updated to match. They are driven by
`reproduce/experiments/run_cluster_experiment.py`, which puts their directory on
`sys.path` and redirects `FIG_DIR` to `reproduce/outputs/figures/cluster/`, so a
regenerated Chapter 3 figure lands with the rest of the evidence.

With them actually running, Table 3.6 reproduces exactly and Table 3.4 needed two
cells corrected (circles/naive-block 0.10 → 0.00, bridged/naive-block 0.07 →
0.08). Table 3.5 had drifted further and is re-quoted: light 0.51 → **0.47**,
top-m-only 0.74 → **0.61**, fps-only 0.39 → **0.37**, and the fraction ≥ 0.9 for
top-m-only 0.72 → **0.60**. The conclusion is unaffected and in fact strengthened
— the principled combination still reaches mean ARI 1.00 with min 1.00, and the
gap to each single ingredient is now wider.

**Note 1 — Table 3.1's 4,096-point pair is a cited measurement, by choice.**
The headline row (classical 124 s vs pVAT 2.56 s, ~48x) comes from the NAFIPS
work, not from this harness. `table_3_1_pvat_scaling.py` caps its cubic reference
at N <= 1024 (`REPRO_NAIVE_CAP`); at that size it reports 23.18 s vs 0.033 s, a
ratio of 692x, and the three-arm generator produces the ~48x figure at N = 500
and 1,000.

Lifting the cap to 4,096 is one flag, and is deliberately not done: the cubic arm
costs ~64x more at 4x the size, so reproducing that cell is several hours of
compute to re-derive a constant factor. The chapter's claim is the *scaling* -- a
cubic-to-quadratic exponent drop and a feasible problem size moving from ~5,000
to >130,000 points -- and the exponent is established by the swept grid and the
three-arm decomposition, both of which run in minutes. Cite the row; do not
re-run it.

**Note 11 — the grid was re-taken on one host, and two claims did not survive.**
§3.4 spanned two machines: the swept grid came from the development laptop
(4-core i7-1185G7, 16 GB, `powersave`) while the memory ceilings came from the
workstation (i9-14900HX, 32 logical cores, 96 GB, RTX 4080 Laptop). Checklist B5
asked for one host; `outputs/full-14900hx-*` is the workstation run.

*What the re-take settles.* The ~45–50% wall-clock swing is **thermal and
laptop-specific**. Three runs here put the 1,024-point classical arm at 13.7,
14.2 and 14.5 s — a 6% spread against the laptop's 22.2/31.7/21.3 s.

*What it costs.* The ratio is **not** machine-invariant, which §3.4 asserts it is.
The 1,024-point speedup reads 1,129x on the laptop and 660–700x here, a 40% move,
because the classical arm is interpreted Python and mergeVAT is compiled, so the
two arms do not respond to a change of host by the same factor. Ratios still
travel far better than seconds, and the reporting standard stands; the
justification for it needs weakening from "survive a change of machine" to
"stable within a host, and far more portable than seconds".

*What is retired.* The stage-two **10 ms fixed cost, the flat plateau above
N ≈ 750, and the parity band where stage two loses to stage one** are properties
of the laptop, not the kernel. Across four independent runs here stage two is
monotone in N (0.5 ms at N=750, 8.4 ms at N=3,000) and beats stage one by
8.1–17.7x at every size, including 17.3x in the band said to collapse. The fitted
exponents are 3.14–3.19 (classical), 1.84–1.87 (stage one) and **1.93–1.95**
(stage two) — the last a *cleaner* confirmation of the quadratic claim than the
laptop's 2.12, which the chapter itself calls "right for the wrong reason"
because the plateau contaminated the fit. **CHECKLIST C2b, which asks what the
10 ms is, should be rescoped: the thing to explain is why the laptop had it.**

---

## Chapter 4 — MoG FIS

| Table | Generator | Output | Status |
|---|---|---|---|
| 4.1 Value of the transform | `table_hyperparam_normalization.py` | `outputs/table_hyperparam_normalization.{md,csv}` | **reproduced** at 10 seeds |
| 4.2 Output partitioning | `table_g5_output_partitioning.py` | `outputs/table_g5_output_partitioning.{md,csv}` | **reproduced** — claim retracted, note 3 |
| 4.3 Partitioning vs skew | `table_g5b_skew_sweep.py` | `outputs/table_g5b_skew_sweep.{md,csv}` | **reproduced** — hypothesis refuted, note 4 |
| 4.4 What MoG achieves | `table_4_1_mog_baselines.py` + `table_hyperparam_normalization.py` | `outputs/table_4_1.{md,csv}` | **reproduced** at 10 seeds |
| 4.5 Baseline comparison | `table_4_1_mog_baselines.py` | `outputs/table_4_1.{md,csv}` | **reproduced**; ANFIS/GA-FIS still absent |
| 4.6 Anomaly operating curve | `table_4_4_openset.py` (`REPRO_THETA_SWEEP=0.5,...,1.1`) | `outputs/table_4_4b_theta_sweep.{md,csv}` | **reproduced** — note 6 |
| 4.7 Vs dedicated detectors | `table_4_4_openset.py` | `outputs/table_4_4_openset.{md,csv}` | **reproduced** — note 6 |
| *(no prose table)* | `table_norm_conorm_matrix.py` | `outputs/table_norm_conorm_matrix.{md,csv}` | backs `TNORM_REEVALUATION_RESULTS.md` |

**Note 2.** Re-quoted at 10 seeds: 1st order 0.658 → 0.783 (Δ +0.125), 2nd order
0.796 → 0.829 (Δ +0.033). The table now also carries the CART and Random Forest
rows, which are the control that gives it force — both are rank-based, so a
monotone feature transform is worth exactly +0.001 and +0.000 to them, against
+0.125 for the fuzzy model.

**Note 3 — the claim was retracted, not just re-quoted.** The prose read a
crossover near four buckets off this table ("at three buckets uniform wins
outright; by six, quantile is ahead"). It does not survive. At 10 seeds the
largest gap anywhere in the 18-configuration sweep is 0.012 in R², against
seed-to-seed deviations of 0.02–0.03, and the 6-bucket/2nd-order pair agrees to
three decimals. The `min bucket n` diagnostic (132/343/75/257/39/171) reproduces
exactly, so the bucket-starvation *mechanism* is intact — Concrete's skew of
+0.42 is simply too mild for it to reach the aggregate error.

**Note 4 — the hypothesis was refuted, and Goal G5 is reopened.** The prose
claimed quantile's advantage "grows monotonically with skew (+0.003 → +0.201)".
At 10 seeds Q−U is negative in every row past symmetry: +0.000, −0.016, −0.068,
−0.291, −2.413, −11.811. The right reading is in the spreads, not the means:
quantile's deviation explodes (±0.208, ±0.990, ±4.448, ±21.155) while uniform's
stays bounded and its mean decays smoothly toward zero. Quantile under heavy skew
does not become less accurate, it becomes *unstable* — a few catastrophic splits
drag the mean — and a 3-seed run simply missed them. Ch 7's G5, previously marked
"settled (complete)" with "quantile by default", is reopened; the recommendation
is withdrawn and the diagnosis kept.

**Note 5.** Re-quoted at 10 seeds. Tables 4.4/4.5 now read flat R² −0.005 / 0.783
/ 0.829, rising to 0.869 at full second order, with CART 0.825 ± 0.047 and RF
0.909 ± 0.018. The old 0.797/0.904 pair appears in no archive and came from a
3-seed run predating every run under `outputs/`. The zeroth-order figure is worth
keeping visible rather than dropping: with constant consequents the model is no
better than predicting the mean, which makes first-order consequents a
requirement here rather than a refinement.

**Note 6 — re-quoted twice, and the ordering flipped both times.** Tables 4.6/4.7
originally quoted the pre-`pin_extremes` path. Current values are 10-seed, from
`outputs/seeds10-2026-08-01/` (tribble-fis `23bfdbc`, hamacher, Glass).

The operating point moved θ = 0.80 / J = +0.155 (2 seeds, pre-fix) → θ = 0.70 /
+0.261 (5 seeds) → a flat band of +0.222…+0.239 across θ = 0.5–0.8 peaking at
θ = 0.60 (10 seeds). More usefully: there is no sharp optimum to tune to, so the
knob is forgiving.

Table 4.7's ranking flipped twice — complement-rule-leads, then isolation-forest
-by-0.038, then level to 0.002 at ten seeds. Three orderings from the same
experiment is the tell that all three were noise: the across-class deviations are
roughly twice the largest gap in the table. Do not quote a winner from it. The
one-class SVM trailing at +0.076 is the only separation exceeding its own error
bar.

**A knob to know about.** `REPRO_THETA_SWEEP` is a comma-separated list of θ
values, *not* a boolean. `REPRO_THETA_SWEEP=1` is a valid list of one and emits a
single row at θ = 1.0, where the boost saturates the aggregate and every cell is
legitimately zero — output that reads exactly like a null result. Both sweeps in
this pass were initially run that way. Use
`REPRO_THETA_SWEEP=0.5,0.6,0.7,0.8,0.9,0.99,1.1`.

---

## Chapter 5 — topological membership generation

| Table | Generator | Output | Status |
|---|---|---|---|
| 5.1 The battery | `run_all.py` → `table_5_x_ch5_selection.py` | `outputs/table_5_1_battery.{md,csv}` | **reproduced** — after the note-9 correction |
| 5.2 Multi-scale recovery | `run_all.py` → `table_5_x_ch5_selection.py` | `outputs/table_5_2_multiscale.{md,csv}` | **reproduced** |
| 5.3 Selection comparison | `run_all.py` → `table_5_x_ch5_selection.py` | `outputs/table_5_3_selection.{md,csv}` | **reproduced** |

Chapter 5 is the best-behaved chapter in the proposal in design: one deterministic
driver, one seeded JSON of record, every figure regenerated from it. Two gaps
turned up when that design was actually exercised.

**The driver did not run.** `run_all.py` crashed partway through figure generation
with `NameError: name 'row' is not defined` at `fig_membership`. A commit on
2026-07-20 changed that figure's grid from `subplots(2, 2)` to `subplots(2, 1)`
while the function body still addressed `axes[row, 0]`, `axes[0, 1]`, and
`axes[1, 1]`. Because `results.json` is written *after* every figure, the crash
meant the whole numeric run was discarded on each invocation — the JSON on disk
was last successfully written on 2026-07-20 and nothing since could regenerate it.
The chapter's claim that the driver "writes the results and every figure
referenced below" was therefore not true when written. Fixed by restoring the 2×2
grid and the `enumerate` that supplies `row`, matching the identical pattern
`fig_transform` already uses. After the fix the driver completes, regenerates 16
of its 17 figures (`fig11_scaling` is behind an opt-in `--scaling` flag), and
rewrites `results.json` **byte-identical to the 2026-07-20 file** — so the
chapter's numbers were never wrong, only unreproducible.

**The tables were hand-transcribed.** `reproduce/tables/table_5_x_ch5_selection.py`
closes that: it does no computation, only renders the JSON, so a stale prose cell
now shows as a diff.

**Note 9 — what the renderer caught in Table 5.1.** The prose table had the
bridged-Gaussians row wrong in two ways, both now fixed. The `0.00 (chaining)`
cell sat under *NERFCM on D\**, but 0.00 is `iVAT_SL_ari` — plain single-linkage.
Both NERFCM columns actually score 1.00 on that dataset. And three cells were
dashed as "not run in that configuration" when the driver did run them
(`NERFCM_D_ari` = 1.00 on bridged, 0.98 on varying_density). The table now carries
an explicit single-linkage column, which is also what the chapter's own prose
describes.

The same row exposed a genuine scoring ambiguity worth keeping in view: the
set-cover on bridged Gaussians scores **0.982** over the points it covers and
**0.001** over all points with the uncovered 47% counted as unassigned. Both are
in `results.json`, under `main_table.cover_ari` and
`persistence_methods.bridged_gaussians.methods.persistence_gap.ari` respectively.
The prose quotes the all-points figure, which is the conservative reading; the
renderer emits both so the choice is visible rather than implicit.

---

## Chapter 6 — hierarchical / refined FIS

| Table | Generator | Output | Status |
|---|---|---|---|
| 6.1 Model family, one protocol | `table_concrete_reconciliation.py` | `outputs/table_concrete_reconciliation.{md,csv}` | **reproduced** — HME caveat, note 7 |
| 6.2 External baselines | `table_6_1_model_family.py` | `outputs/table_6_1.{md,csv}` | **reproduced** at 10 seeds — note 8 |
| 6.3 Interpretability | *none* | — | **ungenerated** — structural by design |
| 6.4 Memory augmentation | `tribble-fis/tests/test_double_pendulum.py` | none | **ungenerated** — entry point unconfirmed |

**Note 7 — one seed in ten destroys this cell, and that is the finding.** Table
6.1 is re-quoted at 10 seeds: flat 2nd-refined 0.875 ± 0.019, fuzzy tree
0.688 ± 0.056, CART 0.826 ± 0.047, RF 0.909 ± 0.018. The mixture-of-experts row
is the exception. Under log+standardized features at library defaults the
harness reports **R² = −220.9 ± 665.0**, because on seed 9 the model predicts up
to 10,536 MPa on a target that never exceeds ~82. The other nine seeds give
0.805 ± 0.059 (RMSE 7.15 ± 0.74) with nothing anomalous about them; the five-seed
protocol did not contain the offending split and reported a clean 0.813 ± 0.039.

The prose quotes the nine-seed figure with the divergence disclosed in a
footnote, on the grounds that a mean of −220.9 describes the failure rather than
the model and hiding it would be worse than either. Two consequences worth
carrying: the HME gating solve needs a numerical guard before the hierarchy can
be recommended for use, and — more broadly — this is the sharpest evidence in the
proposal that a 5-seed mean cannot establish stability.

**Note 8.** Re-quoted at 10 seeds. This table runs everything at **raw features
and library defaults**, which is why every cell sits below Table 6.1's; the two
must not be read as one series. Its PhiUSIIL column is now filled and shows the
dataset is saturated — CART and RF both reach 1.000, the fuzzy models 0.970–0.997
— so PhiUSIIL discriminates between these methods hardly at all and should carry
no weight in the comparison.

---

## Chapter 7

Table 7.1 is a goals-and-status matrix, not a measurement. No generator applies.

---

## Appendix A.4 — feature scoring

| Table | Generator | Output | Status |
|---|---|---|---|
| A.1 Feature ranking by scorer | `reproduce/tables/table_a1_feature_scoring.py` | `outputs/table_a1_feature_ranking.{md,csv}` | **reproduced** — all 20 cells identical to `main-d0efefc` |
| A.2 Accuracy and fit time vs features kept | `reproduce/tables/table_a1_feature_scoring.py` | `outputs/table_a2_feature_count.{md,csv}` | **reproduced within a host**; the bhattacharyya accuracies are **not host-portable** — note 12 |

**Note 12 — one arm of A.2 moves between environments, and it is the arm the
appendix is least resting on.** Against `main-d0efefc`, at the same `tribble-fis`
commit, the same ten seeds and an *identical* A.1 ranking, every bhattacharyya
accuracy in A.2 sits higher: +0.017 at 4 features, +0.033 at 5, +0.043 at 7,
+0.040 at 10, +0.029 at 15, +0.030 at 20. Wasserstein and composite agree to
within 0.0002 everywhere.

This is not nondeterminism. Two complete sweeps on this host
(`full-14900hx-2026-08-02` and `full-14900hx-r2`) reproduce **every one of those
accuracies exactly**; only fit times move. It is an environment difference, and
it could not be narrowed further because no archive before this one recorded the
numeric stack — `PROVENANCE.txt` now carries numpy/scipy/sklearn and the BLAS
build for that reason.

The arm that moved is the ill-conditioned one, which is what a BLAS or threading
difference would look like: bhattacharyya's own ranking scores 0.4267 at one
feature, so its models are fitted on poor features and sit where small numerical
differences change the outcome. **Do not quote A.2's bhattacharyya cells to four
decimals across machines.** A.4's actual argument is untouched — it rests on
wasserstein 0.9967 against bhattacharyya 0.4267 at a single feature, a gap of
0.57 against a host effect of 0.04.

---

## Verification runs behind this map

| Run | Label | What it establishes |
|---|---|---|
| Ten-table sweep, 5 seeds | `outputs/audit-2026-08-01/` | All ten generators green. Five tables came back **byte-identical** to `postfix-pr29`; `table_4_1` differed only in wall-clock timing. The harness is deterministic *on a fixed host and numeric stack* (measured below), so a mismatch in this map is prose-vs-harness rather than run-to-run noise — but see note 12 before reading one across machines. |
| Ten-table sweep, 10 seeds | `outputs/seeds10-2026-08-01/` | The run the chapters are now quoted at. |
| Ch5 driver | `gated-minimax-selection/outputs/` | Runs to completion after the `fig_membership` fix; `results.json` reproduces the 2026-07-20 file exactly. |
| Workstation sweep, 10 seeds | `outputs/full-14900hx-2026-08-02/` | First pass on the i9-14900HX. 13 generators green. Superseded by r2 for citation: its tables carry a degraded machine block (`ram: unknown`), and it lacks Table 4.4b. |
| **Workstation sweep, 10 seeds (run of record)** | `outputs/full-14900hx-r2/` | **The citable run.** All 13 generators green in one pass, 14 tables including the θ curve, correct machine block on every table, and the numeric stack recorded. This is the single-host re-take checklist B5 asked for — note 11. |

**The two workstation sweeps are also the determinism test, and they pass.**
Comparing them cell by cell: `table_concrete_reconciliation` (34),
`table_hyperparam_normalization` (48), `table_norm_conorm_matrix` (57),
`table_g5_output_partitioning` (126), `table_g5b_skew_sweep` (48),
`table_3_2_memory_precision` (32), `table_4_4_openset` (9), `table_6_1` (16),
`table_a1_feature_ranking` (20) and all three Chapter 5 tables (64) are
**byte-identical across two independent full runs**. Every cell that moved is a
wall clock — `table_3_1`, `table_3_1_three_arm`, and A.2's fit-time halves — and
all of those are within noise.

So "the harness is deterministic" is now a measured claim rather than an
assumption, with one boundary worth stating precisely: it is deterministic **on
one host with one numeric stack**. Note 12 is the counterexample across hosts,
and note 11 is why wall-clock cells must never be diffed as if they were results.

Before this pass, `tribble-fis` was checked out at `d0d6714` — the *pre-fix*
baseline — while the parent repo pins `23bfdbc`. Anything run in that state
silently reproduces the superseded numbers. `run_all_tables.sh` records the
submodule SHA in `PROVENANCE.txt` for exactly this reason; check it before
trusting a run.

## The pattern

Two systematic causes accounted for nearly every drifted row, and both are now
resolved:

1. **Seed count.** The prose tables were transcribed from 3-seed runs. Every
   numbered table is now quoted at ten (`common.SEEDS`), and the difference was
   not cosmetic — it retracted a crossover (note 3), refuted a hypothesis and
   reopened a goal (note 4), and exposed a catastrophic failure mode that five
   seeds never sampled (note 7).
2. **The `pin_extremes` fix.** Anything quoting the open-set path or the refined
   consequent solve from before tribble-fis PR #29 was superseded; those tables
   are re-quoted from post-fix runs.

Neither was a defect in the harness. What remains outstanding is narrow:

- **Table 3.1's headline 4,096-point pair** has no in-repo provenance (note 1).
- **Tables 3.2 and 3.3** have no generator; 3.3 needs a GPU host.
- **Table 6.3** is structural by design and 6.4's entry point is unconfirmed.
- **ANFIS and GA-FIS adapters** are still absent, so those cells stay `N/A`.

## What a reader should distrust

Three lessons from this pass, worth applying to any number added later.

**A five-seed mean does not establish stability.** Note 7 is the clean example: a
model that is excellent nine times out of ten and catastrophic the tenth reads as
a solid 0.813 ± 0.039 if the tenth split is not in the sample.

**A conclusion can be reproducible and still wrong.** Notes 3 and 4 came from
generators that ran correctly and deterministically every time. The harness was
never broken; the sample was too small to support the story built on it.

**Silence is not success.** Three separate failures in this pass — a submodule on
the wrong commit, a driver crashing before it wrote its results, and experiments
dying on import — all produced output that looked plausible or exited zero. So
did `REPRO_THETA_SWEEP=1`, which is a valid θ list of one and emits a table of
zeros that reads exactly like a null result. Check the provenance, not the exit
status.
