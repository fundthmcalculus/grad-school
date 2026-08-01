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
| **traced** | No table generator, but the numbers trace to a named script's findings file. |
| **ungenerated** | No script produces this table. Hand-authored or structural. |

---

## Chapter 3 — pVAT

| Table | Generator | Output | Status |
|---|---|---|---|
| 3.1 Reorder time | `reproduce/tables/table_3_1_pvat_scaling.py`, `table_3_1_reorder_three_arm.py` | `outputs/table_3_1.{md,csv}`, `outputs/table_3_1_three_arm.{md,csv}` | **drifted** — see note 1 |
| 3.2 Memory footprint | *none* | — | **ungenerated** |
| 3.3 GPU speedups | `tribble-cluster/experiments/{boruvka_gpu,gpu_vat}.py` | findings only | **ungenerated** — needs a GPU host |
| 3.4 Adversarial ARI | `tribble-cluster/experiments/adversarial_eval.py` | `experiments/findings/ADVERSARIAL_EVAL_FINDINGS.md` | **reproduced** — two cells corrected, note 10 |
| 3.5 Stitch ablation | `tribble-cluster/experiments/principled_stitch.py` | `experiments/findings/GAPS_FINDINGS.md` | **reproduced** — re-quoted, note 10 |
| 3.6 Non-metric agreement | `tribble-cluster/experiments/hardening_eval.py` | `experiments/findings/HARDENING_FINDINGS.md` | **reproduced** — cells match |

**Note 10 — these three could not be run as the manifest invoked them.** All four
registered tribble-cluster experiments do `from experiments.blockwise_vat import
...`, which needs the submodule *root* on `sys.path`. Invoked by path — `python
experiments/adversarial_eval.py`, the form the manifest used — Python puts
`experiments/` on the path instead and every one of them dies with
`ModuleNotFoundError: No module named 'experiments'` before doing any work. The
manifest now uses `python -m experiments.<name>`, which is the only form that
runs; see the `_uvm` helper.

With them actually running, Table 3.6 reproduces exactly and Table 3.4 needed two
cells corrected (circles/naive-block 0.10 → 0.00, bridged/naive-block 0.07 →
0.08). Table 3.5 had drifted further and is re-quoted: light 0.51 → **0.47**,
top-m-only 0.74 → **0.61**, fps-only 0.39 → **0.37**, and the fraction ≥ 0.9 for
top-m-only 0.72 → **0.60**. The conclusion is unaffected and in fact strengthened
— the principled combination still reaches mean ARI 1.00 with min 1.00, and the
gap to each single ingredient is now wider.

**Note 1.** Table 3.1's headline row — 4,096 points, classical 124 s vs pVAT
2.56 s, ~48× — is reproduced by neither generator. `table_3_1_pvat_scaling.py`
caps the cubic reference at N ≤ 1024 (`REPRO_NAIVE_CAP`), so it has no 4,096-point
classical measurement to compare against; at N = 1024 it reports 23.18 s vs
0.033 s, a ratio of 692×, not 48×. `table_3_1_reorder_three_arm.py` does produce
a ~48× figure (47.8× at N = 500, 49.4× at N = 1,000) but at three orders of
magnitude smaller N. The prose pair appears to be an external measurement carried
over from the NAFIPS work. It needs either a citation to that source or a harness
run that reproduces it; right now it is the one headline number in Chapter 3 with
no in-repo provenance.

---

## Chapter 4 — MoG FIS

| Table | Generator | Output | Status |
|---|---|---|---|
| 4.1 Value of the transform | `table_hyperparam_normalization.py` | `outputs/table_hyperparam_normalization.{md,csv}` | **drifted** — note 2 |
| 4.2 Output partitioning | `table_g5_output_partitioning.py` | `outputs/table_g5_output_partitioning.{md,csv}` | **drifted** — note 3 |
| 4.3 Partitioning vs skew | `table_g5b_skew_sweep.py` | `outputs/table_g5b_skew_sweep.{md,csv}` | **drifted** — note 4 |
| 4.4 What MoG achieves | `table_4_1_mog_baselines.py` | `outputs/table_4_1.{md,csv}` | **drifted** — note 5 |
| 4.5 Baseline comparison | `table_4_1_mog_baselines.py` | `outputs/table_4_1.{md,csv}` | **drifted** — note 5 |
| 4.6 Anomaly operating curve | `table_4_4_openset.py` (`REPRO_THETA_SWEEP=1`) | `outputs/table_4_4b_theta_sweep.{md,csv}` | **stale** — note 6 |
| 4.7 Vs dedicated detectors | `table_4_4_openset.py` | `outputs/table_4_4_openset.{md,csv}` | **stale** — note 6 |
| *(no prose table)* | `table_norm_conorm_matrix.py` | `outputs/table_norm_conorm_matrix.{md,csv}` | backs `TNORM_REEVALUATION_RESULTS.md` |

**Note 2.** Prose quotes 1st order 0.646 → 0.797 and 2nd order 0.783 → 0.845. The
harness reports 0.664 → 0.776 and 0.784 → 0.817. The *direction and rough
magnitude* of the transform effect survive; the exact cells do not.

**Note 3 — a conclusion changes here, not just a number.** The prose reads a
crossover near four buckets off this table: "At three buckets uniform wins
outright; by six, quantile is ahead." Under the 5-seed harness the spread does
not support it — at 3 buckets/1st order uniform leads 0.781 to 0.779, at 6
buckets/2nd order it is 0.839 to 0.840, and both gaps are far inside the ±0.02–0.03
seed deviation. The `min bucket n` diagnostic column (132/343/75/257/39/171) *does*
match exactly, so the bucket-starvation mechanism is intact; it is the accuracy
crossover built on top of it that the reproducible run does not show.

**Note 4 — likewise.** The prose claims quantile's advantage "grows monotonically
with skew (+0.003 → +0.201)". The 5-seed harness gives Q−U of +0.003, −0.001,
+0.008, +0.270, +0.019, −1.514 across its skew grid: not monotone, and the last
two rows have standard deviations larger than the effect (±0.749, ±3.483). The
strong middle result (+0.270 at skew +9.74) is real and larger than the prose's
+0.201. The monotonicity framing is what fails.

**Note 5.** Prose Tables 4.4/4.5 quote flat R² 0.797 / 0.845 / 0.881 and
references CART 0.797 ± 0.029, RF 0.904 ± 0.014. `table_4_1_mog_baselines.py`
reports flat R² 0.644 ± 0.015 and RF 0.913 ± 0.014; the reconciliation table
reports CART 0.816 ± 0.037. No archived run contains 0.797/0.904. Those cells
come from a 3-seed run that predates every archive under `outputs/`.

**Note 6.** Tables 4.6 and 4.7 quote the pre-`pin_extremes` code path. Corrected
values are in `outputs/openset-postfix/` (tribble-fis `23bfdbc`, 5 seeds, hamacher,
Glass). Best operating point moves from θ = 0.80 / J = +0.155 to **θ = 0.70 /
J = +0.261**, and at the matched θ = 0.99 point the complement rule (+0.170) now
**trails isolation forest (+0.208)** rather than leading it.

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
| 6.1 Model family, one protocol | `table_concrete_reconciliation.py` (+ `table_hyperparam_normalization.py` for the tuned HME row) | `outputs/table_concrete_reconciliation.{md,csv}` | **drifted** — note 7 |
| 6.2 External baselines | `table_6_1_model_family.py` | `outputs/table_6_1.{md,csv}` | **drifted** — note 8 |
| 6.3 Interpretability | *none* | — | **ungenerated** — structural by design |
| 6.4 Memory augmentation | `tribble-fis/tests/test_double_pendulum.py` | none | **ungenerated** — entry point unconfirmed |

**Note 7.** Prose Table 6.1 quotes flat 0.881 ± 0.001 / HME 0.862 ± 0.022 / tree
0.717 ± 0.039 / CART 0.797 ± 0.029 / RF 0.904 ± 0.014. The reconciliation table
gives flat-2nd-refined 0.868 ± 0.020, HME 0.813 ± 0.039, tree 0.714 ± 0.046, CART
0.816 ± 0.037, RF 0.913 ± 0.014. The HME row is the one to watch: 0.862 is the
*demo-tuned* HME from the hyperparameter table (which reports 0.826 ± 0.026 at
5 seeds), not the reconciliation's library-default arm. One prose table is
currently drawing from two generators at two different configurations.

**Note 8.** `table_6_1_model_family.py` runs the fuzzy arms at raw preprocessing
and library defaults — its flat R² is 0.644, which is not comparable to Table
6.1's uniform-protocol number. It is usable as an external-baseline source
(CART/RF/M5) and not much else. The filename predates the prose renumbering; it
now feeds Table 6.2.

---

## Chapter 7

Table 7.1 is a goals-and-status matrix, not a measurement. No generator applies.

---

## Verification runs behind this map

| Run | Label | What it establishes |
|---|---|---|
| Ten-table sweep, 5 seeds | `outputs/audit-2026-08-01/` | All ten generators green. Five tables came back **byte-identical** to `postfix-pr29`; `table_4_1` differed only in wall-clock timing. The harness is deterministic, so every mismatch in this map is prose-vs-harness, not run-to-run noise. |
| Ten-table sweep, 10 seeds | `outputs/seeds10-2026-08-01/` | The run the chapters are now quoted at. |
| Ch5 driver | `gated-minimax-selection/outputs/` | Runs to completion after the `fig_membership` fix; `results.json` reproduces the 2026-07-20 file exactly. |

Before this pass, `tribble-fis` was checked out at `d0d6714` — the *pre-fix*
baseline — while the parent repo pins `23bfdbc`. Anything run in that state
silently reproduces the superseded numbers. `run_all_tables.sh` records the
submodule SHA in `PROVENANCE.txt` for exactly this reason; check it before
trusting a run.

## The pattern

Two systematic causes account for nearly every **drifted** row:

1. **Seed count.** The prose tables were transcribed from 3-seed runs; the harness
   defaults to 5 (`common.SEEDS`). Several Chapter 4 conclusions are inside the
   seed-to-seed spread at 5 seeds and only look decisive at 3.
2. **The `pin_extremes` fix.** Anything quoting the open-set path or the refined
   consequent solve from before tribble-fis PR #29 is superseded.

Neither is a defect in the harness. Both mean the same thing: **the prose numbers
predate the reproducible pipeline, and re-quoting them from it is the outstanding
work.**
