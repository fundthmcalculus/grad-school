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
| 3.4 GPU speedups | `reproduce/tables/table_3_4_gpu_speedups.py` | `outputs/table_3_4_gpu_speedups.{md,csv}` | **drifted** — measured on the card the chapter names; the exactness claim holds, three of the four speedup rows do not read as quoted — note 15 |
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

**Note 15 — Table 3.4 is measured now, and the exactness claim survives while
three of the four speedup rows do not.** `table_3_4_gpu_speedups.py` runs on the
card §3.4 names (RTX 4080 Laptop, 12 GB, driver 610.74, CuPy 14.1.1 on CUDA
runtime 12.9, compute capability 8.9) against the same host's CPU — the "32-core
CPU" of that table is this machine's 32 logical cores, not a separate box. Ten
seeds, every device timing synchronised, all kernels warmed before measurement
(the first `boruvka_mst_device` call spends ~0.4 s compiling its RawModule, 13x
the N=16,000 kernel time, which would have made the small-N cells fiction). The
run of record is `outputs/gpu-table34-2026-08-02/`, 1332 s through
`run_all_tables.sh`; every figure below is from its CSV, checked against it
programmatically rather than read off. Two full runs hours apart agree within 2%
on most swept rows and under 1% at N = 32,000, but not everywhere: at N = 8,000 —
the smallest, noisiest point on the grid — the MST ratio moved 11% (8.5× → 7.6×)
and the matched front end 18% (3.4× → 2.8×). Quote the large-N cells.

*What holds, and holds more strongly than quoted.* The exactness column is the
chapter's central GPU claim and it reproduces everywhere it is claimed. The VAT
ordering derived from the device Borůvka MST is elementwise identical to the
compiled Cython serial Prim reference at float64 at **every** N and **every** one
of the ten seeds; FCM agrees on 100.0% of hard labels (the chapter claims
">99%") with centres within 5e-5 of the CPU fixed point, and within 6e-13 of the
matched-formulation arm.

*Row 1 — the MST.* Quoted "≈5× at N = 32,000, growing with N". Measured
**6.3×** at N = 32,000 — the chapter *understates* it — but it does **not** grow
with N: 5.4× (4,000), 7.6× (8,000), 7.7× (16,000), 6.3× (32,000). It peaks in the
middle of the grid and falls off at the top. The CPU arm is O(n²) dense Prim and
the device arm is O(n² log n) Borůvka rounds, so a non-increasing ratio is what
the algorithms predict; "growing with N" belongs to the front-end row, not this
one.

*Row 2 — the front end.* Quoted "≈4.8–6.6× end to end". With **matched work** —
both arms producing an ordering and nothing else — it reads 2.3×, 2.8×, 3.9×,
**4.9×** across the same grid. It does grow with N, and only the largest cell
reaches the bottom of the quoted band. The band is reproducible only by letting
the CPU arm additionally materialise the reordered n × n matrix
(`compute_vat_c`) that the GPU arm never builds; that unmatched pair reads
5.6–11.8×, i.e. *above* the quoted band. The chapter's cell therefore matches
neither arm: it sits between a comparison that is fair and one that is not.

*Row 3 — Fuzzy C-Means, the row to fix.* Quoted **30–56×**. Measured 14.4×,
24.1×, 41.0× at N = 50,000 / 200,000 / 500,000 against `fcm.fuzzy_c_means` —
below the quoted range at the two smaller sizes. More importantly, that ratio is
**not a device speedup**: `fcm.fuzzy_c_means` is a NumPy broadcasting
implementation materialising (n, k, d) and (n, k, k) temporaries, while
`gpu.fuzzy_c_means_gpu` uses the gram identity and two GEMMs. Run the GPU's own
formulation in NumPy/BLAS on the CPU and the same three sizes read **1.3×, 2.1×,
3.7×**. About an order of magnitude of the quoted figure is the rewrite, not the
card. This is note 11's hazard again — a ratio between arms that differ *in kind*
rather than in device — and it is why the generator carries both CPU arms.
Caveat on all six FCM cells: with identical initial centres and convergence test
the iteration count to the fixed point ranges from 11 to the 100-iteration cap
across the ten seeds, so the seconds carry a spread as large as their mean. Read
the CSV before quoting one.

*Row 4 — pairwise distances, and the loss is real.* The negative result
reproduces cleanly and is the best-behaved row in the table: float64 at d = 10
runs at **0.30×** of the CPU (i.e. 3.3x slower), d = 50 at 0.38×, and even
d = 784 at 0.64×; float32 at d = 10 is 0.35×. The GPU wins at d = 200 (1.08×
f64, 1.07× f32) and at d = 784 f32 (1.54×), rising to 2.06× and **4.18×** with
`high_precision=False`. Two departures from the quoted "1.3–2.5×, exact": the
upper end is higher than quoted, and **the fastest cells are not exact** —
`high_precision=False` accumulates in the input dtype and deviates from the CPU
kernel by ~1e-4. Exactness in this row holds for the high-precision mode only
(max |Δ| = 0 at float32, ≤4.3e-14 at float64).

*The one row that is a demonstration, not an estimate.* The largest float32
device-resident front end that fits the card, N = 48,000 (9.22 GB resident), is
recorded once with its hardware, precision and footprint per §7.2 rather than as
a ten-seed mean — and it should be read that way, because runs of identical code
on the same seed put it at **1.06×, 3.50× and 3.49×**, a 3.3x move at the VRAM
edge between the first and the rest. Its ordering agreement is **0.99992**, not
1.0: about four of 48,000
positions differ from the serial reference. That is a benign tie-break, not an
error — the Prim total of the two orderings is identical to every digit printed
(relative difference 0.0e+00), so the device found a different member of a set of
equally valid minimum spanning trees. The "bit-identical" claim is exact at
float64 at every size tested and at float32 up to 32,000; it is not universal.

*What could not be measured.* §3.4 predicts a datacenter card with full-rate FP64
would flip the pairwise-distance loss. There is no such card on this host, so
that prediction stays exactly what the chapter calls it — untested. No cell
estimates or extrapolates it, and the generator does not model it.

---

## Chapter 4 — MoG FIS

| Table | Generator | Output | Status |
|---|---|---|---|
| 4.1 Value of the transform | `table_hyperparam_normalization.py` | `outputs/table_hyperparam_normalization.{md,csv}` | **reproduced** at 10 seeds; **now three arms, and the prose's column label is wrong — note 16** |
| 4.2 Output partitioning | `table_g5_output_partitioning.py` | `outputs/table_g5_output_partitioning.{md,csv}` | **reproduced** — claim retracted, note 3 |
| 4.3 Partitioning vs skew | `table_g5b_skew_sweep.py` | `outputs/table_g5b_skew_sweep.{md,csv}` | **reproduced** — hypothesis refuted, note 4 |
| 4.4 What MoG achieves | `table_4_1_mog_baselines.py` + `table_hyperparam_normalization.py` | `outputs/table_4_1.{md,csv}` | **reproduced** at 10 seeds |
| 4.5 Baseline comparison | `table_4_1_mog_baselines.py` (+ `table_hyperparam_normalization.py` for the full-2nd row) | `outputs/table_4_1.{md,csv}` | **reproduced**; ANFIS/GA-FIS still absent; the two MoG rows are from two different code paths — note 14 |
| 4.6 Anomaly operating curve | `table_4_4_openset.py` (`REPRO_THETA_SWEEP=0.5,...,1.1`) | `outputs/table_4_4b_theta_sweep.{md,csv}` | **reproduced** — note 6 |
| 4.7 Vs dedicated detectors | `table_4_4_openset.py` | `outputs/table_4_4_openset.{md,csv}` | **reproduced** — note 6 |
| *(no prose table)* | `table_norm_conorm_matrix.py` | `outputs/table_norm_conorm_matrix.{md,csv}` | backs `TNORM_REEVALUATION_RESULTS.md` |

**Note 2.** Re-quoted at 10 seeds: 1st order 0.658 → 0.783 (Δ +0.125), 2nd order
0.796 → 0.829 (Δ +0.033). The table now also carries the CART and Random Forest
rows, which are the control that gives it force — both are rank-based, so a
monotone feature transform is worth exactly +0.001 and +0.000 to them, against
+0.125 for the fuzzy model. **Superseded on the normalization axis by note 16:
the column this note calls "normalized" is min-max, not z-score, and the table
now has three arms.**

**Note 16 — Table 4.1's "log+std" column never measured standardization, and the
third arm now shows the mislabel was lucky.** `gauss_math.standard_transform`,
behind every "log+std" / "standardized" / "normalized" figure in this document,
computed `(X − min)/(max − min)` — **min-max to [0,1]**, never z-score, despite
the name. `tribble-fis` PR #67 (`a385a1a`) split it into honestly-named
`UnitScalar` (min-max) and `StandardScalar` (z-score) and deleted the original,
which forced the migration and made the missing arm cheap to measure.

*The migration moved no number.* `UnitScalar(log_dynamic_range=2)` is bit-for-bit
identical to the deleted pair (`max|diff| = 0.0` exactly, same detected log
features `['Slag','Age']`), and re-running all five affected generators at ten
seeds against the run of record left `table_concrete_reconciliation` (34 cells),
`table_hyperparam_normalization` (48), `table_g5_output_partitioning` (126) and
`table_g5b_skew_sweep` (48) **byte-identical** — 256 cells, zero movement.
`table_4_1` matches `outputs/warmup-discarded/` on every accuracy cell, with only
its three wall-clocks moving ≤0.01 s inside their own bars.

*The measured facts* (`outputs/norm-three-arm-a385a1a/`, ten seeds):

| row | raw | log+min-max | log+z-score | Δ z−mm |
|---|---|---|---|---|
| CART (control) | 0.825 ± 0.047 | 0.826 ± 0.047 | 0.826 ± 0.046 | -0.000 |
| Random Forest (control) | 0.909 ± 0.018 | 0.909 ± 0.019 | 0.909 ± 0.018 | -0.000 |
| flat MoG-TSK 1st | 0.646 ± 0.039 | 0.772 ± 0.034 | 0.087 ± 0.089 | **-0.685** |
| flat MoG-TSK 2nd | 0.779 ± 0.036 | 0.824 ± 0.043 | 0.781 ± 0.045 | -0.043 |
| flat MoG-TSK full-2nd | 0.790 ± 0.054 | 0.859 ± 0.039 | 0.819 ± 0.058 | -0.040 |
| mixture of experts (demo-tuned) | 0.768 ± 0.029 | 0.834 ± 0.025 | 0.706 ± 0.024 | **-0.128** |

**The rank-based control extends cleanly to three levels**, which is what licenses
reading the rest: CART, Random Forest and both fuzzy-tree rows move by at most
**0.002** between the two normalized arms (against ±0.018–0.056 seed spreads), and
the two scalers' outputs are monotone-equivalent at Spearman 1.000000000000 per
feature.

**Min-max is therefore the correct default, not just the one that happened to
run.** It is best-or-tied in 8 of 9 rows (uniquely best in 5; the sole row where
z-score leads is `fuzzy tree / library default` by +0.002, i.e. noise, and it is
itself an invariance control). Under z-score the 1st-order flat MoG (0.087) is
*worse than raw features* (0.646) — the transform the prose claims would have
destroyed Chapter 4's headline model. Not a ridge artifact (sweeping `l2_reg`
1e-2 → 0 moves the gap by 0.001) and not the scale-dependent BIC membership-count
choice (pinning `n_gaussians` to give both arms an identical rule base still
leaves −0.407/−0.524/−0.634 at n=2/3/4). It underfits on *train* too (MSE 0.030
vs 0.009), consistent with Gaussian memberships and the `[0,1]`-pinned extreme
bucket means assuming a bounded, non-negative domain.

**The prose is mislabelled, not false**, and the relabel is an author decision
that is still open — see `outputs/NORMALIZATION_THREE_ARM.md` §4 for the three
directions and what each costs, and CHECKLIST **A9**. No prose label has been
changed. Nothing in the prose asserts the arithmetic (a sweep for `z-score`,
`zero mean`, `unit variance`, `μ=0`, `σ=1`, "divide by the standard deviation" and
related phrasings returns nothing); §4.3's "target standardized to $[0,1]$" and
the bucket means pinned at 0.0/1.0 are already correct and already say min-max.

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

**Note 5.** Re-quoted at 10 seeds. Tables 4.4/4.5 read flat R² **−0.334 / 0.772 /
0.824** at TSK orders 0 / 1 / 2, with CART 0.825 ± 0.047 and RF 0.909 ± 0.018 —
all four from the run of record `full-14900hx-r2`, and matching the prose. The old
0.797/0.904 pair appears in no archive and came from a 3-seed run predating every
run under `outputs/`. The zeroth-order figure is worth keeping visible rather than
dropping: with constant consequents the model is no better than predicting the
mean, which makes first-order consequents a requirement here rather than a
refinement.

*This note itself was stale and is corrected above.* It previously read
"−0.005 / 0.783 / 0.829, rising to 0.869 at full second order." Those are
`seeds10-2026-08-01` / `full-2026-08-02` values, superseded by `main-d0efefc` and
then by `full-14900hx-r2`, and none of them is what the prose says. A provenance
map quoting numbers a reader cannot find in the prose or in the run of record
inverts its own purpose — it becomes a fourth version of the table. Cross-check
this file's inline figures against the run of record when the run of record moves.

**Note 14 — Table 4.5's two MoG rows are not the same measurement.**

*(Timing note, since this row's clock changed.* The 1st-order cell read
1.04 ± 0.62 s until the generator gained a discarded warm-up fit. That ±60% was
never seed spread: Concrete is the first arm the process fits, so seed 0 carried
import, JIT, BLAS thread-pool spin-up and first-touch allocation at **3.68x** the
mean of the other nine, and dropping it alone moved the spread from ±0.641 s to
±0.021 s. The PhiUSIIL row in the same table was never affected, because it is
fitted second — that asymmetry is what identified it. Post-fix the cell reads
0.84 ± 0.01 s (`outputs/warmup-discarded/`), and **every accuracy in the table is
byte-identical across the change**, which is the check that the warm-up consumes no
shared randomness. A ±60% error bar on the headline cell of a speed claim is worth
one discarded fit.)* The
1st-order row (R² 0.780 ± 0.029, 0.84 ± 0.01 s) comes from
`table_4_1_mog_baselines.py`, which drives the
`MixtureOfGaussiansFuzzyRegressor` estimator. The full-second-order row's
R² 0.859 ± 0.039 comes from **`table_hyperparam_normalization.py`** (Table 4.1's
study), which drives `solve_tsk_consequents` / `predict_tsk` directly and
standardizes the target. Two implementations of the same idea, in adjacent rows of
a table whose caption promises "identical splits."

The gap is measurable, not theoretical. `table_4_1_mog_baselines.py` now carries a
timed full-2nd arm on the estimator path; at ten seeds it scores
**R² 0.840 ± 0.049 in 0.83 ± 0.01 s** (`outputs/table45-full2nd/`) against the
functional path's 0.859 ± 0.039 — a 0.019 difference between two things the
document treats as one number. So the row's `*pending*` training-time cell was
**not** filled with the 0.83 s: pairing that time with the 0.859 above it would be
the same quiet mismatch, one generator over, as the earlier version of this table
that paired a normalized accuracy with a raw-feature time. The cell is marked and
the reason stated; filling it properly means timing the functional pipeline, which
no generator does.

Two related timing observations from the same run, both wall-clock rather than
result, and neither yet acted on:

- **Full second order is not slower than first order** here — 0.83 s against
  1.04 s — and it is far steadier (±0.01 against ±0.62).
- **That ±0.62 is a warm-up artifact, not variance.** The Concrete arm runs first
  in the process and absorbs first-fit costs; the arms that run after it come back
  at ±0.01 to ±0.03 in the same run, and the same pattern holds in every archive
  (Concrete first at ±0.62, PhiUSIIL second at ±0.02). A ±60% deviation on the
  headline cell of a table whose whole point is a speed claim should not be read
  as seed-to-seed spread. The fix is a discarded warm-up fit before timing begins;
  it would change Table 4.5's quoted time, so it is flagged here rather than done
  in passing.

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
| 6.3 Interpretability | *none* | — | **ungenerated** — structural by design; the counts row is checklist C9 / Goal G6 |
| 6.4 Memory augmentation | `AnalyticalDynamics/test_double_pendulum.py`, `AnalyticalDynamics/test_atwood_machine.py` | none | **ungenerated** — entry point now located, but single-seed and outside the harness — note 13 |

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
no weight in the comparison. Its **M5 row is blocked on a dependency, not on an
experiment**: the generator already imports `m5py` optionally and would fill the
row unattended, but `m5py` does not load against scikit-learn 1.9.0 —
`ImportError: cannot import name 'DTYPE' from 'sklearn.tree._classes'` — and
pinning an older scikit-learn to rescue two cells would move every other number
in the chapter. Measured on this host 2026-08-02.

**Note 13 — Table 6.4's entry point is located, and it is still not quotable.**
This row said `tribble-fis/tests/test_double_pendulum.py`, and Chapter 6's
reproduction paragraph said the experiment lived in `tribble-fis`. **Neither is
true** — no such file exists in that submodule. The scripts are in *this*
repository, at `AnalyticalDynamics/test_double_pendulum.py` and
`AnalyticalDynamics/test_atwood_machine.py`; the Atwood one simulates its own
trajectories and reports R² and RMSE for a single-step and a window-of-3
(memory-augmented) model, which is exactly the comparison the table wants.

Finding them does not fill the table. Both run at one fixed `random_state=42`,
neither goes through `reproduce/common.py`, and neither emits a CSV — so what
they print is a one-seed point estimate, the thing the ten-seed floor exists to
keep out of the document. Wiring them into the harness is checklist **C7**, which
also owes a reconciliation the located code makes more urgent: Table 6.4's two
double-pendulum pairs disagree about the target's scale (0.92/0.045 implies
σ ≈ 0.159, 0.96/0.028 implies σ ≈ 0.140), so the headline "38% error reduction"
should be read as an order of magnitude and not as two significant figures until
both rows are re-measured under one protocol.

---

## Chapter 7

Table 7.1 is a goals-and-status matrix, not a measurement. No generator applies.

---

## Appendix A.4 — feature scoring

| Table | Generator | Output | Status |
|---|---|---|---|
| A.1 Feature ranking by scorer | `reproduce/tables/table_a1_feature_scoring.py` | `outputs/table_a1_feature_ranking.{md,csv}` | **reproduced** — all 20 cells identical to `main-d0efefc` |
| A.2 Accuracy and fit time vs features kept | `reproduce/tables/table_a1_feature_scoring.py` | `outputs/table_a2_feature_count.{md,csv}` | **reproduced within a host**; the bhattacharyya accuracies are **not host-portable** — note 12 |

**Note 12 — one arm of A.2 moves between environments, it is the arm the appendix
is least resting on, and the cause is now measured rather than guessed.** Against
`main-d0efefc`, at the same `tribble-fis` commit, the same ten seeds and a
*byte-identical* A.1 ranking, A.2's bhattacharyya accuracies sit higher on this
host from four features on: +0.0174 at 4, +0.0325 at 5, +0.0427 at 7, +0.0402 at
10, +0.0288 at 15, +0.0300 at 20. At one and two features the two runs agree
exactly and at three this host is 0.0002 *lower* — the divergence appears only
once the model has enough features to fit something, so whatever causes it acts on
the fit and not on the ranking or the data. (An earlier version of this note said
"every" accuracy sits higher, and put the control columns' agreement at 0.0002
everywhere. Composite's largest deviation is indeed 0.0001, but **wasserstein's is
0.0017**, at 15 features — eight times the figure quoted. Two orders below
bhattacharyya's 0.0427, so the argument holds and the number did not.)

This is not nondeterminism. Two complete sweeps on this host
(`full-14900hx-2026-08-02` and `full-14900hx-r2`) reproduce **every one of those
accuracies exactly**; only fit times move.

**The standing hypothesis — a BLAS/threading difference — was tested and it does
not hold.** `reproduce/experiments/run_note12_threading.py`, ten seeds, full grid,
all three scorers, one variable at a time; write-up in
`outputs/NOTE12_THREADING.md`, per-setting tables in `outputs/note12-threading/`.

| Variable | Range | Did the knob bite? | Accuracy effect | Verdict |
|---|---|---|---|---|
| Thread count (`OMP`/`OPENBLAS`/`MKL`…) | 1 → 32 | yes — 140% runtime spread | **0.000000** in all 27 cells | **refuted** |
| BLAS kernel family (`OPENBLAS_CORETYPE`) | Haswell → Katmai (SSE-only) | **no** — 1.6% runtime spread | **0.000000** in all 27 cells | **inconclusive** |

Thread count is excluded outright: it moves this generator's wall clock by 2.4×
and moves the reported accuracy by nothing. The kernel-family sweep is reported as
inconclusive rather than as a second refutation because its manipulation check
failed — dropping OpenBLAS from AVX2 to SSE-only changed runtime by 1.6%, so the
variable loaded and then did nothing this workload can feel, and an unchanged
accuracy says nothing either way.

That failed check is the more useful finding, because it undercuts the *framing*
rather than one branch of it: a computation this insensitive to which vector
instruction path executes it is not spending its time in the BLAS, so "a BLAS
difference" is an unlikely explanation for a 0.043 swing in it. What is left is
the part of the stack this workload does use and that does differ — numpy 2.4.6 /
scipy 1.17.1 / scikit-learn 1.9.0 here against an **unrecorded** stack there.
`main-d0efefc/PROVENANCE.txt` has no machine block at all, and `logs/` has no
`table_a1_feature_scoring.log`: those A.2 numbers came from a hand run outside the
orchestrator. The generator itself is byte-identical between that archive's
`grad-school` commit and now, so code, commit, seeds and data are all ruled out.

The next experiment is cheap and no longer needs a second machine: re-run the
generator on this host against pinned older library versions (`uv run --with
'numpy==2.1.*' --with 'scikit-learn==1.5.*'`). If a downgrade reproduces the
archive column, note 12 is solved. One direction stays closed —
`OPENBLAS_CORETYPE=SkylakeX` faults on Raptor Lake rather than falling back, so
the AVX-512 kernels the suspected archive host (i7-1185G7, Tiger Lake) would have
used cannot be tested from here.

**Do not quote A.2's bhattacharyya cells to four decimals across machines** — the
guidance is unchanged. Its positive half is now stronger than before: within one
environment that column survives four thread counts, four BLAS kernel families and
two independent full sweeps, all bit-identical. It is reproducible on a fixed
environment and not portable off it. A.4's actual argument is untouched either way
— it rests on wasserstein 0.9967 against bhattacharyya 0.4267 at a single feature,
a gap of 0.57 against an environment effect of 0.043 and a thread-count effect of
exactly zero.

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
one host with one numeric stack**, and note 12 now sharpens both halves of that:
A.2's sensitive column is invariant to thread count (1 → 32) and to BLAS kernel
family (AVX2 → SSE-only) on this host, so "one host" is a stronger guarantee than
it was, while the across-host difference survives every in-environment knob
measured. Note 12 is the counterexample across hosts,
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
- **Table 3.4** now has a generator and is **drifted**, not ungenerated (note 15).
  This line previously read "Tables 3.2 and 3.3 have no generator; 3.3 needs a
  GPU host" — the same off-by-one numbering documented above, naming two tables
  that both have generators and reproduce.
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
