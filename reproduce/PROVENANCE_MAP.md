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

## 2026-08-22 — whole-document status against latest `main` + latest submodules

A full ten-seed sweep ran on the current pins (`full-2026-08-22`, 16/16 generators
green, zero failures) and `check_prose.py` was run against it and against the
previous run of record. The two numbers together are the finding:

| ± pairs in the prose | vs `goal-8h-2026-08-11-fullsuite` (old pin) | vs `full-2026-08-22` (current pin) |
|---|---:|---:|
| match a cell | **156** | **59** |
| drifted | 10 | 23 |
| untraceable | 44 | 128 |

**The document reproduces against the pin it was written under and does not
reproduce against today's.** The cause is a single upstream function — see
checklist **B14** and [`outputs/WASSERSTEIN_REGRESSION.md`](outputs/WASSERSTEIN_REGRESSION.md).
`stats_numba.wasserstein_distance` drops the $dx$ weighting, making it
dimensionless and scale-invariant, which corrupts the feature-differentiation
screen every `gauss_math`-based table runs through. Correcting that one function
at the current pin returns PhiUSIIL to $0.997 \pm 0.001$ exactly. **Do not
re-quote any Chapter 4 or Chapter 6 accuracy cell from `full-2026-08-22`.**

Unaffected and re-confirmed identical across the same diff: every Chapter 5 table
(they never touch `gauss_math`) and `table_a7_regression_scale`. That is the
control that makes the rest attributable.

**Two rows change status on their own merits, independent of B14:**

- **Table 3.3** moves from *reproduced* to *reproduced, and two cells are new
  results*: the matrix-free rows read 1.000 (exact) at float64 and 0.999 ± 0.002
  at float32, against 0.001 ± 0.001 before. Upstream repaired
  `vat_prim_mst_seq`; the repair is inside the pinned SHA and is *not* inside
  `e3c27e6`, which §3.4's source permalinks still cite. Prose and generator both
  updated; see B6/G4d.
- **Table 3.4** stays **drifted**, and the reason has changed. The CPU FCM fix
  E2c was waiting for has landed (clustering #75/#72), so the device row has moved
  again, as E2c predicted it would. On this host the matched FCM arm now reads
  **2.03× / 3.02× / 2.86×** against the prose's 1.24× / 2.35× / 3.71× and the
  archive's 2.64× / 2.86× / 4.22× — three runs, three answers, every one with a
  spread of the same order as its mean. **The FCM rows are not quotable at
  single-run precision and the fix has not changed that.** What would: `n_iter_`
  and `converged` are now exposed (E2c's second ask), so the generator can control
  for iteration count instead of letting an 11-to-100-iteration spread dominate
  the timing. That is the concrete next step for **E2b**.

⚠️ **Also on this host:** the compiled kernels are built by **gcc/mingw**, not
MSVC, because the host lost its Visual C++ toolchain (checklist **B15**). Chapter 3
timings from `full-2026-08-22` are therefore not comparable to earlier archives —
compiler, compiler flags and library code all moved at once. Exactness columns are
unaffected and reproduce.

---

## Chapter 3 — pVAT

| Table | Generator | Output | Status |
|---|---|---|---|
| 3.1 Reorder time | `reproduce/tables/table_3_1_pvat_scaling.py`, `table_3_1_reorder_three_arm.py` | `outputs/table_3_1.{md,csv}`, `outputs/table_3_1_three_arm.{md,csv}` | **reproduced** for the swept grid; headline row **cited** — note 1; re-taken on one host — note 11 |
| 3.2 Complexity fit | `reproduce/tables/table_3_1_reorder_three_arm.py` | `outputs/table_3_1_complexity_fit.{md,csv}` | **reproduced** — exponents confirm; stage-two plateau does **not** reproduce, note 11 |
| 3.3 Memory footprint | `reproduce/tables/table_3_2_memory_precision.py` | `outputs/table_3_2_memory_precision.{md,csv}` | **reproduced** — all 32 cells identical to `main-d0efefc` |
| 3.4 GPU speedups | `reproduce/tables/table_3_4_gpu_speedups.py` | `outputs/table_3_4_gpu_speedups.{md,csv}` | **drifted** — measured twice on the card the chapter names, hours apart and now inside the sweep; the exactness claim holds, three of the four speedup rows do not read as quoted — notes 15, 18 |
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
exponents are 3.14–3.20 (classical), 1.84–**1.88** (stage one) and **1.93–1.97**
(stage two), across five runs now including `full-2026-08-03` — the last a *cleaner* confirmation of the quadratic claim than the
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

*Row 3 — Fuzzy C-Means, now fixed in the chapter.* Quoted **30–56×** until 2026-08-04; §3.4 now prints the matched arm, 1.24× / 2.35× / 3.71× at `uniform-2026-08-03`, and reports the unmatched pair separately as the reformulation result it is. Measured 14.4×,
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
| 4.2 Output partitioning | `table_g5_output_partitioning.py` | `outputs/table_g5_output_partitioning.{md,csv}` | **reproduced** at three consequent orders; **G5 settled on the 0th-order rows — note 19** |
| 4.3 Partitioning vs skew | `table_g5b_skew_sweep.py` | `outputs/table_g5b_skew_sweep.{md,csv}` | **reproduced** — hypothesis refuted, note 4 |
| 4.4 What MoG achieves | `table_4_1_mog_baselines.py` + `table_hyperparam_normalization.py` | `outputs/table_4_1.{md,csv}` | **reproduced** at 10 seeds |
| 4.5 Baseline comparison | `table_4_1_mog_baselines.py` (+ `table_hyperparam_normalization.py` for the full-2nd row) | `outputs/table_4_1.{md,csv}` | **reproduced**; ANFIS/GA-FIS still absent; the two MoG rows are from two different code paths — note 14 |
| 4.6 Anomaly operating curve | `table_4_4_openset.py` (`REPRO_THETA_SWEEP=0.5,...,1.1`) | `outputs/table_4_4b_theta_sweep.{md,csv}` | **stale** — every cell moved under tribble-fis #72; the band and the operating point are both superseded — note 18 |
| 4.7 Vs dedicated detectors | `table_4_4_openset.py` | `outputs/table_4_4_openset.{md,csv}` | **stale** — three of nine cells moved beyond noise under #72; note 6's instruction not to quote a winner still stands — note 18 |
| 4.11 BETH anomaly detection | `table_4_11_beth_anomaly.py` | `outputs/table_4_11_beth_anomaly.{md,csv}`, `outputs/table_4_11_beth_fa_sweep.{md,csv}` | **reproduced** at 10 seeds (new, grad-school #95); prose slot at §4.4 — note 22 |
| 4.11(c) BETH feature reduction | `table_4_11c_beth_feature_reduction.py` | `outputs/table_4_11c_beth_feature_reduction.{md,csv}` | **reproduced** at 10 seeds (new, #95); prose slot at §4.4 — note 23 |
| 4.11(d) BETH matched sample size | `table_4_11d_beth_sample_scaling.py` | `outputs/table_4_11d_beth_sample_scaling.{md,csv}` | **reproduced** at 10 seeds (new, #95); corrects (c)'s timing; prose slot at §4.4 — note 23 |
| 4.11(e) BETH knob validation | `table_4_11e_beth_boost_sweep.py` | `outputs/table_4_11e_beth_boost_sweep.{md,csv}` | **reproduced** at 10 seeds (new, #95); prose slot at §4.3–§4.4 — note 24 |
| *(no prose table)* | `table_norm_conorm_matrix.py` | `outputs/table_norm_conorm_matrix.{md,csv}` | backs `TNORM_REEVALUATION_RESULTS.md` |

**Note 22 — BETH is a one-class benchmark; the supervised table #95 asked for could
not have been produced, and the library's default anomaly score is unreadable on it.**

*Why there are no supervised arms.* grad-school #95 specified "train on the BETH
training split, report AUC vs. RF / ANFIS." Counting `evil` per shipped split settles
it: **train 763,144 rows / 0 positives, val 188,967 / 0, test 188,967 / 158,432.** Every
positive BETH ships is in the test split. A supervised RF fits the training split
without raising anything and predicts the constant 0 — 16.2% accuracy, AUC 0.5 — so the
failure mode is a *plausible-looking number*, not a crash, which is the class of defect
this map exists for. `table_4_11_beth_anomaly.py` runs the one-class protocol BETH
supports and emits **no supervised rows at all**. **Isolation Forest is the
Random-Forest family's one-class detector** — an unlabelled forest of random trees
scoring by isolation depth — and is the RF-shaped comparison a one-class task admits.
(A literal `RandomForestClassifier` can be pressed into one-class service by the
Shi–Horvath synthetic-contrast construction; that is a different estimator and is
deliberately not implemented.)

*The fuzzy arm is the library's own estimator.* `tribblefis.one_class.TribbleOneClassDetector`
(an sklearn `OutlierMixin`), not a hand-assembly of `create_gaussian_membership_dict` +
`AnomalyParameters` + `simple_gaussian_predict`. The first revision of this generator
did assemble it by hand, bending the *multi-class* path into single-class use. The two
agree exactly on the operating point (det 0.9930 / false alarm 0.1502 either way), so
nothing was wrong — but a second copy of a library capability is how two tables drift
apart, and the hand-rolled version could only emit a hard label, which cost the table a
real ROC-AUC and concealed the next paragraph.

*The default score saturates at eight features, not sixty.* `TribbleOneClassDetector`
offers `score="complement"` (`1 − max firing`, the library default and the formulation
Chapter 4 argues for) and `score="surprisal"` (`Σ −log membership`). Under the product
t-norm these are **monotone transforms of one another, so in exact arithmetic their
ROC-AUC is identical**. Measured on BETH: **complement 0.928, surprisal 0.990**, with
Spearman between the two scores of only **0.812**. The diagnostic is score resolution
against the ceiling the data sets — the test split holds **4,002 distinct feature
vectors**, so 4,002 is the most distinct scores anything can produce; surprisal
recovers **3,997 (99.9%)**, the complement only **1,508 (37.7%)**. The library's
docstring puts the onset of this "past roughly 60 features"; BETH reaches it at
**eight**, because the log-scaled process/thread identifiers are heavy-tailed enough
that a typical point's summed z² already exceeds float64's resolution against 1.0. **So
the saturation threshold is a property of the tails, not of the feature count.** Read
the surprisal row. Both are emitted, because dropping the complement row would hide a
caveat Chapter 4's own default walks into. (Note that most test *rows* are tied at the
maximum under both scores — 75% and 85% — but that is BETH repeating itself, not a
numerical fault. The distinct-score count is the diagnostic; the tied-row fraction is
not, and an earlier draft of this note had that backwards.)

*Two of the ten columns `load_beth()` returns are not features.* `sus` is BETH's second
**label** — the heuristic suspicion annotation — and it is 1 for **158,432 of 158,432**
evil rows, so alone it detects at 1.000/0.427 and any arm given it is scoring the
annotator. `timestamp` is a per-capture session clock whose ranges separate the three
files rather than the behaviour. Both are dropped before any fit, matching
`FuzzySystemsExperiments/beth-anomaly.py`, leaving 8 features. The drop is in the
generator, **not** in `load_beth()`, because `table_4_4_openset.py` shares that loader
and narrowing it would silently move an already-archived table.

*The calibration does not transfer.* The threshold is the (1 − budget) quantile of the
detector's scores on the benign validation split — exact, and it touches no positive
label. At a 1% budget it realizes **0.0100 on validation and 0.1500 on test, 15×**. Both
splits are benign-only draws from one capture, so this is a property of BETH's benign
test rows, not of any detector. A false-alarm budget set on BETH's validation split
cannot be believed on its test split; every arm is matched at the validation rate, so
they stay comparable to each other.

*The 1% default is not the best operating point.* Table 4.11(b) sweeps the budget and
the **tightest setting wins**: J **+0.870 at a 0.1% budget** against +0.843 at 1% and
+0.755 at 10%, with detection flat at 0.993 throughout — all of the movement is in false
alarms. The same sweep is where the saturation bites hardest: at the 0.1% budget the
complement arm collapses to **det = 0.000** while surprisal holds 0.993, which is the
strict-operating-point failure the library docstring predicts and that AUC alone hides.

*Isolation Forest reads as broken at the matched operating point and is not.* It scores
det=0.001 ± 0.003 at fa=0.021 ± 0.005 — worse than chance by Youden's J — while its
score-based **AUC is 0.898 ± 0.005**. Its ranking is fine; what fails is threshold
placement, because `contamination` fixes the cut on the *training* score distribution
and BETH's test benign rows sit elsewhere. Reporting the two columns together is what
makes that legible; either alone would mislead.

*A supervised separability probe exists and is deliberately not in the table.* It has to
train on the only split with positives, and the test split's 188,967 rows hold just
**4,002 distinct feature vectors (2.12%)**, so an i.i.d. row split trains on a copy of
nearly every row it scores (AUC 1.000 — a lookup table succeeding). A GroupShuffleSplit
on feature-vector identity still scores 0.999 AUC because both halves are one capture.
It is printed to stdout only: a cell that is safe to read only alongside its note is a
cell that gets quoted without the note.

*Operational note.* `n_jobs=-1` on this 32-core host hung the machine and killed the
process with SIGSEGV at a nondeterministic seed, with no Python traceback. That was
thread oversubscription — loky spawns one interpreter per core on Windows and each
builds its own BLAS pool, against an OpenBLAS compiled for fewer threads — and **not**
memory: peak resident set was 521 MB of 95.6 GB, and BETH's three frames are 91 MB
combined. Diagnosing it as memory pressure would have led to downsampling the training
split, which is precisely what #95 forbids. The generator caps `n_jobs` at 8
(`REPRO_BETH_N_JOBS`) and BLAS threads at 8 (`REPRO_BLAS_THREADS`, set before the numpy
import) and completes ten seeds in 1m39s.

**Note 23 — a wall-clock is only a comparison when the work is the same, and
Table 4.11(c)'s first version was not.** `table_4_11c_beth_feature_reduction.py` swept
feature count 2→8 and reported a training-time column across four arms. Those arms were
not doing comparable work: the fuzzy detector was fitted on all **763,144** benign rows,
the one-class SVM on a **20,000**-row cap (libsvm is O(n²)–O(n³)), and Isolation Forest
on 763,144 rows nominally but with **`max_samples=256`**, so each tree was built from 256
rows. Three different experiments, one column. Its flat Isolation Forest line was that
default, not a property of the algorithm.

`table_4_11d_beth_sample_scaling.py` is the correction: **one subsample per (n, seed),
handed to all five arms** — same rows, same count — for n = 1,000…20,000. The ceiling is
20,000 because that is where the SVM's cap sat; going higher would drop it out of the
comparison and recreate the defect. Isolation Forest deliberately appears **twice**, at
`max_samples=256` (so (c)'s number stays traceable) and at `max_samples=n` (matched work);
reporting only the first repeats the mistake, reporting only the second silently changes
what "Isolation Forest" means between two tables.

*Two claims the matched sweep retracted.* (a) The fuzzy arm was **not** the slowest to
train — at n=1,000 it fits in 0.081 s against both forests' ~0.15 s, and its 2.63 s in (c)
was entirely the 763k-row fit. (b) Its inference advantage over the SVM exists **only at
the SVM's cap**: SVM scoring grows **11×** (0.117 s → 1.280 s) with support-vector count
while the fuzzy arms and the default forest are flat in n, so at n=1,000 the SVM is
marginally *faster*. Both statements had been published in an artifact before the
correction; the artifact now carries the retraction on its face.

*What the correction also revealed.* `max_samples=256` was costing Isolation Forest a
great deal: at n=20,000 with `max_samples=n` it reaches **AUC 0.978** against 0.896 at the
default, and the **best operating point measured anywhere in this family is that arm at
n=2,000, J +0.879** — better than the fuzzy arm's +0.845 and the SVM's +0.833. Its
apparent uselessness in Tables 4.11 and 4.11(c) was substantially a library default.

*The fuzzy arm needs 1,000 rows.* AUC **0.9903** at n=1,000 against **0.9905** on the full
763,144-row split. The full-split fit buys nothing measurable, which makes #95's
"no downsampling" instruction moot for this arm — worth knowing before budgeting compute,
and an argument for quoting (d) rather than (c) on cost.

*Ten seeds earned their keep again.* At n=1,000 two arms are coin flips, not middling
detectors: matched-work Isolation Forest reports detection **0.563 ± 0.470** and the
complement score **0.794 ± 0.419** — some seeds detect nearly everything, others nearly
nothing. Both settle by n=2,000. A single-seed run would have reported either as a clean
number, which is the failure mode non-negotiable 2 exists to prevent.

*What (c) is still good for.* Its quality columns stand, and they carry the finding that
**AUC and the operating point disagree about feature reduction**: AUC says keep all 8
(0.9905), Youden's J at a 1% budget says keep **6** (+0.914 against +0.843), because
adding `userId` at k=7 lifts detection 0.933→0.993 while pushing false alarms
0.019→0.137. Its within-arm timing trend is also valid. Only the cross-arm timing reading
was wrong, and the emitted table now says so in its own note.

**Note 24 - the boost theta is a threshold parameterisation, and in the one-class
configuration the norm/conorm choice is inert.** `table_4_11e_beth_boost_sweep.py` asks,
for each arm's operating-point knob: at a **matched false-alarm rate**, does turning it
detect more than simply moving the threshold on that arm's own continuous score? On BETH
the answer is no for all three, and the largest absolute delta-detection across each grid
sits at or below that arm's own detection seed-spread:

| knob | kind | largest abs delta | detection seed-spread |
|---|---|---|---|
| Tribble boost theta | threshold on firing (derived below) | 0.0012 | 0.0006 |
| iForest `contamination` | pure threshold - the method's control | 0.0003 | 0.0007 |
| OC-SVM `nu` | **refits per value** | 0.0006 | 0.0000 |

*For theta this is provable, and the table measures the derivation rather than asserting
it.* `gauss_math._anomaly_argmax` forms the anomaly column as
`complement(conorm(clip(class_firing + theta, 0, 1)))`. With **one** known class there is
exactly one class column, and `t_conorm(x, None, ...)` aggregates *column-wise* - so the
conorm is the **identity**, and the anomaly label wins exactly when
**`firing < (1 - theta)/2`**. Two consequences the chapter should absorb:

1. **theta is a hard threshold on firing strength** in the one-class setting and cannot
   express any decision a threshold cannot. Section 4.3 argues the weaker multi-class
   version - that at theta=0.99 the rule degenerates to a max-membership rejector; the
   one-class reduction makes it total at **every** theta, not just the shipped default.
   Describe theta as a threshold parameterisation, not as a mechanism.
2. **`REPRO_ANOM_CONORM` is inert in this configuration.** There is one column to
   aggregate, so the conorm family cannot change a single decision. A conorm sweep run on
   a one-class BETH fit measures nothing - which matters because section 4.3 sweeps conorm
   families and `table_norm_conorm_matrix.py` exists to do exactly that on multi-class
   data.

*The Isolation Forest row is a control, not a finding.* `contamination` provably only sets
`offset_` from a quantile of the training scores and never touches the trees, so its delta
**must** be ~0. That it comes back 0.0003 is what licenses reading a non-zero delta
elsewhere as real - and it is why one fit per seed is correct rather than a shortcut.

*`nu` is the informative negative.* It enters libsvm's QP objective, so every value is a
genuinely different fitted model - the only knob in the table that could have beaten
thresholding by trading one decision surface for another. It does not: the refit moves the
surface and buys nothing over sliding along a fixed one.

*Secondary findings.* theta's J is **monotone** (+0.160 at theta=0 rising to +0.769 at
theta=0.999), so on BETH there is no interior optimum and the shipped 0.99 default is
near-best. **The proposal's usable band of theta = 0.5-0.8 does not transfer** - it came
from Glass and RT-IOT2022. This is consistent with note 22's finding that the tightest
false-alarm budget gave the best J, which is exactly what a threshold-in-disguise would do.
The best operating point anywhere in this table is **iForest at `contamination=0.005`,
J +0.864**; `contamination=0.001` is another coin flip (detection **0.464 +/- 0.481**).

*What this does not say.* None of the knobs is useless - choosing an operating point is
what they are for, and note 22 shows how much that choice is worth. The claim is narrower:
none of them **adds** discriminative power over a threshold on the same score. Every arm is
fitted on the same 20,000-row benign subsample per seed, so note 23's sample-count
confound is not re-introduced, and no wall-clock is reported because this table is about
decision curves.

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
largest gap anywhere in the 18-configuration sweep is 0.022 in R² (3 buckets /
1st order, uniform 0.795 ± 0.035 against quantile 0.773 ± 0.033), against
seed-to-seed deviations of ±0.023–0.043 — so even the widest separation sits
inside one standard deviation of the arms producing it. The `min bucket n`
diagnostic (132/343/75/257/39/171) reproduces exactly, so the bucket-starvation
*mechanism* is intact — Concrete's skew of +0.42 is simply too mild for it to
reach the aggregate error.

*This note said 0.012 and "the 6-bucket/2nd-order pair agrees to three decimals",
and both were wrong against the run of record: the maximum is 0.022 and that pair
reads 0.840 against 0.849. The prose had it right and the map did not, which is
note 5's failure mode in the opposite direction — worth recording, because a reader
who trusts this file over the chapter would have been misled. The retraction itself
is unaffected: an effect below the noise floor is not a crossover whether it is
0.012 or 0.022.*

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
| 5.1 The battery | `run_all.py` → `table_5_1_3_ch5_tables.py` | `outputs/table_5_1_battery.{md,csv}` | **reproduced** — after the note-9 correction |
| 5.2 Multi-scale recovery | `run_all.py` → `table_5_1_3_ch5_tables.py` | `outputs/table_5_2_multiscale.{md,csv}` | **reproduced** |
| 5.3 Selection comparison | `run_all.py` → `table_5_1_3_ch5_tables.py` | `outputs/table_5_3_selection.{md,csv}` | **reproduced** |
| 5.4 Goal G1 scaling decision rule | `table_5_4_ch5_g1_scaling.py` (own computation) | `outputs/table_5_4_ch5_g1_scaling.{md,csv}` + `_raw.csv` | **reproduced** — two-stage vs. flat only; the decision rule's third arm (one-pass) is unimplemented, stated in the table's own note |

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

**The tables were hand-transcribed.** `reproduce/tables/table_5_1_3_ch5_tables.py`
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
| ~~6.4 Memory augmentation~~ | `AnalyticalDynamics/test_double_pendulum.py`, `AnalyticalDynamics/test_atwood_machine.py` | none | **DESCOPED 2026-08-04 — no longer a table in the document.** §6.3.6, Table 6.4, Figure 6.3 and Goal C7 are removed; that work continues outside the proposal. Notes 13 and 17 stay as the record of why it was never quotable |
| §6.3.5 refinement study | `reproduce/optimizers/run_study.py` | `outputs/table_opt_hotstart.{md,csv}`, `…_traces.csv` | **reproduced** — new; supersedes the two-optimizer evidence behind §6.3.5 |

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

**Note 13 — Table 6.4's entry point is located, and it is still not quotable.** *(Retained
as a record; Table 6.4 was descoped from the document on 2026-08-04, so nothing below is
owed to the proposal.)*
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

**Note 17 — Table 6.4 is also blocked on a defect, not just on effort.** *(Retained as a
record of a live upstream bug; the table itself was descoped on 2026-08-04.)*
`MimoGaussianPredictorMemory.predict_trajectory` returns its initial window
unchanged for every `(window_size, memory_size)` pair: it slices exactly
`window_size` rows of history, `prepare_sequences` then computes the last row's
long-term average over an interval that is empty at exactly that row, and the
method's own NaN guard breaks the loop at step zero. Verified at (3,1), (4,2),
(10,4), (2,1). The one-step `predict` path is unaffected. Until the slice is
widened to `window_size + memory_size` there is no iterated rollout to measure,
which is also why Figure 6.3 is a placeholder.

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
| Workstation sweep, 10 seeds | `outputs/full-14900hx-r2/` | The previous run of record, at `tribble-fis d0efefc`. Superseded by `full-2026-08-03` — note 18. |
| Full trace, 10 seeds | `outputs/full-2026-08-03/` | The previous run of record, at `tribble-fis 4b33a0d`. Superseded by `uniform-2026-08-03` — note 19.  <!-- superseded; the original entry follows -->
| &nbsp; | &nbsp; | **Was the citable run.** All 13 table generators green, the GPU table included in the same pass for the first time (1380 s of the 48 min), 19 of 19 drawable figures, the Chapter 5 driver and its opt-in scaling benchmark, and the Chapter 3 cluster experiments. At `tribble-fis 4b33a0d`, which is three functional PRs past the previous run of record — note 18. Read its `PROVENANCE.txt` in full: the archive step was recovered with `--archive-only` and says so. |
| **Full trace, 10 seeds (run of record)** | `outputs/uniform-2026-08-03/` | **The citable run.** All 14 table generators green at ten seeds, at `tribble-fis 6ddb802`. Differs from `full-2026-08-03` by one library default: `partition_output` cuts the target at equal width rather than equal frequency with pinned extremes — note 19. Its `table_g5_output_partitioning` was backfilled to add 0th order, the axis that settled G5. Read its `PROVENANCE.txt` addendum: the header records `tribble-fis 1a83df8`, a squash-merged branch commit, and the run spans two SHAs whose `src/` trees are identical. |
| Preprocessing control | `outputs/splitfirst-2026-08-03/` | Table 6.1's flat arms with the target scale, output partition and feature scaler fit on the training fold only. Bounds the transductive-preprocessing defect at inside-the-seed-spread on every exposed row — `outputs/SPLIT_FIRST_LEAK.md`. |
| Full trace, 10 seeds | `outputs/verify-fis1435811-2026-08-23/` | First full sweep at the current pin (`tribble-fis 1435811`, grad-school `fc64728`), and the first citable ten-seed archive there. All 17 generators green; preflight all-invariants-pass — its `preflight.txt` records the `INSTALL-FRESH` guard passing on a freshly rebuilt wheel, which the first attempt caught stale (note 21). Every table byte-identical to `bumped-0764bc5-2026-08-22` except the timing tables (noise) and RT-IOT2022's `table_4_4b`, which is a **correction**, not a regression — note 21. |

**Note 18 — the 2026-08-03 full trace, and the one result in it that changes a
chapter's claim.** `tribble-fis` moved `d0efefc` → `4b33a0d` between the previous run
of record and this one: PR #67 and #73 (the scaling split and `log_features`, both
byte-identical without `log_features`), #68 (the Ruspini guard), and **#72**, which is
the one that moves numbers — `find_optimal_gaussians` now scores each candidate
component count off the k-means partition it implies instead of fitting four EM
mixtures and discarding them. Full cell-by-cell diff in `outputs/FIX_IMPACT.md`.

*Mostly it is good news.* Chapter 4 and 6 accuracies rose and their spreads tightened:
flat 2nd-refined 0.864 ± 0.046 → **0.877 ± 0.037**, flat 1st-refined 0.836 ± 0.054 →
**0.866 ± 0.029**, flat full-2nd under log+min-max 0.859 ± 0.039 → **0.873 ± 0.020**.
Training time roughly halved — Concrete 1.04 ± 0.62 s → **0.43 ± 0.01 s**, PhiUSIIL
0.64 → **0.28 s** — and Table 4.1 now carries a timed full-2nd row on the estimator
path, which fills note 14's `*pending*` cell. `table_3_2_memory_precision` (32 cells),
`table_a1_feature_ranking` (20) and all three Chapter 5 tables are byte-identical.
Exponents hold: classical 3.20 exactly, stage two 1.97 → 1.95, stage one 1.86 →
**1.88** — which puts stage one just outside the 1.84–1.87 range §3.4 quotes, so that
range needs widening to 1.84–1.88 across five runs.

*The exception is Table 4.6, and it is not cosmetic.* The θ operating curve moved at
every θ: detection rate up 0.084–0.152, false-alarm rate up **0.151–0.221**, and the
net J therefore *down* 0.040–0.103. The band §4.4 quotes as **+0.222…+0.239 peaking
at θ = 0.60** now reads **+0.119…+0.154 peaking at θ = 0.80**. About 35% of the
achievable separation is gone and the operating point has moved two steps along the
sweep. The chapter's qualitative reading survives — there is still no sharp optimum, so
the knob is still forgiving — but every figure in it is superseded, and the honest
summary is now weaker: the complement rule detects less cleanly than the previous run
suggested. Table 4.7's ranking should be re-read on the same basis, and note 6's
standing instruction not to quote a winner from it still applies.

*Table 3.4 is no longer drifted, it is measured.* The GPU table ran inside the sweep
for the first time and reproduces note 15 throughout: the MST ratio is non-monotone
(5.4 / 8.4 / 7.4 / 6.5 across the grid, so "growing with N" remains wrong), the matched
front end grows (2.3 / 3.5 / 3.9 / 5.0) while the unmatched pair reads 5.7–12.2, FCM
against the NumPy-broadcasting arm reads 13.1 / 24.7 / 38.9 against **1.2 / 2.3 / 3.8**
for its own formulation on the CPU — so the order of magnitude between "device" and
"rewrite" reproduces — and the pairwise-distance loss is intact, with the GPU losing at
d = 10 and d = 50 in both precisions. The N = 48,000 float32 demonstration reads 3.7×
at ordering agreement **0.99992**, with the two Prim totals identical to every printed
digit (42023.180315, relative difference 0.0e+00). That is the tie-break case §3.2 now
states as an assumption rather than a hypothetical.

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

**Note 19 — the output partition, and the two chapter claims that were artifacts
of it.** `partition_output` cut the target with `pd.qcut` and then overwrote the two
extreme bucket centroids with the observed min and max. `uniform-2026-08-03` is the
same sweep with equal-width cuts and per-bucket centroids, one library default
different (`tribble-fis` #81), and it moves two claims that three prior studies had
looked straight at.

**Why three studies missed it.** `table_g5_output_partitioning` ran 126 cells over
three schemes and six configurations — at 1st and 2nd consequent order only. There
the three arms span 0.009 in R² against seed spreads of ±0.018 to ±0.027, so G5 was
left open with no scheme recommended, correctly given what was measured.
`solve_tsk_consequents` holds the first and last rules' *constant* terms at the
centroids it is handed, as an exact equality constraint. At 0th order that constant
is a rule's entire output, so the same three arms span **0.828**: uniform
0.394 ± 0.065, pure quantile 0.242 ± 0.070, pinned quantile −0.434 ± 0.241, ordering
preserved at three, four and six buckets. Decomposed, the boundary scheme is worth
0.152 and the pinning 0.676. The generator now runs 0th order by default.

**Claim 1, Chapter 4's consequent-order ladder.** §4.3 read −0.434 at 0th order as
evidence that first-order consequents were a requirement rather than a refinement.
They are not: the flat arm is 0.394 ± 0.065 under equal-width cuts, and the ladder
0.394 / 0.796 / 0.841 / 0.861 has its knee between first and second order. Chapter 1
and Chapter 2 both carried the negative figure in their opening arguments and are
re-quoted.

**Claim 2, Chapter 4's boundedness argument.** §4.1 reported that real z-scoring
collapsed the first-order model to 0.014 ± 0.195, below its raw-feature score, and
built the case that a bounded input domain was load-bearing. Under equal-width cuts
that cell is 0.713 ± 0.035 and Δ z-score − raw flips from −0.651 to +0.018. The
interaction was the two pinned extreme rules against features z-scoring leaves
unbounded. What survives is a preference worth 0.083 at first order, plus a real
variance cost at full second order (±0.115 against ±0.026). The `n_gaussians`-pinned
figures −0.407/−0.524/−0.634 quoted from the old arm are withdrawn.

**What did not move.** `table_g5_output_partitioning` and `table_g5b_skew_sweep` are
identical across all 174 cells, since both implement their own arms and are the
evidence for the change rather than consumers of the default. The four rank-based
control rows of `table_concrete_reconciliation` move by exactly 0.000, which is the
check that the switch touched only what it claimed to. Chapter 3's movement is
timing variance plus two fitted exponents (classical 3.20 → 3.15, stage one
1.88 → 1.86). Chapter 5 is untouched.

**The lesson is not "more seeds".** This survived ten seeds, three schemes, six
configurations and four archives. It was a regime never entered, and no amount of
repetition inside the wrong regime finds that. What found it was reading the solved
coefficients instead of the scores.

**Note 20 — why the Table 3.4 diagnosis sat here for four archives without
reaching the chapter.** Note 11 above identified all three defective rows,
including the FCM row's order-of-magnitude formulation confound, and the chapter
went on printing **30–56×** anyway. The gap is mechanical, not editorial:
`check_prose` compares `mean ± std` pairs against archive CSVs, and every
defective cell in Table 3.4 was written as a *range* — "30–56×",
"≈4.8–6.6×", "1.3–2.5×", "≈5× ... growing with N". None of them is a
mean-and-spread pair, so the archive checker never looked at any of them, and the
one check that would have caught a four-archive-stale headline number ran clean the
whole time.

Two consequences worth acting on. Ranges are now avoided in that table in favour of
per-condition values, which the checker can see. And a note in `PROVENANCE_MAP.md`
is not a correction: it records that someone knew, which is worse than not knowing
if the chapter is what gets read. Findings belong in the prose or in a tracked
checklist item, not only here.

**Note 21 — the RT-IOT2022 operating curve now exists at ten seeds, and it
corrects an inconsistency in the prior archive rather than regressing.** C17
(grad-school #139) hoisted the θ-independent work out of `theta_sweep` — the
screen, the membership dict and the class-rule firing are all θ-independent, so
the sweep now builds the model once per (held-out class, seed) and varies θ over
the anomaly step alone via `simple_gaussian_predict_sweep` (tribble-fis #176).
`table_4_4_openset` fell from **3h38m to 82 min** on the workstation of record, so
the ten-seed RT-IOT2022 curve that §4.4 calls "roughly a day of compute" and
skips is now part of a routine full sweep.

*The restructure is bit-identical, proven at scale.* `complement_rule_sweep`
matched per-θ `complement_rule` on **0 of 182,867 rows per θ**, across all 7 θ and
3 held-out classes, at the current pin — the check C17's own verification ran only
at 2 classes × 2 seeds. So the θ-sweep optimisation itself moves nothing.

*The prior RT-IOT2022 `table_4_4b` was internally inconsistent with its own
headline.* In `bumped-0764bc5-2026-08-22`, the operating curve read detection
**0.777** at θ = 0.99 while Table 4.7b's complement-rule row read **0.798** over
the same leave-one-class-out folds; false-alarm agreed (0.431) but detection did
not. The new sweep makes them agree — both 0.798 — so the 17-cell "beyond noise"
move `compare_runs` reports for `table_4_4b` against `bumped-0764bc5` is that
correction. Full cell diff in this archive's `FIX_IMPACT.md`.

*The finding §4.4 was missing.* Tuning θ **narrows but does not close** the gap to
the baselines on RT-IOT2022: J peaks at **θ = 0.80 (+0.391**, detection 0.901,
false alarm 0.510), still below Isolation Forest's **+0.534** at θ = 0.99. That
answers the open question §4.4 and Figure 4.2 both flag as the missing RT-IOT2022
row. Curve in `table_4_4b_theta_sweep.{md,csv}`; headline in `table_4_4_openset.{md,csv}`.

*Caveat — these are current-pin measurements, not a re-quote.* Per **D8**,
`_kmeans_labels_1d` swapped k-means++ for a uniform-random start (tribble-fis
#95), and it reaches the complement rule through
`create_gaussian_membership_dict → fit_gaussians → fit_gaussian_mixture_1d`. So
these digits measure the current code, not the code the prose's Table 4.7b quotes;
**do not overwrite Table 4.7b from this archive until D8 is settled.** The
qualitative reading (the complement rule loses to isolation forest and no swept θ
closes the gap) is robust to any plausible D8 shift — the deficit is 0.14–0.17
against an init effect that moved the five-seed headline by ~0.03 — but the exact
values are not. C4/D8 in `research/proposal-defense/CHECKLIST.md` carry the
decision.

---

**Note 25 — two loaders were passing label-derived columns as features, and one
of them was carrying a whole result.** *(2026-08-27, grad-school#92 follow-up.)*

Both were the same mistake in two shapes: a column that is *about* the answer got
into `X` because it happened to be numeric.

*RT-IOT2022 — the index column.* `load_rt_iot2022` sliced `df.iloc[:, :-1]` and
kept every numeric column, so the CSV's unnamed leading column survived as an
82nd feature. It is **not a row number.** The file concatenates the twelve
per-class captures and the counter restarts at zero for each, so any value above
8,107 identifies `DOS_SYN_Hping` and nothing else — a decision tree on that column
alone scores 0.706. What saves Table 4.4 is unrelated luck: that row uses
`mog_classifier`'s default `top_n = 5` antecedent screen, and the index column
never ranked in the top five, so the same five features are chosen either way.
Re-running the generator both ways on the host of record, ten seeds, same day:

| | with index | without | |
|---|---|---|---|
| MoG accuracy | 0.927 ± 0.002 | 0.927 ± 0.002 | unchanged |
| MoG train time | 4.04 ± 0.67 s | 3.68 ± 0.06 s | overlapping |
| RF reference | 0.999 ± 0.000 | 0.998 ± 0.000 | −0.001 |

**Table 4.7b is the row genuinely at risk, and it is NOT re-quoted here.** It runs
the screen over *all* features and keeps all of them, so it consumed the leaky
column directly — and Note 21 / `OPENSET_COST_2026-08-22.md` establish that its
result is sensitive to the screen's **order**, which removing a column perturbs.
Its published +0.394 vs +0.537 therefore still measures leaky data. The re-run is
~82 min at the current pin (down from 3h38m, per Note 21) and is owed.

*Bike Sharing — the target's own addends.* `load_bikeshare` already dropped
`instant` for exactly the reason above, which is why the RT-IOT2022 miss is a
lesson learned once and not generalised. But it left `casual` and `registered` in
`X`, and those two **sum exactly to the target `cnt` on all 17,379 rows.** The
model was being handed the answer in two pieces. That is why the Random Forest
reference on this row read a perfect **1.000**, which should have been the tell.
Re-measured with both dropped, ten seeds:

| | leaked | corrected |
|---|---|---|
| MoG R² | 0.962 ± 0.002 | **0.620 ± 0.014** |
| RF reference | 1.000 ± 0.000 | **0.944 ± 0.004** |

The old numbers are **superseded, not revised**: they measure a model's ability to
add two columns, not to predict demand. DATASETS.md's "demonstrating fuzzy
regression scaling on real urban dynamics" was resting on that.

*A separate drift this exposed, which is nobody's leak.* The prose quoted MoG
train time on RT-IOT2022 as **37.42 ± 0.64 s**. Re-running the *unfixed* loader
today gives **4.04 ± 0.67 s** on the same host — so the ~10× is pre-existing drift
between the prose and the current code/pins, not an effect of this change. Table
4.4's timing is re-quoted from the corrected run of record (**4.24 ± 0.68 s**,
RF **0.998 ± 0.000**); *which* pin bump bought the 10× is not yet identified, and
until it is, this note is the only account of it.

*Guarded, so it cannot be a third dataset's turn.* `reproduce/test_dataset_loaders.py`
now fails if any loader returns an index-like column, and cross-checks every
loader's modelled width against `dataset_specs.yaml`, applying that file's own
`drop_columns` so a loader may still hand back columns its generator drops (BETH
returns `sus`/`timestamp` for `table_4_11` to remove).

*Note 25, addendum (2026-08-27) — the RT-IOT2022 index column costs Table 4.7b
0.019 J, measured rather than assumed.* Two single-seed `table_4_4_openset` runs,
`REPRO_SEEDS=0`, same host, same day, differing only in whether the loader drops
the unnamed column:

| Method | with index (82 feat.) | without (81 feat.) | Δ J |
|---|---|---|---|
| Complement rule | +0.391 | +0.372 | **−0.019** |
| One-class SVM | +0.444 | +0.437 | −0.007 |
| Isolation Forest | +0.517 | +0.517 | 0.000 |

The matched control matters here: neither the prose's five-seed +0.394/+0.537 nor
the three-seed +0.333/+0.513 sitting in `reproduce/outputs/` is a valid baseline
for a one-seed probe, and comparing against either would have charged seed
variance to the fix. Against the control, the deficit to Isolation Forest *widens*
from 0.126 to 0.145 — close to the 0.143 the prose already reports — so the
qualitative reading is unchanged and the full re-quote is deferred to #184 rather
than rushed. One seed is one seed: this bounds the effect, it does not replace the
five-seed table.

---

**Note 26 — D8's Table 4.7b blocker is already cleared by the pin the repository
is on, and nobody noticed.** *(2026-08-27.)*

Note 21 and CHECKLIST **D8** both say, in terms, *do not overwrite Table 4.7b
until D8 is settled*: `_kmeans_labels_1d` had swapped sklearn's k-means++ for a
single uniform-random start (tribble-fis #95), and it reaches the complement rule
through `create_gaussian_membership_dict → fit_gaussians → fit_gaussian_mixture_1d`.
Both of the causes that blocked a re-quote have since landed upstream, and **both
are ancestors of the currently pinned `353162c`**:

| cause | fixed by | in the pin? |
|---|---|---|
| B14 — `wasserstein_distance` ignored the CDF gap width | `5253aa0` (#171) | yes |
| D8 — `_kmeans_labels_1d` single-start init | `353162c` (#191) | **it *is* the pin** |

tribble-fis #191 ("swap safe candidates, keep hot-path ones") put
`sklearn.cluster.KMeans` back with `n_init="auto"`, restoring k-means++, on
2026-08-25. Verified in the running environment rather than from the diff: the
`tribblefis` the harness imports resolves to the submodule source tree, and
`inspect.getsource(gauss_math._kmeans_labels_1d)` at that pin contains `KMeans(`
and not `kmeans_1d(`.

**This is the third instance of one failure mode**, and it is worth naming as
such rather than fixing quietly a third time. §7.3 already records it for BETH:
`TribbleOneClassDetector` arrived upstream and retired a stated blocker while the
proposal went on recording it as open. `check_prose.py` watches for *numbers*
drifting between prose and harness; it cannot see a **capability or a fix**
arriving in a submodule and silently retiring a recorded blocker. Note 22's
lesson — a pin bump should ask which recorded blockers the new pin removes — was
written and then not applied to D8, which the very next pin bump had settled.

Consequence: the ten-seed RT-IOT2022 re-run of `table_4_4_openset` at this pin is
no longer a "measurement of the current code" to be reported beside the prose. It
is a legitimate run of record for Table 4.7b, and the D8 hold on re-quoting is
lifted. What D8 still carries is unrelated to this table: the Chapter 3 compiler
question (**B15**) and §4.3.2/G5 absorbing the `pin_extremes` default flip
(**#102**).

---

**Note 27 — Table 4.7b re-quoted at ten seeds on the de-leaked loader, and the
favourable ten-seed result did not survive.** *(2026-08-27, closes the
substantive half of grad-school#184.)*

`table_4_4_openset.py`, RT-IOT2022, leave-one-class-out, ten seeds, θ = 0.99,
94 minutes on the host of record, at pin `353162c` with `load_rt_iot2022` no
longer passing the CSV's unnamed index column as a feature.

| method | archive, 5 seeds | 2026-08-22, 10 seeds | **now, 10 seeds, de-leaked** |
|---|---:|---:|---:|
| Complement rule | +0.394 | +0.515 | **+0.366** |
| One-class SVM | +0.408 | +0.271 | **+0.410** |
| Isolation Forest | +0.537 | +0.579 | **+0.535** |

Detection / false-alarm for the new column: complement 0.804 ± 0.270 / 0.438 ±
0.085; one-class SVM 0.845 ± 0.225 / 0.435 ± 0.061; Isolation Forest 0.966 ±
0.145 / 0.431 ± 0.063.

**The result that disappeared was the flattering one.** CHECKLIST **C4** carried
the 2026-08-22 column as "the margin narrows sharply" — 0.064 to Isolation Forest,
and the complement rule *overtaking* the one-class SVM. Neither survives. The
margin is **0.169**, wider than the five-seed archive's 0.143, and the complement
rule trails the SVM again. Worth stating plainly because the direction of the
correction is unusual: the leak and the stale pin were together flattering the
construction, and removing them costs it the one open-set result that read as a
win.

**What is and is not attributed.** Two changes separate the 2026-08-22 column
from this one — the leaky feature is gone, and the pin restored k-means++ in
`_kmeans_labels_1d` (#191, note 26). A matched single-seed control (same seed,
host and day, differing only in the column) puts the leak's own cost at **0.019
$J$**, so most of the −0.149 is most plausibly the init restoration. That is an
*inference from one seed against a ±0.27 spread, not a measurement*; separating
them properly needs a matched ten-seed run at the old pin, which is not planned.
The document does not depend on the split: this column is measured on correct
data at a settled pin, which is the only claim Table 4.7b makes.

**Unchanged by the re-run:** the spreads (±0.15–±0.27) still mean no separation
in the table clears its own error bar, exactly as §4.4 already says of Table 4.7,
and §4.4's reading — the free, no-second-model property survives at scale and
accuracy parity does not — is strengthened rather than altered.

**Still owed (#184):** the RT-IOT2022 θ-sweep behind Fig 4.2's missing row was not
regenerated in this pass (`REPRO_THETA_SWEEP` was not set), so §4.4's sentence
that Table 4.6's sweep "was not run here" still stands, and note 21's ten-seed
sweep figures remain pre-de-leak.

---

**Note 28 — the RT-IOT2022 θ-sweep is now run on the de-leaked loader, and it
closes #184's open question: tuning θ does not rescue the complement rule.**
*(2026-08-27, run of record `outputs/rtiot-deleaked-2026-08-27/`.)*

Note 27 left this owed: the ten-seed re-quote of Table 4.7b did not regenerate
the θ-sweep, so note 21's sweep figures (peak **+0.391** at θ = 0.80, Isolation
Forest **+0.534**) were still pre-de-leak. They are now superseded. The sweep and
the Table 4.7b headline were produced in **one** `table_4_4_openset.py`
invocation (`REPRO_THETA_SWEEP=0.5,0.6,0.7,0.8,0.9,0.99,1.1`, 2h09m), so the
θ = 0.99 sweep row and the headline complement-rule cell agree by construction —
the internal inconsistency note 21 recorded for the previous `table_4_4b` (sweep
detection 0.777 vs headline 0.798) cannot recur.

| θ | detection | false alarm | J | | de-leaked vs note 21 |
|---|---|---|---|---|---|
| 0.80 (peak) | 0.914 | 0.518 | **+0.396** | | +0.391 → +0.396 |
| 0.99 (shipped) | 0.804 | 0.438 | +0.366 | | — |
| Isolation Forest | 0.966 | 0.431 | **+0.535** | | +0.534 → +0.535 |

The correction moved the digits by thousandths and left every conclusion intact.
The peak is at θ = 0.80 both before and after — the same operating point the
Glass sweep (Table 4.6) finds — so the knob's best setting transfers across
datasets. What does not transfer is the verdict: the best tuned J is **0.139
below Isolation Forest**, which leads at every θ. So the gap Table 4.7b reports is
**fundamental to this dataset, not an artefact of the shipped operating point** —
which is the question §4.4 and Fig 4.2's missing row were both waiting on.
Written into prose as **Table 4.7c** (§4.4). This closes the substantive half of
grad-school#184; the only remainder is whether to redraw Fig 4.2 with a second
(RT-IOT2022) panel, a presentation choice, not a measurement gap.
