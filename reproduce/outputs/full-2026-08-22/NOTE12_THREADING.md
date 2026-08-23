# Note 12, measured: what moves Table A.2's bhattacharyya column

Host `NEX-210200`, i9-14900HX (32 logical cores), Windows 11, Python 3.13.7,
numpy 2.4.6 / scipy 1.17.1 / scikit-learn 1.9.0, OpenBLAS `scipy-openblas`
0.3.31.188.0 (numpy's copy) and 0.3.30 (scipy's copy), `tribble-fis` at
`d0efefc`. Measured 2026-08-02. Runner:
`reproduce/experiments/run_note12_threading.py`. Per-setting tables under
`reproduce/outputs/note12-threading/`.

Everything below is at the **full ten seeds** (`0..9`), the full `K_GRID`
(1,2,3,4,5,7,10,15,20) and all three scorers. Nothing was reduced.

---

## What note 12 claimed

Table A.2 comes from `reproduce/tables/table_a1_feature_scoring.py`. Against the
archive `reproduce/outputs/main-d0efefc/`, at the same `tribble-fis` commit and
the same ten seeds, its **bhattacharyya** accuracies sit up to +0.043 higher on
this host, while **wasserstein** and **composite** were said to agree to within
0.0002. Two full sweeps here reproduce every bhattacharyya accuracy exactly, so it
is not run-to-run nondeterminism.

The standing explanation was "a BLAS or threading difference," on the reasoning
that bhattacharyya is the ill-conditioned arm — its own ranking scores 0.4267
accuracy at one feature, so its models are fitted on poor features and sit where
small numerical differences change an outcome. Plausible, and never tested.

## Design, registered before running

Hold code, commit, dataset, sample size, seed list and scorer list fixed. Vary
exactly one environment variable per run, each run in its own subprocess (these
variables are read when the BLAS loads and cannot be changed inside a live
interpreter) writing to its own output directory.

Two axes:

1. **Thread count** — `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
   `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` all set
   together, over {1, 2, 8, 32}.
2. **BLAS kernel family** — `OPENBLAS_CORETYPE` over
   {Haswell, Sandybridge, Nehalem, Prescott}. The loaded OpenBLAS is a
   DYNAMIC_ARCH build that picks a family from CPUID; `threadpool_info()` reports
   `architecture: 'Haswell'` here. Thread count only changes how the same
   arithmetic is *scheduled*; kernel family changes vector width and accumulation
   order, i.e. the arithmetic itself.

Read the **accuracy** halves of the cells. Fit seconds are expected to move and
mean nothing as a result.

**Confirms** — bhattacharyya accuracies move by an amount on the order of the
archive gap (any cell ≥ 0.001) while wasserstein and composite stay put.
**Refutes** — all three columns identical, or under 0.001, at every setting. Then
that variable cannot produce a 0.043 shift on this host and is removed as an
explanation; it does not become more likely by elimination.
**A different story** — wasserstein and composite move too. They are the internal
control; if the control moves, "the harness is deterministic on one host with one
numeric stack" needs qualifying, because neither of these variables is currently
pinned and both are part of a host's configuration.

Two controls beyond the two scorer columns:

- **A.1's ranking** is the *input* control. It is computed once per run from the
  same data and must be byte-identical at every setting; if it moves, the A.2
  difference is downstream of a changed input and the framing changes.
- **A manipulation check.** Total A.2 fit seconds per run must respond to the
  variable. An invariance result is worthless if the knob never bit, and "the
  environment variable was ignored" is indistinguishable from "this variable does
  not matter." This project has been caught by that shape before —
  `REPRO_THETA_SWEEP=1` is a valid list of one that emits a table of zeros
  reading exactly like a null result.

---

## Result 1 — thread count: REFUTED

All 27 accuracy cells (3 scorers × 9 feature counts) are **identical to four
decimals at all four thread counts**. Max |Δ| across the sweep is exactly
`0.000000` in every column.

| scorer | max abs Δ accuracy, T ∈ {1,2,8,32} | verdict |
|---|---:|---|
| wasserstein | 0.000000 | identical |
| bhattacharyya | 0.000000 | identical |
| composite | 0.000000 | identical |

The manipulation check says the variable really was varied — total A.2 fit
seconds, summed over scorers and feature counts, mean per seed:

| threads | total fit s | vs T=1 |
|---:|---:|---:|
| 1 | 17.49 | 1.00× |
| 2 | 17.88 | 1.02× |
| 8 | 19.22 | 1.10× |
| 32 | 42.02 | **2.40×** |

A.1's ranking was byte-identical at every thread count, and identical to the
run-of-record archive `full-14900hx-r2`.

And the gap to `main-d0efefc` did not budge at any setting:

| threads | max abs Δ vs archive: wasserstein | bhattacharyya | composite |
|---:|---:|---:|---:|
| 1 | 0.0017 | 0.0427 | 0.0001 |
| 2 | 0.0017 | 0.0427 | 0.0001 |
| 8 | 0.0017 | 0.0427 | 0.0001 |
| 32 | 0.0017 | 0.0427 | 0.0001 |

**Conclusion: the threading half of note 12's hypothesis is refuted on this
host.** Thread count moves wall-clock by 2.4× and moves the reported accuracy by
nothing at all. It cannot account for +0.0427.

A side finding worth keeping: **32 threads is 2.4× slower than 1** for this
generator. These are small per-class Gaussian fits, and the default (24 threads
here) is spending more on synchronization than it recovers. The run-of-record
archive's A.2 fit times are ~2× the T=1 times measured here, consistent with it
having run at the default. That is a wall-clock observation only — the harness's
timing cells already carry the caution that absolute seconds are host-dependent —
but anyone treating A.2's fit-time halves as a cost model should know a thread cap
halves them without changing a single accuracy.

## Result 2 — BLAS kernel family: INCONCLUSIVE, and informative anyway

`OPENBLAS_CORETYPE` verifiably takes effect on this host: requesting Haswell,
Sandybridge, Nehalem and Prescott makes `threadpool_info()` report `Haswell`,
`Sandybridge`, `Nehalem` and `Katmai` respectively, for both loaded OpenBLAS
copies. `Katmai` is SSE-only — about as far from the native AVX2 Haswell kernels
as this build can be pushed.

All 27 accuracy cells are again **identical to four decimals at all four kernel
families**, max |Δ| exactly `0.000000` in every column, A.1's ranking
byte-identical throughout, and the gap to `main-d0efefc` unchanged at 0.0427.

**But the manipulation check fails, so this is not a refutation.**

| OPENBLAS_CORETYPE | reported arch | total fit s |
|---|---|---:|
| Haswell (native) | Haswell | 31.93 |
| Sandybridge | Sandybridge | 32.18 |
| Nehalem | Nehalem | 31.86 |
| Prescott | Katmai (SSE only) | 31.68 |

Spread: **1.6%**, against 140% on the thread axis. Dropping OpenBLAS from AVX2 to
SSE-only did not measurably change how long this generator takes. The variable
loaded — `threadpool_info()` proves that — and then made no difference to the
work, so "accuracy did not change either" is uninformative about kernel families.
The runner now says `INCONCLUSIVE` here rather than `REFUTED`, on a 5%
manipulation floor, and would have printed a clean false negative without that
gate.

**The failed check is the more useful result.** If swapping the entire vector
instruction path changes runtime by 1.6%, this workload spends almost no time in
BLAS kernels at all. It is many small per-class, per-feature Gaussian fits, not
large matrix products. That undercuts the *framing* of note 12, not just its
threading half: "a BLAS difference" is an unlikely explanation for a 0.043 swing
in a computation that is demonstrably insensitive to which BLAS is doing the
arithmetic.

What is left is the part of the stack that this workload *does* spend its time
in and that genuinely differs between environments: numpy's own reduction and
`einsum` paths, scipy's optimizers, and scikit-learn's mixture initialization and
`train_test_split` — all version-dependent, none BLAS-dependent. Here that is numpy 2.4.6, scipy
1.17.1, scikit-learn 1.9.0; in `main-d0efefc` it is unrecorded.

---

## Corrections to note 12 as written

Two of the note's own figures do not survive checking against the CSVs.

1. **"Wasserstein and composite agree to within 0.0002 everywhere" is wrong for
   wasserstein.** Composite's largest deviation from the archive is 0.0001, but
   wasserstein's is **0.0017**, at 15 features (archive 0.9957, here 0.9974) —
   eight times the stated bound. Still two orders below bhattacharyya's 0.0427, so
   the note's *argument* holds; the number quoted for the control does not.

2. **"Every bhattacharyya accuracy sits higher" overstates it.** At 1 and 2
   features the two runs agree exactly (0.4267, 0.4527) and at 3 features this
   host is 0.0002 *lower* (0.8455 vs 0.8457). The divergence appears only from 4
   features on:

   | features kept | archive | here | Δ |
   |---:|---:|---:|---:|
   | 1 | 0.4267 | 0.4267 | 0.0000 |
   | 2 | 0.4527 | 0.4527 | 0.0000 |
   | 3 | 0.8457 | 0.8455 | −0.0002 |
   | 4 | 0.8986 | 0.9160 | +0.0174 |
   | 5 | 0.9131 | 0.9456 | +0.0325 |
   | 7 | 0.9183 | 0.9610 | **+0.0427** |
   | 10 | 0.9274 | 0.9676 | +0.0402 |
   | 15 | 0.9477 | 0.9765 | +0.0288 |
   | 20 | 0.9477 | 0.9777 | +0.0300 |

   That shape is informative and the note flattened it. The two runs agree
   exactly where the model is degenerate (one or two poor features) and diverge
   once it has enough features to fit something, peaking at 7. Whatever the cause
   is, it acts on the *fit*, not on the ranking or the data.

## What the archive does and does not pin down

Everything that could be checked from the repository has been, and it all matches:

- `tribble-fis` is at `d0efefc` in both cases — the archive is named for that SHA
  and the submodule is still there.
- `reproduce/tables/table_a1_feature_scoring.py` is **byte-identical** between the
  archive's `grad-school` commit (`e08382d`) and now (`git diff` is empty). The
  only change to `_fuzzy_models.py` in that range is the *addition* of
  `normalize()`, which this generator does not call.
- Both runs record `seeds = [0..9]`.
- A.1's ranking is byte-identical, which pins the sample and the sample size.

So the difference is not code, not commit, not seeds and not data. What the
archive does **not** pin down is the environment, and it pins down less than note
12 implies:

- `main-d0efefc/PROVENANCE.txt` has **no machine block at all** — no host, no CPU,
  no core count, no numeric stack. It predates `common.machine()`, so its A.2
  Markdown carries no machine footer either.
- `main-d0efefc/logs/` has **no `table_a1_feature_scoring.log`**, and that
  generator is absent from the archive's status list. Those A.2 numbers were
  produced by a hand run outside the orchestrator. The seed list is at least
  recorded in the Markdown footer, so that much is pinned; nothing else about the
  invocation is.

Chapter 3 says the pre-workstation numbers came from an i7-1185G7 with 16 GB under
a `powersave` governor. If that is the host, it is the interesting fact in this
whole note: **Tiger Lake has AVX-512 and Raptor Lake does not**, so OpenBLAS would
have selected `SkylakeX` there and `Haswell` here — a different kernel family, not
a different thread count. That is the hypothesis result 2 tests in the only
direction this CPU can be asked to test it.

## What would settle it, and what cannot be settled here

The clean experiment, and the one the harness now makes possible and could not
before: **run A.2 in a second environment with its stack recorded**, and diff.
Two versions of it, in order of cost:

1. **Same host, older libraries.** `uv run --with 'numpy==2.1.*' --with
   'scikit-learn==1.5.*'` (or whatever the suspected archive vintage was) against
   the same generator, same seeds. This is cheap, needs no second machine, and
   after result 2 it is the leading candidate rather than a fallback. If a library
   downgrade reproduces the archive's bhattacharyya column, note 12 is solved.
2. **Second host, stack recorded.** What `PROVENANCE.txt` now exists to enable.
   Necessary if (1) comes back invariant, because then something about the CPU or
   OS is involved that no in-environment knob reaches.

One direction is closed on this host and should not be presented as open:
`OPENBLAS_CORETYPE=SkylakeX` cannot be tested from Raptor Lake. Naming a kernel
family the CPU cannot execute faults rather than falling back, so the AVX-512
direction — the one that matches the suspected archive host — is unmeasurable
here. Given result 2 that matters less than it looked: a workload this insensitive
to SSE-vs-AVX2 is unlikely to care about AVX-512 either.

## Summary

| Variable | Range | Manipulation check | Accuracy effect | Verdict |
|---|---|---|---|---|
| Thread count | 1 → 32 | **PASS** (140% runtime spread) | 0.000000 in all 27 cells | **REFUTED** |
| BLAS kernel family | Haswell → Katmai (SSE) | **FAIL** (1.6% spread) | 0.000000 in all 27 cells | **INCONCLUSIVE** |
| Library versions | not varied | — | — | **untested; now the leading candidate** |

Neither measured variable explains note 12's +0.0427. Thread count is excluded
outright. Kernel family is untested in effect, and the reason it is untested —
this workload barely touches BLAS — argues against the whole BLAS framing.

## Practical guidance (unchanged, and now for a stated reason)

**Do not quote A.2's bhattacharyya cells to four decimals across machines.** That
guidance stands, unchanged and for a better reason. It is no longer "probably
threading"; it is "the cause is somewhere in the numeric environment, thread count
has been excluded by measurement, the BLAS framing is doubtful because the
workload does not use the BLAS much, and the archive that disagrees recorded
nothing about its environment at all."

The corollary is the useful one: **A.2's bhattacharyya column is reproducible on a
fixed environment and is not portable off it.** Within this host it survives four
thread counts, four BLAS kernel families and two independent full sweeps, all
bit-identical. That is a stronger determinism claim than the harness had before,
and it is bounded in exactly the right place.

Appendix A.4's actual argument is untouched. It rests on wasserstein 0.9967
against bhattacharyya 0.4267 at a single feature — a gap of 0.57 against a host
effect of 0.043, and against a thread-count effect of exactly zero.
