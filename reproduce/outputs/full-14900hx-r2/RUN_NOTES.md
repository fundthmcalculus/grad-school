# Workstation sweep — the run of record for the proposal defense

`full-14900hx-r2`, 2026-08-02. Ten seeds, thirteen generators, fourteen tables,
one host, one pass. This is the single-machine re-take that checklist **B5** asks
for; `reproduce/PROVENANCE_MAP.md` note 11 carries the detail.

| | |
|---|---|
| Host | i9-14900HX · 32 logical cores · 95.6 GiB · RTX 4080 Laptop (12 GB) |
| Stack | Python 3.13.7 · numpy 2.4.6 · scipy 1.17.1 · sklearn 1.9.0 · scipy-openblas 0.3.31 |
| Pins | `tribble-fis d0efefc` · `tribble-cluster e3c27e6` · `grad-school 4c4fdbc` |
| Seeds | 0–9 (`common.SEEDS`) |
| Status | 13/13 green |
| Companion | `full-14900hx-2026-08-02/` — first pass, same numbers, superseded for citation |

The host is the machine the proposal calls "the workstation" (appendix, §3.4). Its
memory ceilings and large reorders were already measured here; the swept timing
grid was not, and that is what changed.

## What reproduced

Eleven tables are **byte-identical across two independent full sweeps** on this
host: Concrete reconciliation (34 cells), hyperparameter × normalization (48),
norm/conorm matrix (57), output partitioning (126), skew sweep (48), memory ×
precision (32), open-set detection (9), model family (16), feature ranking (20),
and all three Chapter 5 tables (64). Every cell that moved between the two runs is
a wall clock.

Against the previous run of record (`main-d0efefc`, a different host) the same
tables are also identical, and Concrete and Chapter 6 differ only within noise. The
θ operating curve reproduces its documented shape exactly — a flat band of
+0.222…+0.239 across θ = 0.5–0.8, peaking at θ = 0.60.

So "the harness is deterministic" is now measured rather than assumed, with the
boundary stated: **deterministic on one host with one numeric stack.**

## What did not

**Two Chapter 3 claims are properties of the development laptop, not of the code.**

*The stage-two plateau does not exist here.* §3.2 documents a fixed cost of roughly
10 ms engaging at N ≈ 750, a flat 8–15 ms band from 750 to 3,000, and stage two
losing to stage one across a band around N ≈ 10³. On this host, across five
independent measurements, stage two runs 0.5 ms at N = 750 and 8.4 ms at N = 3,000,
growing monotonically, and beats stage one by **8.1–17.7× at every size** — 17.3×
in the band said to collapse. The fitted exponent is **1.93–1.97** against the
laptop's 2.12/2.13: a *cleaner* confirmation of the quadratic claim, because the
plateau was contaminating the old fit, which the chapter itself calls "right for
the wrong reason". Checklist C2b is rescoped from "what is the 10 ms" to "why did
the laptop have it", and its OpenMP hypothesis is weakened — thread-startup cost
should be at least as visible on 32 cores as on 4.

*The ratio is not machine-invariant.* §3.4 argues that seconds are a property of
the host while the ratio between two arms is a property of the algorithms. Within a
host that holds: three runs here give the 1,024-point classical arm at 13.7 / 14.2
/ 14.5 s, a 6% spread, against the laptop's 22.2 / 31.7 / 21.3 s — so **the ~45–50%
swing was thermal and laptop-specific**, which is what B5 asked. But the ratio
itself moved 1,129× → 660–700× between the two hosts, a 40% change, because the
classical arm is interpreted Python and mergeVAT is compiled and the two do not
respond to a change of machine by the same factor. The reporting standard survives;
its justification needs weakening to "stable within a host, and far more portable
than seconds".

Table 3.1's swept ratios therefore need re-quoting: 28.8× → 28.6×, 398× → 304×,
1,129× → 660×. Tracked as B5b.

**One appendix arm is environment-sensitive.** Table A.2's bhattacharyya
accuracies sit up to +0.043 above `main-d0efefc` at identical code, seeds and
rankings, while wasserstein and composite agree to four decimals — and both sweeps
here reproduce those accuracies exactly, so it is the environment rather than
noise. The arm that moved is the ill-conditioned one. A.4's argument is untouched
(it rests on a 0.57 gap, not a 0.04 one), but those cells should not be quoted to
four decimals across machines. See note 12.

## Harness defects fixed to get here

Each produced plausible output or exited zero, which is why none had been noticed —
the same pattern the working doc's §7 warns about.

1. **PhiUSIIL was missing from `data/`**, so the loader would have fallen back to
   `ucimlrepo` and a different feature set — the documented 0.997 → 0.913
   "regression". Restored from `tribble-fis` history.
2. **Submodules were off their gitlinks in both directions.** `tribble-cluster`
   behind, `tribble-fis` ahead.
3. **The SHA-divergence guard never ran.** `check_submodule_shas` was called 35
   lines above its own definition; with `set -e` off, bash treated it as an unknown
   command and carried on. The guard added for B4 had never fired once.
4. **Two tables died after completing all their compute.** The emitters left the
   encoding to the platform; cp1252 encodes `±` but not `λ` or `Δ`, so
   `table_hyperparam_normalization` and `table_g5b_skew_sweep` crashed in the
   `f.write` that emits them — structurally the same bug as the Chapter 5 driver
   that discarded its own numeric phase. UTF-8 is now pinned.
5. **Three generators existed that no sweep ran**: `table_a1_feature_scoring`,
   `table_3_2_memory_precision`, and Table 4.4b's θ curve (emitted only when
   `REPRO_THETA_SWEEP` is set). Their outputs were in the archives from hand-runs,
   so a sweep reported all-green while silently carrying the older copies forward.
6. **`common.machine()` reported `ram: unknown`** and a registry CPU string under
   the native Windows interpreter `uv` launches, because it reads `/proc`. The
   machine block is the whole point of B1/B2, and it was degraded on the one host
   the chapter calls the workstation.
7. **A generator asserted a conclusion its own table contradicted.** The complexity
   fit's note hardcoded the 10 ms plateau; it is now derived from the measurements.
8. **`--help` was taken as a run label**, which is where the committed
   `reproduce/outputs/--help/` archive came from. Now a usage guard.
9. **No archive recorded the numeric stack**, which is why note 12's difference
   cannot be narrowed further. `PROVENANCE.txt` now carries numpy/scipy/sklearn and
   the BLAS build.

## Traps worth not repeating

**Editing a shell script while bash is executing it.** Bash reads scripts
incrementally; my mid-run edit shifted the byte offsets and the smoke run died with
`name: unbound variable` in a fragment. Nothing was wrong with the script.

**`| head -N` on the harness kills it.** A backfill piped through `head -5` took
SIGPIPE after printing its status line and never reached its archive step, so the
table it had just spent 697 s computing was written to `outputs/` and never
archived. `tail` is safe; `head` is not.

**Repeatable is not portable.** The plateau was measured repeatedly, on one
machine, and read as a property of the kernel. Determinism across runs says nothing
about determinism across hosts, and this project now has a measured example of
each.
