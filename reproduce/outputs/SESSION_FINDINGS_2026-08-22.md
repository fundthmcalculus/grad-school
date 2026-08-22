# Session findings — 2026-08-22 overnight reproduction pass

**Question asked:** does everything in `research/proposal-defense/` still build and
reproduce against the latest `main` and the latest submodules?

**Answer:** the harness does, after a toolchain repair. The document does not, and
the reason is now traced to a single upstream commit acting through two different
functions, plus three separate benign causes and one residue still owed an
explanation.

---

## The one-paragraph version

`check_prose.py` over the whole proposal, same prose, three archives:

| archive | ok | drifted | untraceable |
|---|---:|---:|---:|
| `goal-8h-2026-08-11-fullsuite` (the pin the document was written under) | **156** | 10 | 44 |
| `full-2026-08-22` (current pins, 16/16 generators green at ten seeds) | **59** | 23 | 128 |
| `wasserstein-fixed-2026-08-22` (current pins, one function corrected) | **70** | 23 | 117 |

The document reproduces against the pin it was written under and does not
reproduce against today's. `tribble-fis` **`5237ebe`** (#95, *"Replace
scipy/sklearn stats functions with numba-accelerated implementations"*) is the
largest single cause, and it is not the whole of it.

---

## 1. Nothing ran at all until the toolchain was repaired  (B15 — fixed)

This host has lost its MSVC toolchain: `Microsoft Visual Studio/2022` exists and
is empty. All three submodules carry compiled extensions, so every
`uv run --project …` in `reproduce/` failed during dependency resolution, before
a generator could import numpy.

Three defects had to be worked around. Each was reproduced first.

1. **`optional=True` does not survive `cythonize()`.** `tribble-opt` declares both
   extensions optional so a missing compiler degrades to the numba fallback — its
   `setup.py` docstring says so. Cython builds a *new* `Extension` from a fixed
   list of distutils settings that does not include `optional`. Measured: `True`
   in, `False` out, `same object: False`. Cython is in `build-system.requires`, so
   that documented degradation has never once been reachable.
2. **MSVC flags are chosen by platform, not by compiler.** `tribble-opt`'s
   `setup.py` branches on `platform.system()`, so gcc is handed `/O2 /openmp` and
   reads them as filenames. (`tribble-cluster` gets this right — it branches on
   `self.compiler.compiler_type` — which is why only one submodule needed a shim.)
3. **The editable build path drops `DIST_EXTRA_CONFIG`.** `[build_ext] compiler =
   mingw32` is honoured by `build_wheel` and ignored by `build_editable`, so
   `tribble-clustering` built fine as a git dependency of `tribble-fis` and failed
   as the project itself, in the same shell, seconds apart.

Fixed by `reproduce/hostenv.sh` + `tools/ccshim/`, sourced from
`run_all_tables.sh`. No-op on Linux/macOS and on any Windows host that still has
MSVC — forcing mingw where MSVC exists would silently change the compiler behind
every archived number.

⚠️ **Consequence:** Chapter 3's kernels are now **gcc-built**. Timings from this
host are not comparable to earlier archives. This is B5b/§3.4's host hazard,
extended to compilers.

---

## 2. The defect: `stats_numba.wasserstein_distance`  (B14 — filed, blocked on author)

$W_1 = \int |F_u(x) - F_v(x)|\,dx$. The implementation returns the *mean* of the
CDF gaps over the union support, with no $dx$ weighting — dimensionless, bounded
in $[0,1]$, and **completely scale-invariant**: multiply both samples by 1000 and
scipy's answer scales by 1000 while this one returns the identical `0.245960`.

It feeds the feature-differentiation screen, so it picks the wrong features:

| | archive | current pin | current pin, corrected |
|---|---:|---:|---:|
| PhiUSIIL | 0.997 ± 0.001 | **0.729 ± 0.023** | **0.997 ± 0.001** |
| RT-IOT2022 | 0.927 ± 0.002 | **0.500 ± 0.244** | 0.923 ± 0.011 |

Independently confirmed by a table that was not part of the diagnosis: **Table
A.1's `wasserstein` column collapses toward zero** (rank 2: 0.867 → 0.247; rank 5:
0.471 → 0.049) while rank 1 still reads 1.000 in both, because the top feature is
normalised to 1 by construction — so the column looks healthy exactly where a
reader would glance. Chapter 5's seven tables and `table_a7_regression_scale` come
back **bit-identical** across the same diff, which is the control that makes the
rest attributable.

Full account: [`WASSERSTEIN_REGRESSION.md`](WASSERSTEIN_REGRESSION.md). One-command
reproduction: `reproduce/experiments/diagnose_wasserstein_regression.py`.

**Not filed upstream from here** — outward-facing, left to the author.

---

## 3. The sibling in the same commit: `_kmeans_labels_1d`  (D8)

Table 4.1's three *regression* rows also moved, and moved identically with and
without the wasserstein correction, so they had a separate cause. Bisected to the
**same commit**, acting through a different function:

| tribble-fis commit | 1st order | full-2nd |
|---|---:|---:|
| `80e98d7` (archive pin) | 0.7950 ± 0.0249 | 0.8517 ± 0.0297 |
| `ce4a0fc` (#87) | 0.7950 ± 0.0249 | 0.8517 ± 0.0297 |
| **`5237ebe` (#95)** | **0.8041 ± 0.0297** | **0.8680 ± 0.0277** |
| `141596e` (current) | 0.8078 ± 0.0297 | 0.8666 ± 0.0311 |

Restoring each replacement one at a time, exactly one moves them:
`_kmeans_labels_1d`, back to 0.7986 ± 0.0255 / 0.8521 ± 0.0287 — the full-second-order
row landing on the archive to within 0.0004.

`sklearn.cluster.KMeans` seeds with **k-means++**; the replacement `kmeans_1d`
takes a **single uniform-random start** with no restarts. Different mixture
initialization, different memberships, different $R^2$. **Unlike the wasserstein
defect this is not a wrong answer** — the values went *up* — but it is an
unexplained change to a headline number inside a commit titled as a performance
optimization.

---

## 4. Results that changed on their own merits

- **The matrix-free reorder is built, correct and $O(N)$ in memory** (B6/G4d).
  Upstream repaired `vat_prim_mst_seq`; the repair is inside the pinned SHA and
  *not* inside `e3c27e6`, which §3.4's permalinks cited. G4d's decision rule now
  passes on all three counts: ordering $1.000 \pm 0.000$ at $N \in \{1k, 2k, 5k\}$
  over ten seeds; peak working set flat at **64.7–65.2 MB** from $N = 2{,}000$ to
  $12{,}000$ while the implied matrix grows 36×; wall clock **0.14–0.22×** the
  materialising arm — passing the "order of magnitude slower" threshold in the
  opposite direction. Table 3.3's two "defective" rows are now positive results.
- **E2c landed.** The CPU FCM formulation is fixed upstream and `n_iter_` /
  `converged` are exposed. Table 3.4's device row has moved again exactly as E2c
  predicted, and three runs of that row now give three answers, each with a spread
  of the order of its mean. The fix did not make it quotable; controlling for
  iteration count with the newly exposed fields would.

---

## 5. Things that were quietly not running

Each found by grepping for the *class* of defect rather than re-checking known
instances.

| what | consequence |
|---|---|
| `table_norm_conorm_matrix` used two renamed classes | both flat-MoG rows silently `N/A` since at least the archive — **E1's entire evidence base** |
| `reproduce/optimizers/structure.py` imported two deleted helpers | the optimizer structure study — **§6.3.5's evidence** — raised `ImportError` before doing any work |
| `reproduce/regression_scale/mog_top_p_sweep.py` | same rename |
| Glass moved to `data/`, three call sites did not follow | Table 4.8's Glass row and **the whole of Table 4.9** absent from the archive — **C4's headline measurement had no regenerable output** |
| `table_3_7_g2_downstream` is not in the orchestrator | in every archive with no log; evidence for Ch 3's G2 claim and §5.4's correction |
| `load_openset_data()` prefers RT-IOT2022 | **Tables 4.6 and 4.7 can no longer be regenerated at all** |

All but the last two fixed. With Glass restored, **C4's conclusion holds**: gated
cascade $+0.0431 \pm 0.0495$ against the prose's $+0.031 \pm 0.027$, once B14 is
corrected in the same run.

The shape they share is worth more than any one of them: every failure was
*graceful*. `load_glass()` returned `None` and the rows vanished; the norm/conorm
skip path printed its reason and emitted `N/A`; the generator exited 0 and the
orchestrator reported **ok**. A graceful degradation nobody reads is
indistinguishable from a result.

---

## 6. What is still owed

- **B14** — file upstream; do not quote Ch 4/Ch 6 accuracy from the current pin.
- **D8** — `table_a1` (8 cells), `table_a2` (21) and `table_g5_output_partitioning`
  (11) still move after correcting B14, unexplained. Until they are explained,
  `full-2026-08-22` is a **diagnostic archive, not a run of record**, and the run of
  record stays `goal-8h-2026-08-11-fullsuite`.
- **B13** — extend the pin-bump check to every column of a table. It verified three
  R² values and concluded "byte-identical" while two accuracy columns beside them
  had collapsed. Chapter 8's tally already names the lesson: *repetition is not the
  same thing as coverage.*
- **B16** — the orchestrator's missing table, the `--fast` seed list that no longer
  bounds the suite, `tribble-cluster`'s out-of-sync lock, and
  `uv run --project tribble-opt`, which cannot resolve at all.
- **D7** — seven state-then-walk-it-back passages marked in the prose for
  consolidation.
