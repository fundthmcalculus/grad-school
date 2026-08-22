# `stats_numba.wasserstein_distance` is not the Wasserstein distance

_Found 2026-08-22 during the overnight reproduction pass against latest `main`
and latest submodules. Reproduce with one command:_

```bash
uv run --project tribble-fis python \
    reproduce/experiments/diagnose_wasserstein_regression.py
```

## The symptom

Re-running `reproduce/tables/table_4_1_mog_baselines.py` at ten seeds on this
host, at the current pins, against the archived run of record
(`reproduce/outputs/goal-8h-2026-08-11-fullsuite/`, `tribble-fis` `80e98d7`):

| row | archive (`80e98d7`) | current pin (`141596e`) | move |
|---|---:|---:|---:|
| PhiUSIIL (classification) | 0.997 ± 0.001 | **0.729 ± 0.023** | **−0.268** |
| RT-IOT2022 (12-class) | 0.927 ± 0.002 | **0.500 ± 0.244** | **−0.427** |
| Concrete R² | 0.795 ± 0.025 | 0.808 ± 0.030 | +0.013 |
| Concrete, full 2nd order | 0.852 ± 0.030 | 0.867 ± 0.031 | +0.015 |
| Bike Sharing R² | 0.939 ± 0.004 | 0.965 ± 0.001 | +0.026 |
| MoG train time (Concrete) | 0.41 ± 0.02 s | 0.06 ± 0.00 s | −0.35 s |

Those two accuracies are Chapter 1 §1.2's and Chapter 4 §4.4's headline numbers.
A model that fits six times faster and scores twenty-seven points worse is doing
less work, which is what pointed at the antecedent screen rather than the solver.

## The attribution

Each step holds everything else fixed.

1. **Not the data or the loader.** The PhiUSIIL matrix is frozen to one `.npz`
   before either library is imported. `git log 80e98d7..141596e --
   tribble-tree/demo_phishing.py` is empty, so the loader did not move anyway.
2. **Not the host or the compiler.** Old library and new library run in the same
   shell, minutes apart, on that frozen matrix: **0.9952 ± 0.0014** against
   **0.7405 ± 0.0092**.
3. **Bisected** over the 48 commits in `80e98d7..141596e`. First bad commit
   `5237ebe` — *"Replace scipy/sklearn stats functions with numba-accelerated
   implementations"* (#95). Its parent `ce4a0fc` is good; parentage confirmed.
4. **Isolated within #95** by restoring each replaced function one at a time, at
   the current pin, changing nothing else:

   | restored | PhiUSIIL accuracy |
   |---|---:|
   | *(nothing — current pin)* | 0.7405 ± 0.0092 |
   | `norm_fit` | 0.7405 ± 0.0092 |
   | `norm_pdf` | 0.7405 ± 0.0092 |
   | `jensenshannon_distance` | 0.7405 ± 0.0092 |
   | **`wasserstein_distance`** | **0.9947 ± 0.0017** |
   | `silhouette_score` | 0.7405 ± 0.0092 |
   | `_kmeans_labels_1d` | 0.7405 ± 0.0102 |
   | *(all six)* | 0.9952 ± 0.0014 |

## The defect

The 1-D Wasserstein distance is the integral of the absolute CDF difference
**with respect to $x$**:

$$W_1(u,v) = \int \left| F_u(x) - F_v(x) \right| \, dx.$$

`stats_numba.wasserstein_distance` returns instead the **mean** of the absolute
CDF differences over the union of the support points, with no $dx$ weighting:

```python
return float(np.sum(np.abs(u_quantile_vals - v_quantile_vals)) / len(all_quantiles))
```

That is a different quantity: dimensionless, bounded in $[0,1]$, and — the
manipulation check — **completely invariant to the scale of the data**.

| case | scipy | `stats_numba` | ratio |
|---|---:|---:|---:|
| `u=[0,1] v=[0,2]` (analytic 0.5) | 0.5000 | 0.1667 | 3.00× |
| the same, ×10 (analytic 5.0) | 5.0000 | 0.1667 | 30.00× |
| the same, ×1000 (analytic 500.0) | 5000.0000 | 0.1667 | 30000.00× |
| shift by 3 (analytic 3.0) | 3.0000 | 0.5000 | 6.00× |
| the same, ×100 (analytic 300.0) | 300.0000 | 0.5000 | 600.00× |

```
data ×1      scipy=    1.4255   stats_numba=0.245960
data ×10     scipy=   14.2546   stats_numba=0.245960
data ×100    scipy=  142.5460   stats_numba=0.245960
data ×1000   scipy= 1425.4604   stats_numba=0.245960
```

A distance in the data's own units cannot be invariant to those units. The fix
is to weight each CDF gap by the spacing between consecutive support points —
`np.sum(np.abs(u_cdf - v_cdf)[:-1] * np.diff(all_quantiles))` — which is what
`scipy.stats.wasserstein_distance` does.

## Blast radius

The function feeds `gauss_math._pairwise_label_distance`'s `"composite"` score,
which **is** the feature-differentiation screen. `mog_classifier` runs
`top_n=5`, so a wrong score selects the wrong five features and the whole model
is built on them. Two consequences beyond the accuracy:

- `_pairwise_label_distance`'s own comment says it *"squash[es] the unbounded
  pooled-std-normalized wasserstein distance"*. The value is already bounded in
  $[0,1]$, so both the pooled-std normalization and the squash now operate on a
  quantity they were not designed for, and the composite's three-term balance is
  not the balance that was tuned.
- The same metric is behind **Appendix A.4** and **Tables A.1/A.2**, whose whole
  subject is *why the composite metric earned its keep*.

Every `tribble-fis` table at this pin is therefore suspect, not only Table 4.1.

## Why the existing guard did not catch it

Checklist **B13** verified the `141596e` bump as *"byte-identical across the
bump"* on the strength of `table_4_1`'s three R² values — which do match, then
and now. The two accuracy columns in the same table were not part of that check,
and the regression predates #170 in any case: it entered at #95, inside the same
22-commit window the bump spanned.

B13's own rule is the right one and would have caught this had it been applied
to the whole table: *"if a Gaussian row moves, something else changed too and
the bump is not the explanation."* Chapter 8's closing tally already names the
lesson — **repetition is not the same thing as coverage**. This is a second
instance of it, and the first where the uncovered half was the headline number.

## What to do

1. **File it upstream** against `tribble-fis` (the fix is one line in
   `src/tribblefis/stats_numba.py`). Not filed from here — it is an outward-facing
   action and is left for the author.
2. **Do not quote the current pin** for any Chapter 4 or Chapter 6 accuracy
   number until it lands.
3. The archived numbers for **Table 4.1** are **correct**, not wrong: `80e98d7`
   reproduces 0.9952 ± 0.0014 against the archive's 0.997 ± 0.001, and restoring
   only this one function at the current pin recovers 0.9947 ± 0.0017.
   `reproduce/experiments/run_with_wasserstein_fix.py` re-runs any generator with
   the correction applied in-process, so the question can be settled table by
   table without waiting for upstream.

   ⚠️ **Do not read that as "the fix restores the document."** It does not, and
   the measurement says so. `check_prose.py` over the whole proposal:

   | archive | ok | drifted | untraceable |
   |---|---:|---:|---:|
   | `goal-8h-2026-08-11-fullsuite` (old pin) | **156** | 10 | 44 |
   | `full-2026-08-22` (current pin) | **59** | 23 | 128 |
   | `wasserstein-fixed-2026-08-22` (current pin, this one function corrected) | **70** | 23 | 117 |

   Correcting the defect recovers **11 pairs of the ~97 lost**. Per table it is
   decisive where the mechanism applies — `table_4_1` goes from 2 cells beyond
   noise to **0**, and `table_6_1` from 5 to 1 — and irrelevant where it does not.
   The rest of the gap has four separate causes, each identified and none of them
   this defect: the **gcc-vs-MSVC compiler change** on this host (B15), which moves
   every Chapter 3 timing; the **landed CPU FCM fix** (E2c), which moves Table 3.4
   again exactly as E2c predicted; **rows that were missing and are now restored**
   (the norm/conorm flat-MoG rows, Table 4.8's Glass row, Table 4.9 in its
   entirety), which register as differences because the archive had nothing to
   compare against; and a genuine **unexplained residue** in the feature-scoring
   and output-partitioning tables — `table_a1` 8 cells, `table_a2` 21,
   `table_g5_output_partitioning` 11 — that survives the correction and is owed an
   explanation of its own. That residue is the honest open item this pass leaves
   behind.
4. Extend the pin-bump check to **every column of a table, not the columns that
   happen to be easiest to eyeball** — the concrete change B13 needs.

---

## Independent confirmation, from a table that was not part of the diagnosis

The full ten-seed suite of 2026-08-22 was diffed cell-by-cell against the archived
run of record (`reproduce/compare_runs.py`, output in `FIX_IMPACT.md`). The drift
lands exactly where the mechanism above predicts and nowhere else:

| table | cells beyond noise | why |
|---|---:|---|
| `table_a1_feature_ranking` | 12 of 20 | **the feature screen itself** |
| `table_a2_feature_count` | 26 of 36 | accuracy as a function of screen depth |
| `table_4_8_mf_dedup_sweep` | 126 of 196 | memberships fitted on the screened features |
| `table_4_8_mf_dedup` | 24 of 28 | same |
| `table_norm_conorm_matrix` | 20 of 39 | same |
| `table_hyperparam_normalization` | 18 of 84 | same |
| `table_5_1` … `table_5_4_raw` | **0** | Chapter 5 never touches `gauss_math` |
| `table_a7_regression_scale` | **0** | Random Forest / flat MoG, no screen |

**Table A.1 is the direct read-out.** Its `wasserstein` column is the one that
collapses, and it collapses toward zero — which is what a bounded, scale-invariant
quantity does when asked to rank features by spread:

| rank | before | after |
|---:|---|---|
| 2 | HasSocialNet (0.867) | SpacialCharRatioInURL (**0.247**) |
| 3 | HasCopyrightInfo (0.743) | DegitRatioInURL (**0.075**) |
| 4 | HasDescription (0.629) | LetterRatioInURL (**0.051**) |
| 5 | DomainTitleMatchScore (0.471) | HasSocialNet (**0.049**) |

Rank 1 still reads 1.000 in both, because the top feature is normalised to 1 by
construction — so the column looks healthy exactly where a reader would glance.

**Table A.2 prices it.** The `wasserstein` arm's accuracy falls 0.9967 → **0.6709**
at one feature and 0.9997 → **0.8096** at ten; the `bhattacharyya` arm moves the
other way (0.4267 → 0.9967) as the ranking reshuffles under it.

Appendix **A.4**'s subject is *why the composite metric earned its keep*. At this
pin, one of the three terms in that composite is not the quantity it is named
after, so A.4's argument cannot be evaluated as written until the fix lands.

That Chapter 5's seven tables and the regression-scale table are **bit-identical**
across the same diff is the control: this is not a host effect, a compiler effect,
or general numerical drift. It is one function.

---

## The sibling defect in the same commit

`5237ebe` (#95) replaced six functions. Two of them changed results, through
different paths, which is why neither correction alone explains the whole of
Table 4.1's drift:

| replaced function | reaches | effect |
|---|---|---|
| `wasserstein_distance` | `_pairwise_label_distance` → the feature screen | **classification collapse**: PhiUSIIL 0.997 → 0.729 |
| `_kmeans_labels_1d` | `fit_gaussian_mixture_1d` → the memberships | **regression shift**: Concrete R² 0.795 → 0.804 |

Traced with `reproduce/experiments/diagnose_regression_drift.py`. Restoring
`_kmeans_labels_1d` alone at the current pin returns the regression rows to
0.7986 ± 0.0255 and 0.8521 ± 0.0287 — the full-second-order row landing on the
archive's 0.8517 to within 0.0004 — while every other restoration is inert.

The mechanism is initialization, not arithmetic. `sklearn.cluster.KMeans` seeds
with **k-means++**; the replacement `kmeans_1d` takes a **single uniform-random
start** with no restarts. Different 1-D mixture initialization, different fitted
memberships, different $R^2$.

**This one is not a wrong answer** — the values went *up*, by 0.009 and 0.016 —
so it needs a different response from the wasserstein defect. What it shares is
the shape: an unexplained change to a headline number, arriving inside a commit
whose title is a performance optimization, caught only because someone diffed the
tables afterwards. A single random start is also a weaker guarantee than
k-means++ even on a dataset where it happens to score better, which is worth a
sentence upstream regardless of which number the document ends up quoting.
