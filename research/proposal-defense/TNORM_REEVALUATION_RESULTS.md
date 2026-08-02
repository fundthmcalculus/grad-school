# Re-evaluation results — `tribble-fis` d0d6714 → d4dd392

_Run 2026-08-01. Companion to `TNORM_REEVALUATION_PLAN.md`. Cell-by-cell diff in
`reproduce/outputs/FIX_IMPACT.md`; raw archives in
`reproduce/outputs/baseline-d0d6714/` and `reproduce/outputs/postfix-pr29/`._

Nine tables, five seeds each, both sides run on the same machine from the same
harness. `tribble-cluster` is pinned to the same SHA in both runs, so anything
that moved there is machine noise by construction.

---

> **SUPERSEDED IN PART — read the second addendum first.** Glass has since been
> added, `table_4_4_openset` now runs, and the headline below is wrong: the
> t-norm fix changes the open-set results substantially (Youden's J roughly
> triples at the default operating point). Everything in sections 1–3 still
> stands. See "Addendum 2 — the open-set comparison, finally run".

## Headline

**The t-norm/t-conorm fix (#26) produced no measurable change anywhere in the
harness.** Not because it does not matter — because the only experiment that
exercises the code it touches cannot run here. `table_4_4_openset` is the sole
table using the anomaly path or a non-`min/max` conorm, and it needs
`glass.csv`, which is gitignored, absent, and unreachable (UCI is egress-blocked
in this environment). It produced no output on either side.

Results **did** get stronger in one place, and the cause is a different bug than
the one we set out to measure. The rest is either unchanged or explained by a
dataset appearing in the repository.

| | Table | Verdict |
|---|---|---|
| ✅ | `table_concrete_reconciliation` | **2 cells genuinely improved** — refined 0th order |
| ⚠️ | `table_4_1`, `table_6_1` | 7 cells `N/A` → real numbers, but **not from a code fix** |
| ➖ | `table_3_1`, `table_3_1_three_arm` | 27 cells differ, **all wall-clock**, 0 beyond noise |
| ➖ | `table_g5`, `table_g5b`, `table_hyperparam_normalization` | **bit-identical**, 204 cells |
| ❌ | `table_4_4_openset` | **no output either side** — the one table that mattered |

---

## 1. The real improvement: refinement was pinning bucket means to zero

`table_concrete_reconciliation`, refined arms only:

| Row | Metric | Before | After | Δ |
|---|---|---|---|---:|
| flat MoG-TSK **0th**, refined | R² | 0.135 ± 0.325 | **0.461 ± 0.116** | **+0.326** |
| flat MoG-TSK **0th**, refined | RMSE | 15.04 ± 2.64 | **12.01 ± 0.98** | **−3.03** |
| flat MoG-TSK 1st, refined | R² | 0.822 ± 0.056 | 0.822 ± 0.032 | +0.000 |
| flat MoG-TSK 2nd, refined | R² | 0.853 ± 0.020 | 0.868 ± 0.020 | +0.015 |

Every `closed-form only` arm is **bit-identical**. Only the refined arms moved,
and the effect is concentrated entirely at 0th order.

**Cause.** `d0d6714` — the commit that introduced `pin_extremes`, defaulting it
to `True` — did not update `refine.py`. That module's cross-validated fitness
calls the solver with a placeholder:

```python
y_bucket_mean_dummy = np.zeros(n_output_buckets)   # "solver ignores this arg"
...
solve_tsk_consequents(..., y_bucket_mean_dummy, ...)   # pin_extremes defaults True
```

The comment was true when it was written and false the moment pinning landed.
Every refinement fitness evaluation pinned the first and last bucket means to
**0.0** — values with no relation to the target — so the refinement search was
optimising against a corrupted objective. On `main` the call passes
`pin_extremes=False` explicitly and the placeholder is genuinely ignored.

The size of the effect follows: at 0th order the bucket means *are* the whole
model, so pinning two of three rules to zero is catastrophic; at 1st and 2nd
order the correction terms absorb it, which is why those rows sit within noise.
The standard deviation collapsing (±0.325 → ±0.116) is the clearer tell — the
old numbers were unstable because the objective was wrong, not merely worse.

**This is a real fix and it belongs in the text**, but note what it is not: it
has nothing to do with t-norms. It is fallout from `pin_extremes`, i.e. the
Ch 4 pinning work, and it means the refined 0th-order Concrete numbers measured
before today were measured against a broken refinement objective.

## 2. PhiUSIIL filled in — but not because of any fix

Seven cells went from `N/A` to real values across Tables 4.1 and 6.1
(flat 0.997, fuzzy tree 0.969, HME 0.997, CART 1.000, RF 1.000).

This is **not** a code improvement. `demo_phishing.load_data` reads a committed
CSV, and that file was added by **040e4c7 (#19)**, a literature-index commit:

```
present at d0d6714: NO
present at d4dd392: YES     (added by 040e4c7, 56 MB)
```

`d0d6714` predates it, so the loader raised `FileNotFoundError` and the
`ucimlrepo` fallback hit the egress block. Updating the submodule brought the
dataset along with the fixes. The numbers are legitimate, but they are new
*coverage*, not a changed result, and they must not be described as an
improvement attributable to the fixes.

They also explain the runtime jumps that looked alarming mid-run
(`table_4_1` 8s → 24s, `table_6_1` 26s → 62s): a second dataset is now being fit.

## 3. Everything else is unchanged

- **`table_g5_output_partitioning` (126 cells), `table_g5b_skew_sweep` (48),
  `table_hyperparam_normalization` (30) — bit-identical.** 204 cells, zero drift.
  G5's conclusions and the Ch 6 normalization finding stand exactly as written.
- **`table_3_1` and `table_3_1_three_arm`** differ in 27 cells, every one of them
  wall-clock or a speed-up ratio derived from wall-clock, none beyond seed
  spread. `tribble-cluster` is at the identical SHA in both runs, so this is
  thermal/scheduling variance and nothing else. Worth noting the variance is not
  small — the 256-point pqVAT cell reads 1.271 ± 2.536 s vs 0.031 ± 0.058 s — which
  is exactly the instability the G4 protocol exists to fix.

## 4. What still cannot be answered

**Whether the complement rule got better.** `table_4_4_openset` produced nothing
on either side. This is the only harness experiment touching `t_conorm`'s
reduction branch, the Hamacher formulas, or the anomaly clipping — all four of
#26's defects. Until it runs, the recorded pre-fix baseline stands unrefuted and
unconfirmed: best **J = +0.155 at θ = 0.80**, J = +0.075 at the θ = 0.99 default,
≈31 % detection at 13 % false alarm, statistically indistinguishable from
isolation forest and one-class SVM.

Unblocking it needs one of: commit `glass.csv` (~12 KB, public UCI, and the
harness already claims it is in-repo); allow-list `archive.ics.uci.edu`; or add
an sklearn-bundled arm (`wine`, 178×13×3) as a second dataset. The third works
offline today and is worth having regardless, since `ACTION_ITEMS.md` already
flags Glass's 214 samples as "a stress test, not a demonstration".

**The norm/conorm comparison table** (plan §4) was also blocked when this was
written — norm and conorm were a single coupled knob and the MoG regressor could
not express one at all. `tribble-fis#32` has since landed and the table is built;
see the addendum below. (The HME *gate* turned out not to need exposing: it is
fixed by the model's semantics, not by omission.)

---

## Corrections to the record

- Ch 4 / Ch 6: the **refined 0th-order Concrete** figures predate the refinement
  fix and should be re-quoted from `postfix-pr29` (R² 0.461, RMSE 12.01).
- `ACTION_ITEMS.md` line 65 says six experiments are verified end-to-end. Of the
  nine the harness now runs, **eight produce output**; `table_4_4_openset` does
  not, and `table_3_1_pvat_scaling` only runs with `--with scipy` (scipy sits in
  `tribble-cluster`'s `dev` extra, not its base dependencies).
- Two harness defects were found and fixed during this run, both of which had
  reported success falsely: the run script called a script that wrote **no table**
  `ok` because it exited 0, and `compare_runs.py` was keying rows on their first
  column only, which silently collapsed the six-row reconciliation table to three
  and dropped 100+ cells from the g5 comparison. Both are fixed; the archived
  provenance files are corrected in place with a note.

---

# Addendum — the norm/conorm sweep (De Morgan diagonal)

_Run 2026-08-01 at `tribble-fis` ba87f5a, five seeds. Table and provenance in
`reproduce/outputs/norm-matrix-ba87f5a/`; generator
`reproduce/tables/table_norm_conorm_matrix.py`._

The five De Morgan pairs, across both datasets and all three model families.
Nine rows, no skipped cells.

| Dataset | Model | Metric | min/max | probability | luk | hamacher | einstein | spread¹ |
|---|---|---|---|---|---|---|---|---:|
| Concrete | flat MoG-TSK | R² | 0.644 ± .015 | **0.651** ± .042 | **−3.575** ± .389 | 0.644 ± .017 | 0.634 ± .050 | 0.017 |
| Concrete | flat MoG-TSK | RMSE | 9.84 ± .35 | **9.71** ± .35 | **35.25** ± 1.22 | 9.85 ± .24 | 9.94 ± .46 | 0.23 |
| Concrete | fuzzy tree² | R² | 0.719 ± .043 | 0.720 ± .041 | 0.720 ± .043 | 0.719 ± .042 | 0.720 ± .041 | 0.002 |
| Concrete | fuzzy tree² | RMSE | 8.72 ± .57 | 8.71 ± .55 | 8.69 ± .57 | 8.72 ± .56 | 8.70 ± .56 | 0.03 |
| Concrete | HME experts³ | R² | 0.754 ± .041 | **0.784** ± .041 | **−3.632** ± .571 | 0.763 ± .044 | 0.767 ± .046 | 0.030 |
| Concrete | HME experts³ | RMSE | 8.16 ± .83 | **7.65** ± .80 | **35.42** ± 1.64 | 8.01 ± .86 | 7.94 ± .95 | 0.51 |
| PhiUSIIL | flat MoG | accuracy | 0.997 ± .001 | 0.997 ± .001 | 0.996 ± .002 | 0.997 ± .001 | 0.997 ± .001 | 0.001 |
| PhiUSIIL | fuzzy tree² | accuracy | 0.967 ± .001 | 0.967 ± .001 | 0.967 ± .001 | 0.967 ± .001 | 0.967 ± .001 | 0.000 |
| PhiUSIIL | HME experts³ | accuracy | 0.997 ± .001 | **0.999** ± .001 | **0.930** ± .020 | 0.997 ± .001 | **0.999** ± .001 | 0.069 |

¹ Spread between best and worst mean, **excluding Łukasiewicz**, which would
otherwise swamp every row. ² t-norm only — a tree path is a pure AND, so there is
no OR for a conorm to act on. ³ The experts' operators. The HME *gate* is a
product of partition-of-unity weights by construction and is not a free axis.

## 1. Łukasiewicz is unusable for MoG regression, and predictably so

R² of **−3.6** on Concrete for both the flat model and the HME experts — far
worse than predicting the mean, and an RMSE of 35 MPa against a ~9 MPa baseline.
This is not a bug; it is the nilpotency of `T(x,y) = max(0, x+y−1)`. Concrete has
eight features, and ANDing eight memberships under Łukasiewicz drives the firing
to *exactly* zero for almost every sample. The rules stop firing, the
normalisation falls back to its zero-firing convention, and the model degenerates.

Classification tolerates it far better (0.996 on PhiUSIIL flat) because argmax
over a mostly-zero firing vector still lands on the right class often enough,
while a regression consequent has nothing to interpolate from. The HME experts
row is the exception that shows the mechanism: 0.930 vs 0.997, the one
classification row where enough samples lose all firing to matter.

**Practical upshot:** Łukasiewicz should not be offered as a regression option
without a warning, and it is the one family where the wide-antecedent TSK
construction genuinely breaks.

## 2. Excluding Łukasiewicz, the axis barely matters

This is the honest headline and it argues *against* the 25-cell version:

- **flat MoG on Concrete**: 0.634 → 0.651, a spread of 0.017 against a
  seed-to-seed standard deviation of 0.015–0.050. Inside the noise.
- **fuzzy tree**: 0.002 on Concrete, **0.000** on PhiUSIIL.
- **PhiUSIIL flat MoG**: 0.001.

The fuzzy tree result is not dead plumbing — I checked, because `build_tree`
already has one parameter it ignores (`tribble-fis#31`). Individual predictions
move by up to **3.04 MPa** between families; the aggregate R² simply does not
care. The tree is genuinely robust to this choice.

## 3. Where it does matter: HME experts, and the default is never best

The largest non-Łukasiewicz effect is **HME experts on Concrete: probability
0.784 vs min/max 0.754, +0.030 R²** (RMSE 7.65 vs 8.16). That is ~0.7 σ — real
but not decisive on five seeds. PhiUSIIL agrees in direction: probability and
einstein both 0.999 against min/max's 0.997.

**`min/max` — the library default — does not win a single Concrete row.**
`probability` wins all four. The margins are small, but they are consistent
across two model families and two datasets, which is more than a coin flip.
A default change to `probability` for the TSK regression path is worth
considering; it is also the family the Ruspini models and fuzzy trees already
default to, so it would make the library self-consistent.

## 4. Einstein behaves

Mid-pack throughout, closely tracking probability and hamacher, never
catastrophic, and — unlike hamacher — with no singularity to guard. It ties for
best on PhiUSIIL HME. A safe addition rather than a dramatic one.

## Correction to an earlier claim

PR #32's description reports a synthetic sweep where the operator moved R² from
0.748 (min/max) to 0.888 (probability), implying the pinned default was costing
~0.14 R². **On real Concrete that gap is 0.007**, and 0.030 for the HME experts.
The synthetic figure was real but is not representative; the practical effect of
the regressor plumbing fix is much smaller than that number suggests. What the
fix genuinely buys is the *ability to ask the question* — and the answer, on this
evidence, is "not much, except avoid Łukasiewicz."

## What this implies for the 25-cell version

Given that four of nine rows are flat to within 0.002 and the rest are dominated
by a single pathological family, the 20 mixed pairs are unlikely to pay for
themselves on *these* datasets and metrics. The place a mixed pair should visibly
bite is the anomaly rule, whose complement construction assumes duality — and
that panel is still blocked on Glass. Recommend running the mixed pairs only
against the open-set harness once a dataset is available, rather than across this
matrix.

---

# Addendum 2 — the open-set comparison, finally run

_Glass added 2026-08-01. Controlled before/after: identical dataset, identical
five seeds, identical six held-out classes, identical θ grid; the only variable
is the library. Archives in `reproduce/outputs/openset-prefix/` (`d0d6714`) and
`openset-postfix/` (`23bfdbc`)._

**This overturns the headline at the top of this document.** That conclusion was
correct only in the sense that the experiment could not run; it should not be
read as evidence the fix does nothing.

## The fix roughly triples the complement rule's separation

| θ | pre det | pre FA | **pre J** | post det | post FA | **post J** | ΔJ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.336 | 0.207 | +0.129 | 0.731 | 0.488 | **+0.243** | +0.114 |
| 0.60 | 0.327 | 0.186 | +0.141 | 0.685 | 0.428 | **+0.257** | +0.116 |
| 0.70 | 0.305 | 0.164 | +0.141 | 0.612 | 0.352 | **+0.261** | +0.120 |
| 0.80 | 0.276 | 0.142 | +0.134 | 0.539 | 0.298 | **+0.242** | +0.108 |
| 0.90 | 0.209 | 0.104 | +0.104 | 0.457 | 0.258 | **+0.199** | +0.095 |
| 0.99 *(default)* | 0.121 | 0.064 | +0.057 | 0.360 | 0.190 | **+0.170** | +0.113 |
| 1.10 | 0.000 | 0.000 | +0.000 | 0.000 | 0.000 | +0.000 | 0.000 |

Detection roughly **doubles at every θ**, and J improves by +0.095 to +0.120
across the whole usable range — a uniform shift, not a single lucky operating
point. Peak J moves from +0.141 (θ≈0.6–0.7) to **+0.261 (θ=0.70)**. Saturation
at θ=1.1 is unchanged: the rule stops firing in both.

The pre-fix θ=0.99 figure (+0.057) is close to the +0.075 recorded in
`ACTION_ITEMS.md` from an earlier two-seed run, which is a useful cross-check
that the pre-fix arm really is reproducing the old behaviour.

## Why: the old intermediates were not valid memberships

Probing one split (held-out class 7, seed 0) through the harness's own
`complement_rule`:

| | flagged unknown | max class firing | NaNs | min anomaly membership |
|---|---:|---:|---:|---:|
| pre-fix | **2 / 85** | 1.107 | 128 | **−0.322** |
| post-fix | **32 / 85** | 0.618 | 0 | ~0 |

Pre-fix, the rule fired almost never, and the values it was deciding on included
128 NaNs, firing strengths above 1, and *negative* anomaly memberships. Two of
#26's four defects account for it: **#22** meant the requested `hamacher` conorm
was silently discarded in the array-reduction branch, and **#25** meant the θ
boost was aggregated unclipped, pushing inputs outside the `[0,1]` domain the
operators are defined on. A negative membership can never win an argmax, so the
anomaly label lost by construction on most samples.

## The important caveat: the baselines moved too

| Method | pre J | post J |
|---|---:|---:|
| **Complement rule** | +0.057 | **+0.170** |
| One-class SVM | +0.037 | +0.062 |
| Isolation Forest | +0.032 | **+0.208** |

The baselines are scikit-learn and cannot have been affected by a `tribble-fis`
change. They moved because the protocol **matches each baseline's contamination
to the complement rule's observed false-alarm rate** — that rate went from 0.064
to 0.190, so all three arms are now being read at a different, higher-FA
operating point. The arms remain comparable to each other within a run; they are
not comparable *across* runs.

That reframes the result. In absolute terms the complement rule improved a lot.
In relative terms it **lost its nominal lead**: pre-fix it beat both baselines,
post-fix isolation forest is ahead (+0.208 vs +0.170). The fix made the rule work
properly; it did not make it the best detector on this data.

And the caution from the earlier run still applies — post-fix detection carries a
standard deviation of ±0.331 on a mean of 0.360, wider than the gap between any
two arms. **The three detectors remain statistically indistinguishable.** Glass,
at 214 samples with three classes under 18 members, is a stress test.

## What to change in the text

- Ch 4 §4.3.5: the complement-rule numbers should be re-quoted from
  `openset-postfix`. Best operating point **J = +0.261 at θ = 0.70**, not the
  θ = 0.99 default inherited from `beth-anomaly.py`, which sits well past the
  useful range.
- `ACTION_ITEMS.md` line 93–94: the "best J = +0.155 at θ = 0.80" figures are
  pre-fix and should be retired.
- The claim that the complement rule leads the dedicated detectors should be
  dropped. It is behind isolation forest at a matched operating point, and the
  spread makes all three a tie.

## Methodological note

The first attempt at this comparison silently produced nothing: a stray `cd` into
the submodule made the script path unresolvable, python exited immediately, and
the stale post-fix outputs were still on disk — so the "pre-fix" and "post-fix"
files compared byte-identical and briefly looked like strong evidence that the
fix changed nothing. It was caught only because a single-split probe disagreed
with the table. The check that mattered was reading the run's log rather than
trusting its exit status, and the harness's `no-output` detection (added earlier
today) exists for exactly this failure mode but does not cover a run whose
outputs were left behind by a previous invocation.
