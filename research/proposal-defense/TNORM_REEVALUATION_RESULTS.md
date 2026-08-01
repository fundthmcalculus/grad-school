# Re-evaluation results — `tribble-fis` d0d6714 → d4dd392

_Run 2026-08-01. Companion to `TNORM_REEVALUATION_PLAN.md`. Cell-by-cell diff in
`reproduce/outputs/FIX_IMPACT.md`; raw archives in
`reproduce/outputs/baseline-d0d6714/` and `reproduce/outputs/postfix-pr29/`._

Nine tables, five seeds each, both sides run on the same machine from the same
harness. `tribble-cluster` is pinned to the same SHA in both runs, so anything
that moved there is machine noise by construction.

---

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
  small — the 256-point pVAT cell reads 1.271 ± 2.536 s vs 0.031 ± 0.058 s — which
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

**The norm/conorm comparison table** (plan §4) is also still blocked, for a
different reason: norm and conorm remain a single coupled knob, and the MoG
regressor and HME gate do not expose it at all. That needs the library change in
plan item 5 before the table in item 6 can be built.

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
