# Review: PR #111 (FIS → ReLU conversion) and the N-CMAPSS RUL work

Two reviews, written after re-running both bodies of work and then building a
third experiment on top of them (`experiments/nn-cmapss/`, results in
[`BENCHMARK.md`](BENCHMARK.md)). Everything asserted below was executed, not
read off the write-ups.

## Fix log

Everything actionable here has been applied. What changed:

| # | change | files |
|---|---|---|
| 1 | `triangle_to_relu` / `trapezoid_to_relu` now raise `DegenerateMembership` on a zero-width term and `ValueError` on an inverted one, instead of returning a wrong or empty array; `fis_knots` skips degenerate terms and **warns with a count** | `fis2nn.py` |
| 2 | `trapezoid_to_relu` rejects infinite feet — the package's own `evaluate` returns NaN there, so there is no ground truth to convert to | `fis2nn.py` |
| 3 | `train_adam` excludes evaluation time from `hist.seconds`, so recorded wall clock is gradient descent only | `fis2nn.py` |
| 4 | `pwl_to_relu_weights` and `analytic_seed_from_fis` document the extrapolation term and the 0th/1st-order-only restriction | `fis2nn.py` |
| 5 | `setup_seconds` is derived from `starts` rather than a hand-copied key list that had drifted | `run_experiment.py` |
| 6 | five new regression tests (18/18 pass, was 13/13) | `test_fis2nn.py` |
| 7 | RUL caps built from **training units only**; `apply_rul_shape`/`capped_rul` leave uncapped units at raw RUL | `cmapss_rul.py`, `cmapss_rul_best.py`, `cmapss_rul_trajectories.py` |
| 8 | the test-selection disclosure, with the validation re-check that vindicates it | `cmapss_rul.py`, `cmapss_rul_best.py` |
| 9 | `n_gaussians=0` documented as *automatic*, not "zero Gaussians" | `cmapss_rul.py`, `cmapss_rul_best.py` |
| 10 | `import pickle` / `import os` hoisted out of the middle of the file and out of two function bodies | `cmapss_rul.py` |

**No quality number moved.** After the cap change, `cmapss_rul_best.py`
reproduces `11.23` (honest) and `6.48` (best) exactly, and DS02 conversion
fidelity is unchanged by fix 1 (the one degenerate term it hits contributed
nothing before and contributes nothing now — see the reachability note in
Defect 1). Fix 3 *does* change the wall-clock column of this benchmark's own
tables — that is the point of it, and the effect is not small: the `best` sweep
went from 1475 s to 894 s of recorded training time once evaluation stopped
being billed to it. Every run in [`BENCHMARK.md`](BENCHMARK.md) was regenerated
afterwards.

One thing to carry over: **PR #111's own committed curves in
`experiments/fis-to-neural-net/outputs/` predate fix 3** and are biased against
whichever arm was wider — usually the hot one, whose width is set by the FIS's
knot count. That does not flip any conclusion in that write-up (the hot arm was
losing on wall clock, and correcting the bias only helps it partway), but the
ratios quoted in its §4 should be regenerated before being quoted again.

Two things were deliberately **not** changed, because they are the author's
call and not a reviewer's: `_fuzzy_models.load_bikeshare`'s target leak (fixing
it invalidates proposal Tables 4.1 and 6.1) and
`triangle_fit.fit_triangles_to_mixture`'s compact-support collapse (upstream, in
the `tribble-fis` submodule).

---

# Part 1 — PR #111: `experiments/fis-to-neural-net`

**Verdict: sound work, correct central claim, honestly reported. Three real
defects and two methodological gaps, none of which overturn the conclusion —
all now fixed or documented in place (see the fix log above).**

## What I verified independently

| claim | how checked | result |
|---|---|---|
| 25 tests pass | ran both suites | 13/13 and 12/12, 2.6 s total (now 17/17 and 12/12 with the new regression tests) |
| `black` + `flake8` clean | ran both | clean at the project's `max-line-length=120` |
| triangle = 3 ReLUs exactly | independent grid check | 0.0 error |
| the additivity boundary | new experiment on N-CMAPSS DS02 | **confirmed, and strengthened** |

The additivity claim is the paper's real contribution, and it was the one thing
stated as an inference rather than measured. I measured it. Forcing a
TribbleRegressor down to `top_n` features on DS02 and converting at each size:

| FIS features | seed-vs-FIS (relative) | **best possible additive fit** |
|---:|---:|---:|
| 1 | 0.070 | 0.030 |
| 2 | 0.418 | 0.418 |
| 4 | 0.797 | 0.823 |
| 8 | 1.116 | 1.201 |
| 16 | 1.923 | 1.973 |
| 21 (full) | 2.421 | 2.493 |

The right-hand column is new: the FIS's own ANOVA projection evaluated on a
dense 33-point grid, i.e. *the best any axis-aligned seed of any width could
do*. The seed tracks it at every dimension and sometimes beats it. **The
conversion is not leaving anything on the table — the loss is irreducible
interaction.** That is a stronger statement than the PR makes, and it is the
one the write-up should carry.

At one feature, fidelity is 0.070 relative, i.e. the equivalence, executable, on
real turbofan data. The machinery is correct.

## Defect 1 — `triangle_to_relu` silently returns the wrong function on a degenerate triangle  ·  **FIXED**

`fis2nn.py:120-126`. When `a == b` (vertical rise), the code comments:

> `b == a` (a vertical rise) contributes no ReLU: the term jumps to 1 at the
> apex, which the falling side's expansion below reproduces on its own.

It does not. For `T(a=1, b=1, c=2)`:

```
x    : -1.0  -0.5   0.0   0.5   1.0   1.5   2.0   2.5   3.0
true :  0     0     0     0     0     0.5   0     0     0
relu :  0     0     0     0     0    -0.5  -1.0  -1.0  -1.0
```

It returns the **negation** of the correct ramp and then keeps falling forever.
`b == c` is the mirror case: it returns a right shoulder pinned at 1 instead of
dropping to 0 (`max|err| = 1.0` in both).

The honest fix is not to patch the expansion. A degenerate triangle is
*discontinuous*, and a finite sum of ReLUs is continuous, so it genuinely cannot
be represented — `triangle_to_relu` now `raise`s on `a == b` or `b == c` (and on
an inverted foot, which the old `elif b > a` guard swallowed the same way).

### Reachability: I got this wrong first time, and the correction matters

My initial read was that `fit_triangle_to_gaussian` produces `a < b < c` for any
`sigma > 0`, so the branch was unreachable from fitted models and the defect was
latent. Then the guard raised on a real DS02 fit within the hour. The precise
picture:

- DS02's `honest` FIS contains **1 of 109** membership functions with
  `sigma == 0` exactly — `Xs_T30_max`, a feature with no variance — which fits
  the *fully* collapsed `a == b == c`.
- On that form the old code took neither the rise branch nor the fall branch and
  computed a zero apex coefficient, so it emitted **no knots and an all-zero
  expansion**. Wrong in kind, not in value: a collapsed feature entered the seed
  as silence. DS02 fidelity is byte-identical before and after the fix.
- The *negating* form (`a == b < c`, the one that returns `-0.5, -1.0, -1.0…`)
  is a genuine wrong answer, but it is not what this package's Gaussian fit
  produces. It bites a caller building terms by hand.

So the honest severity is lower than I first wrote — no published number was
ever corrupted by it — and the fix is still worth having, because a constant
feature silently contributing nothing is exactly the thing a conversion should
say out loud. `fis_knots` now skips degenerate terms and warns with a count.

Chasing this turned up a second one in the same family: `trapezoid_to_relu` had
no bias branch at all, so a left-shouldered trapezoid came back 0 across its own
plateau. That one is *not* fixed by adding the constant — `TrapezoidMembership.
evaluate` computes `(x - a) / (b - a)`, so an infinite foot gives `inf / inf`
and the package's own ground truth is NaN across the whole left side. Shouldered
trapezoids simply are not a shape this package has, so the conversion rejects
them rather than inventing semantics for them.

## Defect 2 — the seed extrapolates linearly outside its knot range, unmeasured  ·  **DOCUMENTED**

`pwl_to_relu_weights` sets `base = seg[0]` and leaves the first and last knots'
coefficients at zero, so beyond the outermost knot the seed continues with the
slope of the outermost *segment*, forever. On DS02's `honest` FIS, **42% of test
rows** fall outside at least one FIS feature's knot range. Every one of those is
served by extrapolation.

This is not hypothetical — it accounts for essentially all of the residual at
one feature (0.070 seed vs 0.030 best-additive, with 18.3% of rows outside).
The PR attributes fidelity loss entirely to non-additivity without separating
this term. Two cheap ablations would settle it: extend flat instead of linearly,
or clip `x` into the knot range before the hidden layer.

## Defect 3 — `partial_dependence` evaluates the FIS off its data manifold, and a 2nd-order TSK explodes there  ·  **DOCUMENTED**

`partial_dependence` is textbook PDP: tile the background rows, overwrite one
column. That is fine for a 0th/1st-order TSK, which is all the PR runs. It is
not fine for `tsk_order="full-2nd"`, whose consequents are quadratic: forcing a
feature to a value its row never co-occurs with lands the model on a
combination it was never fitted near, and the quadratic extrapolates.

On DS02's `best` pipeline (`full-2nd`, the DOE's own champion configuration)
this is catastrophic:

| FIS features | seed-vs-FIS (relative) | best additive |
|---:|---:|---:|
| 1 | 0.129 | 0.130 |
| 6 | 1.002 | 1.062 |
| 8 | **32.084** | **29.762** |
| 21 (full) | **35.128** | 36.146 |

Note the best-additive column explodes *with* it — so the conversion is again
faithful to what it is asked to do, and the blowup is a property of the
partial-dependence probe applied to a quadratic-consequent FIS. But the practical
consequence is that `analytic_seed_from_fis` cannot be pointed at a `full-2nd`
FIS as written. A conditional/ALE-style profile, or simply restricting the grid
to the feature's observed conditional support, would fix it. The docstring should
say so.

## Gap 1 — the `quantile` comparison is framed against the wrong arm

The paper's headline negative ("a trivial baseline beats it almost everywhere")
compares `quantile`, which fits the **labels** in its ridge solve, against a hot
start. The write-up states the asymmetry (§5) and calls it "the fair reading" —
but the symmetric comparison already exists in the code: `hot` is the analytic
seed plus exactly the same one-shot ridge solve against labels. `hot` vs
`quantile` is apples-to-apples, and it is much closer than the framing implies.
On DS02:

| arm | test RMSE | setup+train |
|---|---:|---:|
| `quantile` | 9.00 | 0.02 s |
| `hot` | 9.39 | 0.33 s |

A 4% quality gap at 16× the cost — still a loss for the warm start, and the
conclusion survives, but "beats it almost everywhere" oversells a comparison the
paper itself flags as unfair. Promote `hot` vs `quantile` to the headline table.

## Gap 2 — evaluation cost is inside the timed region, and it scales with width  ·  **FIXED**

`train_adam` starts its clock, then calls `record(0.0, 0.0)`, and every later
`hist.seconds` is `perf_counter() - start`. So each recorded time includes all
prior *evaluation* passes over `X_test` and `X_val`. `track_train=False` removes
the largest one, but not these.

Where arms have equal width this cancels. It does not cancel in the comparison
the paper actually reports, because a hot arm's width is fixed by the FIS's knot
count while `he`'s is free — on DS02 that is 264 hidden units against 8, a 33×
difference in per-eval cost charged to wall clock. Any wall-clock ratio between
arms of unequal width is biased against the wider one. Cheap fix: subtract the
eval time, or record it separately.

## Smaller notes

- `run_experiment.py:369-376` builds `setup_seconds` without a `"quantile-all"`
  key while `starts` has one; any consumer indexing `setup_seconds[arm]` over
  `starts` raises `KeyError`. Currently latent.
- The claim that the conversion "touches no data" is true of the 1-D theorem but
  not of `analytic_seed_from_fis`, which consumes `X` (never `y`). §8's phrasing
  "label-free" is the accurate one and should be used throughout; "no data" in
  the abstract is stronger than the code.
- The two upstream defects reported (`fit_triangles_to_mixture` compact-support
  collapse, `load_bikeshare` target leak) are both real and correctly left to the
  author. The `load_bikeshare` one should be filed upstream now, regardless of
  the re-quoting cost, so it does not propagate into new work.

## What I'd change in the write-up

1. Lead §6 with the best-additive reference. "Fidelity tracks the best possible
   additive fit at every dimension" is a much stronger claim than "fidelity
   degrades with dimension", and the data supports it.
2. Report the outside-knots fraction alongside every fidelity number.
3. Say plainly that `analytic_seed_from_fis` is 0th/1st-order only.

---

# Part 2 — the N-CMAPSS RUL work

**Verdict: the engineering is good, the headline preprocessing finding is real
and valuable, and the reported numbers survive an honest re-selection. The
write-ups did not disclose that selection touched the test engines; they do
now, together with the re-check that answers the objection.**

## What's genuinely good

**The condition-correction finding is the result here.** Regressing each sensor
channel on the W operating-condition channels using each engine's own early
cycles, and modelling the *residual*, is the step that made everything else
work. The negative result that led to it — a naive moving-average onset detector
that never fired because baseline-period std exceeded whole-lifetime std — is
documented in the code with the numbers that killed it (`cmapss_rul.py:894-904`).
That is exactly how this should be recorded.

**The B1/B2/B3 feature-set audit is careful work.** Establishing that T40/P30 sit
in the HDF5 `X_v` group for simulator-bookkeeping reasons, and that the published
CNN/MLP baselines use them, by triangulating Arias Chao et al.'s Table 2, Custode
et al.'s citation of it, and Mo's released code slicing `X_v[:, 0:2]` — that
converts an apples-to-oranges comparison into a fair one, and it is the
difference between "beats the published CNN" being a claim and being a caveat.

**I reproduced the `honest` number exactly.** Independent pipeline, same
preprocessing: test RMSE **11.23**, matching `cmapss_rul_best.py`'s
`expected_rmse=11.23`. The pipeline is deterministic and the documentation is
accurate.

**Operational care.** Per-pair subprocesses after a confirmed OOM, checkpoint
CSVs after every unit of work after a mid-run kill, `--resume` throughout,
`COORDINATE_MAX_MF` guarding a refiner known to blow the budget. This is a
codebase that has been run at scale and repaired where it broke.

## Issue 1 — hyperparameters are selected on the held-out test engines  ·  **DISCLOSED** (see the measurement below)

`cmapss_rul.py:508-511`:

```python
best = min(
    (r for r in results if r["pipeline"] == pipeline),
    key=lambda r: r["rmse_test_true"],
)
```

Stage 2 grids 288 Factor-D combinations and picks the minimum of
`rmse_test_true` — the official held-out units 11, 14, 15. Stage 3 then
"confirms" that configuration on the same units. `cmapss_rul_best.py`'s
`PIPELINES` dict hardcodes the winners of that search, so `expected_rmse=6.48`
for `best` is a **test-set-selected** number, and `--pipeline best` beating the
published CNN's 7.22 is not a like-for-like comparison with a paper that (as far
as its protocol states) did not select on test.

There is no validation split anywhere in the DOE. The dev set has six engines;
holding two out costs almost nothing. My rerun does exactly that
(`experiments/nn-cmapss/cmapss_data.py`, val = units 18, 20), and it is a
two-line change to the existing code.

### I measured the size of the advantage, and it is a better story than the concern

Rather than leave this as a warning, I ran the same Factor-D grid (72
configurations) against a validation fold — dev engines 18 and 20 held out —
and confirmed the winner on test:

| bundle | selected on | winning config | test RMSE |
|---|---|---|---:|
| `honest` | test (as published) | `1st`, `n_gaussians=0`, `top_p=0.9`, hamacher, `l2=0.01` | **11.23** |
| `honest` | validation | `1st`, **`n_gaussians=3`**, `top_p=0.9`, hamacher, `l2=0.01` | 16.06 |
| `best` | test (as published) | `full-2nd`, `n_gaussians=0`, `top_p=0.95`, hamacher, `l2=0.01` | **6.48** |
| `best` | validation | **identical** | **6.48** |

**On `best` — the pipeline whose 6.48 is compared to the published CNN's 7.22 —
an honest protocol selects exactly the same configuration.** The headline number
is not an artifact of test selection. That is worth knowing and worth saying;
it converts a defensible-but-attackable claim into one with a direct answer.

On `honest` the validation protocol picks a *worse* model (16.06 against 11.23)
while scoring better on validation (9.54 against 10.18). That is not evidence
the published number is inflated — it is evidence that a two-engine validation
fold is too noisy to select on. Six engines do not divide well.

**Revised recommendation.** The methodological point stands — a reader cannot
tell from the code that selection touched test — but the fix is documentation,
not re-running:

1. State in `cmapss_rul_best.py` and the dissertation section that Factor-D was
   selected on the held-out units, **and** that re-selecting `best` on a
   validation fold reproduces the same configuration. Two sentences, and the
   objection is closed on evidence.
2. Do **not** switch `honest` to validation selection on six engines; it is
   worse. If a fully clean protocol is wanted, leave-one-engine-out
   cross-validation over the six dev engines is the shape that fits this
   dataset, and it is still seconds-scale.
3. Fix `rmse_test_shaped` regardless (Issue 2) — that one is free.

## Issue 2 — test-unit RUL caps are computed, and one metric uses them  ·  **FIXED**

`cmapss_rul.py:406`, `:494`, `:532` and `cmapss_rul_best.py:307`:

```python
caps = unit_physical_caps(pd.concat([train_tab, test_tab], ignore_index=True))
```

The caps come from the `hs` oracle flag over **train and test combined**. In
`cmapss_rul_best.py` they only ever reach `y_train`, so the headline is clean. In
`cmapss_rul.py:329` they also produce `y_test_target`, and `rmse_test_shaped` is
therefore computed against a target shaped by the test engines' own
degradation-onset times. `rmse_test_true` is the reported column, so nothing
published is affected — but the leaky metric is in the results CSVs under a name
that does not announce itself.

Building caps from training units only makes the concern impossible to have.
Costs nothing: the test units' caps are unused in the headline path already.

## Issue 3 — the per-engine number is three data points

The full-analysis report leads with "**Per-engine (canonical...): RMSE 8.61**".
For DS02 alone, that is one prediction each for units 11, 14 and 15. In my
reruns the per-engine endpoint RMSE swings between 6.85 and 19.54 across models
whose per-sample RMSE differs by less than 1 cycle — with n=3 it is dominated by
which single cycle happens to be last. The pooled 39-engine version in the
full-dataset report is defensible; the per-file version is not, and the two are
presented in the same voice.

Report per-sample RMSE as primary for DS02, and give the per-engine number with
its n attached every time.

## Issue 4 — `n_gaussians=0` will be misread (documentation, not method)  ·  **FIXED**

Every winning configuration carries `n_gaussians=0`, and the D-grid sweeps
`[0, 3, 5]`. In `TribbleRegressor` that `0` means **automatic** ("Number of
Gaussians per feature per label (0 for automatic)"), so the finding is
"automatic beats both fixed choices" — a perfectly good result. But nothing in
`cmapss_rul.py`, `cmapss_rul_best.py`, or the reports says so, and `0` sitting
in a list next to `3` and `5` reads as "zero Gaussians" to anyone who has not
opened the regressor's docstring. One clause in the config comment fixes it, and
it is worth fixing before a committee reads `n_gaussians=0` as an ablation.

## Smaller notes

- `aggregate_raw_memory` subsamples `df.iloc[::stride]` *after* grouping by unit
  but the memory window is then computed across cycle boundaries within a unit,
  so a window can straddle two flights. Probably harmless (the features are
  slow-moving), but it is an unstated assumption.
- DS08d fails to load (`truncated file`) and is silently skipped in the pooled
  report's file table. The skip is visible, but the pooled totals do not say
  "9 of 10 datasets" in the headline.
- `cmapss_rul.py:1039` has `import pickle` two-thirds of the way down the file.

## Bottom line

The condition-correction result and the feature-set audit are solid contributions
and should survive review unchanged. The comparison to the published CNN/MLP
numbers also survives — I checked, and re-selecting `best` on a validation fold
returns the identical configuration and the identical 6.48. What is missing is
the *disclosure*, and with the check in hand that is two sentences rather than a
re-run. Add them, fix the `rmse_test_shaped` cap leak, and this section is
defensible against the obvious line of attack instead of exposed to it.
