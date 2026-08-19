# Experiment: overlap modeling — soft output-bucket boundaries for TRIBBLE regression

**Status:** measured, negative · **Started:** 2026-08-18

**Verdict in one line:** overlap's mechanism is real and large against per-bucket
consequent fitting (+0.17 test R² on concrete) and worth nothing against the
firing-weighted solve TRIBBLE ships, which is already a soft-boundary fit — and
the follow-ups explain why. Per-bucket solving makes each rule a far better local
approximator (local R² 0.61 → 0.94) and the blended model worse; compact
antecedent support does **not** close that gap (flat to four decimals from 93%
down to 62% of rules firing per row). The only arm that survives its own control
is blend sharpening on the *global* solve, +0.021 on bikeshare at 60/60 cells.
See [`RESULTS.md`](RESULTS.md).

Four stages: [`run_experiment.py`](run_experiment.py) (overlap),
[`run_local.py`](run_local.py) (per-bucket solving — fit or blend?),
[`run_support.py`](run_support.py) (compact support),
[`run_trapz.py`](run_trapz.py) (the trapezoid fitter's endpoint defect), with
`analyze*.py` generating the tables and
[`diagnose_trapz_defect.py`](diagnose_trapz_defect.py) regenerating the defect
evidence.

Stage 4 found a defect worth fixing upstream regardless of this experiment's
conclusions: `trapz_math_fast.fit_trapezoids_fast` puts the fitted support's left
edge exactly where `TrapezoidMembership.evaluate` returns zero, so any feature with
a mass point at its minimum loses those rows to every rule.

[`RESULTS.md`](RESULTS.md) is the laboratory record: the hypotheses as
registered before the run of record, and how each one scored. Generated tables
are in [`outputs/`](outputs/); the run of record is
[`outputs/results.json.gz.gz`](outputs/results.json.gz).

## Question

TRIBBLE's TSK regressor turns the target into rules by cutting it into
`n_output_buckets` and treating each bucket as one rule. Every per-bucket
quantity is then fitted on a **hard** slice — `y_bucket == r` and nothing else:

| quantity | where | slice |
|---|---|---|
| membership functions `(μ, σ)` per feature | `gauss_math.fit_gaussians` | `X[column][y == label_value]` |
| bucket centroid `y_bucket_mean[r]` | `regression.partition_output` | `y.groupby(y_bucket).mean()` |
| per-rule consequent polynomial (legacy path) | `regression.compute_*_order_corrections` | `y_train["y_bucket"] == rule_id` |

A sample a hair above a bucket edge counts entirely toward rule `r+1` and not at
all toward rule `r`, though it is all but indistinguishable from the sample a
hair below. The proposal under test:

> When you fit the consequent equations, each quantile should have a certain
> percentage of overlap with the data points of the neighbouring quantiles —
> single-sided at the two ends, where there is no neighbour. The result should be
> a smoother, better-fitting model than one built on hard boundary edges.

## What "a percentage of overlap" is taken to mean

Overlap is measured in **rank space as a fraction of the neighbour's point
count**: at `overlap=0.25`, rule `r` is also fitted on the 25% of bucket `r+1`'s
points nearest their shared edge, and the 25% of bucket `r-1`'s nearest theirs.
Rank space rather than value space for two reasons — it is what the request says
("a percentage of the data points"), and it behaves the same under
`output_partition="uniform"` and `"quantile"`, whose bucket widths differ by
construction. Bucket 0 borrows only upward and the last bucket only downward, so
the ends keep a hard outer edge; there is no data past them to blend with.

Two band profiles, because "overlap" does not fix the weighting:

- **flat** — borrowed points count as much as the rule's own (weight 1). The
  literal reading: the slices simply overlap.
- **ramp** — borrowed weights fall linearly from 1 at the shared edge to `1/m` at
  the band's far end, so the rule's membership in rank space is a **trapezoid**:
  a plateau over its own bucket, a shoulder into each neighbour. The smoother of
  the two.

`overlap=0` is the hard partition, so the whole sweep is anchored on the shipped
model rather than on a re-implementation of it — asserted, not assumed, in
[`test_overlap.py`](test_overlap.py).

## Where the overlap can be applied, and one thing that is already soft

This is the part that decided the arm list, and it is worth stating plainly
because it changes what the request can possibly buy.

**The consequent solver TRIBBLE actually ships is already a soft-boundary fit.**
`regression.solve_tsk_consequents_from_firing` does not fit rule `r` on bucket
`r`'s rows. It stacks a design block `w_r ⊙ [1 | basis(X)]` for *every* rule
across *every* sample, where `w_r` is that sample's normalized firing strength,
and solves one ridge system — so every sample already contributes to every rule's
coefficients, weighted continuously. That is exactly what "overlap the fitting
slices" is reaching for, arrived at from the other direction, and it is the
stronger version: the weights come from the antecedents (which is what predict
time also uses) rather than from a rank cut, and the solution is the exact
minimizer of the firing-weighted objective the model is scored on. There is no
hard edge left in it to soften. Overlapping the *data* cannot enter that solve at
all, because the solve has no per-rule row subset to widen.

So the hard edges live in three other places, and each gets its own switch:

| switch | what it softens | reaches predict time? |
|---|---|---|
| `overlap_antecedents` | the `(μ, σ)` fits, hence the firing strengths themselves | **yes** — this is the only one that does |
| `overlap_means` | the bucket centroids the consequents correct from | only through a *pinned* rung (see below) |
| `consequent_fit="local"` | per-rule polynomial fitted on its own overlapped slice | yes, and it replaces the global solve |

Plus one arm that is the same intent expressed as a penalty rather than as shared
rows — `fusion_reg` adds `λ Σ_r ‖c_{r+1} − c_r‖²` to the global solve, pulling
adjacent rules' correction coefficients toward agreement, which is what
unbounded data sharing between neighbours converges to. It keeps the closed form
and the exact firing-weighted structure, which the local arm gives up.

`overlap_means` deserves a note: the global solver **re-derives every unpinned
intercept** as part of its exact optimum, so a centroid handed to it survives
only where it is pinned. With `pin_extremes=False` the switch is not weakly
effective, it is arithmetically inert. That is pinned as a test
(`test_overlap_means_is_inert_under_the_global_solve_without_pinning`) rather
than left for a later reader to rediscover.

## Arms

Six, sharing one split, one scaler, one feature-selection pass and one ridge
strength per cell, so the only thing that differs is the overlap:

    baseline        hard buckets, global firing-weighted ridge      [the shipped model]
    soft-ante       overlapped membership-function fits, global solve
    local-hard      per-rule fit on the hard bucket only            [control for "local"]
    local-overlap   per-rule fit on the overlapped slice            [the literal request]
    full-overlap    overlapped antecedents *and* overlapped per-rule fits
    fusion          global solve + adjacent-consequent agreement penalty

`local-hard` is the arm that makes `local-overlap` readable. Going local is
itself a large change — it throws away the firing weighting — so without that
control any difference would be an unattributable mixture of "local instead of
global" and "soft instead of hard".

Swept: `overlap ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}` × `{flat, ramp}`,
`fusion_reg ∈ {1e-3 … 10}`, `n_output_buckets ∈ {3, 5, 7}` (more buckets means
more edges to soften), `tsk_order ∈ {2nd, full-2nd}`, ten seeds — the repo's seed
standard. `output_partition="quantile"`, since the request is stated about
quantiles; `l2_reg=1e-2`, the value `FuzzySystemsExperiments/concrete.py` runs at.

## Rungs

| dataset | shape | why |
|---|---|---|
| `concrete` | 1,030 × 8 | the reference target of `tribble-fis/consequent-plan.md` |
| `bikeshare` | 17,379 × 12 | large, and a heavily tied target — the case where a rank-space band splits runs of equal `y` |
| `synth-smooth` | 1,500 × 5 | one smooth surface: every bucket edge is arbitrary by construction. The idea's best case |
| `synth-piecewise` | 1,500 × 5 | three genuine regimes with jumps, aligned to the target's tertiles. The idea's worst case |

The two synthetic rungs are there to make the hypothesis falsifiable at the
mechanism level, which no real dataset can do: the claim is that the response
surface does not really change at the cuts, and that claim has a contrapositive.
If overlap helps on `synth-piecewise` too, it is not working for the stated
reason.

`bikeshare`'s shared loader leaks its target (`cnt` is exactly `casual +
registered`) and this driver carries its own leak-free wrapper, as
`experiments/fis-to-neural-net` does and for the same reason — archived proposal
tables quote the shared loader, so it is reported rather than patched. **WEC has
the same problem in a different costume** and is excluded from the default run;
see `RESULTS.md` and the note in `run_experiment.py`.

## Protocol

Each seed draws an 80/20 train/test split, then a 75/25 inner split of the train
fold. Every arm is fitted on the inner-train fold and scored on both validation
and test. `X` is unit-scaled and `y` standardized on the inner-train fold only —
FIS membership functions want bounded inputs (see the measured note in
`FuzzySystemsExperiments/concrete.py`; min-max, never z-score, for `X`).

**The overlap width is never chosen on test.** The headline table picks the width
and band shape per cell on validation R², then reports what that choice scored on
test, paired against the baseline in the same cell. Test R² against width is also
reported, labelled as a diagnostic: reading the best column off that curve is
selection on test, and it is not what a user would get.

## Files

| | |
|---|---|
| [`overlap.py`](overlap.py) | `overlap_weights`, the overlap-aware antecedent fit, the local and fused consequent solvers, and `OverlapTribbleRegressor` |
| [`test_overlap.py`](test_overlap.py) | degeneracy against `TribbleRegressor` and the mechanism properties |
| [`run_experiment.py`](run_experiment.py) | the sweep → `outputs/results.json.gz` |
| [`analyze.py`](analyze.py) | `results.json.gz` → the generated tables and figure |
| [`RESULTS.md`](RESULTS.md) | the record |

Nothing in `tribble-fis` is modified. `OverlapTribbleRegressor` is a parallel
implementation that calls the library's own `partition_output`,
`create_gaussian_membership_dict`, `fit_gaussians`, `tsk_firing_strengths`,
`build_consequent_features`, `solve_tsk_consequents_from_firing` and
`predict_tsk`; the baseline arm is the shipped code path, not a re-derivation of
it. If a result here justifies it, upstreaming is a second step — one that would
belong in `tribble-fis`, which this session cannot push to.

## Reproducing

```bash
python -m pytest experiments/overlap-modeling/test_overlap.py -q
python experiments/overlap-modeling/run_experiment.py          # ~35 min, 4 cores
python experiments/overlap-modeling/analyze.py
```

`--quick` runs concrete only, three seeds, one bucket count and order.
`OVERLAP_SEEDS=0,1,2` overrides the seed set.
