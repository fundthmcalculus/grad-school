# Goal G5, settled: uniform output cuts, and the study that could not see why

Concrete, ten seeds, closed-form consequents, everything except the partition held fixed.
Reproduce with `reproduce/tables/table_g5_output_partitioning.py`, which now runs 0th order
alongside 1st and 2nd.

## The three arms, at every consequent order

| arm | 0th order | 1st order | 2nd order |
|---|---:|---:|---:|
| uniform (equal width, data means) | **+0.394 ± 0.065** | 0.796 ± 0.018 | 0.841 ± 0.021 |
| quantile (equal frequency, data means) | +0.242 ± 0.070 | 0.789 ± 0.026 | 0.836 ± 0.025 |
| quantile + pinned extremes — *the old shipped default* | **−0.434 ± 0.241** | 0.787 ± 0.026 | 0.832 ± 0.027 |

At 1st and 2nd order the three arms span 0.009 and 0.005 against seed spreads of ±0.02
to ±0.03. Nothing separates. At 0th order they span **0.828**, which is 12× the largest
spread involved.

## Where the 0.828 comes from

Two things differ between the top and bottom rows, and the middle row separates them:

| change | Δ R² at 0th order |
|---|---:|
| equal-width → equal-frequency boundaries | −0.152 |
| data bucket means → extremes pinned to observed min/max | **−0.676** |

The pinning costs 4.4× what the boundary scheme costs. It is the dominant term, and the
old default was the only arm carrying it.

## The mechanism, read off the solved coefficients

`solve_tsk_consequents(..., pin_extremes=True)` — its default, and what every generator
here uses — holds the first and last rules' constant terms at the values `partition_output`
supplied, and solves the rest against the residual. The constraint is exact, so those two
numbers survive the solve unchanged:

```
0th order, quantile + pinned
  handed in:  [0.0,    0.4038, 1.0]
  as solved:  [0.0,    0.411,  1.0]     <- ends untouched
  occupancy:  {0: 344, 1: 343, 2: 343}

0th order, quantile (pin_extremes=False)
  handed in:  [0.195,  0.4038, 0.6534]
  as solved:  [0.195,  0.4303, 0.6534]
```

**At 0th order the constant term is the entire output of a rule.** So under the old default
the bottom rule emits the target's global minimum for a bucket of 344 points whose mean is
0.195, and the top rule emits the global maximum for 343 points whose mean is 0.653. Those
are extrema, not representative values, and with nothing else in the consequent there is
nothing to correct them with.

At 1st and 2nd order the same two ends stay pinned at 0.0 and 1.0, but the free middle
constant runs to −0.378 and −1.189 — outside the target's own [0, 1] range. The solve is
spending the intercepts as free parameters and paying for the bias with the linear terms.
That is the compensation, and it is why the pinning costs 0.002 to 0.009 there instead of
0.676.

## What this means for the study

`table_g5_output_partitioning` ran three arms, six configurations, 126 cells, and concluded
that every separation was smaller than the seed spread producing it — so G5 stayed open with
no scheme recommended. That conclusion was correct about what it measured and wrong about
what it implied, because **the study ran only 1st and 2nd order: the two regimes where the
consequent can absorb a bad centroid.** The partition binds hardest exactly where the study
did not look.

The 0th-order rows are now part of the table.

## A claim in the document that is false

Chapter 4 §4.3.2 and Appendix A.2.5 both said the hybrid is not a third option but a defect,
on the grounds that *the closed-form solve re-derives its own bucket means, so the pinned
values are discarded before they can reach inference.*

They are not discarded. The solved coefficients above show `0.0` and `1.0` intact after the
solve, and the hybrid arm's R² differs from pure quantile's in all six 1st/2nd-order
configurations of Table 4.2 (0.787 vs 0.789, 0.832 vs 0.836, 0.797 vs 0.795, 0.848 vs 0.850,
0.808 vs 0.806, 0.852 vs 0.853) — which could not happen if the pinning were inert. The
sentence describes `pin_extremes=False`, which is not the path any generator takes.

The hybrid was a real third scheme all along, and the worst of the three.

## Why pinning existed, and why uniform does not want it

Quantile's extreme buckets are wide, because equal frequency in a sparse tail means a broad
interval. A wide bucket's mean sits well inside it, so the model's reachable output range
shrinks and it can never predict near the true extremes. Pinning was the patch for that.

Equal-width buckets are narrow by construction, so their means already sit near the range
ends and there is nothing to patch. Pinning them to a single most extreme observation would
only import that observation's noise. Hence `pin_extremes=None` resolving to
`method == "quantile"`: each scheme's default is the arm that was actually measured.

## Uniform's own failure mode, and the remedy

Uniform starves. Equal-width bins in a sparse tail can catch almost nothing, and a bucket
holding two samples barely moves an aggregate error — so this is precisely the failure an
accuracy-only check cannot see. `partition_output` now raises `RuntimeWarning` when a bucket
holds fewer than three samples. On a lognormal target at three buckets it fires immediately,
occupancy `{0: 395, 1: 3, 2: 2}`.

Concrete does not trip it: at skew +0.42 the uniform occupancy is `{0: 378, 1: 520, 2: 132}`.

**The remedy is a target transform, not a different partition.** If the output range is
discontinuous or badly non-uniform, apply a monotone map — log, Box-Cox, rank — before
fitting, and invert it for reporting. Monotone maps preserve bucket order, so nothing
downstream that reasons about bucket order is affected. Switching back to quantile trades
starvation, which is loud and bounded, for instability, which is neither: the skew sweep of
Table 4.3 measured quantile's seed-to-seed deviation reaching ±4.3 and ±24.0 while uniform's
mean decayed smoothly with a bounded spread.

## The recommendation

Uniform, with a monotone target transform when the target is badly skewed.

It is the only arm that leaves the flat model usable, it costs nothing measurable at 1st or
2nd order, it tightens six of eight spreads on the full Chapter 4 reconciliation, and its
failure mode is detectable at fit time rather than at seed 7 of 10.
