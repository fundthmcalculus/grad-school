# Membership-shape study: is the trapezoid the knee?

**Status:** measured (part 1 of 3) · **Code:** [`shapes_study.py`](shapes_study.py) · **Results:** [`results.md`](results.md), [`results.csv`](results.csv), [`run.log`](run.log)

## Question

Does TSK model quality stop improving past the trapezoid, unless the shape is
Gaussian or elliptic? Shapes compared: triangle, trapezoid, pentagon,
hexagon, heptagon (piecewise-linear, 3–7 knots), gaussian and elliptic.

## Setup

- First-order TSK, 2 inputs, K=5 fuzzy numbers per input (25 rules), product t-norm, normalised firing.
- Consequents solved by least squares; antecedent shape parameters fitted by `scipy.optimize.least_squares` (finite-difference Jacobian). That is the baseline fitting method, not the method under test.
- Target: `sin(2πx₁)(x₂+0.5) + 0.8·exp(−((x₁−0.6)²+(x₂−0.4)²)/0.02)`, N=500 train with noise sd 0.05, noise-free 40×40 test grid.
- Seeds 0..9 (sample and init jitter), per the ten-seed protocol.

## Results

| shape | antecedent params | test RMSE mean ± sd | **test RMSE median** | train RMSE | seconds per fit |
|---|---:|---:|---:|---:|---:|
| triangle | 30 | 0.0639 ± 0.0565 | **0.0348** | 0.0447 | 7.1 |
| trapezoid | 40 | 0.0401 ± 0.0081 | 0.0389 | 0.0447 | 7.1 |
| pentagon | 50 | 0.0368 ± 0.0039 | 0.0361 | 0.0433 | 10.4 |
| hexagon | 60 | 0.0370 ± 0.0043 | 0.0371 | 0.0437 | 11.8 |
| heptagon | 70 | 0.0358 ± 0.0030 | 0.0354 | 0.0435 | 10.9 |
| gaussian | 20 | 0.3829 ± 1.0510 | 0.0358 | 0.0449 | 11.8 |
| elliptic | 20 | 0.1467 ± 0.2368 | 0.0657 | 0.0534 | 3.4 |

Per-seed test RMSE failures:

| shape | seeds with RMSE > 0.08 |
|---|---|
| triangle | 1 (0.171), 2 (0.170) |
| gaussian | 2 (0.092), 6 (3.373), 8 (0.108) |
| elliptic | 3 (0.170), 5 (0.813) |

Gaussian's 3.37 on seed 6 drives its mean.

## Pre-registered expectations, scored

| # | Expectation | Verdict |
|---|---|---|
| E1 | Triangle→trapezoid gives the largest drop in the n-gon family | **Not supported.** The mean drop (0.024) comes from triangle's two failed fits. On medians the triangle is best (0.0348 vs 0.0389). |
| E2 | Each knot past trapezoid gives a smaller gain, within about one sd of zero | **Supported on means, flat on medians.** Pentagon +0.003, hexagon −0.000, heptagon +0.001; all within one sd. Medians: 0.0389 → 0.0361 → 0.0371 → 0.0354. |
| E3 | Gaussian or elliptic reaches heptagon RMSE with fewer params | **Gaussian: yes on median** (0.0358 at 20 params, vs heptagon 0.0354 at 70). **Not on mean**, since one seed diverges. Elliptic: **no** (median 0.066). |
| E4 | Shape changes move RMSE (manipulation check) | **Yes.** Medians span 0.035–0.066 across shapes. |

## What this says

1. **At median, model quality is flat across the n-gon family and the Gaussian.** No knot count past the triangle buys accuracy here. The claim "trapezoid is the knee" is not supported by these data.
2. **The mean-based knee is an optimizer effect.** Triangle, gaussian and elliptic each fail on a minority of seeds, from which the local optimizer does not recover. Shape choice also changes how often the fit gets stuck, and that changes the mean.
3. **So the practical bottleneck is fitting reliability, not expressive power.** This is the motivation for the fitting work in part 2: a fit that converges reliably (a better gradient, or a smoother surrogate) may move the means more than the shape does.

## Caveats

- One target function, one grid size (K=5), one noise level. The shape ranking may change with the target.
- Test RMSE is measured against the noise-free target. Train RMSE is against noisy y, so it is not comparable to test RMSE.
- Fits were not restarted; a multi-start protocol would likely shrink the failure tails and could change E1 and E3.
- Wall-clock times are for a single laptop-class core with the finite-difference Jacobian and are not a benchmark.

## Not yet done

- **Part 2 (fitting method, "synthetic differentiation")**: term not defined in the repo; see the question in the reply.
- **Part 3 (type-2)**: depends on part 2's definition. Related prior work is in [`experiments/fis-drafter/fisdraft/learned_fou.py`](../fis-drafter/fisdraft/learned_fou.py), which fits a footprint of uncertainty by Winkler interval score.

Reproduce: `python shapes_study.py` (about 15 min, numpy and scipy).
