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

## Part 2: gradient sources (`gradient_study.py`)

Same seeds and model, with four fitting arms. Full table: [`results_gradient.md`](results_gradient.md); per-fit rows: [`results_gradient.csv`](results_gradient.csv).

- **Analytic Jacobian.** Closed-form derivatives of the memberships (knot, centre and width) plus the consequent-solve derivative. Checked against central finite differences at init: max rel. error 5e-6 (gaussian/PL), elliptic excluded from the threshold because its sqrt edge makes the derivative diverge.
- **Synthetic gradient.** A 64-unit tanh MLP maps θ to dL/dθ. It is trained on true gradients from warm-up steps and every 4th step after that. Its prediction supplies the gradient on the other steps (173 true evals in 600 Adam steps).

Medians, test RMSE:

| shape | trf_fd | trf_analytic | adam_exact | adam_synth |
|---|---:|---:|---:|---:|
| triangle | 0.0368 | 0.0338 | 0.1048 | 0.0398 |
| trapezoid | 0.0368 | 0.0362 | 0.0410 | 0.0368 |
| pentagon | 0.0355 | 0.0353 | 0.0406 | 0.0779 |
| hexagon | 0.0345 | 0.0339 | 0.0491 | 0.0632 |
| heptagon | 0.0353 | 0.0337 | 0.0420 | 0.0748 |
| gaussian | 0.0332 | 0.0366 | 0.0333 | 0.0294 |
| elliptic | 0.0587 | 0.0576 | 0.1076 | 0.0733 |

Cost, mean gradient or Jacobian evals and seconds (trf_fd counts residual evals; the others count true gradients):

| arm | evals (range over shapes) | seconds (range) |
|---|---|---|
| trf_fd | 508–3276 | 3.1–48.8 |
| trf_analytic | 26–137 | 0.5–1.8 |
| adam_exact | 600 | 3.2–13.0 |
| adam_synth | 173 | 1.8–5.1 |

What this says:
1. **The analytic Jacobian is the win.** Same accuracy as finite differences (medians match within noise), about 20–50× fewer evals, and 10–30× faster wall clock.
2. **Synthetic gradients are not a replacement here.** They cut true gradient calls to 29% of the exact-Adam budget. They are not a consistent improvement: they beat exact Adam on triangle, gaussian and elliptic medians, and lose on pentagon, hexagon and heptagon. Predictor accuracy was not measured separately, so the cause is open.
3. **Adam arms are not tuned** (fixed lr 0.02, 600 steps). Read them as a fixed-budget comparison, not as the best first-order fit.
4. **Elliptic is the worst shape under every trf arm** (median ≈ 0.058), so the Part 1 ranking holds once fitting is reliable. Gaussian's Part 1 failure was a fitting failure.
5. Medians here differ slightly from Part 1 (e.g. triangle trf_fd 0.0368 vs 0.0348) because the consequent solve now uses a 1e-6 ridge for numerical stability (gaussian Phi has cond ≈ 1e8).

Run note: the first run was killed by a worker restart after 264 of 280 fits. The remaining 16 fits (seed 9, hexagon onward) were run with the same code path and merged with the logged rows, so the table covers all 280 fits.

## Part 3: interval type-2 (`type2_study.py`)

Gaussian footprint of uncertainty: upper σ = exp(s+w), lower σ = exp(s−w), w = exp(v). 30 antecedent parameters. Point prediction is Nie-Tan (average firing); output bounds are the lower- and upper-firing TSK outputs with the same consequents, ordered by min/max. Analytic Jacobian verified against central differences (rel. err 8e-6). Full table: [`results_type2.md`](results_type2.md); per-seed rows: [`results_type2.csv`](results_type2.csv).

| arm | test RMSE median | mean ± sd | interval coverage (noise-free test) | mean width | fitted mean w |
|---|---:|---:|---:|---:|---:|
| it2_collapse (control, w=0) | 0.0392 | 0.0426 ± 0.0132 | — | — | — |
| it2_fou | 0.0376 | 0.0382 ± 0.0055 | 0.898 (per seed 0.82–0.99) | 1.91 | 0.71 (init 0.10) |

Pre-registered checks, scored:

| # | Check | Verdict |
|---|---|---|
| T1 | Collapse control matches type-1 gaussian | **Pass.** Medians 0.0392 vs 0.0366 (type-1, `gradient_study.py`), within the 0.005 tolerance. |
| T2 | Free FOU lowers median RMSE vs control | **Weak.** Median −0.0016. The FOU's clearer effect is the tail: sd 0.013 → 0.006, and the control's seed-6-style outliers do not recur. |
| T3 | Coverage ≥ 90%; width tracks noise | **Coverage: borderline** (mean 0.898). **Width vs noise: not testable as written.** Training noise is homoscedastic (sd 0.05 everywhere), so there is no noise gradient to track. I should have caught that in the pre-registration. |
| T4 | FOU moves away from init | **Pass.** Fitted w averages 0.71 from init 0.10. |

What this says:
1. **The FOU helps mainly by making fits more reliable, not by improving the median.** That matches Parts 1–2: fit reliability, not expressive power, is the bottleneck.
2. **The intervals are too wide to be useful.** Mean width 1.9 on a target spanning about 2.5 units means high coverage is bought with width. Coverage alone does not show a calibrated interval. A proper test would use a calibrated predictive target (e.g. a held-out noisy sample with a width-coverage curve).
3. **Nie-Tan with min/max bounds is a crude output interval.** Proper Karnik–Mendel type reduction would give tighter, better-founded bounds; it is the next thing to add if interval quality matters.

## Part 3b: Karnik–Mendel type reduction and calibration (`km_study.py`)

Output interval by Karnik–Mendel (KM) type reduction, replacing the min/max bounds from Part 3. Heteroscedastic training noise (sd 0.02 where x₁<0.5, 0.15 elsewhere), so width-vs-noise is testable. KM is verified against brute-force switch-point search on 300 random cases (worst diff 0).

**Consequent ridge matters a lot.** At ridge 1e-6, consequent coefficients reach ~120 and KM intervals are ~35 wide, dominated by low-firing rules with ill-determined consequents ([`results_km_ridge1e-6.md`](results_km_ridge1e-6.md)). At ridge 1e-2 the coefficients stay under ~10. The 1e-2 value was chosen after a sweep on seeds 0–1, which are also in the evaluation, so it is tuned rather than held out.

Ridge 1e-2, 10 seeds ([`results_km.md`](results_km.md), [`results_km.csv`](results_km.csv)):

| bounds | cov. noise-free | cov. noisy | mean width | width (low sd) | width (high sd) | Spearman(width, sd) |
|---|---:|---:|---:|---:|---:|---:|
| Nie–Tan min/max | 0.660 | 0.532 | 0.32 | 0.23 | 0.41 | +0.32 |
| Karnik–Mendel | 0.970 | 0.920 | 1.47 | 1.37 | 1.58 | +0.08 |

Pre-registered checks:

| # | Check | Verdict |
|---|---|---|
| K1 | KM matches brute force | **Pass** (exact). |
| K2 | Collapse control has zero width | **Pass.** |
| K3 | KM width tracks local noise (ρ > 0.3) | **Fail.** ρ = +0.08. Nie–Tan reaches +0.32, but it under-covers. |
| K4 | KM covers ≥ 90% of noise-free target | **Pass** (0.970). |
| K5 | KM covers < 50% of noisy targets | **Fail.** Coverage 0.92. The KM intervals are wide enough to act as rough predictive intervals here, which is not what this model is for. |

What this says:
1. **KM with regularised consequents gives sensible-width intervals**, but they do not track local noise. The FOU here reflects a global spread, not a local one.
2. **Nie–Tan bounds are narrower and do track noise** (ρ +0.32), but under-cover (0.66). Neither bound is a calibrated predictive interval.
3. **The interval quality depends on a consequent-regularisation choice** that was not in the pre-registration. Any type-2 interval result should report the ridge alongside it.
