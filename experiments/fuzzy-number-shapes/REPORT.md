# Does a pentagon footprint of uncertainty add anything?

**Decision: no.** Under the setup tested here, the pentagon footprint gives no reliable interval
benefit over a simple post-hoc residual interval. Its only measurable contribution is a slightly
better point prediction, and that is not robust to the optimiser.

Code and raw results: `experiments/fuzzy-number-shapes/`. Parts 1–5 are in `README.md`.

## Question

Is a pentagonal (5-knot) footprint of uncertainty for an interval type-2 TSK model worth
using? "Worth" here means: does it give better calibrated, sharper intervals than the obvious
alternatives, on the same data?

## Setup

- First-order TSK, 2 inputs, K = 5 fuzzy numbers per input, 500 training points, seeds 0–9.
- Target: `sin(2πx₁)(x₂+0.5) + 0.8·exp(−((x₁−0.6)²+(x₂−0.4)²)/0.02)`.
- Two noise models:
  - **x_hetero:** sd 0.02 where x₁ < 0.5, else 0.15. Observable from x.
  - **regime:** sd 0.02 or 0.15 by a hidden 0/1 regime that x does not reveal. Negative control.
- Intervals are scored with the Winkler score at α = 0.1 (nominal 90%) on held-out noisy targets.
- Footprints are calibrated by minimising Winkler on training data, footprint parameters only.

## Methods compared

| id | Point prediction | Interval |
|---|---|---|
| B1 | pentagon, type-1 | ± global 90% quantile of \|training residual\| |
| B2 | pentagon, type-1 | ± 90% quantile of \|residual\| over the 100 nearest training points |
| B3 | pentagon footprint (calibrated) | ± global 90% quantile of \|training residual\| |
| PENT | pentagon footprint (calibrated) | footprint, Karnik–Mendel bounds |
| GAUSS | Gaussian local footprint (calibrated) | footprint, Karnik–Mendel bounds |

B2 uses x, so it is the strongest non-footprint baseline on x_hetero. On regime, B1 and B2 coincide.

## Results

Mean over seeds 0–9. Winkler: lower is better.

| noise | method | coverage | Spearman(width, sd) | Winkler | RMSE (noise-free) |
|---|---|---:|---:|---:|---:|
| x_hetero | B1 | 0.858 | −0.00 | 0.592 | 0.0528 |
| x_hetero | **B2** | 0.856 | **+0.85** | **0.463** | 0.0528 |
| x_hetero | B3 | 0.866 | −0.01 | 0.581 | 0.0502 |
| x_hetero | PENT | 0.803 | +0.59 | 0.589 | 0.0502 |
| x_hetero | GAUSS | 0.814 | +0.70 | 0.638 | 0.0974 |
| regime | B1 | 0.861 | −0.01 | 0.576 | 0.0523 |
| regime | B2 | 0.859 | +0.00 | 0.581 | 0.0523 |
| regime | B3 | 0.872 | +0.01 | 0.565 | 0.0491 |
| regime | PENT | 0.836 | −0.01 | 0.649 | 0.0491 |
| regime | GAUSS | 0.821 | −0.01 | 0.825 | 0.0852 |

Paired per-seed Winkler difference (row minus column; negative favours the row):

| noise | comparison | mean diff | seeds row better |
|---|---|---:|---:|
| x_hetero | PENT vs B2 | **+0.126** | **1/10** |
| x_hetero | PENT vs B1 | −0.003 | 7/10 |
| x_hetero | PENT vs B3 | +0.008 | 7/10 |
| x_hetero | PENT vs GAUSS | −0.050 | 7/10 |
| regime | PENT vs B2 | +0.068 | 0/10 |
| regime | PENT vs GAUSS | −0.175 | 9/10 |

### Robustness check: longer optimiser (x_hetero, seeds 0–2)

The calibrated pentagon stopped at its 40-iteration cap in the main run, so I re-ran it with a
cap of 200.

| seed | PENT, cap 40 (main run) | PENT, cap 200 | B2 |
|---|---:|---:|---:|
| 0 | — | 0.466 (150 iterations, converged) | 0.476 |
| 1 | — | 0.727 (3 iterations, stalled) | 0.430 |
| 2 | — | 0.504 (107 iterations) | 0.454 |

Converging the optimiser helps on some seeds and not others. On seed 1 the optimiser stalls
almost immediately. The pentagon footprint is therefore fragile as well as unhelpful.

## Pre-registered decision rule

| # | Rule | Verdict |
|---|---|---|
| D1 | PENT has the lowest mean Winkler on x_hetero and beats B2 on ≥ 8/10 seeds | **Fail.** B2 is lowest (0.463). PENT beats B2 on 1/10 seeds. |
| D2 | Regime: no method tracks the hidden regime (\|ρ\| < 0.2) | **Pass.** All |ρ| ≤ 0.01. |
| D3 | PENT beats GAUSS on ≥ 7/10 seeds on x_hetero | **Pass,** 7/10 seeds. |

D1 is the rule that decides the question, and it fails. D3 shows the pentagon footprint beats the
Gaussian footprint, but the Gaussian footprint is also not a good competitor (it loses to B2 too).

## What drives the result

1. **Intervals.** B2 takes its width from the residual scale near each point. That tracks
   observable noise almost perfectly (ρ = +0.85). The pentagon footprint's width is only
   loosely related to noise (ρ = +0.59), and it is narrower and under-covering (0.80).
2. **Point prediction.** The pentagon footprint's point prediction is slightly better than
   the type-1 point (0.050 vs 0.053 RMSE). B3 uses that same point with a global interval and
   scores 0.581, essentially tied with PENT (0.589). So the footprint's advantage, if any, lives
   in the point prediction, not in its interval.
3. **Hidden noise.** Under regime noise no interval method tracks the regime. The footprint
   does not change that.

## Caveats

- **One target function**, K = 5, 500 training points. Results may differ elsewhere.
- **Consequent ridge 1e-2** was picked on seeds 0–1, which are also in the evaluation. The
  choice favours neither side, but it is tuned, not held out.
- **Calibration caps.** The main run stopped at 40 iterations. The longer-run check shows
  this matters on some seeds, so PENT's numbers are a lower bound on what calibration can achieve.
- **Baselines use in-sample residuals.** They are simple, which is the point, but a
  held-out residual calibration could tighten them slightly. It would not change the ranking
  by the margin observed (PENT vs B2 is +0.13).
- **B2 uses x.** On a problem where noise is not a function of x, B2 reduces to B1, and the
  footprint's relative standing could differ.

## Recommendation

Do not use the pentagon footprint for intervals in this kind of problem. If an interval is
needed, a type-1 point prediction plus a kNN residual quantile (B2) is simpler, faster, and
better calibrated in this setup. The pentagon footprint's small point-prediction gain is
under-tested and does not justify the optimiser fragility seen here.

If the pentagon footprint is revisited, the next test should be a held-out ridge and a fixed
optimiser budget tuned on separate seeds, with the point and interval effects separated.

## Follow-up: decoupled footprint (`decoupled_fou_study.py`)

The earlier calibrated footprint also moved the point prediction, which cost accuracy (noise-free RMSE 0.062 → 0.097). Here the point comes from the type-1 centres, and the footprint is calibrated for the interval only.

| noise | arm | coverage | Spearman(width, sd) | Winkler | RMSE |
|---|---|---:|---:|---:|---:|
| x_hetero | B2 kNN, Gaussian point | 0.857 | +0.85 | **0.470** | 0.0580 |
| x_hetero | B2 kNN, pentagon point | 0.856 | +0.85 | 0.463 | 0.0528 |
| x_hetero | DEC, Gaussian footprint | 0.793 | +0.69 | 0.611 | 0.0580 |
| x_hetero | DEC, pentagon footprint | 0.768 | +0.55 | 0.684 | 0.0528 |

- The point is now unchanged by the footprint (RMSE equals type-1, checked in code).
- The decoupled footprint still tracks observable noise, but its intervals lose to kNN on every seed (0/10). The gap is about 0.14 Winkler.
- Under-coverage is not overfitting. On seeds 0–2, training coverage (0.84–0.86) matches test coverage (0.80–0.84). The footprint family, under the current optimiser budget, cannot reach the nominal 90% even on its own training data.
- Conclusion: decoupling removes the point-accuracy cost, but a parametric footprint still does not beat a non-parametric residual quantile. A useful type-2 result would need a footprint that can reach nominal coverage: a per-rule or nonparametric footprint, or a hybrid with the kNN residual scale.

## Footprint shortfall: diagnosis and candidate fixes

The calibrated footprint under-covers (0.79 held-out against 0.90 nominal), its width tracks noise only partly, and it loses to kNN on every seed. Three things were checked.

**Diagnosis.** The centre is right: the calibrated interval covers the *noise-free* target 95–98% of the time. The misses are noise tails. On missed points |y − f| ≈ 0.15, close to the high-noise sd, so the intervals are too narrow in the tails. The footprint can only reweight the rule firings, which does not supply noise-level width.

| arm (x_hetero, mean of 10 seeds) | train coverage | held-out coverage | held-out Winkler | Spearman(width, sd) | train–test Winkler gap |
|---|---:|---:|---:|---:|---:|
| B2, kNN residual (reference) | – | 0.857 | **0.470** | +0.85 | – |
| DEC, footprint (baseline) | 0.841 | 0.793 | 0.611 | +0.69 | 0.13 |
| LONG, footprint, 300-iteration cap | 0.842 | 0.793 | 0.611 | +0.69 | 0.13 |
| RULE, per-rule footprint (50 params) | 0.868 | 0.808 | 0.607 | +0.66 | **0.17** |
| ICON, interval consequents (25 params) | **0.914** | 0.846 | 0.498 | **+0.83** | 0.14 |

**Candidate fixes**
1. **Longer optimiser (LONG): rejected.** Identical to the baseline, so the optimiser is not the cause.
2. **Per-rule footprint (RULE): rejected.** No better than the baseline and overfits more. Rule-level firing scales don't add noise-level width.
3. **Interval consequents (ICON): partial fix.** Each rule's consequent becomes an interval yhat ± d_r, and the output bounds are Karnik–Mendel over those. This adds noise-level width directly, and the rules form a grid over x, so local widths can follow local noise. It beats the baseline footprint on all 10 seeds (0.498 vs 0.611), tracks noise (ρ = +0.83), keeps the point unchanged, and does not track the hidden regime (ρ = +0.02). It still loses to kNN on 9 of 10 seeds (+0.029 mean).

**Pre-registered checks (`fou_solutions_study.py`, `icon_study.py`)**
- S1 (RULE reaches nominal coverage): **fail** (0.868 train, 0.808 held-out).
- S2 (LONG beats baseline coverage by ≥ 0.03): **fail** (identical).
- S3 (RULE beats DEC): **fail** (0.607 vs 0.611, a tie).
- S4 (RULE within 0.03 of kNN): **fail** (+0.137).
- S5 (RULE does not track hidden regime): **pass**.
- S6 (RULE overfit gap < 0.05): **fail** (0.17).
- I1 (ICON coverage in [0.85, 0.95]): **marginal fail** (train 0.914 passes, held-out 0.846 is just under 0.85).
- I2 (ICON beats DEC): **pass** (10/10 seeds).
- I3 (ICON within 0.03 of kNN): **marginal pass** on the mean (+0.029), but kNN wins 9/10 seeds.
- I4 (ICON tracks noise, ρ > 0.5): **pass** (+0.83).
- I5 (ICON does not track hidden regime): **pass** (+0.02).
- I6 (identical point across arms): **pass** (checked in code).

**Open issue with ICON.** Its train–test Winkler gap (0.14) is as large as the footprint's, and held-out coverage sits under nominal. The 25 half-widths partly fit training noise. Regularising them (for example, a smoothness prior across neighbouring rules or cross-validated shrinkage towards a global width) is the obvious next step. The kNN baseline has no such gap and still wins.

**Conclusion for type-2.** The best type-2 variant so far is interval consequents, not the firing footprint. It is still not better than the kNN residual interval, so the practical recommendation in this report stands: use kNN residuals unless the regularised interval-consequent model shows a clear gain on held-out data.
