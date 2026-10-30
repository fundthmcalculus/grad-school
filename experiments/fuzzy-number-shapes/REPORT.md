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
