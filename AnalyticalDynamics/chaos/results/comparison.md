# FIS vs the models reported in arXiv:2504.13453

RMSE is in the paper's scaled units (per-trajectory min-max to [0, 1]), so it
is a fraction of each trajectory's own angular range, not degrees. FIS rows
also carry the degree figure.

Two no-learning baselines appear in the holdout tables. They have no
parameters and fit nothing: the holdout IC 2.05 deg sits exactly between the
trained ICs 2.0 and 2.1 deg, so `bracket midpoint` averages those two scaled
trajectories and `nearest trained IC` copies one of them. Any learned model
that does not beat them has demonstrated grid interpolation, not dynamics.

Each cell is scored with the FIS configuration that wins *that* cell; the
configuration is named under the table. The trained-IC and held-out-IC optima
are at opposite ends of the rule-count range, so no single configuration is
best in both.

## double pendulum, frictionless, trained IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | **FIS (ours)** | 0.01264 | 0.998144 |
| 2 | LSTM (paper) | 0.02701 | 0.991527 |
| 3 | GRU (paper) | 0.03838 | 0.982088 |
| 4 | BIRNN (paper) | 0.04015 | 0.980485 |
| 5 | VRNN (paper) | 0.04073 | 0.981180 |
| 6 | FFNN (paper) | 0.0601 | 0.963700 |
| 7 | MLP (paper) | 0.07547 | 0.938207 |
| 8 | SRNN (paper) | 0.0776 | 0.927200 |
| 9 | AR (paper) | 0.1415 | 0.769966 |

FIS configuration: `nb300_full-2nd_g0_uniform_raw_probability_l21e-09`. Rank by RMSE: **1 of 9**. RMSE in degrees: 4.97.

## double pendulum, frictionless, holdout IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | **FIS (ours)** | 0.1839 | 0.619944 |
| 2 | _bracket midpoint (no learning)_ | 0.2256 | 0.439762 |
| 3 | LSTM (paper) | 0.26 | 0.230000 |
| 4 | _nearest trained IC (no learning)_ | 0.3644 | -0.421687 |

FIS configuration: `nb40_1st_g8_uniform_raw_probability`. Rank by RMSE: **1 of 4**. RMSE in degrees: 118.78.

## double pendulum, friction, trained IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | **FIS (ours)** | 0.005503 | 0.999576 |
| 2 | LSTM (paper) | 0.009546 | 0.998731 |
| 3 | GRU (paper) | 0.01496 | 0.996853 |
| 4 | VRNN (paper) | 0.01498 | 0.996860 |
| 5 | BIRNN (paper) | 0.02151 | 0.993485 |
| 6 | MLP (paper) | 0.02356 | 0.992257 |
| 7 | SRNN (paper) | 0.0389 | 0.978743 |
| 8 | FFNN (paper) | 0.04353 | 0.972693 |
| 9 | AR (paper) | 0.1147 | 0.613597 |

FIS configuration: `nb300_full-2nd_g0_uniform_raw_probability_l21e-09`. Rank by RMSE: **1 of 9**. RMSE in degrees: 1.26.

## double pendulum, friction, holdout IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | _bracket midpoint (no learning)_ | 0.0025 | 0.999909 |
| 2 | _nearest trained IC (no learning)_ | 0.01111 | 0.998209 |
| 3 | **FIS (ours)** | 0.01201 | 0.997890 |
| 4 | LSTM (paper) | 0.01529 | 0.996431 |
| 5 | VRNN (paper) | 0.01663 | 0.995972 |
| 6 | BIRNN (paper) | 0.01791 | 0.995323 |
| 7 | GRU (paper) | 0.01813 | 0.995240 |
| 8 | MLP (paper) | 0.02356 | 0.992594 |
| 9 | SRNN (paper) | 0.02816 | 0.987623 |
| 10 | FFNN (paper) | 0.03136 | 0.985698 |
| 11 | AR (paper) | 0.07868 | 0.919674 |

FIS configuration: `nb300_full-2nd_g0_uniform_raw_probability_l21e-09`. Rank by RMSE: **3 of 11**. RMSE in degrees: 3.11.

## triple pendulum, frictionless, trained IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | GRU (paper) | 0.017 | 0.985333 |
| 2 | **FIS (ours)** | 0.01908 | 0.993728 |
| 3 | LSTM (paper) | 0.022 | 0.982643 |
| 4 | BIRNN (paper) | 0.06 | 0.942589 |
| 5 | MLP (paper) | 0.084 | 0.889062 |
| 6 | VRNN (paper) | 0.086 | 0.890089 |
| 7 | FFNN (paper) | 0.11 | 0.804925 |
| 8 | SRNN (paper) | 0.11 | 0.927063 |
| 9 | AR (paper) | 0.15 | 0.699268 |

FIS configuration: `nb300_full-2nd_g0_uniform_raw_probability_l21e-09`. Rank by RMSE: **2 of 9**. RMSE in degrees: 7.29.

## triple pendulum, frictionless, holdout IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | **FIS (ours)** | 0.162 | 0.609140 |
| 2 | _bracket midpoint (no learning)_ | 0.1657 | 0.588166 |
| 3 | _nearest trained IC (no learning)_ | 0.2008 | 0.360535 |

The paper ran no model in this cell. FIS configuration: `nb40_full-2nd_g0_uniform_raw_probability`. Rank by RMSE: **1 of 3**. RMSE in degrees: 59.29.

## triple pendulum, friction, trained IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | **FIS (ours)** | 0.007181 | 0.998865 |
| 2 | GRU (paper) | 0.009113 | 0.998233 |
| 3 | LSTM (paper) | 0.01002 | 0.997459 |
| 4 | BIRNN (paper) | 0.0173 | 0.993719 |
| 5 | FFNN (paper) | 0.01807 | 0.992468 |
| 6 | SRNN (paper) | 0.02377 | 0.986479 |
| 7 | VRNN (paper) | 0.02436 | 0.987377 |
| 8 | MLP (paper) | 0.04407 | 0.970511 |
| 9 | AR (paper) | 0.09655 | 0.795768 |

FIS configuration: `nb300_full-2nd_g0_quantile_raw_probability`. Rank by RMSE: **1 of 9**. RMSE in degrees: 1.41.

## triple pendulum, friction, holdout IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | _bracket midpoint (no learning)_ | 0.0003585 | 0.999997 |
| 2 | _nearest trained IC (no learning)_ | 0.001865 | 0.999932 |
| 3 | **FIS (ours)** | 0.00477 | 0.999503 |
| 4 | GRU (paper) | 0.006497 | 0.999093 |
| 5 | LSTM (paper) | 0.008365 | 0.998542 |
| 6 | MLP (paper) | 0.01655 | 0.994005 |
| 7 | BIRNN (paper) | 0.01822 | 0.992288 |
| 8 | VRNN (paper) | 0.02121 | 0.990385 |
| 9 | FFNN (paper) | 0.02437 | 0.987363 |
| 10 | SRNN (paper) | 0.02966 | 0.981304 |
| 11 | AR (paper) | 0.09074 | 0.812740 |

FIS configuration: `nb300_full-2nd_g0_quantile_raw_probability`. Rank by RMSE: **3 of 11**. RMSE in degrees: 0.95.

## quintuple pendulum, frictionless, trained IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | **FIS (ours)** | 0.02666 | 0.986572 |

The paper ran no model in this cell. FIS configuration: `nb300_full-2nd_g0_uniform_raw_probability_l21e-09`. RMSE in degrees: 7.18.

## quintuple pendulum, frictionless, holdout IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | **FIS (ours)** | 0.2113 | 0.201385 |
| 2 | _bracket midpoint (no learning)_ | 0.2275 | 0.110068 |
| 3 | _nearest trained IC (no learning)_ | 0.2778 | -0.333036 |

The paper ran no model in this cell. FIS configuration: `nb40_1st_g8_uniform_raw_probability`. Rank by RMSE: **1 of 3**. RMSE in degrees: 171.19.

## quintuple pendulum, friction, trained IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | **FIS (ours)** | 0.02458 | 0.987446 |

The paper ran no model in this cell. FIS configuration: `nb300_full-2nd_g0_uniform_raw_probability_l21e-09`. RMSE in degrees: 3.32.

## quintuple pendulum, friction, holdout IC

| Rank | Model | RMSE | R^2 |
|---|---|---|---|
| 1 | _bracket midpoint (no learning)_ | 0.0515 | 0.943335 |
| 2 | _nearest trained IC (no learning)_ | 0.05199 | 0.945460 |
| 3 | **FIS (ours)** | 0.0672 | 0.901591 |

The paper ran no model in this cell. FIS configuration: `nb300_full-2nd_g0_quantile_raw_probability`. Rank by RMSE: **3 of 3**. RMSE in degrees: 11.35.
