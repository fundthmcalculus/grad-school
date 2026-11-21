| noise | method | coverage | cov (low sd) | cov (high sd) | mean width | width ratio hi/lo | Spearman(width, sd) | Winkler | RMSE (noise-free) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x_hetero | B1 | 0.858 | 0.999 | 0.718 | 0.352 | 1.00 | -0.00 | 0.592 | 0.0528 |
| x_hetero | B2 | 0.856 | 0.898 | 0.814 | 0.290 | 3.03 | +0.85 | 0.463 | 0.0528 |
| x_hetero | B3 | 0.866 | 0.999 | 0.734 | 0.363 | 1.00 | -0.01 | 0.581 | 0.0502 |
| x_hetero | PENT | 0.803 | 0.870 | 0.736 | 0.316 | 2.71 | +0.59 | 0.589 | 0.0502 |
| x_hetero | GAUSS | 0.814 | 0.868 | 0.761 | 0.357 | 3.18 | +0.70 | 0.638 | 0.0974 |
| regime | B1 | 0.861 | 0.994 | 0.728 | 0.351 | 1.00 | -0.01 | 0.576 | 0.0523 |
| regime | B2 | 0.859 | 0.993 | 0.725 | 0.351 | 1.00 | +0.00 | 0.581 | 0.0523 |
| regime | B3 | 0.872 | 0.996 | 0.748 | 0.366 | 1.00 | +0.01 | 0.565 | 0.0491 |
| regime | PENT | 0.836 | 0.972 | 0.701 | 0.373 | 1.00 | -0.01 | 0.649 | 0.0491 |
| regime | GAUSS | 0.821 | 0.937 | 0.705 | 0.499 | 0.99 | -0.01 | 0.825 | 0.0852 |

Paired per-seed Winkler difference (row minus column; negative favours the row). Mean, sd over seeds, seeds where row < column.

| noise | comparison | mean diff | sd | seeds row better |
|---|---|---:|---:|---:|
| x_hetero | PENT vs B2 | +0.126 | 0.133 | 1/10 |
| x_hetero | PENT vs B1 | -0.003 | 0.136 | 7/10 |
| x_hetero | PENT vs B3 | +0.008 | 0.138 | 7/10 |
| x_hetero | PENT vs GAUSS | -0.050 | 0.171 | 7/10 |
| x_hetero | B3 vs B2 | +0.118 | 0.017 | 0/10 |
| regime | PENT vs B2 | +0.068 | 0.061 | 0/10 |
| regime | PENT vs B1 | +0.074 | 0.062 | 0/10 |
| regime | PENT vs B3 | +0.085 | 0.067 | 0/10 |
| regime | PENT vs GAUSS | -0.175 | 0.313 | 9/10 |
| regime | B3 vs B2 | -0.017 | 0.021 | 8/10 |

Seeds 0..9, N_train=500. Test: 40x40 grid (x_hetero) or 1600 random points (regime). Winkler at alpha = 0.1. group = high noise (sd > 0.1). B2 uses k = 100 nearest training residuals.
