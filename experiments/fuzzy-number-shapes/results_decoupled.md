| noise | arm | coverage | cov (low sd) | cov (high sd) | mean width | width ratio hi/lo | Spearman(width, sd) | Winkler | RMSE (noise-free) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x_hetero | B2_gauss | 0.857 | 0.909 | 0.804 | 0.290 | 3.01 | +0.85 | 0.470 | 0.0580 |
| x_hetero | B2_pent | 0.856 | 0.898 | 0.814 | 0.290 | 3.03 | +0.85 | 0.463 | 0.0528 |
| x_hetero | DEC_gauss | 0.793 | 0.885 | 0.701 | 0.285 | 2.97 | +0.69 | 0.611 | 0.0580 |
| x_hetero | DEC_pent | 0.768 | 0.860 | 0.677 | 0.306 | 2.20 | +0.55 | 0.684 | 0.0528 |
| regime | B2_gauss | 0.861 | 0.988 | 0.734 | 0.356 | 1.00 | -0.00 | 0.588 | 0.0530 |
| regime | B2_pent | 0.859 | 0.993 | 0.725 | 0.351 | 1.00 | +0.00 | 0.581 | 0.0523 |
| regime | DEC_gauss | 0.819 | 0.956 | 0.681 | 0.354 | 1.00 | +0.00 | 0.673 | 0.0530 |
| regime | DEC_pent | 0.779 | 0.929 | 0.628 | 0.333 | 0.97 | -0.02 | 0.716 | 0.0523 |

Paired per-seed Winkler difference (row minus column; negative favours the row).

| noise | comparison | mean diff | sd | seeds row better |
|---|---|---:|---:|---:|
| x_hetero | DEC_gauss vs B2_gauss | +0.141 | 0.059 | 0/10 |
| x_hetero | DEC_pent vs B2_pent | +0.221 | 0.118 | 0/10 |
| x_hetero | DEC_pent vs DEC_gauss | +0.073 | 0.085 | 2/10 |
| x_hetero | B2_pent vs B2_gauss | -0.007 | 0.030 | 5/10 |
| regime | DEC_gauss vs B2_gauss | +0.085 | 0.046 | 0/10 |
| regime | DEC_pent vs B2_pent | +0.135 | 0.051 | 0/10 |
| regime | DEC_pent vs DEC_gauss | +0.043 | 0.078 | 4/10 |
| regime | B2_pent vs B2_gauss | -0.007 | 0.024 | 7/10 |

Seeds 0..9, N_train=500. Point is the type-1 centre model for every arm; only the interval differs. Winkler at alpha = 0.1. group = high noise (sd > 0.1).
