| noise | variant | arm | coverage | cov (low sd) | cov (high sd) | mean width | width ratio hi/lo | Spearman(width, sd) | Spearman(width, group) | Winkler | RMSE (noise-free) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x_hetero | gauss_global | uncal | 0.940 | 0.984 | 0.897 | 1.604 | 1.16 | +0.08 | +0.08 | 1.714 | 0.0623 |
| x_hetero | gauss_global | cal | 0.811 | 0.902 | 0.720 | 0.372 | 2.17 | +0.47 | +0.47 | 0.681 | 0.0815 |
| x_hetero | gauss_local | uncal | 0.939 | 0.982 | 0.896 | 1.790 | 1.26 | +0.15 | +0.15 | 1.901 | 0.0625 |
| x_hetero | gauss_local | cal | 0.814 | 0.868 | 0.761 | 0.357 | 3.18 | +0.70 | +0.70 | 0.638 | 0.0974 |
| x_hetero | pent_fou | uncal | 0.710 | 0.928 | 0.492 | 0.250 | 0.98 | -0.04 | -0.04 | 0.806 | 0.0471 |
| x_hetero | pent_fou | cal | 0.803 | 0.870 | 0.736 | 0.316 | 2.71 | +0.59 | +0.59 | 0.589 | 0.0502 |
| regime | gauss_global | uncal | 0.915 | 0.978 | 0.852 | 1.326 | 0.98 | -0.01 | -0.01 | 1.475 | 0.0549 |
| regime | gauss_global | cal | 0.817 | 0.941 | 0.693 | 0.429 | 1.00 | -0.01 | -0.01 | 0.760 | 0.0833 |
| regime | gauss_local | uncal | 0.928 | 0.973 | 0.883 | 1.554 | 1.00 | +0.00 | +0.00 | 1.673 | 0.0570 |
| regime | gauss_local | cal | 0.821 | 0.937 | 0.705 | 0.499 | 0.99 | -0.01 | -0.01 | 0.825 | 0.0852 |
| regime | pent_fou | uncal | 0.715 | 0.884 | 0.546 | 0.281 | 0.99 | -0.01 | -0.01 | 0.776 | 0.0446 |
| regime | pent_fou | cal | 0.836 | 0.972 | 0.701 | 0.373 | 1.00 | -0.01 | -0.01 | 0.649 | 0.0491 |

Seeds 0..9, K=5, N_train=500. Test: 40x40 grid (x_hetero) or 1600 random points (regime), with fresh noisy targets. group = high noise (sd > 0.1). Winkler at alpha = 0.1. Calibration: L-BFGS-B on footprint parameters only, maxiter 40.
