| noise | arm | train coverage | held-out coverage | cov (low sd) | cov (high sd) | mean width | Spearman(width, sd) | Winkler held-out | Winkler train |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x_hetero | B2 | nan | 0.857 | 0.909 | 0.804 | 0.290 | +0.85 | 0.470 | nan |
| x_hetero | DEC | 0.841 | 0.793 | 0.885 | 0.701 | 0.285 | +0.69 | 0.611 | 0.477 |
| x_hetero | LONG | 0.842 | 0.793 | 0.884 | 0.702 | 0.285 | +0.69 | 0.611 | 0.476 |
| x_hetero | RULE | 0.868 | 0.808 | 0.903 | 0.713 | 0.286 | +0.66 | 0.607 | 0.441 |
| regime | B2 | nan | 0.861 | 0.988 | 0.734 | 0.356 | -0.00 | 0.588 | nan |
| regime | DEC | 0.846 | 0.819 | 0.956 | 0.681 | 0.354 | +0.00 | 0.673 | 0.565 |
| regime | LONG | 0.845 | 0.818 | 0.957 | 0.680 | 0.353 | -0.00 | 0.672 | 0.560 |
| regime | RULE | 0.858 | 0.813 | 0.946 | 0.678 | 0.346 | +0.00 | 0.676 | 0.510 |

Seeds 0..9, N_train=500. Point is the type-1 Gaussian centre model for every arm. Train columns are n/a (nan, shown as blank in the mean) for B2 because it has no fitted parameters on the training set. Winkler at alpha = 0.1.
