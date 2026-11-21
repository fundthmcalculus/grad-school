| noise | arm | train coverage | held-out coverage | cov (low sd) | cov (high sd) | mean width | Spearman(width, sd) | Winkler | RMSE (noise-free) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x_hetero | B2 | nan | 0.857 | 0.909 | 0.804 | 0.290 | +0.85 | 0.470 | 0.0580 |
| x_hetero | DEC | 0.841 | 0.793 | 0.885 | 0.701 | 0.285 | +0.69 | 0.611 | 0.0580 |
| x_hetero | ICON | 0.914 | 0.846 | 0.915 | 0.776 | 0.281 | +0.83 | 0.498 | 0.0580 |
| regime | B2 | nan | 0.861 | 0.988 | 0.734 | 0.356 | -0.00 | 0.588 | 0.0530 |
| regime | DEC | 0.846 | 0.819 | 0.956 | 0.681 | 0.354 | +0.00 | 0.673 | 0.0530 |
| regime | ICON | 0.901 | 0.849 | 0.978 | 0.720 | 0.358 | +0.02 | 0.619 | 0.0530 |

Paired per-seed Winkler difference (row minus column; negative favours the row).

| noise | comparison | mean diff | sd | seeds row better |
|---|---|---:|---:|---:|
| x_hetero | ICON vs DEC | -0.113 | 0.062 | 10/10 |
| x_hetero | ICON vs B2 | +0.029 | 0.021 | 1/10 |
| regime | ICON vs DEC | -0.054 | 0.042 | 9/10 |
| regime | ICON vs B2 | +0.031 | 0.014 | 0/10 |

Seeds 0..9, N_train=500. Point is the type-1 Gaussian centre model for every arm. Winkler at alpha = 0.1. B2 has no training interval, so its train columns are blank.
