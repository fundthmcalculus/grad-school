# FIS -> ReLU network: measured results

Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] · epochs: 150 · batch: 128 · commit `d830022` · tribble-fis `5b92ec8`

## synth1d

480 train / 120 test · 1 raw features · FIS keeps 1 · hidden width 44

| model | test RMSE | test R2 | epoch-0 RMSE | epochs to FIS parity | total s |
|---|---|---|---|---|---|
| tribble FIS | 0.056 ± 0.002 | 0.975 ± 0.002 | — | — | 0.05 ± 0.00 |
| tribble FIS (triangularized) | 0.057 ± 0.002 | 0.974 ± 0.003 | — | — | — |
| nn-hot-analytic (lr 0.003) | 0.054 ± 0.004 | 0.977 ± 0.003 | 0.055 ± 0.003 | 3 (10/10) | 0.22 ± 0.02 |
| nn-hot (lr 0.01) | 0.053 ± 0.004 | 0.978 ± 0.003 | 0.053 ± 0.004 | 2 (10/10) | 0.22 ± 0.02 |
| nn-quantile (lr 0.01) | 0.054 ± 0.004 | 0.977 ± 0.003 | 0.054 ± 0.003 | 2 (10/10) | 0.16 ± 0.01 |
| nn-elm (lr 0.01) | 0.199 ± 0.014 | 0.685 ± 0.048 | 0.280 ± 0.014 | never (0/10) | 0.17 ± 0.02 |
| nn-he (lr 0.01) | 0.196 ± 0.014 | 0.695 ± 0.048 | 0.439 ± 0.168 | never (0/10) | 0.17 ± 0.02 |
| nn-he-all (lr 0.01) | 0.195 ± 0.012 | 0.698 ± 0.043 | 0.399 ± 0.033 | never (0/10) | 0.16 ± 0.01 |

Cost to reach the quality the best randomly-initialized arm ends at (0.195 RMSE):

| model | epochs | wall-clock s (FIS fit charged to nn-hot) |
|---|---|---|
| nn-hot-analytic | 0.0 ± 0.0 (10/10 seeds) | 0.06 ± 0.01 |
| nn-hot | 0.0 ± 0.0 (10/10 seeds) | 0.06 ± 0.01 |
| nn-quantile | 0.0 ± 0.0 (10/10 seeds) | 0.00 ± 0.00 |
| nn-elm | 117.2 ± 16.0 (4/10 seeds) | 0.13 ± 0.02 |
| nn-he | 117.2 ± 34.6 (8/10 seeds) | 0.13 ± 0.04 |
| nn-he-all | 118.7 ± 26.8 (9/10 seeds) | 0.13 ± 0.03 |

Analytic seed's fidelity to the FIS it was backed out of: relative RMSE 0.030 ± 0.003 of the FIS output's own standard deviation (0 would mean the seed reproduces the FIS exactly).

## concrete

824 train / 206 test · 8 raw features · FIS keeps 8 · hidden width 221

| model | test RMSE | test R2 | epoch-0 RMSE | epochs to FIS parity | total s |
|---|---|---|---|---|---|
| tribble FIS | 7.292 ± 0.462 | 0.809 ± 0.023 | — | — | 0.19 ± 0.00 |
| tribble FIS (triangularized) | 27.901 ± 6.131 | -1.957 ± 1.182 | — | — | — |
| nn-hot-analytic (lr 0.003) | 5.611 ± 0.552 | 0.886 ± 0.024 | 8.035 ± 0.487 | 12 (9/10) | 0.84 ± 0.05 |
| nn-hot (lr 0.003) | 5.699 ± 0.413 | 0.883 ± 0.015 | 5.851 ± 0.447 | 0 (10/10) | 0.85 ± 0.05 |
| nn-quantile (lr 0.0003) | 4.739 ± 0.391 | 0.919 ± 0.012 | 5.471 ± 0.283 | 0 (10/10) | 0.43 ± 0.02 |
| nn-elm (lr 0.003) | 8.883 ± 5.735 | 0.608 ± 0.638 | 10.960 ± 11.286 | 2 (5/10) | 0.56 ± 0.04 |
| nn-he (lr 0.01) | 5.105 ± 0.342 | 0.906 ± 0.014 | 20.659 ± 4.119 | 6 (10/10) | 0.56 ± 0.05 |
| nn-he-all (lr 0.01) | 5.156 ± 0.321 | 0.904 ± 0.015 | 20.876 ± 3.510 | 7 (10/10) | 0.56 ± 0.05 |

Cost to reach the quality the best randomly-initialized arm ends at (5.052 RMSE):

| model | epochs | wall-clock s (FIS fit charged to nn-hot) |
|---|---|---|
| nn-hot-analytic | 123.0 ± 0.0 (1/10 seeds) | 0.74 ± 0.00 |
| nn-hot | 0.0 ± 0.0 (1/10 seeds) | 0.27 ± 0.00 |
| nn-quantile | 18.4 ± 16.2 (8/10 seeds) | 0.05 ± 0.05 |
| nn-elm | never (0/10) | — |
| nn-he | 100.8 ± 21.6 (9/10 seeds) | 0.39 ± 0.10 |
| nn-he-all | 102.6 ± 25.6 (9/10 seeds) | 0.38 ± 0.09 |

Analytic seed's fidelity to the FIS it was backed out of: relative RMSE 0.294 ± 0.058 of the FIS output's own standard deviation (0 would mean the seed reproduces the FIS exactly).

## wec

1854 train / 464 test · 301 raw features · FIS keeps 12 · hidden width 373

| model | test RMSE | test R2 | epoch-0 RMSE | epochs to FIS parity | total s |
|---|---|---|---|---|---|
| tribble FIS | 905801.582 ± 329042.765 | -89.728 ± 59.645 | — | — | 1.58 ± 0.07 |
| tribble FIS (triangularized) | 2150408.034 ± 1504455.258 | -687.568 ± 1001.267 | — | — | — |
| nn-hot-analytic (lr 0.0003) | 468904.265 ± 396863.043 | -33.129 ± 59.367 | 885752.070 ± 1009210.534 | 1 (9/10) | 4.40 ± 0.21 |
| nn-hot (lr 0.0003) | 235007.195 ± 124196.771 | -6.528 ± 8.911 | 228924.630 ± 126036.401 | 0 (10/10) | 4.45 ± 0.18 |
| nn-quantile (lr 0.0003) | 58390.657 ± 30839.544 | 0.561 ± 0.499 | 57949.687 ± 31991.099 | 0 (10/10) | 1.00 ± 0.07 |
| nn-elm (lr 0.01) | 647777.691 ± 560557.219 | -71.527 ± 104.809 | 727764.340 ± 574635.830 | 6 (8/10) | 1.93 ± 0.07 |
| nn-he (lr 0.01) | 66192.740 ± 23775.823 | 0.499 ± 0.289 | 157136.361 ± 56161.473 | 0 (10/10) | 1.92 ± 0.07 |
| nn-he-all (lr 0.003) | 18541.751 ± 1766.932 | 0.966 ± 0.005 | 133117.399 ± 31280.263 | 0 (10/10) | 7.20 ± 0.55 |

Cost to reach the quality the best randomly-initialized arm ends at (18541.751 RMSE):

| model | epochs | wall-clock s (FIS fit charged to nn-hot) |
|---|---|---|
| nn-hot-analytic | never (0/10) | — |
| nn-hot | never (0/10) | — |
| nn-quantile | never (0/10) | — |
| nn-elm | never (0/10) | — |
| nn-he | never (0/10) | — |
| nn-he-all | 143.4 ± 6.9 (10/10 seeds) | 6.88 ± 0.55 |

Analytic seed's fidelity to the FIS it was backed out of: relative RMSE 1.170 ± 0.705 of the FIS output's own standard deviation (0 would mean the seed reproduces the FIS exactly).

## bikeshare

13903 train / 3476 test · 12 raw features · FIS keeps 12 · hidden width 275

| model | test RMSE | test R2 | epoch-0 RMSE | epochs to FIS parity | total s |
|---|---|---|---|---|---|
| tribble FIS | 103.110 ± 2.519 | 0.680 ± 0.014 | — | — | 0.51 ± 0.03 |
| tribble FIS (triangularized) | 263.598 ± 2.503 | -1.089 ± 0.022 | — | — | — |
| nn-hot-analytic (lr 0.0003) | 57.471 ± 3.768 | 0.900 ± 0.013 | 180.195 ± 62.003 | 2 (10/10) | 12.33 ± 0.53 |
| nn-hot (lr 0.0003) | 53.475 ± 2.552 | 0.914 ± 0.008 | 99.864 ± 1.777 | 0 (10/10) | 12.68 ± 0.72 |
| nn-quantile (lr 0.003) | 56.968 ± 0.883 | 0.902 ± 0.003 | 100.080 ± 2.045 | 0 (10/10) | 6.94 ± 0.29 |
| nn-elm (lr 0.01) | 64.060 ± 5.078 | 0.876 ± 0.021 | 121.625 ± 1.381 | 27 (10/10) | 11.97 ± 0.58 |
| nn-he (lr 0.01) | 49.594 ± 1.021 | 0.926 ± 0.003 | 228.155 ± 44.154 | 9 (10/10) | 12.12 ± 0.51 |
| nn-he-all (lr 0.01) | 50.031 ± 1.660 | 0.925 ± 0.004 | 240.038 ± 33.001 | 8 (10/10) | 12.16 ± 0.48 |

Cost to reach the quality the best randomly-initialized arm ends at (49.233 RMSE):

| model | epochs | wall-clock s (FIS fit charged to nn-hot) |
|---|---|---|
| nn-hot-analytic | never (0/10) | — |
| nn-hot | never (0/10) | — |
| nn-quantile | never (0/10) | — |
| nn-elm | never (0/10) | — |
| nn-he | 140.2 ± 13.6 (6/10 seeds) | 11.30 ± 1.04 |
| nn-he-all | 135.2 ± 10.3 (5/10 seeds) | 10.91 ± 0.90 |

Analytic seed's fidelity to the FIS it was backed out of: relative RMSE 1.027 ± 0.442 of the FIS output's own standard deviation (0 would mean the seed reproduces the FIS exactly).

