# FIS -> ReLU network: measured results

Seeds: [0, 1] · epochs: 150 · batch: 128 · commit `5e1aab2` · tribble-fis `5b92ec8`

## concrete

824 train / 206 test · 8 raw features · FIS keeps 8 · hidden width 232

| model | test RMSE | test R2 | epoch-0 RMSE | epochs to FIS parity | total s |
|---|---|---|---|---|---|
| tribble FIS | 6.977 ± 0.489 | 0.805 ± 0.018 | — | — | 0.22 ± 0.00 |
| tribble FIS (triangularized) | 32.086 ± 3.082 | -3.138 ± 0.603 | — | — | — |
| nn-hot (lr 0.0003) | 5.213 ± 0.545 | 0.887 ± 0.036 | 5.735 ± 0.489 | 0 (2/2) | 0.80 ± 0.01 |
| nn-quantile (lr 0.0003) | 4.390 ± 0.238 | 0.918 ± 0.019 | 5.175 ± 0.224 | 0 (2/2) | 0.45 ± 0.04 |
| nn-elm (lr 0.003) | 6.612 ± 0.121 | 0.773 ± 0.046 | 6.946 ± 0.060 | 0 (1/2) | 0.60 ± 0.05 |
| nn-he (lr 0.01) | 4.921 ± 0.062 | 0.905 ± 0.005 | 20.137 ± 0.740 | 7 (2/2) | 0.60 ± 0.05 |
| nn-quantile-all (lr 0.0003) | 4.390 ± 0.238 | 0.918 ± 0.019 | 5.175 ± 0.224 | 0 (2/2) | 0.44 ± 0.03 |
| nn-he-all (lr 0.01) | 5.097 ± 0.044 | 0.903 ± 0.003 | 19.809 ± 5.139 | 7 (2/2) | 0.59 ± 0.06 |

Cost to reach the quality the best randomly-initialized arm ends at (4.921 RMSE):

| model | epochs | wall-clock s (FIS fit charged to nn-hot) |
|---|---|---|
| nn-hot | 15.0 ± 0.0 (1/2 seeds) | 0.29 ± 0.00 |
| nn-quantile | 23.5 ± 23.5 (2/2 seeds) | 0.08 ± 0.08 |
| nn-elm | never (0/2) | — |
| nn-he | 102.5 ± 27.5 (2/2 seeds) | 0.40 ± 0.08 |
| nn-quantile-all | 23.5 ± 23.5 (2/2 seeds) | 0.07 ± 0.07 |
| nn-he-all | 89.5 ± 40.5 (2/2 seeds) | 0.34 ± 0.13 |

Conversion fidelity (read-out solved against the FIS's own predictions rather than the labels): relative RMSE 0.176 ± 0.011 of the FIS output's own standard deviation.

