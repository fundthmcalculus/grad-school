# Does the gating choice reach the conversion? (measured)

Seeds: [0, 1, 2, 3, 4]. **seed fidelity** is the analytic seed's RMSE against the FIS it was backed out of, relative to that FIS output's own standard deviation -- 0 means the seed reproduces the FIS exactly.

## synth1d

| norm family | FIS test RMSE | seed test RMSE | seed fidelity | hidden |
|---|---|---|---|---|
| min/max | 0.057 ± 0.003 | 0.056 | 0.031 | 44 |
| probability | 0.057 ± 0.003 | 0.055 | 0.030 | 44 |
| luk | 0.057 ± 0.002 | 0.055 | 0.030 | 44 |
| hamacher | 0.057 ± 0.003 | 0.055 | 0.031 | 44 |
| einstein | 0.057 ± 0.003 | 0.055 | 0.030 | 44 |

## concrete

| norm family | FIS test RMSE | seed test RMSE | seed fidelity | hidden |
|---|---|---|---|---|
| min/max | 6.829 ± 0.378 | 10.755 | 0.513 | 224 |
| probability | 7.138 ± 0.573 | 7.994 | 0.313 | 224 |
| luk | 34.029 ± 0.690 | 43.893 | 1.283 | 224 |
| hamacher | 6.831 ± 0.375 | 10.494 | 0.504 | 224 |
| einstein | 7.388 ± 0.786 | 9.014 | 0.450 | 224 |

