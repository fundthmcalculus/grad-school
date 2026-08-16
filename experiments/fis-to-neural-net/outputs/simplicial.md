# The tetrahedral construction

Seeds: [0, 1, 2, 3, 4]. **fidelity** is the converted model's RMSE against the FIS it came from, relative to that FIS output's own standard deviation: 0 means the conversion reproduces the FIS exactly. The first-order additive construction is the number to beat.

## synth1d — 480 rows, 1 features

FIS test RMSE 0.057, fit in 0.03 s. Additive seed: fidelity **0.030**, 45 units, 0.01 s.

### Where the consequents come from (full-dimensional grid)

| K | vertices | dense grid | rows/vertex | `c_v = FIS(v)` | support-weighted | projected |
|---|---|---|---|---|---|---|
| 2 | 4 | 3 | 120.00 | 0.569 | 0.643 | 0.246 |
| 4 | 6 | 5 | 80.00 | 0.350 | 0.415 | 0.151 |
| 8 | 10 | 9 | 48.00 | 0.207 | 0.171 | 0.086 |

### Hybrid: additive main effects + tetrahedral interactions on the top *k* features

Grid resolution is chosen automatically to keep roughly 10 rows behind every vertex.

| k | K | vertices | dense grid | rows/vertex | ReLU units | depth | fidelity | test RMSE | +1 solve | total s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 26 | 25 | 18.5 | 78 | 3 | **0.021** | 0.056 | 0.054 | 0.04 |

## concrete — 824 rows, 8 features

FIS test RMSE 7.138, fit in 0.10 s. Additive seed: fidelity **0.313**, 244 units, 0.06 s.

### Where the consequents come from (full-dimensional grid)

| K | vertices | dense grid | rows/vertex | `c_v = FIS(v)` | support-weighted | projected |
|---|---|---|---|---|---|---|
| 2 | 1020 | 6.56e+03 | 0.81 | 1.806 | 0.382 | 0.444 |
| 4 | 2886 | 3.91e+05 | 0.29 | 1.668 | 0.367 | 0.371 |
| 8 | 5162 | 4.3e+07 | 0.16 | 1.420 | 0.643 | 0.698 |

### Hybrid: additive main effects + tetrahedral interactions on the top *k* features

Grid resolution is chosen automatically to keep roughly 10 rows behind every vertex.

| k | K | vertices | dense grid | rows/vertex | ReLU units | depth | fidelity | test RMSE | +1 solve | total s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 26 | 25 | 31.7 | 78 | 3 | **0.264** | 7.584 | 7.115 | 0.17 |
| 2 | 8 | 63 | 81 | 13.3 | 317 | 3 | **0.257** | 7.401 | 7.009 | 0.17 |
| 3 | 3 | 62 | 64 | 13.4 | 431 | 4 | **0.265** | 7.470 | 7.650 | 0.17 |
| 4 | 2 | 92 | 81 | 9.0 | 828 | 4 | **0.597** | 11.498 | 25.849 | 0.17 |
| 5 | 2 | 204 | 243 | 4.0 | 2248 | 5 | **0.642** | 11.795 | 19.464 | 0.18 |

## wec — 1854 rows, 12 features

FIS test RMSE 845937.867, fit in 0.83 s. Additive seed: fidelity **1.471**, 349 units, 0.42 s.

### Where the consequents come from (full-dimensional grid)

| K | vertices | dense grid | rows/vertex | `c_v = FIS(v)` | support-weighted | projected |
|---|---|---|---|---|---|---|
| 2 | 573 | 5.31e+05 | 3.31 | 8.694 | 0.974 | 0.897 |
| 4 | 915 | 2.44e+08 | 2.17 | 8.636 | 1.003 | 0.821 |
| 8 | 1788 | 2.82e+11 | 1.18 | 5.758 | 1.006 | 0.818 |

### Hybrid: additive main effects + tetrahedral interactions on the top *k* features

Grid resolution is chosen automatically to keep roughly 10 rows behind every vertex.

| k | K | vertices | dense grid | rows/vertex | ReLU units | depth | fidelity | test RMSE | +1 solve | total s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 16 | 25 | 121.4 | 48 | 3 | **1.169** | 915061.038 | 626736.706 | 1.26 |
| 2 | 24 | 70 | 625 | 28.1 | 348 | 3 | **1.278** | 1066440.409 | 1043532.145 | 1.27 |
| 3 | 21 | 152 | 11340 | 12.7 | 1065 | 4 | **1.744** | 1593046.400 | 831022.112 | 1.30 |
| 4 | 9 | 155 | 21121 | 12.0 | 1393 | 4 | **1.579** | 1214885.219 | 877994.907 | 1.31 |
| 5 | 5 | 167 | 17046 | 11.1 | 1839 | 5 | **1.301** | 1304110.883 | 921443.368 | 1.31 |

## bikeshare — 13903 rows, 12 features

FIS test RMSE 103.665, fit in 0.29 s. Additive seed: fidelity **1.101**, 272 units, 0.10 s.

### Where the consequents come from (full-dimensional grid)

| K | vertices | dense grid | rows/vertex | `c_v = FIS(v)` | support-weighted | projected |
|---|---|---|---|---|---|---|
| 2 | 8192 | 5.31e+05 | 1.70 | 1.606 | 0.534 | 1.137 |
| 4 | 8192 | 2.44e+08 | 1.70 | 1.316 | 0.623 | 0.685 |
| 8 | 8192 | 2.82e+11 | 1.70 | 1.029 | 0.914 | 1.841 |

### Hybrid: additive main effects + tetrahedral interactions on the top *k* features

Grid resolution is chosen automatically to keep roughly 10 rows behind every vertex.

| k | K | vertices | dense grid | rows/vertex | ReLU units | depth | fidelity | test RMSE | +1 solve | total s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 26 | 25 | 534.7 | 78 | 3 | **0.995** | 177.549 | 117.583 | 0.51 |
| 2 | 24 | 561 | 625 | 24.8 | 2804 | 3 | **0.972** | 175.197 | 130.586 | 1.18 |
| 3 | 16 | 922 | 4913 | 15.1 | 6457 | 4 | **1.174** | 201.982 | 154.489 | 1.91 |
| 4 | 6 | 783 | 2401 | 17.8 | 7049 | 4 | **0.959** | 172.658 | 118.041 | 2.19 |
| 5 | 4 | 906 | 3125 | 15.3 | 9964 | 5 | **0.911** | 167.615 | 104.917 | 2.49 |

