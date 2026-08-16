# FIS-aligned vertices, and interaction-chosen subspaces

Seeds: [0, 1, 2, 3, 4]. **fidelity** is the conversion's RMSE against the FIS it came from, relative to that FIS output's own standard deviation -- 0 means it reproduces the FIS exactly. Two choices vary independently: where the lattice vertices sit (`lattice` = data bounding box, `warped` = on the FIS's own knots) and how the correction's subspace is chosen (`importance` = top-k by differentiation score, `interaction` = features spanned by the highest-lift pairs).

## concrete — 824 rows, 8 features

Additive seed fidelity **0.313**. Warp costs 195 extra ReLU units across all axes.

| k | subspace | geometry | K | vertices | rows/vertex | ReLU units | fidelity | vs additive | s |
|---|---|---|---|---|---|---|---|---|---|
| 2 | importance | lattice | 8 | 63 | 13.3 | 317 | **0.257** | +18% | 0.01 |
| 2 | importance | warped | 8 | 68 | 12.2 | 394 | **0.264** | +16% | 0.01 |
| 2 | interaction | lattice | 8 | 63 | 13.3 | 313 | **0.265** | +15% | 0.01 |
| 2 | interaction | warped | 7 | 65 | 13.0 | 380 | **0.265** | +15% | 0.01 |
| 3 | importance | lattice | 3 | 62 | 13.4 | 431 | **0.265** | +15% | 0.01 |
| 3 | importance | warped | 3 | 68 | 12.9 | 557 | **0.255** | +19% | 0.01 |
| 3 | interaction | lattice | 3 | 68 | 12.1 | 477 | **0.352** | -12% | 0.01 |
| 3 | interaction | warped | 2 | 50 | 17.7 | 427 | **0.257** | +18% | 0.01 |
| 4 | importance | lattice | 2 | 92 | 9.0 | 828 | **0.597** | -90% | 0.01 |
| 4 | importance | warped | 2 | 103 | 8.0 | 1031 | **0.258** | +18% | 0.02 |
| 4 | interaction | lattice | 2 | 95 | 8.7 | 853 | **0.628** | -100% | 0.01 |
| 4 | interaction | warped | 2 | 102 | 8.1 | 1025 | **0.255** | +19% | 0.01 |

## wec — 1854 rows, 12 features

Additive seed fidelity **1.471**. Warp costs 275 extra ReLU units across all axes.

| k | subspace | geometry | K | vertices | rows/vertex | ReLU units | fidelity | vs additive | s |
|---|---|---|---|---|---|---|---|---|---|
| 2 | importance | lattice | 24 | 70 | 28.1 | 348 | **1.278** | +13% | 0.03 |
| 2 | importance | warped | 19 | 152 | 12.3 | 799 | **6.442** | -338% | 0.03 |
| 2 | interaction | lattice | 24 | 77 | 30.5 | 385 | **1.696** | -15% | 0.03 |
| 2 | interaction | warped | 18 | 148 | 13.2 | 776 | **2.854** | -94% | 0.03 |
| 3 | importance | lattice | 21 | 152 | 12.7 | 1065 | **1.744** | -19% | 0.05 |
| 3 | importance | warped | 6 | 155 | 12.1 | 1149 | **2.509** | -71% | 0.03 |
| 3 | interaction | lattice | 21 | 123 | 15.3 | 864 | **1.226** | +17% | 0.05 |
| 3 | interaction | warped | 6 | 152 | 12.8 | 1122 | **5.804** | -295% | 0.03 |
| 4 | importance | lattice | 9 | 155 | 12.0 | 1393 | **1.579** | -7% | 0.06 |
| 4 | importance | warped | 3 | 158 | 11.8 | 1509 | **1.611** | -10% | 0.03 |
| 4 | interaction | lattice | 14 | 168 | 11.2 | 1514 | **1.066** | +28% | 0.07 |
| 4 | interaction | warped | 3 | 158 | 11.7 | 1504 | **2.056** | -40% | 0.03 |

## bikeshare — 13903 rows, 12 features

Additive seed fidelity **1.101**. Warp costs 240 extra ReLU units across all axes.

| k | subspace | geometry | K | vertices | rows/vertex | ReLU units | fidelity | vs additive | s |
|---|---|---|---|---|---|---|---|---|---|
| 2 | importance | lattice | 24 | 561 | 24.8 | 2804 | **0.972** | +12% | 0.95 |
| 2 | importance | warped | 24 | 539 | 25.8 | 2768 | **0.932** | +15% | 0.47 |
| 2 | interaction | lattice | 24 | 561 | 24.8 | 2804 | **0.972** | +12% | 0.52 |
| 2 | interaction | warped | 24 | 539 | 25.8 | 2768 | **0.932** | +15% | 0.45 |
| 3 | importance | lattice | 16 | 922 | 15.1 | 6457 | **1.174** | -7% | 1.38 |
| 3 | importance | warped | 19 | 1065 | 13.4 | 7553 | **1.912** | -74% | 1.30 |
| 3 | interaction | lattice | 16 | 922 | 15.1 | 6457 | **1.174** | -7% | 0.84 |
| 3 | interaction | warped | 19 | 1065 | 13.4 | 7553 | **1.912** | -74% | 1.28 |
| 4 | importance | lattice | 6 | 783 | 17.8 | 7049 | **0.959** | +13% | 0.96 |
| 4 | importance | warped | 8 | 1116 | 12.9 | 10163 | **1.459** | -32% | 1.69 |
| 4 | interaction | lattice | 8 | 1347 | 10.3 | 12123 | **1.531** | -39% | 1.47 |
| 4 | interaction | warped | 7 | 825 | 16.9 | 7546 | **1.637** | -49% | 1.46 |

