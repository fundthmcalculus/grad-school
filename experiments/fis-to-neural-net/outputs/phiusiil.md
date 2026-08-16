# PhiUSIIL at full scale

160,340 train / 28,296 val / 47,159 test · 50 numeric features · seeds [0, 1, 2] · 15 epochs · batch 512

Error rate, not accuracy: everything on this dataset lands near 99%, and a single threshold on `URLSimilarityIndex` alone scores 0.9914, so accuracy cannot separate the arms. All arms see the same log + min-max scaled inputs (fitted on the training fold); on raw features the FIS scores 0.730 rather than 0.994, so the scaling is load-bearing.

| model | test error | log loss | epoch-0 error | setup s | s/epoch | total s |
|---|---|---|---|---|---|---|
| tribble FIS (5 features, 25 MFs) | 0.0060 | 0.0308 | — | 2.8 | — | 2.8 |
| nn-hot (lr 0.001) | 0.0001 | 0.0017 | 0.0035 | 3.2 | 0.3 | 8 |
| nn-hot-all (lr 0.001) | 0.0001 | 0.0147 | 0.0034 | 3.2 | 0.5 | 11 |
| nn-hot-anova (lr 0.01) | 0.0003 | 0.0200 | 0.5721 | 2.8 | 0.3 | 7 |
| nn-quantile (lr 0.01) | 0.0003 | 0.0066 | 0.4279 | 1.3 | 0.2 | 4 |
| nn-he (lr 0.01) | 0.0003 | 0.0054 | 0.4279 | 1.3 | 0.3 | 6 |
| nn-he-all (lr 0.01) | 0.0002 | 0.0005 | 0.4279 | 1.3 | 0.5 | 8 |

Conversion: 72 hidden units, 0.4 s, seeded error 0.0035 before a single gradient step -- against the FIS's own 0.0060. The partial-dependence route that worked for regression seeds 0.5721 instead: the FIS saturates, so its profile averages are dominated by clipped extremes.

## Time to target

Wall clock to *first* reach each test error rate, with scaling, the FIS fit and the conversion all charged to `nn-hot`. Targets are absolute error rates every arm faces identically.

| arm | err <= 0.050 | err <= 0.020 | err <= 0.010 | err <= 0.007 | err <= 0.005 |
|---|---|---|---|---|---|
| `hot` | 3.2s | 3.2s | 3.2s | 3.2s | 3.2s |
| `hot-all` | 3.2s | 3.2s | 3.2s | 3.2s | 3.2s |
| `hot-anova` | 2.9s | 2.9s | 3.1s | 3.2s | 3.4s |
| `quantile` | 1.3s | 1.3s | 1.3s | 1.4s | 1.5s |
| `he` | 1.3s | 1.3s | 1.4s | 1.5s | 1.5s |
| `he-all` | 1.3s | 1.3s | 1.3s | 1.3s | 1.4s |

Speedup of `nn-hot` over each other arm at the same target:

| arm | err <= 0.050 | err <= 0.020 | err <= 0.010 | err <= 0.007 | err <= 0.005 |
|---|---|---|---|---|---|
| vs `hot-all` | 1.0x | 1.0x | 1.0x | 1.0x | 1.0x |
| vs `hot-anova` | 0.9x | 0.9x | 1.0x | 1.0x | 1.1x |
| vs `quantile` | 0.4x | 0.4x | 0.4x | 0.4x | 0.5x |
| vs `he` | 0.4x | 0.4x | 0.4x | 0.5x | 0.5x |
| vs `he-all` | 0.4x | 0.4x | 0.4x | 0.4x | 0.4x |

