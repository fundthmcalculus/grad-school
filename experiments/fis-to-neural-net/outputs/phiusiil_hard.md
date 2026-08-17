# PhiUSIIL at full scale — without the dominant feature

160,340 train / 28,296 val / 47,159 test · 49 numeric features · seeds [0, 1, 2] · 15 epochs · batch 512

**`URLSimilarityIndex` excluded.** Thresholded on its own it scores 0.9914 on the test fold, within a fraction of a point of every model below, so with it present the dataset cannot distinguish initializations at all. All arms see the same log + min-max scaled inputs (fitted on the training fold); on raw features the FIS scores 0.730 rather than 0.994, so the scaling is load-bearing.

| model | test error | log loss | epoch-0 error | setup s | s/epoch | total s |
|---|---|---|---|---|---|---|
| tribble FIS (5 features, 25 MFs) | 0.0488 | 0.1263 | — | 2.5 | — | 2.5 |
| nn-hot (lr 0.003) | 0.0116 | 0.0551 | 0.0510 | 3.1 | 0.2 | 7 |
| nn-hot-all (lr 0.003) | 0.0002 | 0.0184 | 0.0435 | 2.9 | 0.4 | 9 |
| nn-hot-anova (lr 0.003) | 0.0116 | 0.0422 | 0.0620 | 2.6 | 0.2 | 6 |
| nn-quantile (lr 0.01) | 0.0130 | 0.0376 | 0.4279 | 1.2 | 0.2 | 4 |
| nn-he (lr 0.01) | 0.0119 | 0.0335 | 0.4279 | 1.2 | 0.2 | 5 |
| nn-he-all (lr 0.01) | 0.0001 | 0.0006 | 0.4279 | 1.2 | 0.4 | 7 |

Conversion: 63 hidden units, 0.5 s, seeded error 0.0510 before a single gradient step -- against the FIS's own 0.0488. The partial-dependence route that worked for regression seeds 0.0620 instead: the FIS saturates, so its profile averages are dominated by clipped extremes.

## Time to target

Wall clock to *first* reach each test error rate, with scaling, the FIS fit and the conversion all charged to `nn-hot`. Targets are absolute error rates every arm faces identically.

| arm | err <= 0.050 | err <= 0.020 | err <= 0.010 | err <= 0.007 | err <= 0.005 |
|---|---|---|---|---|---|
| `hot` | 3.1s | 3.1s | never | never | never |
| `hot-all` | 2.9s | 2.9s | 2.9s | 2.9s | 2.9s |
| `hot-anova` | 2.6s | 2.6s | never | never | never |
| `quantile` | 1.2s | 1.2s | never | never | never |
| `he` | 1.2s | 1.2s | never | never | never |
| `he-all` | 1.2s | 1.2s | 1.2s | 1.2s | 1.2s |

Speedup of `nn-hot` over each other arm at the same target:

| arm | err <= 0.050 | err <= 0.020 | err <= 0.010 | err <= 0.007 | err <= 0.005 |
|---|---|---|---|---|---|
| vs `hot-all` | 0.9x | 0.9x | hot never | hot never | hot never |
| vs `hot-anova` | 0.8x | 0.8x | hot never | hot never | hot never |
| vs `quantile` | 0.4x | 0.4x | hot never | hot never | hot never |
| vs `he` | 0.4x | 0.4x | hot never | hot never | hot never |
| vs `he-all` | 0.4x | 0.4x | hot never | hot never | hot never |

