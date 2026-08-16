# The warm start on a slow-converging problem — frictionless n=2

Frictionless n=2 double-pendulum time-step operator, 43,400 train / 9,300 test rows, 3 inputs · seeds [0, 1] · 10 epochs · batch 128.

R2 0.9 is unreachable here, not merely slow: widening the network from 128 to 1024 units moves its ceiling only from 0.725 to 0.771, and the FIS plateaus in the same place. Without damping the trajectories separate exponentially, so past some horizon in `t` the operator is not a learnable function of a 0.1-degree initial-condition grid. The targets below are the ones this problem admits. The FIS fit and the conversion are charged to the hot arms.

| model | R2 at start | best R2 | setup s | s/update |
|---|---|---|---|---|
| tribble FIS (32 buckets, 238 MFs) | — | 0.6696 | 3.12 | — |
| nn-hot (lr 0.001) | 0.5562 | 0.6254 | 5.71 | 7.24 ms |
| nn-hot-anova (lr 0.003) | 0.5580 | 0.6672 | 3.76 | 7.00 ms |
| nn-quantile (lr 0.003) | 0.5682 | 0.6576 | 0.45 | 2.74 ms |
| nn-he (lr 0.01) | -1.7690 | 0.6086 | 0.00 | 5.75 ms |

Conversion: 706 hidden units. The projection seed starts at R2 0.5562 against the FIS's own 0.6696; the partial-dependence seed starts at 0.5580.

## Time to target — the whole question

Wall clock to *first* reach each R2, FIS fit and conversion included in the hot arms' totals.

| arm | R2 0.4 | R2 0.5 | R2 0.6 | R2 0.65 | R2 0.7 |
|---|---|---|---|---|---|
| `hot` | 5.71s | 5.71s | 7.95s | never | never |
| `hot-anova` | 3.76s | 3.76s | 7.75s | 21.87s | never |
| `quantile` | 0.45s | 0.45s | 0.97s | 8.33s | never |
| `he` | 9.67s | 11.57s | 18.27s | never | never |

Updates to the same targets (setup excluded):

| arm | R2 0.4 | R2 0.5 | R2 0.6 | R2 0.65 | R2 0.7 |
|---|---|---|---|---|---|
| `hot` | 0 | 0 | 310 | never | never |
| `hot-anova` | 0 | 0 | 570 | 2,590 | never |
| `quantile` | 0 | 0 | 190 | 2,880 | never |
| `he` | 1,700 | 2,040 | 3,190 | never | never |

Speedup of `nn-hot` over each arm, wall clock at the same target:

| arm | R2 0.4 | R2 0.5 | R2 0.6 | R2 0.65 | R2 0.7 |
|---|---|---|---|---|---|
| vs `hot-anova` | 0.66x | 0.66x | 0.98x | hot never | hot never |
| vs `quantile` | 0.08x | 0.08x | 0.12x | hot never | hot never |
| vs `he` | 1.70x | 2.03x | 2.30x | hot never | hot never |

