# The warm start on a slow-converging problem

Damped n=2 double-pendulum time-step operator, 43,400 train / 9,300 test rows, 3 inputs · seeds [0, 1, 2] · 20 epochs · batch 128.

This is the problem `find_slow_problem.py` measured at **3,444 updates** for a from-scratch network to reach R2 0.9 — against **25** for PhiUSIIL. The FIS fit and the conversion are charged to the hot arms.

| model | R2 at start | best R2 | setup s | s/update |
|---|---|---|---|---|
| tribble FIS (16 buckets, 60 MFs) | — | 0.8746 | 0.84 | — |
| nn-hot (lr 0.001) | 0.8747 | 0.9352 | 1.65 | 1.96 ms |
| nn-hot-anova (lr 0.003) | 0.8785 | 0.9361 | 0.91 | 2.10 ms |
| nn-quantile (lr 0.001) | 0.9387 | 0.9387 | 0.64 | 1.99 ms |
| nn-he (lr 0.01) | -2.1208 | 0.8773 | 0.00 | 2.08 ms |

Conversion: 181 hidden units. The projection seed starts at R2 0.8747 against the FIS's own 0.8746; the partial-dependence seed starts at 0.8785.

## Time to target — the whole question

Wall clock to *first* reach each R2, FIS fit and conversion included in the hot arms' totals.

| arm | R2 0.8 | R2 0.85 | R2 0.9 | R2 0.93 | R2 0.95 |
|---|---|---|---|---|---|
| `hot` | 1.71s | 1.81s | 2.45s | 7.85s | never |
| `hot-anova` | 0.91s | 1.10s | 1.96s | 5.81s | never |
| `quantile` | 0.64s | 0.64s | 0.64s | 0.64s | never |
| `he` | 9.85s (2/3) | 10.47s (2/3) | 11.62s (2/3) | never | never |

Updates to the same targets (setup excluded):

| arm | R2 0.8 | R2 0.85 | R2 0.9 | R2 0.93 | R2 0.95 |
|---|---|---|---|---|---|
| `hot` | 27 | 67 | 347 | 2,987 | never |
| `hot-anova` | 0 | 80 | 433 | 2,073 | never |
| `quantile` | 0 | 0 | 0 | 0 | never |
| `he` | 4,590 | 4,870 | 5,410 | never | never |

Speedup of `nn-hot` over each arm, wall clock at the same target:

| arm | R2 0.8 | R2 0.85 | R2 0.9 | R2 0.93 | R2 0.95 |
|---|---|---|---|---|---|
| vs `hot-anova` | 0.53x | 0.61x | 0.80x | 0.74x | hot never |
| vs `quantile` | 0.38x | 0.36x | 0.26x | 0.08x | hot never |
| vs `he` | 5.75x | 5.79x | 4.74x | only hot arrives | hot never |

