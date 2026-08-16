# Time to quality

Seconds of wall clock for each initialization to *first* reach a target test RMSE, averaged over seeds. The TRIBBLE fit and the conversion are charged to the `hot` arms. Targets are multiples of the best RMSE any arm reached on that seed, so every arm faces the same bar.

`speedup` is he-all's time divided by that arm's time at the same target: >1 means faster to the same quality. Cells read `never` when the arm did not reach the target within the epoch budget; a parenthesised `(k/n)` means only k of n seeds got there and the mean covers those.

## synth1d

Wall-clock seconds to target (mean over seeds):

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 |
| `hot` | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| `quantile` | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 (9/10) |
| `elm` | never | never | never | never | never |
| `he` | never | never | never | never | never |
| `he-all` | never | never | never | never | never |

Speedup over `he-all` at the same target:

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | only arm to arrive | only arm to arrive | only arm to arrive | only arm to arrive | only arm to arrive |
| `hot` | only arm to arrive | only arm to arrive | only arm to arrive | only arm to arrive | only arm to arrive |
| `quantile` | only arm to arrive | only arm to arrive | only arm to arrive | only arm to arrive | only arm to arrive |
| `elm` | never | never | never | never | never |
| `he` | never | never | never | never | never |

Epochs to the same targets (mean over seeds):

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 3.20 | 0.00 | 0.00 | 2.20 | 13.10 |
| `hot` | 1.60 | 0.00 | 0.00 | 0.00 | 1.80 |
| `quantile` | 2.20 | 0.00 | 0.00 | 0.00 | 12.22 (9/10) |
| `elm` | never | never | never | never | never |
| `he` | never | never | never | never | never |
| `he-all` | never | never | never | never | never |

Fixed costs: `hot-analytic` setup 0.06s + 1ms/epoch, `hot` setup 0.06s + 1ms/epoch, `quantile` setup 0.00s + 1ms/epoch, `elm` setup 0.00s + 1ms/epoch, `he` setup 0.00s + 1ms/epoch, `he-all` setup 0.00s + 1ms/epoch

Best RMSE reached by any arm: 0.051; FIS: 0.056.

## concrete

Wall-clock seconds to target (mean over seeds):

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 0.32 (9/10) | 0.40 | 0.63 (7/10) | never | never |
| `hot` | 0.29 | 0.29 | 0.31 (4/10) | never | never |
| `quantile` | 0.00 | 0.00 | 0.01 | 0.10 | 0.18 (8/10) |
| `elm` | 0.01 (5/10) | 0.05 (6/10) | 0.03 (2/10) | never | never |
| `he` | 0.02 | 0.04 | 0.19 | 0.34 (6/10) | 0.32 (3/10) |
| `he-all` | 0.02 | 0.04 | 0.15 (9/10) | 0.31 (5/10) | 0.36 (2/10) |

Speedup over `he-all` at the same target:

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 0.1x | 0.1x | 0.2x | never | never |
| `hot` | 0.1x | 0.2x | 0.5x | never | never |
| `quantile` | inf | inf | 22.3x | 3.1x | 2.0x |
| `elm` | 3.7x | 0.8x | 5.5x | never | never |
| `he` | 1.0x | 1.0x | 0.8x | 0.9x | 1.1x |

Epochs to the same targets (mean over seeds):

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 11.67 (9/10) | 32.70 | 93.43 (7/10) | never | never |
| `hot` | 0.00 | 0.00 | 10.00 (4/10) | never | never |
| `quantile` | 0.00 | 0.00 | 2.40 | 34.50 | 61.38 (8/10) |
| `elm` | 1.80 (5/10) | 14.17 (6/10) | 6.50 (2/10) | never | never |
| `he` | 6.40 | 11.10 | 51.50 | 86.50 (6/10) | 89.00 (3/10) |
| `he-all` | 6.80 | 11.70 | 41.89 (9/10) | 79.80 (5/10) | 99.00 (2/10) |

Fixed costs: `hot-analytic` setup 0.28s + 4ms/epoch, `hot` setup 0.29s + 4ms/epoch, `quantile` setup 0.00s + 3ms/epoch, `elm` setup 0.00s + 4ms/epoch, `he` setup 0.00s + 4ms/epoch, `he-all` setup 0.00s + 4ms/epoch

Best RMSE reached by any arm: 4.501; FIS: 7.292.

## wec

Wall-clock seconds to target (mean over seeds):

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 2.45 (9/10) | never | never | never | never |
| `hot` | 2.49 | never | never | never | never |
| `quantile` | 0.00 | 0.00 (3/10) | 0.00 (1/10) | never | never |
| `elm` | 0.07 (8/10) | never | never | never | never |
| `he` | 0.00 | 0.18 (2/10) | never | never | never |
| `he-all` | 0.00 | 1.62 | 3.02 | 5.11 | 5.96 |

Speedup over `he-all` at the same target:

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 0.0x | never | never | never | never |
| `hot` | 0.0x | never | never | never | never |
| `quantile` | inf | inf | inf | never | never |
| `elm` | 0.0x | never | never | never | never |
| `he` | inf | 9.2x | never | never | never |

Epochs to the same targets (mean over seeds):

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 0.67 (9/10) | never | never | never | never |
| `hot` | 0.00 | never | never | never | never |
| `quantile` | 0.00 | 0.00 (3/10) | 0.00 (1/10) | never | never |
| `elm` | 5.50 (8/10) | never | never | never | never |
| `he` | 0.00 | 14.00 (2/10) | never | never | never |
| `he-all` | 0.00 | 33.90 | 63.20 | 106.30 | 124.20 |

Fixed costs: `hot-analytic` setup 2.47s + 13ms/epoch, `hot` setup 2.49s + 13ms/epoch, `quantile` setup 0.00s + 7ms/epoch, `elm` setup 0.00s + 13ms/epoch, `he` setup 0.00s + 13ms/epoch, `he-all` setup 0.00s + 48ms/epoch

Best RMSE reached by any arm: 18421.934; FIS: 905801.582.

## bikeshare

Wall-clock seconds to target (mean over seeds):

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 0.82 | 1.57 | 5.13 (9/10) | 4.83 (1/10) | 9.50 (1/10) |
| `hot` | 0.95 | 1.67 | 3.10 | 7.95 (7/10) | 7.37 (1/10) |
| `quantile` | 0.00 | 0.29 | 1.95 | never | never |
| `elm` | 2.13 | 6.61 (9/10) | 10.26 (3/10) | never | never |
| `he` | 0.72 | 2.19 | 4.03 | 6.43 | 8.95 |
| `he-all` | 0.69 | 2.36 | 4.09 | 6.97 | 9.28 |

Speedup over `he-all` at the same target:

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 0.8x | 1.5x | 0.8x | 1.4x | 1.0x |
| `hot` | 0.7x | 1.4x | 1.3x | 0.9x | 1.3x |
| `quantile` | 149.1x | 8.2x | 2.1x | never | never |
| `elm` | 0.3x | 0.4x | 0.4x | never | never |
| `he` | 1.0x | 1.1x | 1.0x | 1.1x | 1.0x |

Epochs to the same targets (mean over seeds):

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 2.10 | 12.00 | 58.78 (9/10) | 51.00 (1/10) | 108.00 (1/10) |
| `hot` | 0.10 | 9.30 | 27.30 | 89.43 (7/10) | 84.00 (1/10) |
| `quantile` | 0.10 | 6.20 | 41.80 | never | never |
| `elm` | 26.70 | 82.22 (9/10) | 133.00 (3/10) | never | never |
| `he` | 8.90 | 27.20 | 49.90 | 79.60 | 110.90 |
| `he-all` | 8.50 | 29.10 | 50.50 | 86.00 | 114.70 |

Fixed costs: `hot-analytic` setup 0.65s + 78ms/epoch, `hot` setup 0.94s + 78ms/epoch, `quantile` setup 0.00s + 46ms/epoch, `elm` setup 0.00s + 80ms/epoch, `he` setup 0.00s + 81ms/epoch, `he-all` setup 0.00s + 81ms/epoch

Best RMSE reached by any arm: 49.137; FIS: 103.110.

