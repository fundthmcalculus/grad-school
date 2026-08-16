# N-CMAPSS full-dataset RUL analysis

One combined training set and one combined held-out test set, pooled from every N-CMAPSS file's own official train/test unit split. Condition-correction is fit per file on that file's own training units only; the scaler and per-unit RUL cap are fit once on the pooled training table only. No test-set information is used to fit anything.

NASA score is exponential in per-sample error, so a handful of large-outlier predictions dominate the sum -- on a large, heterogeneous pooled test set this can inflate the score by many orders of magnitude. Treat RMSE as the primary metric; NASA score is included for completeness, not as a normalized number.

## Files processed

| dataset | status | seconds |
|---|---|---:|
| DS01 | ok | 16.2 |
| DS02 | ok | 13.2 |
| DS03 | ok | 20.8 |
| DS04 | ok | 20.5 |
| DS05 | ok | 14.0 |
| DS06 | ok | 13.8 |
| DS07 | ok | 14.6 |
| DS08a | ok | 16.8 |
| DS08c | ok | 12.7 |
| DS08d | skipped: OSError('Unable to synchronously open file (truncated file: eof = 2885034848, sblock->base_addr = 0, stored_eof = 2885034880)') | nan |

## Pipeline: `honest`

- Training rows: 4,535  |  pooled test rows: 2,938
- **RMSE (combined test set): 15.95**  |  NASA score: 14,106
- Fit time: 1.06s

Per-dataset test RMSE (same trained model, broken out by source file):

| dataset | RMSE | n |
|---|---:|---:|
| DS01 | 13.69 | 341 |
| DS02 | 16.76 | 202 |
| DS03 | 14.61 | 438 |
| DS04 | 21.77 | 344 |
| DS05 | 14.94 | 327 |
| DS06 | 15.41 | 322 |
| DS07 | 17.77 | 344 |
| DS08a | 13.61 | 383 |
| DS08c | 13.12 | 237 |

## Pipeline: `best`

- Training rows: 50,000 (subsampled from 221,345 pooled rows, seed=42)  |  pooled test rows: 128,208
- **RMSE (combined test set): 17.70**  |  NASA score: 1,169,750,028,202,100,871,724,246,499,328
- Fit time: 23.89s

Per-dataset test RMSE (same trained model, broken out by source file):

| dataset | RMSE | n |
|---|---:|---:|
| DS01 | 11.91 | 13678 |
| DS02 | 9.16 | 6270 |
| DS03 | 12.30 | 21261 |
| DS04 | 17.33 | 18014 |
| DS05 | 12.43 | 12811 |
| DS06 | 32.75 | 12613 |
| DS07 | 18.73 | 14351 |
| DS08a | 17.19 | 18618 |
| DS08c | 16.87 | 10592 |

## Pipeline: `honest_full_tuned`

- Training rows: 4,535  |  pooled test rows: 2,938
- **RMSE (combined test set): 15.95**  |  NASA score: 14,106
- Fit time: 0.94s

Per-dataset test RMSE (same trained model, broken out by source file):

| dataset | RMSE | n |
|---|---:|---:|
| DS01 | 13.69 | 341 |
| DS02 | 16.76 | 202 |
| DS03 | 14.61 | 438 |
| DS04 | 21.77 | 344 |
| DS05 | 14.94 | 327 |
| DS06 | 15.41 | 322 |
| DS07 | 17.77 | 344 |
| DS08a | 13.61 | 383 |
| DS08c | 13.12 | 237 |

## Pipeline: `best_full_tuned`

- Training rows: 50,000 (subsampled from 221,345 pooled rows, seed=42)  |  pooled test rows: 128,208
- **RMSE (combined test set): 16.18**  |  NASA score: 24,123,921,303,008,108,544
- Fit time: 5.04s

Per-dataset test RMSE (same trained model, broken out by source file):

| dataset | RMSE | n |
|---|---:|---:|
| DS01 | 13.47 | 13678 |
| DS02 | 11.29 | 6270 |
| DS03 | 14.74 | 21261 |
| DS04 | 18.18 | 18014 |
| DS05 | 14.25 | 12811 |
| DS06 | 24.35 | 12613 |
| DS07 | 18.85 | 14351 |
| DS08a | 12.16 | 18618 |
| DS08c | 12.98 | 10592 |

## Pipeline: `best_full_de`

- Training rows: 50,000 (subsampled from 221,345 pooled rows, seed=42)  |  pooled test rows: 128,208
- **RMSE (combined test set): 15.56**  |  NASA score: 63,863,658,777
- Fit time: 27.07s

Per-dataset test RMSE (same trained model, broken out by source file):

| dataset | RMSE | n |
|---|---:|---:|
| DS01 | 12.03 | 13678 |
| DS02 | 9.06 | 6270 |
| DS03 | 12.25 | 21261 |
| DS04 | 17.33 | 18014 |
| DS05 | 12.39 | 12811 |
| DS06 | 20.89 | 12613 |
| DS07 | 18.70 | 14351 |
| DS08a | 15.63 | 18618 |
| DS08c | 16.62 | 10592 |

## Pipeline: `best_full_de_minmax`

- Training rows: 50,000 (subsampled from 221,345 pooled rows, seed=42)  |  pooled test rows: 128,208
- **RMSE (combined test set): 15.21**  |  NASA score: 593,988
- Fit time: 18.88s

Per-dataset test RMSE (same trained model, broken out by source file):

| dataset | RMSE | n |
|---|---:|---:|
| DS01 | 13.63 | 13678 |
| DS02 | 10.30 | 6270 |
| DS03 | 13.65 | 21261 |
| DS04 | 19.08 | 18014 |
| DS05 | 13.33 | 12811 |
| DS06 | 14.89 | 12613 |
| DS07 | 20.08 | 14351 |
| DS08a | 13.43 | 18618 |
| DS08c | 12.78 | 10592 |

Total wall time: 230.8s
