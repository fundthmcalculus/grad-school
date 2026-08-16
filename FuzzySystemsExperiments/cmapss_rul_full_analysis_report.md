# N-CMAPSS full-dataset RUL analysis

One combined training set and one combined held-out test set, pooled from every N-CMAPSS file's own official train/test unit split. Condition-correction is fit per file on that file's own training units only; the scaler and per-unit RUL cap are fit once on the pooled training table only. No test-set information is used to fit anything.

NASA score is exponential in per-sample error, so a handful of large-outlier predictions dominate the sum -- on a large, heterogeneous pooled test set this can inflate the score by many orders of magnitude. Treat RMSE as the primary metric; NASA score is included for completeness, not as a normalized number.

## Files processed

| dataset | status | seconds |
|---|---|---:|
| DS01 | ok | 16.1 |
| DS02 | ok | 13.5 |
| DS03 | ok | 21.0 |
| DS04 | ok | 20.7 |
| DS05 | ok | 14.2 |
| DS06 | ok | 14.0 |
| DS07 | ok | 14.9 |
| DS08a | ok | 17.8 |
| DS08c | ok | 13.4 |
| DS08d | skipped: OSError('Unable to synchronously open file (truncated file: eof = 2885034848, sblock->base_addr = 0, stored_eof = 2885034880)') | nan |

## Pipeline: `honest`

- Training rows: 4,535  |  pooled test rows: 2,938
- **RMSE (combined test set): 15.95**  |  NASA score: 14,106
- Fit time: 0.99s

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
- Fit time: 23.56s

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
- Fit time: 0.95s

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
- Fit time: 5.13s

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
- Fit time: 27.08s

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

Total wall time: 212.1s
