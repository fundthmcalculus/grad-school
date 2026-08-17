# N-CMAPSS RUL grid results

`cmapss_rul_best.py --grid`: the `honest` (physical sensors only, 18 channels) and `best` (physical + 2 virtual, the literature-matching 20-channel set) pipelines, run across every N-CMAPSS dataset file available locally.

Both pipelines use the exact hyperparameters found by the DOE's grid search on DS02 -- **not re-tuned per dataset**. This table is a zero-shot generalization check, not a per-dataset best case; a dataset-specific sweep would likely do better where RMSE is high.

NASA score is exponential in per-sample error (`exp(|error|/13)` or `exp(|error|/10)`), so a handful of large-outlier predictions dominate the sum and can inflate the score by many orders of magnitude on a dataset the model generalizes to poorly. Treat RMSE as the primary comparison metric across datasets; NASA score is included for completeness, not as a normalized cross-dataset number.

| dataset | pipeline | RMSE | NASA score | load (s) | correction (s) | aggregate (s) | fit (s) | total (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| N-CMAPSS_DS01-005.h5 | best | 9.67 | 32,720 | 2.5 | 3.6 | 6.1 | 21.34 | 33.7 |
| N-CMAPSS_DS01-005.h5 | honest | 11.53 | 916 | 2.9 | 3.4 | 2.7 | 0.17 | 9.2 |
| N-CMAPSS_DS02-006.h5 | best | 6.48 | 10,965 | 2.1 | 3.2 | 5.6 | 1.05 | 12.0 |
| N-CMAPSS_DS02-006.h5 | honest | 11.23 | 620 | 2.4 | 3.2 | 2.3 | 0.17 | 8.2 |
| N-CMAPSS_DS03-012.h5 | best | 11.21 | 54,783 | 3.6 | 5.0 | 8.5 | 16.96 | 34.6 |
| N-CMAPSS_DS03-012.h5 | honest | 13.83 | 1,497 | 3.8 | 5.1 | 3.7 | 0.24 | 12.8 |
| N-CMAPSS_DS04.h5 | best | 15.86 | 1,085,307 | 3.5 | 5.4 | 8.3 | 22.84 | 40.4 |
| N-CMAPSS_DS04.h5 | honest | 18.59 | 2,626 | 3.7 | 4.4 | 3.6 | 0.51 | 12.2 |
| N-CMAPSS_DS05.h5 | best | 11.40 | 32,027 | 2.2 | 3.2 | 5.9 | 6.56 | 17.9 |
| N-CMAPSS_DS05.h5 | honest | 11.97 | 864 | 2.5 | 3.0 | 2.5 | 0.22 | 8.3 |
| N-CMAPSS_DS06.h5 | best | 21.98 | 81,198,018,160 | 2.1 | 3.2 | 5.5 | 11.26 | 22.2 |
| N-CMAPSS_DS06.h5 | honest | 13.53 | 957 | 2.5 | 2.9 | 2.5 | 0.21 | 8.1 |
| N-CMAPSS_DS07.h5 | best | 17.29 | 63,374 | 2.5 | 3.4 | 6.2 | 20.06 | 32.5 |
| N-CMAPSS_DS07.h5 | honest | 16.88 | 1,458 | 2.7 | 3.3 | 2.7 | 0.26 | 8.9 |
| N-CMAPSS_DS08a-009.h5 | best | 18.98 | 5,200,287,607,274 | 2.9 | 3.8 | 7.1 | 18.11 | 32.3 |
| N-CMAPSS_DS08a-009.h5 | honest | 13.05 | 1,403 | 3.2 | 3.7 | 3.0 | 0.35 | 10.3 |
| N-CMAPSS_DS08c-008.h5 | best | 12.26 | 399,978 | 2.0 | 3.0 | 5.3 | 12.44 | 22.8 |
| N-CMAPSS_DS08c-008.h5 | honest | 10.63 | 542 | 2.4 | 2.9 | 2.3 | 0.20 | 7.7 |

## Skipped / failed

- N-CMAPSS_DS08d-010.h5 / honest: failed: subprocess exit code 1
- N-CMAPSS_DS08d-010.h5 / best: failed: subprocess exit code 1
