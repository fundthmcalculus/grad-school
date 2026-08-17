# N-CMAPSS RUL across all datasets

The DS02 best-case pipeline (`cmapss_ds02_rul.py`), applied unchanged to every N-CMAPSS file on its own train/test split -- a zero-shot generalization check, not a per-dataset best case. `per-sample RMSE` is the figure comparable to published baselines; `monotone RMSE` is the recommended output after the running-minimum clamp.

| dataset | test engines | per-sample RMSE | monotone RMSE | seconds |
|---|---:|---:|---:|---:|
| DS01-005 | 4 | 9.63 | 10.10 | 16.2 |
| DS02-006 | 3 | 6.48 | 6.45 | 9.3 |
| DS03-012 | 6 | 11.18 | 13.91 | 18.7 |
| DS04 | 4 | 15.78 | 16.16 | 21.5 |
| DS05 | 4 | 11.38 | 12.35 | 11.5 |
| DS06 | 4 | 30.37 | 27.83 | 12.4 |
| DS07 | 4 | 17.23 | 16.61 | 13.7 |
| DS08a-009 | 6 | 15.83 | 11.40 | 17.7 |
| DS08c-008 | 4 | 12.05 | 12.01 | 10.8 |
| **mean of 9** | | **14.44** | **14.09** | |

Skipped:
- DS08d-010: skipped: OSError
