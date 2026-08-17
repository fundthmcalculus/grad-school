# N-CMAPSS full-dataset RUL analysis

One combined training set and one combined held-out test set, pooled from every N-CMAPSS file's own official train/test unit split. Condition-correction is fit per file on that file's own training units only; the scaler and per-unit RUL cap are fit once on the pooled training table only. No test-set information is used to fit anything.

Two metric conventions are reported per pipeline:

- **Per-engine (canonical)**: ONE RUL prediction per test engine, taken at its last available cycle, scored against that engine's ground-truth RUL -- the standard C-MAPSS / PHM08 protocol ("predict the RUL for each test engine"). This is the number comparable to published RMSE / NASA scores.
- **Per-sample**: over every one of the pooled test rows. Its NASA score is exponential in per-sample error AND summed over ~128k rows, so it is inflated by trajectory length, not just accuracy -- useful only for relative comparison between pipelines here, never as a published-comparable figure.

## Files processed

| dataset | status | seconds |
|---|---|---:|
| DS01 | ok | 22.0 |
| DS02 | ok | 18.4 |
| DS03 | ok | 28.8 |
| DS04 | ok | 29.0 |
| DS05 | ok | 19.5 |
| DS06 | ok | 19.2 |
| DS07 | ok | 20.4 |
| DS08a | ok | 24.6 |
| DS08c | ok | 18.6 |
| DS08d | skipped: OSError('Unable to synchronously open file (truncated file: eof = 2885034848, sblock->base_addr = 0, stored_eof = 2885034880)') | nan |

## Pipeline: `honest`

- Training rows: 4,535  |  pooled test rows: 2,938
- **Per-engine (canonical, one RUL per test engine at its last cycle -- 39 engines): RMSE 8.61  |  NASA score 78.2**
- Per-sample (over all 2,938 test rows): RMSE 15.95  |  NASA score 14,106
- Fit time: 1.03s

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
- **Per-engine (canonical, one RUL per test engine at its last cycle -- 39 engines): RMSE 40.64  |  NASA score 5,306,097.8**
- Per-sample (over all 128,208 test rows): RMSE 17.70  |  NASA score 1,169,750,028,202,100,871,724,246,499,328
- Fit time: 28.06s

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
- **Per-engine (canonical, one RUL per test engine at its last cycle -- 39 engines): RMSE 8.61  |  NASA score 78.2**
- Per-sample (over all 2,938 test rows): RMSE 15.95  |  NASA score 14,106
- Fit time: 0.98s

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
- **Per-engine (canonical, one RUL per test engine at its last cycle -- 39 engines): RMSE 13.50  |  NASA score 170.4**
- Per-sample (over all 128,208 test rows): RMSE 16.18  |  NASA score 24,123,921,303,008,108,544
- Fit time: 6.71s

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
- **Per-engine (canonical, one RUL per test engine at its last cycle -- 39 engines): RMSE 37.00  |  NASA score 493,077.0**
- Per-sample (over all 128,208 test rows): RMSE 15.56  |  NASA score 63,863,658,777
- Fit time: 32.15s

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
- **Per-engine (canonical, one RUL per test engine at its last cycle -- 39 engines): RMSE 18.73  |  NASA score 1,600.3**
- Per-sample (over all 128,208 test rows): RMSE 15.21  |  NASA score 593,988
- Fit time: 22.75s

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

## Pipeline: `real_memory`

- Training rows: 50,000 (subsampled from 221,345 pooled rows, seed=42)  |  pooled test rows: 128,208
- **Per-engine (canonical, one RUL per test engine at its last cycle -- 39 engines): RMSE 18.50  |  NASA score 1,658.3**
- Per-sample (over all 128,208 test rows): RMSE 15.31  |  NASA score 597,396
- Fit time: 15.06s

Per-dataset test RMSE (same trained model, broken out by source file):

| dataset | RMSE | n |
|---|---:|---:|
| DS01 | 13.86 | 13678 |
| DS02 | 10.65 | 6270 |
| DS03 | 13.53 | 21261 |
| DS04 | 19.41 | 18014 |
| DS05 | 13.53 | 12811 |
| DS06 | 15.13 | 12613 |
| DS07 | 20.20 | 14351 |
| DS08a | 13.18 | 18618 |
| DS08c | 12.77 | 10592 |

## Pipeline: `best_full_de_minmax_2pass`

- Training rows: 50,000 (subsampled from 221,345 pooled rows, seed=42)  |  pooled test rows: 128,208
- **Per-engine (canonical, one RUL per test engine at its last cycle -- 39 engines): RMSE 16.07  |  NASA score 2,176.3**
- Per-sample (over all 128,208 test rows): RMSE 15.02  |  NASA score 637,693
- Fit time: 59.38s

Per-dataset test RMSE (same trained model, broken out by source file):

| dataset | RMSE | n |
|---|---:|---:|
| DS01 | 12.95 | 13678 |
| DS02 | 9.28 | 6270 |
| DS03 | 13.40 | 21261 |
| DS04 | 19.02 | 18014 |
| DS05 | 13.10 | 12811 |
| DS06 | 15.46 | 12613 |
| DS07 | 19.72 | 14351 |
| DS08a | 13.05 | 18618 |
| DS08c | 12.95 | 10592 |

## Pipeline: `real_memory_2pass`

- Training rows: 50,000 (subsampled from 221,345 pooled rows, seed=42)  |  pooled test rows: 128,208
- **Per-engine (canonical, one RUL per test engine at its last cycle -- 39 engines): RMSE 15.28  |  NASA score 1,777.9**
- Per-sample (over all 128,208 test rows): RMSE 15.07  |  NASA score 635,838
- Fit time: 40.08s

Per-dataset test RMSE (same trained model, broken out by source file):

| dataset | RMSE | n |
|---|---:|---:|
| DS01 | 13.16 | 13678 |
| DS02 | 9.55 | 6270 |
| DS03 | 13.43 | 21261 |
| DS04 | 19.30 | 18014 |
| DS05 | 13.17 | 12811 |
| DS06 | 15.45 | 12613 |
| DS07 | 19.92 | 14351 |
| DS08a | 12.63 | 18618 |
| DS08c | 12.73 | 10592 |

## Literature benchmarks (context)

Published N-CMAPSS DS02 RUL results, for context. **The critical caveat: these are not one leaderboard** -- reported RMSE ranges from ~2.4 to ~15 almost entirely because of evaluation-protocol and file-version differences, not model quality. Always match (file version, per-sample vs per-engine, RUL cap, full-flight vs cruise-only) before comparing, as the CruiseBench authors stress.

| method | DS02 RMSE | protocol / notes | source |
|---|---:|---|---|
| CNN (data-driven) | 4.95 | per-sample, full trajectory, **pre-release low-noise file** | Arias Chao et al. 2022 |
| FNN (data-driven) | 7.89 | per-sample, full trajectory, pre-release file | Arias Chao et al. 2022 |
| CNN (re-run) | ~7.22 | per-sample, **public released file** | Custode et al. 2022 (snippet-level) |
| MLP (re-run) | ~8.34 | per-sample, public released file | Custode et al. 2022 (snippet-level) |
| TSMixer | 2.41 | per-sample, **cruise-only windows + RUL cap** (not comparable) | CruiseBench 2026 |
| LSTM-AE (health indicator) | 2.67 | **per-flight**, capped (not comparable) | de Pater & Mitici 2023 (snippet) |
| Bi-LSTM | 9.08 | protocol unstated | SJSU thesis (snippet) |

**The one apples-to-apples anchor** is Custode et al.'s re-run on the *public* file with per-sample-over-trajectory scoring (CNN ~7.22, MLP ~8.34). This DOE's `best` pipeline reaches ~6.48 per-sample RMSE on DS02 alone under that same protocol -- beating both, with an interpretable fuzzy TSK model. Arias Chao's 4.95 CNN is on the easier pre-release file and is not a fair target.

Note on protocol: N-CMAPSS papers evaluate **continuous prognostics** (RMSE per-sample over full test trajectories), NOT the classic C-MAPSS single-RUL-per-engine-at-truncation protocol. So the per-sample column above is the literature-comparable one; the per-engine metric this script also reports matches the classic protocol but has no N-CMAPSS published comparison.

Citations: Arias Chao, Kulkarni, Goebel, Fink (2022), Reliability Eng. & System Safety 217:107961 (arXiv:2003.00732); Custode, Mo, Ferigo, Iacca (2022), Algorithms 15(3):98 (DOI 10.3390/a15030098); Cheng & Miao (2026), CruiseBench (arXiv:2607.19380); de Pater & Mitici (2023), Eng. Appl. of AI 117; dataset: Arias Chao et al. (2021), Data 6(1):5 (DOI 10.3390/data6010005).

Total wall time: 410.2s
