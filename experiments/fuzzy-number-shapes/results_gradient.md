| arm | shape | test RMSE median | test RMSE mean ± sd | gradient evals (mean) | seconds (mean) |
|---|---|---:|---:|---:|---:|
| trf_fd | triangle | 0.0368 | 0.0503 ± 0.0425 | 1441 | 14.00 |
| trf_fd | trapezoid | 0.0368 | 0.0366 ± 0.0041 | 1746 | 21.73 |
| trf_fd | pentagon | 0.0355 | 0.0361 ± 0.0034 | 2237 | 35.69 |
| trf_fd | hexagon | 0.0345 | 0.0363 ± 0.0062 | 2818 | 48.76 |
| trf_fd | heptagon | 0.0353 | 0.0354 ± 0.0043 | 2248 | 48.65 |
| trf_fd | gaussian | 0.0332 | 0.0406 ± 0.0163 | 3276 | 17.80 |
| trf_fd | elliptic | 0.0587 | 0.0704 ± 0.0362 | 508 | 3.10 |
| trf_analytic | triangle | 0.0338 | 0.0386 ± 0.0118 | 34 | 0.87 |
| trf_analytic | trapezoid | 0.0362 | 0.0405 ± 0.0183 | 36 | 1.09 |
| trf_analytic | pentagon | 0.0353 | 0.0366 ± 0.0038 | 30 | 1.23 |
| trf_analytic | hexagon | 0.0339 | 0.0366 ± 0.0062 | 32 | 1.51 |
| trf_analytic | heptagon | 0.0337 | 0.0340 ± 0.0032 | 32 | 1.73 |
| trf_analytic | gaussian | 0.0366 | 0.0390 ± 0.0080 | 137 | 1.81 |
| trf_analytic | elliptic | 0.0576 | 0.0705 ± 0.0390 | 26 | 0.45 |
| adam_exact | triangle | 0.1048 | 0.1334 ± 0.1084 | 600 | 5.89 |
| adam_exact | trapezoid | 0.0410 | 0.1166 ± 0.1256 | 600 | 7.42 |
| adam_exact | pentagon | 0.0406 | 0.1159 ± 0.1249 | 600 | 9.43 |
| adam_exact | hexagon | 0.0491 | 0.0866 ± 0.0707 | 600 | 11.12 |
| adam_exact | heptagon | 0.0420 | 0.1415 ± 0.1398 | 600 | 12.95 |
| adam_exact | gaussian | 0.0333 | 0.0369 ± 0.0121 | 600 | 3.22 |
| adam_exact | elliptic | 0.1076 | 0.1586 ± 0.1493 | 600 | 3.55 |
| adam_synth | triangle | 0.0398 | 0.1215 ± 0.1874 | 173 | 2.66 |
| adam_synth | trapezoid | 0.0368 | 0.0413 ± 0.0141 | 173 | 3.23 |
| adam_synth | pentagon | 0.0779 | 0.0773 ± 0.0146 | 173 | 3.81 |
| adam_synth | hexagon | 0.0632 | 0.0698 ± 0.0294 | 173 | 4.54 |
| adam_synth | heptagon | 0.0748 | 0.1180 ± 0.1573 | 173 | 5.14 |
| adam_synth | gaussian | 0.0294 | 0.0336 ± 0.0148 | 173 | 1.82 |
| adam_synth | elliptic | 0.0733 | 0.2223 ± 0.2362 | 173 | 1.94 |

Seeds 0..9, K=5, N_train=500. Adam: 600 steps, lr 0.02. Synth: warm-up 30, true gradient every 4th step.
