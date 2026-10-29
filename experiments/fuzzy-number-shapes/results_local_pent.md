| variant | RMSE median | RMSE mean ± sd | cov. noise-free | cov. noisy | mean width | width low sd | width high sd | Spearman(width, sd) | mean abs(v1) | mean w | undefined grid points |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gauss_global | 0.0644 | 0.0623 ± 0.0109 | 0.979 | 0.940 | 1.604 | 1.549 | 1.660 | +0.08 | 0.000 | 0.166 | 0.00% |
| gauss_local | 0.0635 | 0.0625 ± 0.0056 | 0.973 | 0.940 | 1.790 | 1.565 | 2.014 | +0.15 | 1.013 | 0.226 | 0.00% |
| pent_fou | 0.0477 | 0.0471 ± 0.0054 | 0.877 | 0.711 | 0.250 | 0.253 | 0.247 | -0.04 | 2.361 | 0.100 | 0.08% |
| pent_collapse | 0.0532 | 0.0528 ± 0.0075 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.00 | 50.000 | 0.000 | 0.00% |
| gauss_collapse | 0.0576 | 0.0580 ± 0.0086 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | +0.00 | 0.000 | 0.000 | 0.00% |

Seeds 0..9, K=5, N_train=500, heteroscedastic noise sd 0.02 (x1<0.5) / 0.15. Consequent ridge 1e-2. KM output bounds. mean abs(v1) is the fitted local slope for gauss_local and the fitted v for pent variants, averaged over fuzzy numbers.
