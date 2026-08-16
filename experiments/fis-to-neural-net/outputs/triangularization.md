# Why triangularization collapses (H2)

Concrete, `top_n` swept over TRIBBLE's own feature selector, 5 seeds. **dead rows** = test rows whose total firing strength is <= 1e-6 across every rule, which `regression._normalize_firing_strengths` maps to a prediction of 0.

| features kept | Gaussian RMSE | triangular RMSE | Gaussian dead rows | triangular dead rows |
|---|---|---|---|---|
| 1 | 15.19 | 15.66 | 0.0% | 1.0% |
| 2 | 14.29 | 15.44 | 0.0% | 1.3% |
| 3 | 12.89 | 14.13 | 0.0% | 1.6% |
| 4 | 8.82 | 12.56 | 0.0% | 5.0% |
| 5 | 8.80 | 12.81 | 0.0% | 5.3% |
| 6 | 7.88 | 12.50 | 0.1% | 6.0% |
| 7 | 7.32 | 22.23 | 0.3% | 35.4% |
| 8 | 7.14 | 32.07 | 0.3% | 70.7% |

Other datasets at their experiment settings:

| dataset | features kept | Gaussian RMSE | triangular RMSE | triangular dead rows |
|---|---|---|---|---|
| bikeshare | 12 | 103.67 | 263.76 | 100.0% |
| wec | 12 | 845937.87 | 2401073.00 | 18.3% |
