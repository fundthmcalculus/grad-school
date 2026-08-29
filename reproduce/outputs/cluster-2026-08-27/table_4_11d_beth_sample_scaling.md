**Table 4.11(d) — BETH: matched training-set size across every arm**

| Method | Training samples | Train time (s) | Inference time (s) | ROC-AUC | Detection rate | False-alarm rate | Detection − false alarm |
|---|---|---|---|---|---|---|---|
| Tribble one-class (surprisal) | 1,000 | 0.081 ± 0.007 | 0.133 ± 0.006 | 0.9903 ± 0.0015 | 0.993 ± 0.000 | 0.147 ± 0.002 | +0.846 |
| Tribble one-class (surprisal) | 2,000 | 0.075 ± 0.008 | 0.133 ± 0.005 | 0.9906 ± 0.0012 | 0.993 ± 0.000 | 0.148 ± 0.006 | +0.845 |
| Tribble one-class (surprisal) | 5,000 | 0.089 ± 0.008 | 0.136 ± 0.005 | 0.9900 ± 0.0007 | 0.993 ± 0.000 | 0.144 ± 0.004 | +0.849 |
| Tribble one-class (surprisal) | 10,000 | 0.101 ± 0.007 | 0.134 ± 0.003 | 0.9903 ± 0.0005 | 0.993 ± 0.000 | 0.144 ± 0.003 | +0.849 |
| Tribble one-class (surprisal) | 20,000 | 0.129 ± 0.013 | 0.129 ± 0.003 | 0.9900 ± 0.0006 | 0.993 ± 0.000 | 0.144 ± 0.004 | +0.849 |
| Tribble one-class (complement) | 1,000 | 0.079 ± 0.007 | 0.120 ± 0.006 | 0.9242 ± 0.0029 | 0.794 ± 0.419 | 0.118 ± 0.062 | +0.677 |
| Tribble one-class (complement) | 2,000 | 0.073 ± 0.006 | 0.120 ± 0.004 | 0.9267 ± 0.0015 | 0.993 ± 0.000 | 0.148 ± 0.006 | +0.845 |
| Tribble one-class (complement) | 5,000 | 0.089 ± 0.008 | 0.121 ± 0.003 | 0.9284 ± 0.0003 | 0.993 ± 0.000 | 0.144 ± 0.004 | +0.849 |
| Tribble one-class (complement) | 10,000 | 0.101 ± 0.008 | 0.122 ± 0.002 | 0.9279 ± 0.0009 | 0.993 ± 0.000 | 0.144 ± 0.003 | +0.849 |
| Tribble one-class (complement) | 20,000 | 0.128 ± 0.014 | 0.116 ± 0.004 | 0.9283 ± 0.0004 | 0.993 ± 0.000 | 0.144 ± 0.004 | +0.849 |
| Isolation Forest (max_samples=256, library default) | 1,000 | 0.158 ± 0.014 | 0.648 ± 0.022 | 0.8798 ± 0.0530 | 0.005 ± 0.007 | 0.019 ± 0.005 | -0.014 |
| Isolation Forest (max_samples=256, library default) | 2,000 | 0.153 ± 0.005 | 0.650 ± 0.013 | 0.8904 ± 0.0188 | 0.003 ± 0.004 | 0.019 ± 0.003 | -0.016 |
| Isolation Forest (max_samples=256, library default) | 5,000 | 0.166 ± 0.007 | 0.643 ± 0.013 | 0.8997 ± 0.0173 | 0.004 ± 0.004 | 0.020 ± 0.005 | -0.016 |
| Isolation Forest (max_samples=256, library default) | 10,000 | 0.187 ± 0.012 | 0.647 ± 0.021 | 0.8956 ± 0.0116 | 0.005 ± 0.006 | 0.021 ± 0.003 | -0.017 |
| Isolation Forest (max_samples=256, library default) | 20,000 | 0.218 ± 0.004 | 0.635 ± 0.017 | 0.8963 ± 0.0082 | 0.000 ± 0.001 | 0.018 ± 0.004 | -0.017 |
| Isolation Forest (max_samples=n, matched work) | 1,000 | 0.152 ± 0.012 | 0.707 ± 0.036 | 0.9491 ± 0.0489 | 0.563 ± 0.470 | 0.032 ± 0.015 | +0.531 |
| Isolation Forest (max_samples=n, matched work) | 2,000 | 0.150 ± 0.008 | 0.701 ± 0.024 | 0.9844 ± 0.0031 | 0.936 ± 0.004 | 0.057 ± 0.025 | +0.879 |
| Isolation Forest (max_samples=n, matched work) | 5,000 | 0.156 ± 0.004 | 0.785 ± 0.055 | 0.9846 ± 0.0033 | 0.941 ± 0.008 | 0.064 ± 0.023 | +0.877 |
| Isolation Forest (max_samples=n, matched work) | 10,000 | 0.175 ± 0.011 | 0.832 ± 0.031 | 0.9809 ± 0.0060 | 0.942 ± 0.014 | 0.070 ± 0.031 | +0.873 |
| Isolation Forest (max_samples=n, matched work) | 20,000 | 0.200 ± 0.015 | 0.868 ± 0.015 | 0.9781 ± 0.0100 | 0.949 ± 0.012 | 0.086 ± 0.033 | +0.863 |
| One-class SVM | 1,000 | 0.001 ± 0.000 | 0.117 ± 0.006 | 0.9913 ± 0.0098 | 0.993 ± 0.001 | 0.153 ± 0.026 | +0.840 |
| One-class SVM | 2,000 | 0.002 ± 0.000 | 0.180 ± 0.009 | 0.9923 ± 0.0099 | 0.994 ± 0.001 | 0.160 ± 0.039 | +0.833 |
| One-class SVM | 5,000 | 0.008 ± 0.001 | 0.377 ± 0.015 | 0.9952 ± 0.0010 | 0.994 ± 0.001 | 0.159 ± 0.030 | +0.835 |
| One-class SVM | 10,000 | 0.026 ± 0.001 | 0.668 ± 0.012 | 0.9959 ± 0.0008 | 0.993 ± 0.000 | 0.142 ± 0.007 | +0.851 |
| One-class SVM | 20,000 | 0.095 ± 0.003 | 1.280 ± 0.021 | 0.9959 ± 0.0008 | 0.993 | 0.140 ± 0.001 | +0.853 |

> **Every arm is fitted on the identical rows**: one subsample is drawn per (n, seed) from the 763,144-row benign training split and handed to all five arms, so the training-time column is a comparison rather than five different experiments. This corrects Table 4.11(c), where Tribble saw all 763,144 rows, the one-class SVM saw a 20,000-row cap, and Isolation Forest saw 763,144 rows nominally but built each tree from 256 (`max_samples` default) — so (c)'s flat Isolation Forest training line was that default, not a property of the algorithm. n tops out at 20,000 because that is where the SVM's cap sat in (c); going higher would drop it out of the comparison. **Isolation Forest appears twice on purpose**: at `max_samples=256` (the library default, so (c)'s number stays traceable) and at `max_samples=n` (every tree built from all n rows — the arm doing the same work as the others). Quality is reported at every n because 'how much data does this need?' is what a sample sweep is really asking. Inference time is re-measured at every n: it is independent of n for the fuzzy arms and both forests, but **not** for the one-class SVM, whose support-vector count grows with the training set, so its scoring cost rises with n while everyone else's stays flat. Features held fixed at all 8: processId, threadId, parentProcessId, userId, mountNamespace, eventId, argsNum, returnValue — see Table 4.11(c) for the sweep over feature count at fixed sample size. Threshold is the 0.990 quantile of each arm's own benign-validation scores (1.0% budget); scored on the full 188,967-row test split. All cells are mean ± sample std across common.SEEDS ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) — every arm is stochastic here, including the fuzzy ones, because which n rows are drawn is itself random.

> Generated by `reproduce/`; seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. `N/A` marks a cell whose method/dataset was unavailable.

> **Machine.** host: NEX-210200 · os: Windows-11 · cpu: Intel(R) Core(TM) i9-14900HX · cores: 32 physical, 32 logical · ram: 95.6 GiB · gpu: NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB · python: 3.13.7
>
> Wall-clock times are machine-dependent; ratios are not. Markdown tables report normalized ratios where a timing is involved, and the companion CSV carries the absolute seconds.
