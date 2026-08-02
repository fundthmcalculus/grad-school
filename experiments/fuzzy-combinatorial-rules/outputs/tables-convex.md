# Results (convex) — t-norm `min`, disjunction `sum`, lambda 1.0

### iris — 150 samples, 4 features, 3 classes → 3 rules

**Test accuracy** (mean ± std over 10 seeds)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.933 ± 0.024 | 0.884 ± 0.050 | 0.933 ± 0.030 |
| mst_mf | 0.949 ± 0.026 | 0.916 ± 0.036 | 0.944 ± 0.033 |
| mst_core | 0.929 ± 0.045 | 0.929 ± 0.036 | 0.936 ± 0.025 |
| greedy | 0.956 ± 0.028 | 0.958 ± 0.029 | 0.964 ± 0.030 |
| anneal | 0.956 ± 0.028 | 0.958 ± 0.029 | 0.949 ± 0.034 |
| exhaustive | 0.953 ± 0.027 | 0.958 ± 0.029 | 0.949 ± 0.034 |
| _tree_ | 0.947 ± 0.027 | 0.947 ± 0.027 | 0.947 ± 0.027 |
| _nearest_centroid_ | 0.898 ± 0.032 | 0.898 ± 0.032 | 0.898 ± 0.032 |

**MFs selected per rule** (of 12 available at k=3 / 20 available at k=5 / 28 available at k=7)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 7.1 ± 0.2 | 11.0 ± 0.3 | 15.0 ± 0.3 |
| mst_mf | 8.2 ± 0.2 | 11.6 ± 0.5 | 16.0 ± 0.6 |
| mst_core | 8.0 ± 0.5 | 10.6 ± 0.5 | 15.1 ± 0.3 |
| greedy | 9.6 ± 0.2 | 16.3 ± 0.3 | 23.0 ± 0.1 |
| anneal | 9.2 ± 0.2 | 14.3 ± 0.4 | 17.5 ± 1.4 |
| exhaustive | 8.5 ± 0.2 | 12.1 ± 0.4 | 16.1 ± 0.5 |

**Training objective** (sum of the C one-vs-rest margins; this is what every selector optimises)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 1.883 ± 0.042 | 2.378 ± 0.087 | 2.669 ± 0.050 |
| mst_mf | 1.940 ± 0.050 | 2.408 ± 0.058 | 2.692 ± 0.064 |
| mst_core | 1.964 ± 0.042 | 2.432 ± 0.094 | 2.675 ± 0.058 |
| greedy | 2.046 ± 0.035 | 2.525 ± 0.055 | 2.703 ± 0.043 |
| anneal | 2.059 ± 0.032 | 2.577 ± 0.055 | 2.743 ± 0.032 |
| exhaustive | 2.059 ± 0.032 | 2.577 ± 0.055 | 2.743 ± 0.032 |

**Fraction of antecedents that are contiguous** (1.0 = every rule reads as a linguistic term)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| mst_mf | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| mst_core | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| greedy | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| anneal | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| exhaustive | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

**Fraction of test samples where no rule fires**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.009 |
| mst_mf | 0.000 ± 0.000 | 0.002 ± 0.007 | 0.004 ± 0.009 |
| mst_core | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.009 |
| greedy | 0.000 ± 0.000 | 0.002 ± 0.007 | 0.000 ± 0.000 |
| anneal | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.009 |
| exhaustive | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.009 |

**Fit seconds per model**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 |
| mst_mf | 0.010 ± 0.001 | 0.014 ± 0.001 | 0.018 ± 0.001 |
| mst_core | 0.025 ± 0.001 | 0.026 ± 0.000 | 0.027 ± 0.002 |
| greedy | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.002 |
| anneal | 0.504 ± 0.020 | 0.442 ± 0.010 | 0.369 ± 0.009 |
| exhaustive | 0.006 ± 0.002 | 0.288 ± 0.014 | 2.585 ± 0.121 |


### wine — 178 samples, 13 features, 3 classes → 3 rules

**Test accuracy** (mean ± std over 10 seeds)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.885 ± 0.030 | 0.913 ± 0.028 | 0.913 ± 0.020 |
| mst_mf | 0.906 ± 0.021 | 0.906 ± 0.046 | 0.926 ± 0.035 |
| mst_core | 0.861 ± 0.043 | 0.896 ± 0.039 | 0.913 ± 0.017 |
| greedy | 0.902 ± 0.030 | 0.920 ± 0.039 | 0.939 ± 0.020 |
| anneal | 0.906 ± 0.031 | 0.937 ± 0.024 | 0.946 ± 0.024 |
| exhaustive | N/A | N/A | N/A |
| _tree_ | 0.917 ± 0.040 | 0.917 ± 0.040 | 0.917 ± 0.040 |
| _nearest_centroid_ | 0.700 ± 0.037 | 0.700 ± 0.037 | 0.700 ± 0.037 |

**MFs selected per rule** (of 39 available at k=3 / 65 available at k=5 / 91 available at k=7)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 28.6 ± 1.2 | 44.1 ± 2.0 | 64.7 ± 0.7 |
| mst_mf | 34.1 ± 1.0 | 48.8 ± 2.2 | 67.2 ± 1.6 |
| mst_core | 28.9 ± 2.0 | 44.4 ± 2.7 | 65.1 ± 1.6 |
| greedy | 34.7 ± 0.4 | 58.0 ± 0.7 | 82.1 ± 1.5 |
| anneal | 34.3 ± 0.7 | 52.9 ± 0.9 | 72.3 ± 1.8 |
| exhaustive | N/A | N/A | N/A |

**Training objective** (sum of the C one-vs-rest margins; this is what every selector optimises)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 1.518 ± 0.057 | 2.210 ± 0.055 | 2.475 ± 0.050 |
| mst_mf | 1.795 ± 0.063 | 2.391 ± 0.053 | 2.557 ± 0.069 |
| mst_core | 1.564 ± 0.040 | 2.261 ± 0.057 | 2.523 ± 0.062 |
| greedy | 1.901 ± 0.052 | 2.396 ± 0.107 | 2.517 ± 0.088 |
| anneal | 1.925 ± 0.042 | 2.608 ± 0.027 | 2.740 ± 0.020 |
| exhaustive | N/A | N/A | N/A |

**Fraction of antecedents that are contiguous** (1.0 = every rule reads as a linguistic term)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| mst_mf | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| mst_core | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| greedy | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| anneal | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| exhaustive | N/A | N/A | N/A |

**Fraction of test samples where no rule fires**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.011 ± 0.012 | 0.028 ± 0.024 | 0.039 ± 0.013 |
| mst_mf | 0.000 ± 0.000 | 0.017 ± 0.013 | 0.028 ± 0.012 |
| mst_core | 0.006 ± 0.008 | 0.020 ± 0.019 | 0.048 ± 0.026 |
| greedy | 0.000 ± 0.000 | 0.017 ± 0.015 | 0.024 ± 0.012 |
| anneal | 0.000 ± 0.000 | 0.006 ± 0.012 | 0.019 ± 0.017 |
| exhaustive | N/A | N/A | N/A |

**Fit seconds per model**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.007 ± 0.001 |
| mst_mf | 0.039 ± 0.002 | 0.058 ± 0.002 | 0.078 ± 0.003 |
| mst_core | 0.046 ± 0.002 | 0.046 ± 0.001 | 0.047 ± 0.001 |
| greedy | 0.025 ± 0.002 | 0.040 ± 0.007 | 0.057 ± 0.013 |
| anneal | 0.635 ± 0.023 | 0.539 ± 0.017 | 0.498 ± 0.014 |
| exhaustive | N/A | N/A | N/A |


### glass — 214 samples, 9 features, 6 classes → 6 rules

**Test accuracy** (mean ± std over 10 seeds)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.395 ± 0.064 | 0.375 ± 0.061 | 0.414 ± 0.079 |
| mst_mf | 0.374 ± 0.075 | 0.422 ± 0.078 | 0.425 ± 0.076 |
| mst_core | 0.362 ± 0.085 | 0.415 ± 0.101 | 0.471 ± 0.077 |
| greedy | 0.411 ± 0.051 | 0.463 ± 0.072 | 0.437 ± 0.092 |
| anneal | 0.425 ± 0.087 | 0.538 ± 0.067 | 0.534 ± 0.085 |
| exhaustive | N/A | N/A | N/A |
| _tree_ | 0.654 ± 0.050 | 0.654 ± 0.050 | 0.654 ± 0.050 |
| _nearest_centroid_ | 0.422 ± 0.056 | 0.422 ± 0.056 | 0.422 ± 0.056 |

**MFs selected per rule** (of 27 available at k=3 / 45 available at k=5 / 63 available at k=7)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 18.0 ± 0.6 | 24.2 ± 1.5 | 33.1 ± 1.7 |
| mst_mf | 18.0 ± 0.4 | 27.8 ± 2.0 | 36.3 ± 3.3 |
| mst_core | 18.3 ± 0.6 | 24.5 ± 0.9 | 31.1 ± 1.1 |
| greedy | 21.9 ± 0.4 | 38.1 ± 0.5 | 54.6 ± 1.0 |
| anneal | 19.6 ± 0.4 | 29.4 ± 0.7 | 39.1 ± 1.5 |
| exhaustive | N/A | N/A | N/A |

**Training objective** (sum of the C one-vs-rest margins; this is what every selector optimises)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 2.577 ± 0.115 | 3.238 ± 0.156 | 3.520 ± 0.139 |
| mst_mf | 2.587 ± 0.176 | 3.232 ± 0.214 | 3.566 ± 0.169 |
| mst_core | 2.802 ± 0.111 | 3.454 ± 0.089 | 3.716 ± 0.111 |
| greedy | 2.757 ± 0.107 | 3.454 ± 0.186 | 3.612 ± 0.265 |
| anneal | 2.944 ± 0.064 | 3.713 ± 0.065 | 4.050 ± 0.062 |
| exhaustive | N/A | N/A | N/A |

**Fraction of antecedents that are contiguous** (1.0 = every rule reads as a linguistic term)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| mst_mf | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| mst_core | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| greedy | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| anneal | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| exhaustive | N/A | N/A | N/A |

**Fraction of test samples where no rule fires**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.018 ± 0.023 | 0.032 ± 0.026 | 0.043 ± 0.033 |
| mst_mf | 0.023 ± 0.010 | 0.020 ± 0.015 | 0.032 ± 0.025 |
| mst_core | 0.020 ± 0.021 | 0.029 ± 0.022 | 0.069 ± 0.040 |
| greedy | 0.003 ± 0.009 | 0.008 ± 0.016 | 0.029 ± 0.021 |
| anneal | 0.015 ± 0.014 | 0.014 ± 0.011 | 0.048 ± 0.030 |
| exhaustive | N/A | N/A | N/A |

**Fit seconds per model**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.011 ± 0.001 | 0.011 ± 0.001 | 0.010 ± 0.000 |
| mst_mf | 0.045 ± 0.003 | 0.060 ± 0.001 | 0.073 ± 0.002 |
| mst_core | 0.072 ± 0.003 | 0.073 ± 0.003 | 0.075 ± 0.010 |
| greedy | 0.032 ± 0.003 | 0.049 ± 0.004 | 0.063 ± 0.008 |
| anneal | 1.156 ± 0.027 | 0.975 ± 0.024 | 0.863 ± 0.035 |
| exhaustive | N/A | N/A | N/A |

