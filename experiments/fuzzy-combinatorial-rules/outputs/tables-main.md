# Results (main) — t-norm `min`, disjunction `sum`, lambda 1.0

### iris — 150 samples, 4 features, 3 classes → 3 rules

**Test accuracy** (mean ± std over 10 seeds)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.933 ± 0.024 | 0.884 ± 0.050 | 0.933 ± 0.030 |
| mst_mf | 0.949 ± 0.026 | 0.916 ± 0.036 | 0.944 ± 0.033 |
| mst_core | 0.929 ± 0.045 | 0.929 ± 0.036 | 0.936 ± 0.025 |
| greedy | 0.956 ± 0.028 | 0.953 ± 0.025 | 0.951 ± 0.033 |
| anneal | 0.956 ± 0.028 | 0.958 ± 0.029 | 0.949 ± 0.034 |
| exhaustive | 0.953 ± 0.027 | 0.958 ± 0.029 | N/A |
| _tree_ | 0.947 ± 0.027 | 0.947 ± 0.027 | 0.947 ± 0.027 |
| _nearest_centroid_ | 0.898 ± 0.032 | 0.898 ± 0.032 | 0.898 ± 0.032 |

**MFs selected per rule** (of 12 available at k=3 / 20 available at k=5 / 28 available at k=7)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 7.1 ± 0.2 | 11.0 ± 0.3 | 15.0 ± 0.3 |
| mst_mf | 8.2 ± 0.2 | 11.6 ± 0.5 | 16.0 ± 0.6 |
| mst_core | 8.0 ± 0.5 | 10.6 ± 0.5 | 15.1 ± 0.3 |
| greedy | 9.6 ± 0.2 | 15.4 ± 0.4 | 22.4 ± 0.6 |
| anneal | 9.6 ± 0.2 | 13.9 ± 0.3 | 18.3 ± 1.6 |
| exhaustive | 8.5 ± 0.2 | 12.1 ± 0.4 | N/A |

**Training objective** (sum of the C one-vs-rest margins; this is what every selector optimises)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 1.883 ± 0.042 | 2.378 ± 0.087 | 2.669 ± 0.050 |
| mst_mf | 1.940 ± 0.050 | 2.408 ± 0.058 | 2.692 ± 0.064 |
| mst_core | 1.964 ± 0.042 | 2.432 ± 0.094 | 2.675 ± 0.058 |
| greedy | 2.057 ± 0.030 | 2.548 ± 0.071 | 2.740 ± 0.031 |
| anneal | 2.059 ± 0.032 | 2.577 ± 0.055 | 2.743 ± 0.032 |
| exhaustive | 2.059 ± 0.032 | 2.577 ± 0.055 | N/A |

**Fraction of antecedents that are contiguous** (1.0 = every rule reads as a linguistic term)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| mst_mf | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| mst_core | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| greedy | 1.00 ± 0.00 | 0.88 ± 0.06 | 0.92 ± 0.07 |
| anneal | 1.00 ± 0.00 | 0.97 ± 0.06 | 0.95 ± 0.07 |
| exhaustive | 1.00 ± 0.00 | 1.00 ± 0.00 | N/A |

**Fraction of test samples where no rule fires**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.009 |
| mst_mf | 0.000 ± 0.000 | 0.002 ± 0.007 | 0.004 ± 0.009 |
| mst_core | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.009 |
| greedy | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| anneal | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.009 |
| exhaustive | 0.000 ± 0.000 | 0.000 ± 0.000 | N/A |

**Fit seconds per model**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 |
| mst_mf | 0.010 ± 0.000 | 0.013 ± 0.001 | 0.016 ± 0.001 |
| mst_core | 0.025 ± 0.002 | 0.023 ± 0.001 | 0.022 ± 0.001 |
| greedy | 0.005 ± 0.001 | 0.014 ± 0.001 | 0.020 ± 0.003 |
| anneal | 0.552 ± 0.015 | 0.597 ± 0.031 | 0.587 ± 0.017 |
| exhaustive | 0.015 ± 0.020 | 3.755 ± 0.233 | N/A |


### wine — 178 samples, 13 features, 3 classes → 3 rules

**Test accuracy** (mean ± std over 10 seeds)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.885 ± 0.030 | 0.904 ± 0.020 | 0.913 ± 0.023 |
| mst_mf | 0.906 ± 0.021 | 0.902 ± 0.045 | 0.922 ± 0.036 |
| mst_core | 0.861 ± 0.043 | 0.889 ± 0.038 | 0.907 ± 0.025 |
| greedy | 0.911 ± 0.027 | 0.931 ± 0.026 | 0.954 ± 0.028 |
| anneal | 0.902 ± 0.025 | 0.941 ± 0.020 | 0.944 ± 0.012 |
| exhaustive | N/A | N/A | N/A |
| _tree_ | 0.917 ± 0.040 | 0.917 ± 0.040 | 0.917 ± 0.040 |
| _nearest_centroid_ | 0.700 ± 0.037 | 0.700 ± 0.037 | 0.700 ± 0.037 |

**MFs selected per rule** (of 39 available at k=3 / 65 available at k=5 / 91 available at k=7)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 28.6 ± 1.2 | 44.4 ± 2.2 | 64.4 ± 0.8 |
| mst_mf | 34.1 ± 1.0 | 48.7 ± 2.2 | 66.8 ± 1.6 |
| mst_core | 28.9 ± 2.0 | 44.3 ± 2.6 | 64.6 ± 1.9 |
| greedy | 34.0 ± 0.4 | 56.4 ± 0.9 | 80.0 ± 0.7 |
| anneal | 34.0 ± 0.5 | 53.9 ± 1.4 | 74.6 ± 2.0 |
| exhaustive | N/A | N/A | N/A |

**Training objective** (sum of the C one-vs-rest margins; this is what every selector optimises)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 1.518 ± 0.057 | 2.203 ± 0.059 | 2.471 ± 0.046 |
| mst_mf | 1.795 ± 0.063 | 2.389 ± 0.053 | 2.549 ± 0.070 |
| mst_core | 1.564 ± 0.040 | 2.254 ± 0.064 | 2.523 ± 0.060 |
| greedy | 1.906 ± 0.045 | 2.565 ± 0.046 | 2.687 ± 0.023 |
| anneal | 1.928 ± 0.042 | 2.607 ± 0.026 | 2.748 ± 0.019 |
| exhaustive | N/A | N/A | N/A |

**Fraction of antecedents that are contiguous** (1.0 = every rule reads as a linguistic term)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 1.00 ± 0.00 | 0.99 ± 0.01 | 0.98 ± 0.01 |
| mst_mf | 1.00 ± 0.00 | 0.99 ± 0.01 | 0.97 ± 0.03 |
| mst_core | 1.00 ± 0.00 | 0.99 ± 0.01 | 0.99 ± 0.01 |
| greedy | 1.00 ± 0.00 | 0.97 ± 0.02 | 0.90 ± 0.02 |
| anneal | 1.00 ± 0.00 | 0.97 ± 0.02 | 0.89 ± 0.06 |
| exhaustive | N/A | N/A | N/A |

**Fraction of test samples where no rule fires**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.011 ± 0.012 | 0.028 ± 0.024 | 0.039 ± 0.013 |
| mst_mf | 0.000 ± 0.000 | 0.017 ± 0.013 | 0.028 ± 0.012 |
| mst_core | 0.006 ± 0.008 | 0.020 ± 0.019 | 0.048 ± 0.026 |
| greedy | 0.000 ± 0.000 | 0.004 ± 0.011 | 0.015 ± 0.014 |
| anneal | 0.000 ± 0.000 | 0.004 ± 0.007 | 0.013 ± 0.014 |
| exhaustive | N/A | N/A | N/A |

**Fit seconds per model**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.005 ± 0.000 | 0.007 ± 0.002 | 0.006 ± 0.000 |
| mst_mf | 0.034 ± 0.001 | 0.048 ± 0.001 | 0.068 ± 0.003 |
| mst_core | 0.039 ± 0.001 | 0.038 ± 0.001 | 0.038 ± 0.001 |
| greedy | 0.029 ± 0.003 | 0.076 ± 0.009 | 0.133 ± 0.016 |
| anneal | 0.654 ± 0.019 | 0.749 ± 0.035 | 0.808 ± 0.022 |
| exhaustive | N/A | N/A | N/A |


### glass — 214 samples, 9 features, 6 classes → 6 rules

**Test accuracy** (mean ± std over 10 seeds)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.400 ± 0.071 | 0.380 ± 0.052 | 0.400 ± 0.072 |
| mst_mf | 0.375 ± 0.072 | 0.366 ± 0.085 | 0.354 ± 0.068 |
| mst_core | 0.369 ± 0.075 | 0.398 ± 0.087 | 0.438 ± 0.074 |
| greedy | 0.398 ± 0.055 | 0.538 ± 0.088 | 0.528 ± 0.063 |
| anneal | 0.414 ± 0.077 | 0.549 ± 0.062 | 0.555 ± 0.062 |
| exhaustive | N/A | N/A | N/A |
| _tree_ | 0.654 ± 0.050 | 0.654 ± 0.050 | 0.654 ± 0.050 |
| _nearest_centroid_ | 0.422 ± 0.056 | 0.422 ± 0.056 | 0.422 ± 0.056 |

**MFs selected per rule** (of 27 available at k=3 / 45 available at k=5 / 63 available at k=7)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 17.9 ± 0.6 | 23.2 ± 1.7 | 31.0 ± 1.8 |
| mst_mf | 18.2 ± 0.6 | 28.5 ± 0.9 | 35.9 ± 3.2 |
| mst_core | 18.3 ± 0.5 | 23.8 ± 1.0 | 29.9 ± 1.5 |
| greedy | 21.1 ± 0.4 | 34.8 ± 0.7 | 50.0 ± 0.6 |
| anneal | 20.0 ± 0.3 | 33.1 ± 1.0 | 44.3 ± 2.4 |
| exhaustive | N/A | N/A | N/A |

**Training objective** (sum of the C one-vs-rest margins; this is what every selector optimises)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 2.594 ± 0.119 | 3.266 ± 0.162 | 3.571 ± 0.121 |
| mst_mf | 2.592 ± 0.182 | 3.327 ± 0.195 | 3.640 ± 0.160 |
| mst_core | 2.808 ± 0.113 | 3.478 ± 0.110 | 3.753 ± 0.100 |
| greedy | 2.810 ± 0.111 | 3.765 ± 0.062 | 4.133 ± 0.070 |
| anneal | 2.958 ± 0.071 | 3.786 ± 0.073 | 4.155 ± 0.060 |
| exhaustive | N/A | N/A | N/A |

**Fraction of antecedents that are contiguous** (1.0 = every rule reads as a linguistic term)

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.99 ± 0.01 | 0.92 ± 0.03 | 0.83 ± 0.04 |
| mst_mf | 0.99 ± 0.01 | 0.94 ± 0.02 | 0.87 ± 0.04 |
| mst_core | 0.99 ± 0.01 | 0.95 ± 0.02 | 0.87 ± 0.04 |
| greedy | 0.96 ± 0.03 | 0.77 ± 0.05 | 0.65 ± 0.03 |
| anneal | 0.97 ± 0.02 | 0.78 ± 0.05 | 0.68 ± 0.06 |
| exhaustive | N/A | N/A | N/A |

**Fraction of test samples where no rule fires**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.020 ± 0.023 | 0.038 ± 0.032 | 0.058 ± 0.038 |
| mst_mf | 0.022 ± 0.012 | 0.020 ± 0.021 | 0.031 ± 0.028 |
| mst_core | 0.020 ± 0.021 | 0.038 ± 0.021 | 0.072 ± 0.041 |
| greedy | 0.003 ± 0.009 | 0.011 ± 0.014 | 0.012 ± 0.015 |
| anneal | 0.012 ± 0.012 | 0.012 ± 0.017 | 0.034 ± 0.026 |
| exhaustive | N/A | N/A | N/A |

**Fit seconds per model**

| selector | k=3 | k=5 | k=7 |
|---|---|---|---|
| mass | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 |
| mst_mf | 0.038 ± 0.001 | 0.055 ± 0.004 | 0.070 ± 0.003 |
| mst_core | 0.059 ± 0.001 | 0.065 ± 0.007 | 0.067 ± 0.007 |
| greedy | 0.040 ± 0.003 | 0.115 ± 0.009 | 0.207 ± 0.012 |
| anneal | 1.216 ± 0.055 | 1.372 ± 0.019 | 1.478 ± 0.036 |
| exhaustive | N/A | N/A | N/A |

