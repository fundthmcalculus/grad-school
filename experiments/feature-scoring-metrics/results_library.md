# Feature-scoring metrics: results

Mean ± s.d. over 10 seeds. `score_s` is the time to score all features on the training split; `fit_s` is the classifier fit on the selected features; `kept` is the mean number of features the classifier kept after its redundancy filter.

## Breast Cancer

| k | metric | accuracy | score s | fit s | kept |
|---|---|---|---|---|---|
| 2 | lib:ks | 0.915 ± 0.038 | 0.016 | 0.041 | 2.0 |
| 2 | lib:hellinger | 0.897 ± 0.043 | 0.020 | 0.036 | 2.0 |
| 2 | lib:jensen_shannon | 0.891 ± 0.034 | 0.019 | 0.032 | 2.0 |
| 2 | lib:mutual_info | 0.886 ± 0.028 | 0.056 | 0.031 | 2.0 |
| 2 | lib:higher_order | 0.704 ± 0.097 | 0.021 | 0.043 | 2.0 |
| 4 | lib:jensen_shannon | 0.925 ± 0.024 | 0.019 | 0.080 | 4.0 |
| 4 | lib:ks | 0.922 ± 0.038 | 0.016 | 0.088 | 4.0 |
| 4 | lib:hellinger | 0.919 ± 0.031 | 0.020 | 0.091 | 4.0 |
| 4 | lib:mutual_info | 0.918 ± 0.031 | 0.056 | 0.102 | 4.0 |
| 4 | lib:higher_order | 0.839 ± 0.042 | 0.021 | 0.085 | 4.0 |
| 5 | lib:hellinger | 0.933 ± 0.025 | 0.020 | 0.110 | 5.0 |
| 5 | lib:jensen_shannon | 0.933 ± 0.025 | 0.019 | 0.141 | 5.0 |
| 5 | lib:ks | 0.933 ± 0.025 | 0.016 | 0.117 | 5.0 |
| 5 | lib:mutual_info | 0.933 ± 0.025 | 0.056 | 0.130 | 5.0 |
| 5 | lib:higher_order | 0.873 ± 0.046 | 0.021 | 0.118 | 5.0 |
| 8 | lib:ks | 0.932 ± 0.023 | 0.016 | 0.204 | 8.0 |
| 8 | lib:mutual_info | 0.923 ± 0.033 | 0.056 | 0.184 | 8.0 |
| 8 | lib:jensen_shannon | 0.918 ± 0.030 | 0.019 | 0.194 | 8.0 |
| 8 | lib:hellinger | 0.914 ± 0.026 | 0.020 | 0.199 | 8.0 |
| 8 | lib:higher_order | 0.892 ± 0.023 | 0.021 | 0.157 | 8.0 |
| 10 | lib:ks | 0.928 ± 0.027 | 0.016 | 0.198 | 10.0 |
| 10 | lib:hellinger | 0.927 ± 0.024 | 0.020 | 0.217 | 10.0 |
| 10 | lib:mutual_info | 0.927 ± 0.024 | 0.056 | 0.210 | 10.0 |
| 10 | lib:jensen_shannon | 0.925 ± 0.024 | 0.019 | 0.185 | 10.0 |
| 10 | lib:higher_order | 0.894 ± 0.030 | 0.021 | 0.260 | 10.0 |

## PhiUSIIL

| k | metric | accuracy | score s | fit s | kept |
|---|---|---|---|---|---|
| 2 | lib:hellinger | 0.882 ± 0.042 | 0.141 | 0.092 | 2.0 |
| 2 | lib:jensen_shannon | 0.882 ± 0.042 | 0.119 | 0.090 | 2.0 |
| 2 | lib:ks | 0.882 ± 0.042 | 0.103 | 0.100 | 2.0 |
| 2 | lib:mutual_info | 0.882 ± 0.042 | 1.372 | 0.080 | 2.0 |
| 2 | lib:higher_order | 0.692 ± 0.104 | 0.053 | 0.099 | 2.0 |
| 4 | lib:hellinger | 0.955 ± 0.012 | 0.141 | 0.188 | 4.0 |
| 4 | lib:jensen_shannon | 0.955 ± 0.012 | 0.119 | 0.208 | 4.0 |
| 4 | lib:ks | 0.955 ± 0.012 | 0.103 | 0.243 | 4.0 |
| 4 | lib:mutual_info | 0.955 ± 0.012 | 1.372 | 0.199 | 4.0 |
| 4 | lib:higher_order | 0.843 ± 0.069 | 0.053 | 0.218 | 4.0 |
| 5 | lib:hellinger | 0.964 ± 0.008 | 0.141 | 0.260 | 5.0 |
| 5 | lib:jensen_shannon | 0.964 ± 0.008 | 0.119 | 0.273 | 5.0 |
| 5 | lib:ks | 0.964 ± 0.008 | 0.103 | 0.249 | 5.0 |
| 5 | lib:mutual_info | 0.964 ± 0.008 | 1.372 | 0.288 | 5.0 |
| 5 | lib:higher_order | 0.848 ± 0.064 | 0.053 | 0.238 | 5.0 |
| 8 | lib:hellinger | 0.973 ± 0.006 | 0.141 | 0.466 | 8.0 |
| 8 | lib:jensen_shannon | 0.973 ± 0.006 | 0.119 | 0.464 | 8.0 |
| 8 | lib:mutual_info | 0.965 ± 0.005 | 1.372 | 0.419 | 8.0 |
| 8 | lib:ks | 0.965 ± 0.008 | 0.103 | 0.378 | 8.0 |
| 8 | lib:higher_order | 0.929 ± 0.021 | 0.053 | 0.320 | 8.0 |
| 10 | lib:hellinger | 0.981 ± 0.003 | 0.141 | 0.541 | 10.0 |
| 10 | lib:jensen_shannon | 0.974 ± 0.005 | 0.119 | 0.521 | 10.0 |
| 10 | lib:mutual_info | 0.973 ± 0.004 | 1.372 | 0.510 | 10.0 |
| 10 | lib:ks | 0.965 ± 0.008 | 0.103 | 0.434 | 10.0 |
| 10 | lib:higher_order | 0.937 ± 0.015 | 0.053 | 0.432 | 10.0 |

## Wine

| k | metric | accuracy | score s | fit s | kept |
|---|---|---|---|---|---|
| 2 | lib:hellinger | 0.869 ± 0.081 | 0.013 | 0.052 | 2.0 |
| 2 | lib:jensen_shannon | 0.869 ± 0.081 | 0.013 | 0.055 | 2.0 |
| 2 | lib:mutual_info | 0.864 ± 0.076 | 0.033 | 0.052 | 2.0 |
| 2 | lib:ks | 0.842 ± 0.066 | 0.010 | 0.075 | 2.0 |
| 2 | lib:higher_order | 0.800 ± 0.070 | 0.018 | 0.055 | 2.0 |
| 4 | lib:jensen_shannon | 0.947 ± 0.042 | 0.013 | 0.106 | 4.0 |
| 4 | lib:hellinger | 0.944 ± 0.041 | 0.013 | 0.101 | 4.0 |
| 4 | lib:mutual_info | 0.944 ± 0.045 | 0.033 | 0.124 | 4.0 |
| 4 | lib:ks | 0.942 ± 0.044 | 0.010 | 0.136 | 4.0 |
| 4 | lib:higher_order | 0.906 ± 0.073 | 0.018 | 0.104 | 4.0 |
| 5 | lib:ks | 0.969 ± 0.028 | 0.010 | 0.130 | 5.0 |
| 5 | lib:mutual_info | 0.958 ± 0.040 | 0.033 | 0.158 | 5.0 |
| 5 | lib:hellinger | 0.947 ± 0.036 | 0.013 | 0.147 | 5.0 |
| 5 | lib:jensen_shannon | 0.947 ± 0.036 | 0.013 | 0.152 | 5.0 |
| 5 | lib:higher_order | 0.928 ± 0.033 | 0.018 | 0.157 | 5.0 |
| 8 | lib:hellinger | 0.961 ± 0.033 | 0.013 | 0.225 | 8.0 |
| 8 | lib:jensen_shannon | 0.961 ± 0.033 | 0.013 | 0.238 | 8.0 |
| 8 | lib:ks | 0.961 ± 0.033 | 0.010 | 0.199 | 8.0 |
| 8 | lib:mutual_info | 0.956 ± 0.033 | 0.033 | 0.239 | 8.0 |
| 8 | lib:higher_order | 0.953 ± 0.026 | 0.018 | 0.237 | 8.0 |
| 10 | lib:hellinger | 0.956 ± 0.033 | 0.013 | 0.321 | 10.0 |
| 10 | lib:higher_order | 0.956 ± 0.035 | 0.018 | 0.312 | 10.0 |
| 10 | lib:jensen_shannon | 0.956 ± 0.033 | 0.013 | 0.267 | 10.0 |
| 10 | lib:ks | 0.956 ± 0.033 | 0.010 | 0.307 | 10.0 |
| 10 | lib:mutual_info | 0.950 ± 0.029 | 0.033 | 0.290 | 10.0 |
