# Result tables (generated — do not edit)

Regenerated from `data/*.csv` by `export_report.py`. 10 seeds, 7,500 generations.

## auroc_FalsePremise

| detector                  |   auroc_raw_mean |   auroc_raw_std |   auroc_matched_mean |   auroc_matched_std |
|:--------------------------|-----------------:|----------------:|---------------------:|--------------------:|
| FIS · centroid (PCA-free) |           0.8833 |          0.0146 |               0.9058 |              0.0174 |
| OneClassSVM · centroid    |           0.8021 |          0.0201 |               0.7892 |              0.0222 |
| Mahalanobis · stats       |           0.8152 |          0.0143 |               0.7603 |              0.0228 |
| IsolationForest · stats   |           0.767  |          0.015  |               0.7178 |              0.0223 |
| perplexity                |           0.728  |          0.0132 |               0.5797 |              0.0446 |
| mean entropy              |           0.7034 |          0.0136 |               0.5585 |              0.0451 |
| n_tokens (control)        |           0.8429 |          0.0109 |               0.5    |              0      |
| FIS · PCA (64 comp)       |           0.6153 |          0.0725 |               0.454  |              0.089  |

## auroc_TriviaQA

| detector                  |   auroc_raw_mean |   auroc_raw_std |   auroc_matched_mean |   auroc_matched_std |
|:--------------------------|-----------------:|----------------:|---------------------:|--------------------:|
| mean entropy              |           0.6697 |          0.0121 |               0.6729 |              0.0143 |
| perplexity                |           0.6703 |          0.0119 |               0.6692 |              0.0158 |
| Mahalanobis · stats       |           0.6408 |          0.0119 |               0.64   |              0.0193 |
| IsolationForest · stats   |           0.6151 |          0.0086 |               0.6094 |              0.0161 |
| FIS · PCA (64 comp)       |           0.5299 |          0.0204 |               0.5168 |              0.0272 |
| OneClassSVM · centroid    |           0.5205 |          0.0166 |               0.5152 |              0.0222 |
| n_tokens (control)        |           0.5858 |          0.0122 |               0.5    |              0      |
| FIS · centroid (PCA-free) |           0.4975 |          0.0141 |               0.4988 |              0.0135 |

## cost

| detector                  |   feat_ms |   fit_ms |   score_ms_per_1k |   n_mfs |   total_train_ms |
|:--------------------------|----------:|---------:|------------------:|--------:|-----------------:|
| n_tokens (control)        |      0    |     0    |              0.22 |   nan   |             0    |
| perplexity                |      0    |     0    |              0.31 |   nan   |             0    |
| mean entropy              |      0    |     0    |              0.26 |   nan   |             0    |
| Mahalanobis · stats       |      1.05 |     1.63 |              0.69 |   nan   |             2.68 |
| IsolationForest · stats   |      1.05 |    94.3  |              6.97 |   nan   |            95.35 |
| FIS · PCA (64 comp)       |     94.44 |   954.92 |              1.36 |    24   |          1049.35 |
| OneClassSVM · centroid    |   1152.08 |     6.01 |              4.74 |   nan   |          1158.09 |
| FIS · centroid (PCA-free) |   1152.08 |  1065.79 |              1.79 |    29.9 |          2217.88 |

## label_counts

| family       |   abstain |   correct |   hallucination |
|:-------------|----------:|----------:|----------------:|
| falsepremise |        69 |         0 |            1431 |
| triviaqa     |       291 |      1761 |            3948 |

## Headline

```json
{
  "n_seeds": 10,
  "FalsePremise": {
    "fis_matched_mean": 0.9058,
    "fis_matched_std": 0.0174,
    "best_rival": "OneClassSVM \u00b7 centroid",
    "best_rival_matched_mean": 0.7892,
    "paired_delta_mean": 0.1166,
    "paired_delta_std": 0.0165,
    "paired_delta_min": 0.095,
    "seeds_won": 10,
    "seeds_total": 10
  },
  "TriviaQA": {
    "fis_matched_mean": 0.4988,
    "fis_matched_std": 0.0135,
    "best_rival": "mean entropy",
    "best_rival_matched_mean": 0.6729,
    "paired_delta_mean": -0.1741,
    "paired_delta_std": 0.0242,
    "paired_delta_min": -0.2132,
    "seeds_won": 0,
    "seeds_total": 10
  },
  "n_generations": 7500
}
```