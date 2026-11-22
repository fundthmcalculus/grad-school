# Feature-scoring metrics: which ones matter, and why

**Status:** designed, predictions registered, not yet run.

## Question

The MoG classifier keeps the top-`k` features by a differentiation score. Four
scores are built in (`wasserstein`, `bhattacharyya`, `composite`,
`higher_order`). Higher-order beats the rest at `k` = 4–5 on PhiUSIIL, and it
costs more time. Which score is responsible for that, and which other standard
feature-scoring metrics would do as well or better? For each metric, the write-up
must say *why* it works or fails, not only whether it does.

## Protocol

- **Datasets:** PhiUSIIL (binary, 47 features, the disputed set), Breast Cancer
  (binary, 30), Wine (3-class, 13). The last two are bundled with scikit-learn, so
  they need no download.
- **Seeds:** `0..9`, one 80/20 split per seed (`random_state = seed`), as in Table 4.1.
- **Scoring uses the training split only.** Scores never see the test rows.
- **Selection rule, the same for every metric:** rank features by score, keep the
  top `k`. No redundancy filter, so no metric is advantaged by an extra step.
  The built-in scores are ranked with `calculate_gaussian_correlation` (the score
  the classifier uses), and then top-`k` is taken directly.
- **Classifier:** `TribbleClassifier(top_n=k, scoring_method="wasserstein")` on
  the selected columns only, so it cannot re-select. Its `top_n` already equals
  the number of columns, so the internal selection is the external one.
- **Grid:** `k` in {2, 4, 5, 8, 10}, capped at the feature count.
- **Multi-class:** one-vs-rest per class, averaged over classes. Mutual information
  and ANOVA F are native multi-class.

## Candidate metrics, and what each one measures

| Metric | Measures | Why it might matter | Why it might fail |
|---|---|---|---|
| Wasserstein (built in) | CDF gap integrated over x | Scale-aware, uses the whole distribution | Weights all of x equally; binary features reduce to a single gap |
| Bhattacharyya (built in) | Overlap of Gaussian fits | Cheap; bounded | Fits a Gaussian, so it ignores shape and misreads skewed features |
| Composite (built in) | Bhattacharyya + Wasserstein | Combines an overlap view with a transport view | Inherits both blind spots |
| Higher-order (built in) | Mean, spread, skewness, kurtosis gaps | Sees shape, which Gaussians miss | Heavy tails make moment gaps large and noisy |
| **ANOVA F** | Between-class over within-class variance | Standard filter baseline; cheap | Mean-only; variance inflated by outliers |
| **Mutual information** | Any statistical dependence (kNN estimator) | Catches non-monotone and shape effects; no distribution assumed | Estimator noise at small n; sensitive to k-NN settings |
| **Kolmogorov–Smirnov** | Largest CDF gap | Bounded, rank-based, scale-free | Uses one point of the CDF, so it ignores most of the distribution |
| **Energy distance** | Squared CDF gap integrated over x | Like Wasserstein but squared, so it emphasizes large gaps | Squaring makes it dominated by tails and outliers |
| **Hellinger** (20 quantile bins) | Bin-wise square-root overlap | Bounded; robust to tails because bins are quantile-based | Bin count matters; coarse for binary features |
| **Jensen–Shannon** (20 quantile bins) | Bin-wise divergence from the mixture | Bounded and symmetric; well-behaved at zero counts | Same binning sensitivity as Hellinger |

## Predictions (registered before running)

These are written before any run. Each has a reason, and a result that contradicts
it is a finding to explain, not a result to hide.

1. **Higher-order vs composite at `k` = 4–5 on PhiUSIIL.** Predicted: higher-order
   stays slightly ahead at `k` = 4 and composite matches it at `k` = 5. Reason: the
   seed-0 inspection showed higher-order picks heavy-tailed counts, which are the
   features where moment gaps are large.
2. **Energy distance behaves like higher-order on PhiUSIIL at small `k`.** Reason:
   squaring the CDF gap weights large gaps, and the heavy-tailed count features
   have the largest gaps. Predicted to be close to higher-order in accuracy, and
   to fall off on Wine and Breast Cancer, where the signal is in the bulk.
3. **Kolmogorov–Smirnov behaves like the Wasserstein arm on binary features.**
   Reason: on a binary feature the KS statistic is the difference between the two
   class proportions, which picks the most imbalanced flags. Predicted to fail on
   PhiUSIIL at small `k` in the same way as Wasserstein.
4. **Mutual information is the strongest baseline on Breast Cancer and Wine.**
   Reason: those datasets have informative bulk shapes, which MI catches with no
   distributional assumption. Predicted to be the most accurate at `k` = 4 on at
   least one of them.
5. **ANOVA F is the weakest on PhiUSIIL.** Reason: its variance denominator is
   inflated by the heavy-tailed count features, so it ranks them down.
6. **Scoring time.** Predicted order, cheapest first: ANOVA F, KS, Hellinger,
   Jensen–Shannon, Bhattacharyya, Wasserstein, energy, higher-order, mutual
   information. Reason: the mutual-information kNN estimator is the most expensive
   per feature, and the moment and CDF routines are O(n log n) per feature.

## Outputs

- `results.md` — one table per dataset: metric, `k`, accuracy mean ± s.d., mean
  scoring time, mean fit time, features kept.
- `results.csv` — the same, machine-readable.
- An explanation section, added after the run, that covers each metric's result
  and whether the prediction above held.
