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

## Results (run 2026-11-22, 10 seeds; full tables in `results.md`, raw in `results.csv`)

At `k` = 4 (mean accuracy):

| metric | PhiUSIIL | Breast Cancer | Wine |
|---|---|---|---|
| mutual_info | 0.955 | 0.922 | 0.947 |
| ks | 0.955 | 0.919 | 0.942 |
| hellinger | 0.955 | 0.925 | 0.950 |
| jensen_shannon | 0.955 | 0.925 | 0.950 |
| anova_f | 0.792 | 0.928 | 0.950 |
| energy | 0.924 | 0.900 | 0.897 |
| bhattacharyya | 0.818 | 0.931 | 0.914 |
| composite | 0.762 | 0.928 | 0.950 |
| higher_order | 0.869 | 0.839 | 0.919 |
| wasserstein | 0.438 | 0.928 | 0.922 |

Mean fit time per scorer is not reported separately: the classifier fit is
dominated by the classifier and is similar across metrics. Scoring time per
dataset (all features, one seed) varies by orders of magnitude: KS took 22 s on
PhiUSIIL, mutual information 1.7 s, Hellinger and Jensen–Shannon about 0.2 s, and
the built-in scores 0.04–0.12 s.

## Against the predictions

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Higher-order ahead of composite at k = 4, matches it at k = 5 | At k = 4 higher-order (0.869) is ahead of composite (0.762), but behind hellinger, JS, KS and MI (0.955). At k = 5 composite (0.919) is ahead of higher-order (0.893). | **Partly wrong.** Higher-order is not the best on PhiUSIIL. |
| 2 | Energy near higher-order on PhiUSIIL | Energy 0.924 at k = 4, above higher-order. Energy is also below the best on Breast Cancer and Wine. | **Wrong on accuracy**, right that it favours tails. |
| 3 | KS behaves like Wasserstein on binary features | KS 0.955 at k = 4, Wasserstein 0.438. | **Wrong.** |
| 4 | MI strongest on Breast Cancer and Wine | Bhattacharyya is best on Breast Cancer (0.931), and MI is not first on Wine (JS, Hellinger and composite tie at 0.950). | **Wrong.** |
| 5 | ANOVA F weakest on PhiUSIIL | ANOVA F 0.792 at k = 4 and 0.626 at k = 5, above Wasserstein and Bhattacharyya-free candidates but below the binned and MI metrics. | **Partly right.** Weak, not the weakest. |
| 6 | Scoring-time order, with KS cheap | KS took 22 s on PhiUSIIL, far above the others. | **Wrong on KS.** Mutual information was the second-slowest, as predicted. |

## Protocol confound found while reading the results

The classifier applies its own redundancy filter (`correlation_threshold` 0.85)
after the external selection. Higher-order keeps fewer features than requested
on PhiUSIIL (3.0–3.3 of 4–5 at `k` = 4–5, against 4.0–5.0 for the other
metrics). Its top features are correlated count features, so the filter removes
them. The accuracy reported here for higher-order therefore includes a loss from
that filter, not only from the score. The sweep in the paper used the classifier's
own selection, which can differ. Both protocols are reported; neither is the
clean test of the score alone. A clean test needs the filter off.

## What is established, and what is not

**Established by these runs**

- On PhiUSIIL at `k` = 4, four metrics (MI, KS, Hellinger, Jensen–Shannon) reach
  0.955, and higher-order reaches 0.869. Higher-order is not the best candidate here.
- Wasserstein is the only metric that fails badly at small `k` on PhiUSIIL
  (0.438 at `k` = 4), and it recovers by `k` = 15–20.
- On Breast Cancer, the spread among the top metrics is 0.92–0.93, inside the seed
  spread (about ±0.03). No ranking among them is supported there.
- The scoring-cost ordering is set by the implementation, not the metric: KS
  calls `ks_2samp` once per feature and class (which computes p-values), and the
  binned metrics do a single histogram per feature.

**Hypotheses, not yet tested**

- *Binned metrics avoid binary features.* With quantile edges, a binary feature
  collapses to one or two bins. This should make Hellinger and Jensen–Shannon
  prefer continuous features. The runs did not record which features each metric
  kept, so this is untested.
- *Why Wasserstein and KS disagree.* For a binary feature, the one-vs-rest
  Wasserstein distance and the KS statistic are the same quantity (the difference
  in class proportions). Their large disagreement on PhiUSIIL must therefore come
  from non-binary features. Which features each metric selects at `k` = 4 would
  show this. Not recorded.

**Next step, to settle the hypotheses:** rerun with the classifier's redundancy
filter disabled, and log the selected feature names per metric, seed and `k`.
Then the explanation for each metric can rest on which features it picks, not
only on accuracy.

## Filter-off rerun (2026-11-22): correction

The rerun with `correlation_threshold=1.0` (filter off) and per-run feature
logging is committed in `results.csv` / `results.md`. The filter-on results are in
commit `26b68b9`.

Correction to the confound note above. Removing the filter **raised** the
higher-order result at k = 4 on PhiUSIIL from 0.869 (3.0 features kept) to 0.843
(4.0 kept). So the filter was helping higher-order on that dataset, not hurting it.
Every other metric kept the requested number of features on PhiUSIIL under both
settings, and their accuracies did not change.

| PhiUSIIL, k = 4 | filter on | filter off |
|---|---|---|
| mutual_info, ks, hellinger, jensen_shannon | 0.955 | 0.955 |
| energy | 0.924 | 0.924 |
| higher_order | 0.869 (3.0 kept) | 0.843 (4.0 kept) |
| composite | 0.762 | 0.762 |
| wasserstein | 0.438 | 0.438 |

On Breast Cancer and Wine the filter changes a few values, several within the
seed spread. Composite on Wine drops from 0.950 to 0.931 with the filter off, and
energy on Breast Cancer drops from 0.900 to 0.874.

The conclusion holds. Higher-order is not the best candidate on PhiUSIIL, and the
filter does not explain its gap. The redundancy filter is still a real problem:
it changes which features the classifier uses, silently. The `tribble-fis` issue
should be read in that light. Its "accuracy comparisons mix score and filter
effects" point is true, but the example it gives runs the other way here.

Still open: the feature-level logging in `results.csv` (the `features` column) has
not yet been checked against the hypotheses about binary features. That is the next
analysis step.

## Library scorers (2026-11-22, tribble-fis `dd72d5e`)

tribble-fis #253 added `hellinger`, `jensen_shannon`, `ks` and `mutual_info` as built-in `scoring_method`s. To
rerun this experiment through the library instead of this file's reference implementations:

```bash
FSM_METRICS="lib:ks,lib:hellinger,lib:jensen_shannon,lib:mutual_info,lib:higher_order" \
FSM_OUT=results_library uv run --project tribble-fis python experiments/feature-scoring-metrics/explore.py
```

The output is in `results_library.csv` / `results_library.md`. `lib:higher_order` is a control: it is the same
library path as `higher_order` in `results.csv`.

| Scorer | Same features as reference (PhiUSIIL / Breast Cancer / Wine) | Max abs. accuracy change in a single run | Scoring time per fit (lib vs ref) |
|---|---|---|---|
| `higher_order` (control) | 100% / 100% / 100% | 0 | 0.03 s vs 0.07 s |
| `ks` | 100% / 100% / 100% | 0 | 0.04 s vs 9.5 s |
| `hellinger` | 100% / 90% / 86% | 0.056 (Wine, one seed) | 0.06 s vs 0.09 s |
| `jensen_shannon` | 100% / 98% / 88% | 0.028 | 0.05 s vs 0.11 s |
| `mutual_info` | 100% / 98% / 86% | 0.056 (Wine, one seed) | 0.49 s vs 0.78 s |

- Mean accuracies agree to within 0.009 everywhere, well inside the seed spread (±0.03–0.08). PhiUSIIL is unchanged
  in every run, so every conclusion above stands.
- The Breast Cancer and Wine differences have known causes:
  - The library bins Hellinger and JS on quantiles of the pooled class-vs-rest sample, with inverted-CDF edges and
    `searchsorted`. The reference bins on the whole feature with `np.histogram`.
  - The library's MI uses `random_state=0` on a standardised, mean-imputed matrix. The reference used the split
    seed.
- KS's speedup comes from computing the statistic directly instead of calling `scipy.stats.ks_2samp`, which also
  computes p-values.

