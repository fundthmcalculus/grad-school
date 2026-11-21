---
title: "Feature Engineering as Fuzzy Model Construction: Answer-First Mixtures of Gaussians for Interpretable Rule Bases"
author: "[Author names — IFSA-EUSFLAT template, anonymised for review]"
bibliography: references.bib
---

<!-- DRAFT v0.3. Numbers are tagged in NUMBERS.md. [PROVISIONAL] marks a value whose current source is not yet re-run. -->

# Abstract

Fuzzy rule-based models are interpretable only if their inputs are prepared for the membership functions they receive,
and the usual pipeline leaves that preparation implicit. We treat feature engineering as part of model construction.
Inputs are transformed onto a bounded domain, features are scored for class separation, and each class is then described
by a one-dimensional Gaussian mixture per retained feature. The product over features defines one rule per class, so the
rule base grows linearly in features and classes rather than exponentially in input sets. On a regression benchmark the
construction matches ANFIS at first order and trains in a fraction of a second. On a phishing benchmark the shipped
configuration scores 0.440, and a sweep over feature count and scorer reaches 0.985, a ceiling chosen on the test splits.
The gap is a feature-selection effect, and the scorer determines which features make a readable rule base possible at
all. A leakage audit shows that an earlier near-perfect single-clause result came from a label-derived feature.

**Keywords:** fuzzy inference systems, Gaussian mixtures, feature engineering, interpretability, leakage

# 1. Introduction

A fuzzy rule-based model is usually argued for on interpretability: each rule is a readable conjunction of linguistic
terms that a practitioner can inspect and edit. That argument holds only for the rules actually produced. Grid-based
construction partitions every input into a few fuzzy sets and forms a rule for every combination, so the rule count is
the product of per-input set counts. With a dozen inputs and three sets each, that is over half a million rules. The
usual remedies (genetic tuning, gradient descent as in ANFIS, clustering) either reduce the rule count slowly or depend
on an initial structure.

We take a different view. Feature engineering is normally treated as preprocessing, with the fuzzy model built on
whatever it receives. For Gaussian membership functions that separation is wrong. A membership function is defined by a
location and width in the feature's own units, so a monotone rescaling changes what the model learns. Equally, a feature
screen that consults the labels can pass the label through. We therefore treat the transform, the screen and the fit as
one construction whose output is the rule base.

The contributions are:

1. An **answer-first** construction for classification: segment the data by class, score features per class, and fit one
   rule per class, so the number of rules equals the number of answers.
2. A **per-feature, per-class Gaussian-mixture antecedent**, combined within a feature by a t-conorm and across features
   by a t-norm, with parameters growing linearly in the feature count.
3. An empirical account of the **input transform**, of the **feature scorer and feature count**, and of a **leakage
   failure** that changed an earlier headline result. Readability, not only accuracy, depends on which features are kept.

# 2. Background and related work

Building a fuzzy model from the output side is not new. Sugeno and Yasukawa [@sugeno1993qualitative] identify a model by
clustering the output and working back to input structure. Wang and Mendel [@wang1992generating] generate rules from
data by a different route, and Chiu [@chiu1994fuzzy] places rules by subtractive clustering. ANFIS [@jang1993anfis] tunes
a Takagi–Sugeno model by gradient descent. Genetic tuning of fuzzy premises is surveyed in [@cordon2001genetic;
@herrera2008genetic].

The construction is closely related to Gaussian naive Bayes [@hastie2009elements]. With one Gaussian per feature and
class and a product t-norm, the rule score for class $k$ is $\log p_k(x) + \sum_j \log(\sqrt{2\pi}\,\sigma_{jk})$. The
argmax is the naive-Bayes decision under uniform priors, plus per-class normalising constants that favour classes whose
retained features are wider. We claim no new density estimator. The contribution is an *interpretable* FIS, with real
membership functions and linguistic rules, that is cheap to fit because of the independence factorisation.

Feature-separation scorers draw on divergences between class-conditional distributions: the Bhattacharyya coefficient
[@bhattacharyya1943], the Jensen–Shannon divergence [@lin1991divergence], and the Wasserstein distance [@villani2008optimal].
Leakage in the sense used here, a feature that encodes the target through the data-collection process, is treated in
[@kaufman2012leakage].

# 3. Method

## 3.1 Preparing the inputs

Features whose dynamic range spans more than two orders of magnitude are log-scaled automatically, and every feature is
then mapped onto $[0,1]$ by min–max scaling. We compare this with z-scoring ($\mu = 0$, $\sigma = 1$) in §4.

## 3.2 Scoring features for class separation

The screen ranks features by how well their class-conditional distributions separate, keeping the best few. For $N$
samples, $M$ features and $K$ classes, the data are segmented by class and each feature is scored. Four scorers are
compared:

- **Wasserstein** (shipped default in `tribble-fis`): the 1-Wasserstein distance between class-conditional samples, in
  the data's own units.
- **Bhattacharyya**: the Bhattacharyya measure between class-conditional distributions.
- **Composite**: a consensus of four measures of separation, namely Bhattacharyya divergence, Jensen–Shannon distance,
  histogram overlap and histogram correlation. The composite is $\tfrac12(\bar a + \bar g)$, the mean of the arithmetic
  and geometric means of the four terms. The geometric term approaches zero if *any* measure is zero, so a feature scores
  well only when all four agree.
- **Higher-order**: the `higher_order` scorer added to `tribble-fis` in tribble-fis #239. Its definition is in the library
  documentation and is not restated here; we report only its measured behaviour.

The composite evaluated here is the `method="composite"` in the pinned `tribble-fis` (`2bdf875`). It is not verified to be
identical to the original four-metric blend, whose exact weighting has not been re-measured, and it is reported as such.
The screen costs $\mathcal{O}(M K^2)$ pairwise class comparisons: linear in features, quadratic in classes.

**The screen uses labels, so it is a leakage surface.** Any feature that encodes the target will score highly, and §5.3
shows this happening.

## 3.3 One rule per class

For each retained feature and class, a one-dimensional Gaussian mixture of up to a few components is fit. Components
are combined *within* a feature by a t-conorm, so a class is recognised if one Gaussian fires or another does. Features
are combined *across* by a t-norm:

$$
R_k:\quad \text{IF } \bigwedge_{j \in \mathcal{F}} \bigvee_{c} \mu_{jkc}(x_j) \text{ THEN } y = k,
$$

where $\mathcal{F}$ is the retained feature set and $c$ indexes mixture components. Prediction is the argmax over the $K$
rules. There is no grid, so there is no exponential rule base.

## 3.4 Correction rules

Feature independence costs accuracy where classes are confused. A small cascade adds a binary expert per confused pair,
gated by anomaly level and confidence margin. Flattening the cascade recovers only part of the gain, and we do not
headline it (§5.5).

# 4. Why the transform is not optional

Gaussian membership is not invariant to rescaling. An affine map can be absorbed into a re-fitted centre and width, but
a skewed feature's Gaussian fits the skew rather than the structure. Axis-aligned trees are invariant to any monotone
map, since they split on rank, which gives a control: if a reference tree moves under the transform, the plumbing is at
fault.

**Table 1.** [PROVISIONAL] Effect of the input transform on Concrete, R² mean ± s.d. over ten seeds. Source is the
backfilled `pins-synced-2026-08-27` archive, not re-run at `2bdf875` (NUMBERS.md, T1).

| Model                                | raw features  | log + min–max | log + z-score | Δ min–max − raw |
|--------------------------------------|--------------:|--------------:|--------------:|----------------:|
| Flat MoG-TSK, 1st order              | 0.695 ± 0.031 | 0.796 ± 0.019 | 0.713 ± 0.037 | +0.101 |
| Flat MoG-TSK, 2nd order              | 0.804 ± 0.032 | 0.841 ± 0.022 | 0.827 ± 0.029 | +0.037 |
| Flat MoG-TSK, full 2nd order         | 0.830 ± 0.027 | 0.861 ± 0.028 | 0.809 ± 0.121 | +0.030 |
| Fuzzy tree (library default)         | 0.583 ± 0.071 | 0.689 ± 0.059 | 0.691 ± 0.058 | +0.106 |
| Mixture of experts (library default) | 0.631 ± 0.120 | 0.758 ± 0.062 | 0.758 ± 0.062 | +0.126 |
| CART (control)                       | 0.825 ± 0.049 | 0.826 ± 0.049 | 0.826 ± 0.049 | +0.001 |
| Random Forest (control)              | 0.909 ± 0.019 | 0.909 ± 0.020 | 0.909 ± 0.019 | +0.000 |

Bounding is worth about a tenth of $R^2$ at first order. Centring costs 0.08 against bounding at first order, and its
main effect is instability at the top of the order ladder (±0.121 against ±0.028). The controls move by at most 0.001.

# 5. Experiments

## 5.1 Feature scorers and feature count on PhiUSIIL

**Table 2.** Test accuracy against features kept, PhiUSIIL, ten seeds, leak-free. Verified at `tribble-fis` `2bdf875`
(`reproduce/outputs/bump-2bdf875-2026-11-21/table_a2_feature_count.md`, appendix A.2). That table uses a 20,000-row sample.

| Features kept | Wasserstein | Bhattacharyya | Composite | Higher-order |
|--------------:|------------:|--------------:|----------:|-------------:|
| 1  | 0.5733 | **0.7924** | **0.7924** | 0.6364 |
| 2  | 0.5733 | 0.7924 | 0.7924 | 0.7781 |
| 3  | 0.5733 | 0.8106 | 0.7924 | 0.8603 |
| 4  | 0.4384 | 0.8175 | 0.7624 | **0.9011** |
| 5  | 0.4401 | 0.8181 | 0.9186 | **0.9261** |
| 7  | 0.6372 | 0.8191 | 0.9496 | **0.9361** |
| 10 | 0.7204 | 0.9575 | 0.9761 | 0.9373 |
| 15 | 0.9571 | 0.9755 | 0.9766 | 0.9632 |
| 20 | 0.9700 | 0.9799 | **0.9827** | 0.9767 |

The one-feature result is the same as before the bump: no single feature is enough under any scorer. The best one-feature
accuracy is 0.7924. Wasserstein is non-monotone (0.5733 at three features, 0.4384 at four, 0.6372 at seven). We do not offer
a mechanism for this. Higher-order is best at four and five features, and composite from seven on.

**Table 3.** Top five features by scorer, PhiUSIIL, normalised scores. Verified at `2bdf875` (`table_a1_feature_ranking.md`).

| Rank | Wasserstein | Bhattacharyya | Composite | Higher-order |
|-----:|---|---|---|---|
| 1 | HasSocialNet (1.000) | IsHTTPS (1.000) | IsHTTPS (1.000) | NoOfLettersInURL (1.000) |
| 2 | HasCopyrightInfo (0.858) | HasSocialNet (0.848) | HasSocialNet (0.975) | URLLength (0.852) |
| 3 | HasDescription (0.726) | NoOfQMarkInURL (0.810) | HasTitle (0.827) | LineOfCode (0.564) |
| 4 | DomainTitleMatchScore (0.544) | IsDomainIP (0.761) | NoOfCSS (0.800) | NoOfExternalRef (0.500) |
| 5 | HasSubmitButton (0.539) | NoOfEqualsInURL (0.744) | NoOfSelfRef (0.795) | NoOfPopup (0.417) |

The scorers disagree at rank 1, and the disagreement is not a scale artifact. IsHTTPS is constant across the legitimate
class, so the scorers that rank it first are ranking a property of collection rather than of phishing.

## 5.2 Shipped configuration against the sweep, and against the baselines

The shipped MoG configuration uses the Wasserstein scorer and the top five features. Table 4 reports it beside the best
cell of a sweep over scorer and feature count (`top_n` from 1 to 47), and beside ANFIS, GA-FIS and a Random Forest.

**Table 4.** Scores, mean ± s.d. over ten seeds. Verified at `2bdf875` (`table_4_1.md`, `table_4_1c_topn_sweep.md`), except
where marked. *Best sweep cell* is chosen on the same test splits it is scored on, so it is a **ceiling, not a result**.

| Dataset (task) | MoG shipped | MoG best sweep cell\* | ANFIS | GA-FIS | RF |
|---|---|---|---|---|---|
| Concrete (R²)        | 0.799 ± 0.027 | 0.799 ± 0.027 (k = 8) | 0.799 ± 0.112 | 0.896 ± 0.038 | 0.909 ± 0.020 |
| Bike Sharing (R²)    | 0.620 ± 0.014 | 0.655 ± 0.012 (Wasserstein, k = 8) | 0.538 ± 0.092 | 0.545 ± 0.047 | 0.944 ± 0.004 |
| PhiUSIIL (accuracy)  | 0.440 ± 0.181 | 0.985 ± 0.003 (Wasserstein, k = 30) | 0.998 ± 0.001 | 0.998 ± 0.001 | 1.000 ± 0.000 |
| RT-IOT2022 (12-class) | 0.927 ± 0.002 [PROVISIONAL] | not swept | 0.983 ± 0.003 [PROVISIONAL] | 0.988 ± 0.002 [PROVISIONAL] | 0.998 ± 0.000 [PROVISIONAL] |

\* The RT-IOT2022 row is from the older `phiusiil-leakfree-2026-08-30` archive, which failed preflight. The dataset is
absent from the `2bdf875` run, so this row has not been re-run at the current pin.

**Table 5.** PhiUSIIL accuracy by feature count and scorer, from the sweep archive `topn-sweep-2026-11-21`. Full-data cells,
not the 20,000-row sample of Table 2. At k = 20 the values agree with Table 2 to within 0.002.

| k | Wasserstein | Bhattacharyya | Composite | Higher-order |
|--:|---|---|---|---|
| 1  | 0.573 ± 0.003 | 0.792 ± 0.004 | 0.792 ± 0.004 | 0.636 ± 0.062 |
| 3  | 0.573 ± 0.003 | 0.811 ± 0.004 | 0.792 ± 0.004 | 0.860 ± 0.063 |
| 5  | **0.440 ± 0.181** | 0.818 ± 0.022 | 0.919 ± 0.020 | 0.926 ± 0.019 |
| 10 | 0.720 ± 0.063 | 0.958 ± 0.010 | 0.976 ± 0.004 | 0.937 ± 0.021 |
| 20 | 0.970 ± 0.004 | 0.980 ± 0.003 | 0.983 ± 0.004 | 0.977 ± 0.003 |
| 30 | **0.985 ± 0.003** | 0.982 ± 0.002 | 0.982 ± 0.003 | 0.983 ± 0.003 |

The shipped cell (Wasserstein, k = 5) has a seed spread of ±0.181, which is large against its mean. The MoG does not fail
on PhiUSIIL as a model. It fails at one scorer and one feature count, and the sweep's ceiling is 1.3 points below ANFIS. Because the
ceiling is chosen on the test splits, a fair comparison needs a nested selection, which we have not run.

On Concrete the sweep does not improve on the shipped configuration, and GA-FIS is better. On Bike Sharing the shipped
MoG (0.620) beats both ANFIS (0.538) and GA-FIS (0.545). An earlier archive gave ANFIS 0.577 on the same dataset. The two
ANFIS figures differ by more than the reported spread, and ANFIS is nondeterministic run to run, so we quote the current
value and flag the difference.

## 5.3 A leakage failure in the feature screen

An earlier PhiUSIIL result reported a two-rule classifier at 0.997 accuracy with a single clause. The leak was
`URLSimilarityIndex`, a URL's similarity to a whitelist of known-legitimate URLs. It is the label in disguise, and at
AUC 0.996 it was the most separating feature in the file. Two further features, `TLDLegitimateProb` and `URLCharProb`,
were probabilities fitted on the corpus's own labels. All three are now dropped on load. Without them the one-feature
accuracy falls to 0.7924 (Table 2), and the single-clause result does not survive under any scorer.

The lesson for feature engineering is general. A screen that uses labels must be audited for features that are the
label in disguise before accuracy is read from it, and a near-perfect separation on a public dataset is a reason to
audit, not a result. Accuracy-only evaluation did not surface this failure. Readability was the thing it damaged.

## 5.4 Timing

**Table 6.** [PROVISIONAL] Training time in seconds, mean ± s.d. over ten seeds, from the `phiusiil-leakfree-2026-08-30`
archive. That archive failed preflight, so ANFIS and GA-FIS timings have not been re-run at `2bdf875`. The current
`2bdf875` run is on a 4-core VM and reports MoG times only (Concrete 0.24 s, Bike Sharing 0.44 s, PhiUSIIL 0.18 s).

| Dataset (task)        | MoG train   | ANFIS train   | GA-FIS train | Speedup vs slowest fuzzy |
|-----------------------|------------:|--------------:|-------------:|-------------------------:|
| Concrete (regression) | 0.53 ± 0.05 | 7.28 ± 1.12   | 41.11 ± 3.81 | 78× |
| Bike Sharing (regr.)  | 0.59 ± 0.07 | 4.80 ± 0.68   | 8.30 ± 1.72  | 14× |
| PhiUSIIL (classif.)   | 0.13 ± 0.02 | 22.53 ± 2.30  | 25.29 ± 1.55 | 194× |
| RT-IOT2022 (12-class) | 3.64 ± 0.25 | 264.53 ± 25.30 | 51.58 ± 1.25 | 73× |

The construction is one to two orders of magnitude faster than the slowest fuzzy baseline on these four datasets. The
speedup range is provisional until the timings are re-run on one host.

## 5.5 The correction pass

On Glass, the gated correction cascade gained +0.031 ± 0.027 accuracy over the flat base under an earlier measurement.
Under a later dependency revision the gain fell to +0.008 ± 0.011, with a 95% interval spanning zero. We report the
later figure and claim no gain. This row is stale and has not been re-derived at ten seeds on the current pin.

# 6. Limitations

- **Sweep ceiling.** The best sweep cell is chosen on the test splits. A nested selection is needed before any sweep
  result is compared against a baseline.
- **Scorer generality.** The scorer and feature-count results are on one dataset, which is unusually concentrated. The
  broader suite that tests scorer choice where signal is spread is not yet run.
- **Independence.** Where feature interactions carry the class, the flat model leaves accuracy on the table. The
  hierarchical models of the dissertation's Chapter 6 address this; they are not evaluated here.
- **Baselines.** Gaussian naive Bayes is the nearest relative and has not been run head-to-head on identical splits.
- **Pin and run of record.** The current pin is `2bdf875`. No whole-document sweep has been run at that pin, and the run of
  record is still `full-2026-08-22` at `141596e`. RT-IOT2022 is absent from the current run.
- **Provenance.** Table 1 (transform), Table 4's RT-IOT2022 row, Table 6 (timing) and the correction-pass figure are
  provisional until re-run (NUMBERS.md, open items).

# 7. Conclusion

Treating feature engineering as part of fuzzy model construction puts the transform, the scorer, the feature count and
the rule base in one audit trail. The construction is fast and yields one rule per class. Its readability depends on
which features are kept, the scorer and feature count determine that choice, and any sweep over them must be validated
on held-out data before its ceiling is believed. The scorer must also be checked for label leakage before any accuracy is
believed.

# Acknowledgements

[To be completed.]

# References
