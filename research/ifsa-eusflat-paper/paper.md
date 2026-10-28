---
title: "Feature Engineering as Fuzzy Model Construction: Answer-First Mixtures of Gaussians for Interpretable Rule Bases"
author: "[Author names — IFSA-EUSFLAT template, anonymised for review]"
---

<!-- DRAFT v0.1. Target: IFSA-EUSFLAT 2027 full paper (page limit per the conference template; this draft is ~3,400 words, roughly 8 pages before tables). Numbers are tagged in NUMBERS.md. Items marked [PROVISIONAL] must not be cited until re-run. -->

# Abstract

Fuzzy rule-based models are interpretable only if their inputs are prepared for the membership functions they will
receive, and the usual pipeline leaves that preparation implicit. We treat feature engineering as part of model
construction. Inputs are first transformed onto a bounded domain, features are screened for class-discriminative signal
without touching the target labels in the screen, and each class is then described by a one-dimensional Gaussian
mixture per retained feature. The product over features of the resulting memberships defines one rule per class. The
rule base therefore grows linearly in the number of features and classes, not exponentially in the number of input
sets. On a regression benchmark the construction reaches the accuracy of ANFIS at first order and trains in a fraction
of a second; on a 12-class intrusion dataset it trails the baselines, and a leakage audit of the feature screen shows why
a model that appears to separate a dataset perfectly may have been given the label. We report the transform's effect,
the timing, and the failure, with the provenance of each number.

**Keywords:** fuzzy inference systems, Gaussian mixtures, feature engineering, interpretability, leakage

# 1. Introduction

A fuzzy rule-based model is usually argued for on interpretability: each rule is a readable conjunction of linguistic
terms, and a practitioner can inspect and edit it. That argument holds only for the rules that are actually produced.
Grid-based construction, which partitions every input into a few fuzzy sets and forms a rule for every combination,
gives a rule count equal to the product of the per-input set counts. With a dozen inputs and three sets each, that is
over half a million rules before any data is seen. The usual remedies, genetic search, gradient descent (ANFIS), or
clustering, reduce the rule count or tune the sets, but they make training slow or depend on an initial structure.

This paper takes a different view of the same problem. The feature engineering that precedes a model is normally
treated as preprocessing, and the fuzzy model is then built on whatever it receives. For Gaussian membership functions
that separation is wrong: a membership function is defined by a location and a width in the feature's own units, so a
monotone rescaling of an input changes what the model learns. Equally, a feature screen that looks at the labels can
silently pass the label through. We therefore treat the three steps (transform, screen, fit) as one construction with
the rule base as its output.

The contributions are:

1. An **answer-first** construction for classification: segment the data by class, screen features per class, and fit
   one rule per class, so that the number of rules equals the number of answers.
2. A **per-feature, per-class Gaussian-mixture antecedent**, combined within a feature by a t-conorm and across features
   by a t-norm, with parameters growing linearly in the feature count.
3. An empirical account of the **input transform** (bounded versus centred scaling) and of a **leakage failure** in an
   earlier feature screen, with the numbers that changed when the leak was removed.

# 2. Background and related work

Building a fuzzy model from the output side is not new. Sugeno and Yasukawa [@sugeno1993qualitative] identify a model by
first clustering the output and working back to input structure. Wang and Mendel [@wang1992generating] generate rules
from data by a different route, and Chiu [@chiu1994fuzzy] places rules by subtractive clustering. Jang's ANFIS
[@jang1993anfis] tunes a Takagi–Sugeno model by gradient descent on a grid-partitioned or scatter-partitioned
antecedent. Genetic tuning of FIS premises is the common alternative [@TODO-gafis-key].

Our construction is closely related to Gaussian naive Bayes. With one Gaussian per feature and class and a product
t-norm, the rule score for class $k$ is $\log p_k(x) + \sum_j \log(\sqrt{2\pi}\,\sigma_{jk})$. The argmax is therefore
the naive-Bayes decision under uniform priors, plus per-class normalising constants that favour classes whose retained
features are wider. We do not claim a new density estimator. The contribution is that the factorisation yields an
*interpretable* FIS, with real membership functions and linguistic rules, that is cheap to fit, and that the
independence assumption is what makes it cheap.

# 3. Method

## 3.1 Preparing the inputs

Before either step, inputs are transformed. Features whose dynamic range spans more than two orders of magnitude are
log-scaled automatically; every feature is then mapped onto $[0,1]$ by min–max scaling. We compare this with z-scoring
(μ = 0, σ = 1). The motivation is stated in §4 and tested in Table 1. A bounded domain is the domain the construction
assumes; a centred one is not.

## 3.2 Class-conditional feature screening

For a problem with $N$ samples, $M$ features and $K$ classes, the data are segmented by class. For each feature, the
class-conditional distributions are compared pairwise, using a Wasserstein-type distance in the data's own units, and a
feature is retained if it separates at least one pair of classes. The screen costs $\mathcal{O}(M K^2)$ comparisons:
linear in features, quadratic in classes. At $K = 12$ this is not the bottleneck, but a one-versus-rest variant would
reduce it without changing the rule base.

**The screen uses labels, so it is a leakage surface.** Any feature derived from the target, or correlated with it
through the data-collection process, will score highly. We return to this in §5.3.

## 3.3 One rule per class

For each retained feature and class, a one-dimensional Gaussian mixture of up to a few components is fit. Its components
are combined *within* the feature by a t-conorm, so a class is recognised if one Gaussian fires or another does. Features
are combined *across* by a t-norm. The rule for class $k$ is therefore a conjunction of disjunctions:

$$
R_k:\quad \text{IF } \bigwedge_{j \in \mathcal{F}} \bigvee_{c} \mu_{jkc}(x_j) \text{ THEN } y = k,
$$

where $\mathcal{F}$ is the retained feature set and $c$ indexes mixture components. Prediction is the argmax over the $K$
rules. There is no grid, so there is no exponential rule base. A regression variant places the output on $P$ buckets
and regresses within each bucket; we report it only in the appendix.

## 3.4 Correction rules

Feature independence costs accuracy where classes are confused. A small cascade adds a binary expert for each confused
pair, gated by anomaly level and confidence margin. Flattening the cascade into one deployable FIS recovers only part
of the gain. We do not headline this step: its measured gain is at the edge of the noise (§5.2).

# 4. Why the transform is not optional

Gaussian membership is not invariant to rescaling. Under an affine map the centre and width can be re-fitted, but a
skewed feature's Gaussian fits the skew rather than the structure. Axis-aligned trees are invariant to any monotone
map, because they split on rank. That gives a control: if a reference tree moves under the transform, the plumbing is
at fault.

**Table 1.** Effect of the input transform on Concrete (R², mean ± s.d. over ten seeds). [PROVISIONAL — see NUMBERS.md, row T1.]

| Model                          | raw features | log + min–max | log + z-score | Δ min–max − raw |
|--------------------------------|-------------:|--------------:|--------------:|----------------:|
| Flat MoG-TSK, 1st order        | 0.695 ± 0.031 | 0.796 ± 0.019 | 0.713 ± 0.037 | +0.101 |
| Flat MoG-TSK, 2nd order        | 0.804 ± 0.032 | 0.841 ± 0.022 | 0.827 ± 0.029 | +0.037 |
| Flat MoG-TSK, full 2nd order   | 0.830 ± 0.027 | 0.861 ± 0.028 | 0.809 ± 0.121 | +0.030 |
| Fuzzy tree (library default)   | 0.583 ± 0.071 | 0.689 ± 0.059 | 0.691 ± 0.058 | +0.106 |
| Mixture of experts (library default) | 0.631 ± 0.120 | 0.758 ± 0.062 | 0.758 ± 0.062 | +0.126 |
| CART (control)                 | 0.825 ± 0.049 | 0.826 ± 0.049 | 0.826 ± 0.049 | +0.001 |
| Random Forest (control)        | 0.909 ± 0.019 | 0.909 ± 0.020 | 0.909 ± 0.019 | +0.000 |

Three points follow. Bounding is worth about a tenth of $R^2$ at first order. Centring costs 0.08 at first order against
bounding, and its effect is mostly on stability at the top of the order ladder (±0.121 against ±0.028). The controls
move by at most 0.001, so the movement in the fuzzy rows is the model responding to the transform, not the pipeline.

# 5. Experiments

## 5.1 Accuracy and cost against fuzzy baselines

**Table 2.** Training time (s) and accuracy or R², mean ± s.d. over ten seeds. ANFIS and GA-FIS are the fuzzy baselines;
RF is a reference. [PROVISIONAL — the source archive failed its preflight check; see NUMBERS.md, rows T2–T4.]

| Dataset (task)          | MoG train | ANFIS train | GA-FIS train | MoG score | ANFIS score | GA-FIS score | RF score |
|-------------------------|----------:|------------:|-------------:|----------:|------------:|-------------:|---------:|
| Concrete (regression)   | 0.53 ± 0.05 | 7.28 ± 1.12 | 41.11 ± 3.81 | 0.799 ± 0.027 | 0.799 ± 0.112 | 0.896 ± 0.038 | 0.909 ± 0.020 |
| Bike Sharing (regr.)    | 0.59 ± 0.07 | 4.80 ± 0.68 | 8.30 ± 1.72  | 0.620 ± 0.014 | 0.577 ± 0.075 | 0.545 ± 0.047 | 0.944 ± 0.004 |
| PhiUSIIL (classif.)     | 0.13 ± 0.02 | 22.53 ± 2.30 | 25.29 ± 1.55 | 0.440 ± 0.181 | 0.999 ± 0.001 | 0.998 ± 0.001 | 1.000 ± 0.000 |
| RT-IOT2022 (12-class)   | 3.64 ± 0.25 | 264.53 ± 25.30 | 51.58 ± 1.25 | 0.927 ± 0.002 | 0.983 ± 0.003 | 0.988 ± 0.002 | 0.998 ± 0.000 |

On the two regression sets the construction trains one to two orders of magnitude faster than either fuzzy baseline.
At first order it matches ANFIS on Concrete and beats both on Bike Sharing; GA-FIS is better on Concrete. The
construction does not match the baselines on the two classification sets. The PhiUSIIL row is the one that needs
explaining, and §5.3 does.

## 5.2 The correction pass

On Glass, the gated correction cascade gained +0.031 ± 0.027 accuracy over the flat base under an earlier measurement.
Under a later dependency revision the gain fell to +0.008 ± 0.011, with a 95% interval spanning zero. We report the
later figure as the current estimate and do not claim a gain. [NUMBERS.md, row T6 — stale.]

## 5.3 A leakage failure in the feature screen

An earlier PhiUSIIL result reported a two-rule classifier at 0.997 accuracy. The screen's top-scoring feature was a
URL-similarity measure that encodes the label. A leak-free screen gives a different top set: on the leak-free top five
features, four are binary flags and one is a bounded score, and the same construction scores 0.440 ± 0.181 (Table 2).
We conclude that the high accuracy measured the leak, not the method. The lesson for feature engineering is general: a
screen that uses labels must be audited for features that are the label in disguise before any accuracy is read from
it, and a near-perfect separation on a public dataset is a reason to audit, not a result.

# 6. Limitations

- **Independence.** Where feature interactions carry the class, the flat model leaves accuracy on the table. This is the
  case the hierarchical models of the dissertation's Chapter 6 target; it is not addressed here.
- **Scale.** The RT-IOT2022 correction pass has not been run at its intended scale.
- **Baselines.** Gaussian naive Bayes is the nearest relative and has not yet been run head-to-head on identical splits.
- **Seeds.** Every reported cell is ten seeds; no narrowed run is quoted.
- **Provenance.** Table 2 is provisional (§5.1). A clean re-run is required before submission.

# 7. Conclusion

Treating feature engineering as part of fuzzy model construction makes the transform, the screen and the rule base one
object with one audit trail. The construction is fast and yields one readable rule per class. Its accuracy depends on the
input domain, and its feature screen must be checked for label leakage before any accuracy is believed. The
benchmark results here are a mixed picture and are reported as such.

# Acknowledgements

[To be completed.]

# References

<!-- Keys from research/proposal-defense/references.bib. TODO entries marked @TODO-... have no bib entry yet. -->

- [@sugeno1993qualitative] M. Sugeno and T. Yasukawa, "A fuzzy-logic-based approach to qualitative modeling," *IEEE TFS*, 1993.
- [@wang1992generating] L.-X. Wang and J. M. Mendel, "Generating fuzzy rules by learning from examples," *IEEE TSMC*, 1992.
- [@chiu1994fuzzy] S. L. Chiu, "Fuzzy model identification based on cluster estimation," *J. Intelligent & Fuzzy Systems*, 1994.
- [@jang1993anfis] J.-S. R. Jang, "ANFIS: adaptive-network-based fuzzy inference system," *IEEE TSMC*, 1993.
- [@TODO-gafis-key] GA-FIS reference: the exact source used for the GA-tuned baseline must be confirmed.
- [@TODO-nb-key] Gaussian naive Bayes reference (e.g., Hastie, Tibshirani & Friedman), to add.
- [@TODO-feature-screen-key] Feature-selection / leakage reference (e.g., Kaufman et al., 2012, *ACM TKDD*), to confirm.
