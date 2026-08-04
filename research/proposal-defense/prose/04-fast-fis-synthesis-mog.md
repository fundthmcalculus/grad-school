# Chapter 4 — Fast Interpretable FIS Synthesis via Mixtures of Gaussians

## 4.1 Introduction

In *The Hitchhiker's Guide to the Galaxy*, the supercomputer Deep Thought is asked for the Answer to the Ultimate Question of Life, the Universe, and Everything. It thinks for seven and a half million years and returns 42. Only then does anyone realize they never worked out what the Question was. The rest of the story is a search, backward, for a Question that fits the Answer.

This chapter is built on the same inversion, and that is why it is fast. A fuzzy rule has an antecedent (the IF part, a question about the inputs) and a consequent (the THEN part, the answer). The conventional way to build a fuzzy model lays out the questions first: partition every input into fuzzy sets, form a rule for every combination, then search for consequents that fit. I do the opposite. I commit to the answers first, the output classes or a set of output values spread across the range, and only then work backward to the antecedents that select them. That collapses most of the work: the answers are few and known, the space of possible questions enormous. And unlike Deep Thought, the method finishes in seconds instead of geological time, fast enough that the Vogons cannot demolish the Earth before it is done.

The problem is the one I set up in Chapter 1. A fuzzy model built by gridding the inputs has a rule count that is the product of the per-input set counts, which is exponential, and the usual way to fit it (a genetic algorithm, gradient descent, or ANFIS) is either slow to converge or dependent on an initial guess it cannot supply for itself. My claim is that a Mixture-of-Gaussians construction generates both the membership functions and the rules directly from the data, produces on the order of one rule per output class instead of an exponential blowup, and needs no post-hoc genetic search or gradient descent. The contributions: an answer-first (consequent-first) construction for classification and regression; a per-feature, per-class Gaussian-mixture antecedent generator whose parameter count grows linearly, not exponentially; an automatically synthesized *anomaly* rule, the complement of the t-conorm of every explicit rule, giving open-set behavior and a handle on rare classes at no extra training cost; and a demonstration that this trains competitive models in seconds on datasets with hundreds of thousands of rows.

## 4.2 Background and Prior Art

Building a model from the output side is not new. Sugeno and Yasukawa (1993) proposed identifying a fuzzy model by first clustering the *output* and working back to the input structure; that is the instinct I follow. Wang and Mendel (1992) generate rules from data by a different route, and Chiu (1994) uses subtractive clustering to place rules. What I add is a specific, cheap factorization of the antecedents and an explicit path to a compact rule base.

A caveat up front. For classification, fitting an independent Gaussian mixture per feature and per class and combining them is closely related to a Gaussian naive-Bayes model: a class-conditional density estimate with a feature-independence assumption. I am not going to pretend otherwise; what I claim is not a new density estimator. It is that this construction produces an *interpretable fuzzy inference system* (real membership functions, real linguistic rules, editable by hand) extremely fast, and that the naive-Bayes-like factorization is what keeps it so. Where the independence assumption costs accuracy, my intent is to recover it with a small number of correction rules instead of abandoning the factorization. That is a design intention, not a measured result: §4.3.1 concedes I have not isolated how much accuracy the second pass buys, and the concession stands open until Figure 4.3 exists.

## 4.3 Methodology

### Preparing the inputs, and why the transform is not optional

Before either construction, the inputs are transformed: features whose dynamic range spans more than a couple of orders of magnitude are log-scaled automatically, and every feature is then min-max scaled to $[0,1]$. On Concrete the automatic detector selects two, `Slag` and `Age`. `Age` is the unsurprising one, since curing time runs from one day to a year while most features live within a factor of a few.

One naming hazard before the numbers. This step is easy to call "standardization," and it is not standardization. The helper that implemented it, `gauss_math.standard_transform`, computed $(x - \min)/(\max - \min)$, min-max scaling onto the unit interval, and always computed exactly that despite its misleading name. `tribble-fis` has since split it into `UnitScalar` for the min-max transform and `StandardScalar` for genuine z-scoring, and `UnitScalar` reproduces the deleted helper bit for bit: $\max|\Delta| = 0.0$ on every call shape in use, with the same automatically detected log features. The relabelling moves no number in this chapter or in Chapter 6. Throughout I write "min-max scaled to $[0,1]$" or "unit scaled" and never the bare contraction, because Chapter 5 uses "minimax" for an unrelated bottleneck ultrametric and the two words differ by one character.

The transform is worth more than most of the modeling choices in this chapter, and *which* transform it is matters more than the fact that there is one. Measured across ten seeds on identical splits:

**Table 4.1 — What the transform is worth, by model, and what a centred transform costs.** All nine rows the generator emits; the pattern is the point, not any single row. Provenance: all three arms are now the run of record, `reproduce/outputs/full-2026-08-03/table_hyperparam_normalization.csv`, which carries the z-score column that the three-arm side study (`outputs/norm-three-arm-a385a1a/`) first measured.

| Model | Hyperparameters | raw features | log + min-max to $[0,1]$ | log + z-score | Δ min-max − raw | Δ z-score − min-max |
|---|---|---:|---:|---:|---:|---:|
| flat MoG-TSK, 1st order | pipeline default | 0.666 ± 0.041 | 0.787 ± 0.026 | 0.014 ± 0.195 | **+0.121** | **−0.772** |
| flat MoG-TSK, 2nd order | pipeline default | 0.804 ± 0.016 | 0.832 ± 0.027 | 0.787 ± 0.039 | +0.029 | −0.045 |
| flat MoG-TSK, full 2nd | pipeline default | 0.816 ± 0.052 | 0.873 ± 0.020 | 0.835 ± 0.036 | +0.057 | −0.037 |
| fuzzy tree | demo-tuned | 0.712 ± 0.030 | 0.740 ± 0.051 | 0.740 ± 0.051 | +0.028 | −0.000 |
| fuzzy tree | library default | 0.583 ± 0.067 | 0.689 ± 0.056 | 0.691 ± 0.055 | +0.106 | +0.002 |
| mixture of experts | demo-tuned | 0.770 ± 0.035 | 0.829 ± 0.022 | 0.698 ± 0.033 | +0.059 | −0.131 |
| mixture of experts | library default | 0.689 ± 0.066 | 0.760 ± 0.060 | 0.734 ± 0.062 | +0.071 | −0.026 |
| CART (reference) | sklearn default | 0.825 ± 0.047 | 0.826 ± 0.047 | 0.826 ± 0.046 | +0.001 | −0.000 |
| Random Forest (reference) | sklearn default | 0.909 ± 0.018 | 0.909 ± 0.019 | 0.909 ± 0.018 | +0.000 | −0.000 |

Three things fall out.

A **bounded** transform is worth twelve points of $R^2$ to the Gaussian models. A first-order model goes from 0.666 to 0.787 on nothing but a log and a rescale onto $[0,1]$. Any comparison that omits it is not measuring the method.

**The transform is worth essentially nothing to CART and Random Forest**: +0.001 and +0.000. That is not a rounding artifact; it is the control that licenses reading anything else in the table. An axis-aligned decision tree splits on rank, so a monotone transform cannot change whether a feature exceeds a threshold and the induced tree is identical. Min-max and z-score are both strictly monotone per feature, and across CART, Random Forest and both fuzzy-tree rows the largest movement between the two normalized arms is **0.002**, against their own seed spreads of ±0.018 to ±0.056. Column-wise Spearman correlation between the two scalers' outputs is 1.000000000000 on every feature. So when a fuzzy row moves between those columns, it is the model responding, not the plumbing. A Gaussian membership function has no such immunity: defined by a location and a width in the feature's own units, it fits a skewed feature's skew instead of its structure.

**The boundedness is the part that matters, and it is load-bearing.** Under the transform that misleading name would imply, real z-scoring to $\mu = 0$, $\sigma = 1$, the headline model does not merely do slightly worse. It collapses. The first-order flat MoG-TSK falls from 0.787 to **0.014 ± 0.195**, *below* its own raw-feature score of 0.666, with RMSE going from 7.537 ± 0.388 to 16.138 ± 1.196 MPa. Two innocent explanations were ruled out first, both in the three-arm side study and neither re-taken since the component-count change of §4.4. It is not a ridge-scale artifact: sweeping `l2_reg` over 1e-2, 1e-3, 1e-4, 1e-5, 1e-6 and 0 moves the first-order gap by 0.001. And it is not the BIC membership-count choice, which is genuinely not scale-invariant and does pick slightly different rule bases in the two arms; pinning `n_gaussians` so both arms get an *identical* rule base leaves the collapse intact at −0.407, −0.524 and −0.634 for two, three and four components. It underfits the *training* set as well (MSE 0.030 against 0.009 at seed 0), so it is a fitting failure and not an extrapolation failure. Raising the consequent order recovers most of it (−0.772 → −0.045 → −0.037), as one expects if what broke is the first-order affine term.

The claim the data supports: *bounded* normalization helps this construction a great deal, and *centred* normalization does not help it at all. A non-negative input domain bounded on $[0,1]$ is an assumption the construction rests on, not a convenience, and the same assumption the pinned extreme bucket means of §4.3.2 rely on. That is why that fix and this transform belong together.

The asymmetry also explains why my pipeline requires preprocessing the baselines do not, which would otherwise look like an inconsistency between chapters. Chapter 6's fuzzy tree is deliberately fit on *raw* features so its split thresholds stay physically meaningful ("cement ≥ 350", not "cement ≥ 0.42"), and it can afford to be, for the rank-invariance reason above. Nor does the transform smuggle information in: the strongest baseline is unmoved by either arm, so the rescaled features hold no signal the raw ones did not.

### 4.3.1 Classification: one rule per answer

For a classification problem with $N$ samples, $M$ features, and $K$ output classes, the construction runs answer-first. I segment the data by output class (the answers) and for each class I ask which features actually distinguish it, by comparing the per-feature statistics across classes, discarding features with no discriminative signal.

The cost needs stating correctly. `calculate_gaussian_correlation` runs once per feature and, inside each feature, compares the class-conditional distributions for every *pair* of classes, a double loop over $ij < jk$. The number of distribution comparisons is therefore $\mathcal{O}(M \cdot K^2)$: linear in the features and **quadratic in the class count**. It is the one cost here that grows quadratically, along the very axis RT-IOT2022 ($K = 12$) is invoked to showcase. On these sizes it is not the bottleneck, since the comparisons are cheap and $K$ is small, but it would bite first on a many-class problem. Being only a screen, it can be subsampled or restricted to one-versus-rest without touching the rule base.

For each retained feature and class I fit a one-dimensional Gaussian mixture of up to a few components, and that becomes the membership function for that feature under that class. The combination happens at two levels. *Within* a feature, the mixture's components are combined by a fuzzy OR, a t-conorm, so the class is recognized if this Gaussian fires or that one does. *Across* features, those results are combined by a t-norm. The rule for a class is therefore a conjunction of disjunctions, not a single disjunction, which is what `simple_gaussian_predict` implements and what Figure 4.1 draws. Repeating over the $K$ classes gives $K$ rules, each built from at most $M \times p$ Gaussian terms. There is no grid, so there is no exponential rule base.

Finally I evaluate the confusion matrix and add a second, small pass of correction rules where two classes are being confused, the place where the feature-independence assumption is costing me. These corrections are targeted and few, and they keep the rule base readable. I have not yet isolated how much accuracy the pass buys; that before-and-after is a small experiment I owe, and Figure 4.3 is where it belongs.

**Figure 4.1 — Per-feature Gaussian mixtures, and the rule they combine into.** A real fit from the same three calls the harness uses: the top three features for Glass Type 1, each feature's mixture components dashed and their t-conorm bold, the class's own samples a rug against everything else's. The lower panel is the rule's firing strength over every sample, the rule written underneath in the form the construction produces. A single rule is not a classifier, since prediction is the argmax over all $K$ rules, so the overlap is expected. Nor are the degenerate components an artifact of the drawing: a zero-width Gaussian on a single observation is what the automatic component count produces on a 214-sample dataset, visible because the figure fits the model rather than illustrating it.
`![mog-classification](fig/04-mog-classification.png)`

### 4.3.2 Regression: place the answers, regress the questions

For regression the answer-first construction is even more literal. I partition the output range first, and only then find the antecedents. How to place that partition is genuinely open.

Two natural choices trade against each other. **Uniform** partitioning spreads buckets at equal width across the output range, one centroid pinned at each extreme. That gives the more natural function approximation: every rule owns an equal span of the output, so consequents interpolate evenly and the extremes stay covered. But on a skewed target some buckets contain very few samples, and a bucket with almost no data yields a badly estimated rule or none at all. **Quantile** partitioning cuts the output into equal-frequency buckets instead, guaranteeing every rule enough data. The cost is that boundaries crowd where the data is dense and under-resolve where it is sparse, which for a regression target is often precisely the extremes, the values one usually cares most about.

Which wins should depend on the target's skew and the bucket count. I measured it.

**Table 4.2 — Output partitioning on Concrete** (target skew +0.42; 10 seeds; tail RMSE over the true bottom and top deciles; "min bucket" is the smallest training-bucket occupancy). All three arms are printed, including the shipped hybrid. Nothing is marked as a winner in the $R^2$ or tail columns; every separation there is smaller than the seed spread producing it. Rows are from `reproduce/outputs/full-2026-08-03/table_g5_output_partitioning.csv`.

| buckets | order | scheme | R² | tail RMSE | min bucket |
|---:|---|---|---:|---:|---:|
| 3 | 1st | uniform | 0.796 ± 0.018 | 8.10 | 132 |
| 3 | 1st | quantile | 0.789 ± 0.026 | 8.08 | 343 |
| 3 | 1st | hybrid *(shipped)* | 0.787 ± 0.026 | 7.93 | 343 |
| 4 | 1st | uniform | 0.799 ± 0.025 | 7.90 | 75 |
| 4 | 1st | quantile | 0.795 ± 0.024 | 8.30 | 257 |
| 4 | 1st | hybrid *(shipped)* | 0.797 ± 0.024 | 8.18 | 257 |
| 6 | 2nd | uniform | 0.853 ± 0.018 | 6.40 | 39 |
| 6 | 2nd | quantile | 0.853 ± 0.020 | 6.34 | 171 |
| 6 | 2nd | hybrid *(shipped)* | 0.852 ± 0.019 | 6.34 | 171 |

On accuracy this is a null result, and the reason is effect size rather than any ordering. The largest uniform-versus-quantile gap anywhere in the sweep (eighteen rows: six bucket-and-order configurations against three schemes) is 0.007 in $R^2$, at three buckets and first order, where the two arms carry ±0.018 and ±0.026 of their own. Seed-to-seed deviations across that sweep run from ±0.018 to ±0.027, so the *widest* separation the experiment produces is smaller than the *narrowest* single-arm deviation it reports. No configuration clears its own noise floor. Uniform leads at three buckets by 0.007, and at six buckets and second order the two read 0.853 against 0.853, level to every printed digit. Nor is the ordering stable enough to be worth reading: it reshuffled between the previous run of record and this one, on a change to nothing but how the component count is chosen, which is what an effect below the noise floor does.

The starvation mechanism is real even though the accuracy difference is not. Uniform's smallest bucket falls from 132 samples to 75 to 39 as the partition refines, while quantile's floor stays high by construction: 343, 257, 171. That predictable structural difference is why uniform must eventually fail as buckets increase or skew grows. Concrete's skew of +0.42 is too mild for it to bite in the range tested, which is why the next experiment isolates skew directly.

The tails do not settle it either. Uniform holds them in one of the three pairs above, by 0.40 MPa, and loses the other two by 0.02 and 0.06. At the previous run of record it held two of three. An ordering that reshuffles between runs is not a finding; it is the same noise the accuracy column shows, through a different metric.

The hybrid is not a third option. It is a defect: `partition_output` takes equal-frequency boundaries and then pins the two extreme bucket centroids to the observed min and max, but the closed-form consequent solve re-derives its own bucket means, so the pinning never reaches inference. A.6 has the investigation that found it.

What the table shows is the bound that let it hide: **the pinned and unpinned arms differ only by noise.** In the run of record the largest $R^2$ separation between hybrid and quantile anywhere in the sweep is **0.004** (0.832 against 0.836 at three buckets, second order), against the same arms' seed deviations of ±0.018 to ±0.027, and 0.004 is also the bound across every archive under `reproduce/outputs/`, five-seed and ten-seed alike. So pinning the extreme centroids is invisible to every aggregate metric here, and no accuracy number in this table can argue the pin is working.

The consequence was worse than a wasted line of code. With a target min-max scaled to $[0,1]$, the unconstrained solve chose extreme bucket means of $-0.81$ and $-0.31$, both outside the range the target occupies. Those numbers *are* the consequents of the lowest and highest rules, so the rule base was telling a reader "THEN output is $-0.81$" about a quantity that never goes below zero. For a model whose entire justification is that a person can read and edit its rules, consequents corresponding to nothing in the data are a genuine defect that no accuracy metric would have caught.

I have fixed it upstream. The solve now accepts a `pin_extremes` flag, on by default, implemented as an exact linear equality constraint and not a penalty: each rule block's intercept column *is* that rule's bucket mean, so the two pinned columns move to the right-hand side and the remaining coefficients are solved against the residual. The bucket means come back as the intended $0.0$ and $1.0$, a min-max target's actual endpoints and not a coincidence, and accuracy is unchanged to within noise: the archived bound is the 0.004 in $R^2$ above, against seed deviations of ±0.018 to ±0.027. The gain is not accuracy. It is that the model can express the full observed output range by construction, and that its extreme rules say something true.

That still leaves the question the argument started from, because Concrete's skew is only +0.42 and the hypothesis was about skew. No collection of real datasets settles it cleanly either: they differ in dimensionality, noise, and sample size at once, so a gap between two of them is not attributable to skew. So I isolated it. A fixed linear signal is pushed through the strictly monotone map $y = \mathrm{expm1}(\lambda z)/\lambda$, which changes the *shape* of the target while leaving the information in $X$ untouched. A perfect learner would score identically at every $\lambda$; whatever degrades is what the partitioning fails to absorb.

**Table 4.3 — Partitioning against target skew** (synthetic, skew isolated; 4 buckets, 2nd order, 10 seeds; `full-2026-08-03/table_g5b_skew_sweep.csv`).

| target skew | uniform R² | quantile R² | Q − U | uniform tail RMSE | quantile tail RMSE | uniform min bucket |
|---:|---:|---:|---:|---:|---:|---:|
| +0.05 | 0.912 ± 0.009 | 0.911 ± 0.010 | −0.001 | 0.052 | **0.051** | 11 |
| +1.84 | **0.884 ± 0.016** | 0.876 ± 0.035 | −0.008 | 0.066 | **0.062** | 1 |
| +5.32 | **0.731 ± 0.083** | 0.728 ± 0.169 | −0.003 | 0.092 | **0.066** | 1 |
| +10.44 | **0.297 ± 0.126** | 0.183 ± 0.813 | −0.114 | 0.118 | **0.064** | 1 |
| +14.71 | **0.168 ± 0.073** | −2.106 ± 4.274 | −2.274 | 0.111 | **0.062** | 0 |
| +17.71 | **0.084 ± 0.016** | −12.562 ± 24.000 | −12.646 | 0.132 | **0.060** | 0 |

Quantile's mean $R^2$ trails uniform's in every row, and past skew 5 the gap opens hard against it: −0.008, −0.003, −0.114, −2.274, −12.646. That is not a case for "uniform wins" either, and reading the means is the wrong way to read this table.

The finding is that quantile becomes unstable, not inaccurate. Read the standard deviations instead. Quantile's spread explodes: ±0.169 at skew 5, ±0.813 at skew 10, ±4.274 and ±24.000 beyond, while uniform's stays bounded and its mean degrades smoothly toward zero. A mean of −12.6 with a deviation of ±24.0 is not typical behaviour. It is a small number of catastrophic splits dragging an otherwise reasonable distribution. Quantile on a heavily skewed target sometimes produces a usable model and sometimes a disaster, and the three-seed run missed the disasters. Uniform fails, but *predictably*, which for a component inside a larger pipeline is often the more valuable property.

The starvation mechanism survives, and it is still the reason uniform degrades. The last column collapses exactly as predicted: 11 samples at symmetry, *one* by skew 1.8, *zero* past skew 14. Equal-width buckets on a skewed target put nearly every point in the first bucket and leave the rest to estimate rules from almost nothing, which is why uniform's $R^2$ falls from 0.912 to 0.084 across the sweep. What that mechanism does *not* license is the inference that quantile's guaranteed occupancy therefore makes it the better choice: it removes one failure mode and introduces another.

The tails go the other way from the geometry. Even coverage of the output range should hold the *tails* better, and consistently, with small spreads, it does not: uniform's tail error grows from 0.052 to 0.132 while quantile's actually *falls*, 0.051 to 0.060. Coverage of the range is worthless without coverage of the *samples*.

Past skew 14 both schemes fail, quantile's $R^2$ going sharply negative with standard deviations of ±4.3 and ±24.0, so the apparent uniform "win" there is noise between two broken models. At that point the target is so compressed that uniform's smallest bucket is empty outright and the problem needs a target transform, not a better partition.

This is an open question, not a settled recommendation, and Chapter 7 carries it as one. Neither scheme is safe on a heavily skewed target: uniform starves and decays, quantile holds its buckets but goes unstable. What the experiment earned is the negative result and the diagnosis.

Having placed the output partition, I use the same per-feature Gaussian-mixture approach for the antecedents and apply linear regression to obtain first-order Takagi–Sugeno–Kang consequents. That is the Deep Thought move made concrete: answers before questions.

Two empirical observations shape this. Consequent order matters: moving from first to second order and then to the full second-order basis is worth several points of $R^2$ apiece on Concrete (Table 4.4). First order is a floor here, not a sufficient choice. Second, the *antecedent* refinement Chapter 6 §6.4 measures buys progressively less as consequent capacity grows, and almost nothing at the top of the range. That decay is the *structure before search* thesis in miniature: the better the structure-derived model already is, the less a subsequent search finds. The closed-form solve for these consequents is a shared primitive across several model types, deferred to Chapter 6.

### 4.3.3 Inference

Inference is ordinary fuzzy evaluation. For classification I take the class whose rule fires most strongly, $\arg\max_k$, and for regression I defuzzify by the weighted average as usual. Nothing exotic happens at inference; the leverage is all in how cheaply the model was built.

One structural consequence matters for the discussion. Each class rule is estimated from that class's own data and nothing else; no joint fit couples the classes. So new labeled data for one class updates that class's densities without touching any other rule, and a new class can be added by fitting one more rule instead of retraining. That is what makes the construction naturally incremental and semi-supervised-friendly. I have not run a controlled streaming or semi-supervised benchmark, so I state it as a property of the construction and not a measured result.

### 4.3.4 Why the parameters grow linearly

The whole thing stays fast and small because of the factorization: instead of partitioning the joint input space, I condition on the output (for regression, equal-frequency buckets from a quantile split) and fit an independent one-dimensional Gaussian mixture per feature within each bucket or class. Gridding the inputs instead gives a rule count that is a product over the inputs,

$$ N_{rules}^{\text{grid}} = \prod_{i=1}^{M} N_{\mu_i} \sim \mathcal{O}(c^{M}), $$

exponential in the number of features $M$. Conditioning on the output gives

$$ N_{rules}^{\text{MoG}} = K, \qquad N_{params} \sim \mathcal{O}(K \cdot M \cdot p), $$

one rule per class (or output bucket) $K$, with parameters growing *linearly* in the number of features $M$ and the components per mixture $p$. Two things about that second expression. It counts **parameters**, the size of the fitted model, and that is the only quantity for which $\mathcal{O}(K \cdot M \cdot p)$ is the right answer. It is *not* the cost of fitting: the antecedent screen of §4.3.1 compares class-conditional distributions pairwise and so carries a $\mathcal{O}(M \cdot K^2)$ term, quadratic in the class count, the one place this construction is not linear in anything. For twelve classes and eighty-three features (RT-IOT2022, below) the grid form is astronomically large while the factored form is a few thousand parameters and twelve rules, the screen's 66 class pairs per feature still cheap. That choice sidesteps the rule-base explosion: the naive-Bayes instinct traded deliberately for speed and interpretability.

### 4.3.5 The rule for everything else: anomalies and rare classes

The construction has a consequence I did not set out to obtain and then built deliberately. Because every class rule is an explicit fuzzy membership over the input space, I know not just how strongly each class fires but how strongly *anything* fires. So I can synthesize one more rule, automatically: *none of the above*.

Let $\mu_k(x)$ be the firing strength of the rule for class $k$. Aggregating the known classes with the t-conorm $S$ gives the degree to which the model recognizes $x$ as something it has seen; the fuzzy complement of that is the degree to which it does not:

$$ \mu_{\text{anom}}(x) = 1 - S\big(c_1,\; c_2,\; \ldots,\; c_K\big), \qquad c_k = \min\big(\max(\mu_k(x) + \theta,\; 0),\; 1\big). $$

The clip is not decoration. What `gauss_math.tsk_firing_strengths` computes is `np.clip(rule_firing[:, :-1] + threshold, 0.0, 1.0)` and then the complement of the conorm over that.

The $\theta$ term is a boost applied to each known-class firing before aggregation, and the single knob that sets how eager the anomaly rule is. A large $\theta$ inflates the known classes, shrinks the complement, and makes the model reluctant to cry anomaly; a small $\theta$ makes it suspicious of anything less than a confident match. Inference is unchanged, since I take the $\arg\max$ over the $K$ class firings *plus* this one extra, so the anomaly rule wins whenever no known rule fires strongly enough to beat it.

**What the clip does to the rule.** Every t-conorm satisfies $S(1, \cdot) = 1$. For the Hamacher branch used here, $S(x,y) = (x + y - 2xy)/(1 - xy)$, so at $x = 1$ numerator and denominator are both $1 - y$ and the value is exactly 1. One clipped input saturates the aggregate, so $\mu_{\text{anom}}(x) > 0$ only when **every** class firing is below $1 - \theta$. Wherever some class reaches $1 - \theta$, this construction is *exactly* a rejection threshold on $\max_k \mu_k(x)$, and the conorm has no effect on the decision. At $\theta = 0.99$ (the default this chapter inherits from `FuzzySystemsExperiments/beth-anomaly.py`, not the library's own default of 0.5) the bar is a firing of 0.01, essentially always cleared on real data, so the shipped anomaly rule is a max-membership rejector and nothing more. Across the band Table 4.6 sweeps, $\theta = 0.5$ to $0.8$, the clip bites only at firings of 0.5 down to 0.2, so there the conorm value genuinely acts on samples whose every class firing is small. The degeneracy is *total at the default and partial across the sweep*, a reason to prefer an operating point in the swept band over the inherited one, independently of the $J$ argument in §4.4.

The choice of t-conorm is a parameter, and the chapter's configuration is not the library's: the tables here run the **Hamacher** family, inherited with $\theta$ from the same BETH script, while `gauss_data.DefaultNormCornorm` is `probability`. The drastic sum and the Hamacher sum are distinct conorms, and only the latter is implemented, so "the drastic Hamacher conorm" names nothing. Whether the choice of family matters for open-set behaviour is **untested**. `table_norm_conorm_matrix.py` sweeps the five De Morgan families on *accuracy* and says in its own docstring that the interesting case for the anomaly rule "needs the open-set harness rather than this one," which has not been run. So the family ordering by how readily a partial match counts as recognition is a property of the algebra, and no measurement here supports a claim about detection or false alarms.

Three properties make it worth building.

It is **free**. No second model, no separately trained detector, no extra pass over the data. The anomaly rule is an algebraic consequence of the rules I already have, so it costs one add-and-clip, one t-conorm and one subtraction at inference time. Contrast bolting a one-class SVM or isolation forest alongside a classifier and reconciling their disagreements.

It is **interpretable in exactly the way the rest of the model is**, and this, not the decision rule, is what the construction actually adds. Given the clip, at the shipped $\theta$ the *rule* is a threshold on the maximum class membership, which is not a new idea and I will not present it as one. What a bare max-membership rejector cannot supply is the explanation. When it fires, the answer is not an opaque outlier score and not a bare "below threshold." It is "none of the known rules matched this, and here is how close each one came," each near-miss itself a readable linguistic rule over named features, so an operator sees *which* clauses failed and by how much.

It addresses **rare classes**, the failure mode that motivated it. A class with a handful of examples is nearly invisible to accuracy-driven training: a model that ignores it entirely still scores well. But rare and unseen events are frequently the ones that matter, a novel intrusion, an off-nominal flight condition, a failure mode absent from the training set. The complement rule catches these *as a category*, without needing examples of them, which a supervised class rule can never do.

Novelty detection and open-set recognition are established fields: one-class SVMs, isolation forests, Mahalanobis-distance novelty scores and the open-set recognition line all attack this problem directly, and I do not claim to beat them. My claim is narrower than it used to be, and narrower than the equation suggests: in a fuzzy inference system built this way, open-set behavior is a *consequence* and not an addition, written in the same t-conorm and complement the model already uses, inheriting its interpretability for free. The *rule* reduces to the threshold above, so its novelty is small; what is not reducible is that the rejection arrives already explained. §4.4 runs the comparison against a dedicated one-class detector on the same data: the mechanism behaves as described and the three methods come out indistinguishable on the testbed available, supporting parity but not superiority.

## 4.4 Results

The datasets here are public, so unlike Chapter 3's psychiatric set I can name them freely.

On the **PhiUSIIL phishing URL** dataset the model reaches 0.997 ± 0.001 accuracy in under three-tenths of a second, with two rules and a handful of clauses. That is the headline: a readable, two-rule fuzzy classifier, competitive on accuracy, trained in the time it takes to describe it. I temper the "competitive" immediately. CART and a random forest both score a perfect 1.000 here, so PhiUSIIL is saturated and cannot separate these methods on accuracy. What it demonstrates is the *rule count and the training time*, not superiority.

**RT-IOT2022** (123,000 instances, 83 features, 12 output classes) is the chapter's scale *target*, and it is described as one. No timing for it exists: the RT-IOT2022 files are not in this repository, no run under `reproduce/outputs/` touches them, and Table 4.4's own accuracy cell says so. The table prints none. The dataset is here for the shape of the claim, not its confirmation. The answer-first construction should not fall over as the data gets large and multi-class, because the rule base and the parameter count grow with classes times features, not with any product over inputs. The exception is the antecedent screen of §4.3.1, quadratic in the class count at 66 pairs per feature, which is why this dataset is the right place to measure it and why the missing measurement is worth having rather than assuming.

The **BETH** host-telemetry set is where I intend to test the anomaly rule of §4.3.5 in its hardest configuration: train on *benign traffic only*, then show the model a test set containing malicious activity it has never seen. Nothing about the malicious class would be available at training time, so there would be no "attack" rule to fire and detection would come from the complement rule alone. That is open-set recognition, not classification, and where the construction should earn its keep. Because the boost $\theta$ is a single scalar, the operating point would be a one-dimensional sweep, not a retraining problem. All of that is future tense deliberately, and the blocker is not just the missing files. The BETH data are not in the repository, and even with them the harness's protocol does not apply: leave-one-class-out needs at least three classes and BETH is binary, so the experiment needs a purpose-built one-class training path, a research decision before a coding one, recorded in `WORKINGDOC.md` §6. What has been run is the substitute below: the same protocol on Glass, every class withheld in turn.

On the **UCI Concrete Compressive Strength** regression set, the flat model's test $R^2$ is −0.434, 0.787, and 0.832 at TSK orders zero, one, and two, rising to 0.873 with the full second-order basis. The zeroth-order figure deserves a word and not a quiet omission: with constant consequents the model is not merely no better than predicting the mean, it is measurably *worse*, a negative $R^2$, so first-order consequents are not a refinement here but a requirement. From first order on the result is reasonable, and the launch point for the hierarchical models of Chapter 6.

The reconciliation with Chapter 6 is done. A flat-model $R^2$ in the mid-0.60s from the tree-and-mixture experiment reads as a contradiction of the figures above; under one protocol, with identical splits, seeds, and preprocessing, the flat model scores 0.877 ± 0.037 with refinement (`full-2026-08-03/table_concrete_reconciliation.csv`), and the discrepancy was configuration and not disagreement. Chapter 6 Table 6.1 and the table above are the same measurement from two chapters. The remaining trap is Chapter 6's Table 6.2, which deliberately runs everything untuned at raw features and reports the flat model at 0.658 ± 0.040 (`table_6_1.csv`): a different question, not a different answer.

**Table 4.4 — What the Mixture-of-Gaussians construction achieves.** Measured on a single workstation except where a cell is marked *not run*; the baseline comparison is Table 4.5. The rule-base column is structural, not a measurement, as the row with no timing is labelled.

| Dataset (task) | Size (N × M) | Train time | Accuracy / R² | Rule base |
|---|---|---:|---:|---|
| PhiUSIIL (binary classification) | 235K × 54 | 0.28 ± 0.03 s | 0.997 ± 0.001 acc. | 2 rules (K = 2) |
| RT-IOT2022 (12-class) | 123K × 83 | *not run — no timing exists* | *not run — RT-IOT2022 is not in the repository* | 12 by construction (K = 12) |
| Concrete (regression, TSK order 1) | 1,030 × 8 | seconds | R² = 0.787 ± 0.026 | 3 output buckets |
| Concrete (regression, TSK order 2) | 1,030 × 8 | seconds | R² = 0.832 ± 0.027 | 3 output buckets |
| Concrete (regression, full 2nd) | 1,030 × 8 | seconds | **R² = 0.873 ± 0.020** | 3 output buckets |

That RT-IOT2022 row marks its timing instead of estimating it, and labels its rule count a structural consequence because it is one: for classification the count is the number of classes, for regression the number of output buckets, never a product over inputs. A grid over 83 features would be beyond enumeration while this model carries twelve rules, but that arithmetic is all the row illustrates until the dataset is present and measured. A hand-authored cell on a row whose accuracy cell already says the dataset is absent is exactly the failure the reproduction note below exists to prevent, and it is the one a mixed generated-and-transcribed table invites.

**Table 4.5 — Baseline comparison** *(structure fixed; cells to be filled by the reproduction harness).* The speed claim is only persuasive against the methods it displaces, so this is the first experiment owed to the chapter. Every method runs on identical splits, multi-seed with error bars, under the Goal G4 protocol, and every arm in the Concrete column runs on the log-and-min-max-scaled features of §4.3.

Applying the transform uniformly costs the baselines nothing, for the rank-invariance reason Table 4.1 measures at +0.001 and +0.000, so the CART and Random Forest column reads the same either way.

The MoG appears on two rows because only the first is fully instrumented, and that row's provenance is now simpler than it was. Until this pass its accuracy and its clock came from two archives. The previous run of record held Concrete `R² = 0.780 ± 0.029` at **1.04 ± 0.62 s** and predated the warm-up fix, so the wall clocks had to be borrowed from `reproduce/outputs/warmup-discarded/` and the row was an accuracy from one archive beside a clock from another. **The re-take it needed has happened.** The run of record now carries the discarded warm-up fit, and every cell in row 1 comes from one file, `reproduce/outputs/full-2026-08-03/table_4_1.csv`: Concrete `R² = 0.783 ± 0.030` at **0.43 ± 0.01 s**, PhiUSIIL `acc = 0.997 ± 0.001` at **0.28 ± 0.03 s**. Both clocks roughly halved against the previous run, and the ±60% is gone.

The diagnosis behind that fix stays on the record. The ±60% on the 1.04 s was not seed spread. Concrete is the first arm the process fits, so seed 0 absorbed import, JIT, thread-pool spin-up and first-touch allocation, coming in at 3.7× the mean of the other nine. The generator now performs one discarded warm-up fit, and the cell reads 0.43 ± 0.01 s, a spread of 2%, in line with the PhiUSIIL row, which was never affected because it is fitted second. That asymmetry between two rows of one table gave it away. Every accuracy in `warmup-discarded/` was byte-identical to the archive it was taken against, so the warm-up moved the clock and nothing else; see note 14 in `PROVENANCE_MAP.md`. Two figures are excluded from this row on provenance: `seeds10-2026-08-01` scores the same arm at 0.651, so a clock taken from it cannot be paired with the R² above, and a PhiUSIIL time of 0.72 ± 0.04 s appears in no directory under `reproduce/outputs/`.

One missing row a committee is most likely to ask for. §4.2 names Gaussian naive Bayes as this construction's nearest acknowledged relative and concedes the resemblance up front, and then it does not appear here, while CART and Random Forest do. That asymmetry is not defensible on effort: unlike ANFIS and a GA-tuned FIS, which need adapters written, Gaussian naive Bayes is one import from scikit-learn, and no generator under `reproduce/tables/` instantiates it. It is listed below as not run because it is a debt, not because it is hard. Until it is filled, this chapter's accuracy comparison has no row for the model whose factorization it shares, the single cheapest experiment on the list.

| Method | Concrete R² | Concrete train time | PhiUSIIL accuracy | PhiUSIIL train time |
|---|---:|---:|---:|---:|
| **MoG FIS (this work)**, 1st order | 0.783 ± 0.030 | 0.43 ± 0.01 s | **0.997 ± 0.001** | 0.28 ± 0.03 s |
| **MoG FIS**, full 2nd order | **0.873 ± 0.020** | *no comparable timing exists — see below* | — | — |
| ANFIS | N/A (C1) | N/A (C1) | N/A (C1) | N/A (C1) |
| GA-tuned FIS | N/A (C1) | N/A (C1) | N/A (C1) | N/A (C1) |
| Gaussian naive Bayes (nearest relative, §4.2) | *not run — no generator* | *not run* | *not run — no generator* | *not run* |
| CART (reference) | 0.826 ± 0.047 | seconds | 1.000 ± 0.000 | seconds |
| Random Forest (reference) | 0.909 ± 0.019 | seconds | 1.000 ± 0.000 | seconds |

*The full-second-order row's missing time.* This cell was `*pending*` and is now marked instead. Its two halves do not come from the same code path. The **R² of 0.873 ± 0.020 comes from Table 4.1's study** (`table_hyperparam_normalization.py`), which drives `solve_tsk_consequents` / `predict_tsk` directly and min-max scales the target to $[0,1]$; row 1's numbers come from `table_4_1_mog_baselines.py` driving the `MixtureOfGaussiansFuzzyRegressor` estimator. That second generator now carries a timed full-second-order arm inside the run of record, and at ten seeds the **estimator** path measures **R² = 0.842 ± 0.040 in 0.46 ± 0.02 s** (`full-2026-08-03/table_4_1.csv`). Each of those halves is internally consistent; neither is consistent with the 0.873 above them, and the gap between the two implementations has widened from 0.019 to 0.031. Putting 0.46 s next to 0.873 would pair a time from one implementation with an accuracy from another, the same quiet mismatch one generator over. The cell stays empty until the pipeline that produced 0.873 is itself timed. The reading today: on the estimator path full second order buys about +0.06 R² over first order for three hundredths of a second, with the exact pairing still owed.

*Reading the empty cells.* `N/A (C1)` is not a number withheld; it is a measurement that cannot be taken with what is in the repository. `table_4_1_mog_baselines.py` looks for `reproduce/tables/_baseline_anfis.py` and `_baseline_gafis.py` and emits `N/A` when they are absent, which they are. Writing those two adapters is checklist item **C1**, the single most important experiment still owed, because the *orders of magnitude faster* claim in the title and in Chapters 1, 7 and 8 has no fuzzy baseline to be faster **than** until they exist. The eight cells are the shape of that hole, marked rather than filled with a plausible figure.

**The mechanism, measured.** The BETH files are not in the repository, so the harness runs the same protocol on public data: leave-one-class-out, where each class is withheld from training in turn and treated as unseen, averaged over held-out classes and seeds. Sweeping the boost gives the operating curve the section promised.

**Table 4.6 — The anomaly operating curve.** Detection and false alarm as functions of $\theta$, on Glass (6 classes, leave-one-class-out); Hamacher conorm, 10 seeds, from `reproduce/outputs/full-2026-08-03/table_4_4b_theta_sweep.csv`.

| $\theta$ | detection rate | false-alarm rate | detection − false alarm |
|---:|---:|---:|---:|
| 0.50 | 0.838 | 0.719 | +0.119 |
| 0.60 | 0.815 | 0.664 | +0.151 |
| 0.70 | 0.762 | 0.616 | +0.146 |
| **0.80** | 0.700 | 0.546 | **+0.154** |
| 0.90 | 0.625 | 0.475 | +0.150 |
| 0.99 | 0.491 | 0.362 | +0.129 |
| 1.10 | 0.000 | 0.000 | 0.000 |

The curve behaves exactly as §4.3.5 says it should. Raising $\theta$ inflates the known-class firings, shrinks the complement, and monotonically reduces both detection and false alarms; past $\theta = 1.1$ the boost saturates the aggregate and the anomaly rule stops firing altogether. The knob is real and monotone: an operator trades sensitivity against nuisance alarms by turning one scalar, which was the design claim.

Three observations follow, and the first is that this curve has moved. `tribble-fis` PR #72 now scores each candidate component count off the k-means partition it implies instead of fitting and discarding four EM mixtures, and every $\theta$ here moved with it: detection up 0.084 to 0.152, false alarms up 0.151 to 0.221, so the net separation *fell* throughout. The band the previous run put at +0.222…+0.239 peaking at $\theta = 0.60$ now reads +0.119…+0.154 peaking at $\theta = 0.80$. About 35% of the achievable separation is gone and the operating point has moved two steps along the sweep.

Second, the default of $\theta = 0.99$ inherited from the BETH configuration is still *not* a good operating point on this data: it gives about five-sixths of the separation available and sits well down the low-sensitivity end of the curve. Report the curve, and tune $\theta$ per deployment. Third, there is still no sharp optimum to tune *to*: $J$ sits between +0.119 and +0.154 across the whole range from 0.5 to 0.8, so the choice within that band is nearly free and the knob forgiving. That qualitative claim survives the move. The absolute performance does not: at the best setting the rule detects 70% of unseen points at a 55% false-alarm rate, a weaker signal than the previous run suggested and further from a deployable detector, not closer.

Nor should that be over-read in either direction. Glass has 214 samples across six classes, several with fewer than a dozen members, so withholding a class removes much of the little data there is and asks the model to be confident about a space it has barely seen. It is a stress test, not a demonstration. It establishes that the mechanism works as described, not that the complement rule is *competitive*; that requires BETH or a comparable dataset.

**Table 4.7 — Against detectors built for the job** *(θ = 0.99, matched operating points; `full-2026-08-03/table_4_4_openset.csv`).* The baselines' contamination is set to the complement rule's observed false-alarm rate, so all three are compared at the same point on their curves.

| Method | Detection rate | False-alarm rate | Detection − false alarm | Separate model? |
|---|---:|---:|---:|:--:|
| **Complement rule (this work)** | 0.491 ± 0.359 | 0.362 ± 0.262 | +0.129 | no |
| Isolation Forest | 0.493 ± 0.341 | 0.317 ± 0.160 | **+0.176** | yes |
| One-class SVM | 0.399 ± 0.300 | 0.288 ± 0.166 | +0.111 | yes |

Isolation forest nominally leads, now by 0.047 where the previous run showed 0.002, and nothing should be read into that either. The ordering has changed three times across runs: a five-seed run put isolation forest ahead by 0.038, ten seeds made the two level to within a fifth of a percent, and this run separates them again. The standard deviations across held-out classes are roughly five times the *largest* gap in the table and seven times the smallest, so every one of those orderings is noise read as signal. What the table supports is the cheaper claim that motivated the section: the complement rule performs *comparably to purpose-built detectors while requiring no second model*. Three of the nine cells moved beyond noise this run, and no separation in the table now exceeds its own error bar, the one-class SVM's +0.111 included. More than parity needs a testbed larger than 214 samples, and is a goal for completion.

> **Reproduction.** Each table has its own generator under `reproduce/tables/`, emitting Markdown and CSV with mean ± standard deviation across a fixed seed set: Table 4.1 from `table_hyperparam_normalization.py`, Table 4.2 from `table_g5_output_partitioning.py`, Table 4.3 from `table_g5b_skew_sweep.py`, Tables 4.4 and 4.5 from `table_4_1_mog_baselines.py`, Tables 4.6 and 4.7 from `table_4_4_openset.py`. The operating curve is emitted whenever `REPRO_THETA_SWEEP` holds a θ *list*, and `run_all_tables.sh` now defaults it to `0.5,0.6,0.7,0.8,0.9,0.99,1.1`. Note that `=1` is a valid list of *one* and emits a single saturated row of zeros, which reads exactly like a null result. No cell here is left a bare *pending*: every unfilled cell names what blocks it, and any checklist item tracking it. The harness prints exactly what it could not run rather than substituting a guess. But the guesses this pass removed from Table 4.4 were *hand-authored into the prose*, so the guarantee held for the harness and not the transcription: that is the failure mode to watch where a table mixes generated and hand-entered cells, and why Tables 4.4 and 4.5 now name each cell's archive. Per-cell provenance and reconciliation status are tracked in `reproduce/PROVENANCE_MAP.md`.
>
> **TODO — repeatable performance (board-wide standard):** the numbers above are single-machine point estimates. Reproduce under the fixed protocol, with pinned clocks and thermals, multiple seeds and reported error bars, before citation. See `ACTION_ITEMS.md` §A and Chapter 7, Goal G4.

**Figure 4.2 — The open-set operating curve.** Table 4.6 plotted: detection and false-alarm rate against the boost $\theta$, with $J$, their difference, as the shaded band between them, and the inherited default $\theta = 0.99$ and the saturation past $\theta = 1.1$ marked. Drawn from the table's own CSV, not a second run of the sweep, so the two cannot disagree; regenerate the table with a θ list (`REPRO_THETA_SWEEP=0.5,0.6,0.7,0.8,0.9,0.99,1.1`) and the figure follows. To be re-taken on BETH via `plot_anomaly_threshold_sweep` when those data are available *and* a one-class training path exists; as above, the data are necessary but not sufficient (`WORKINGDOC.md` §6).
`![anomaly-sweep](fig/04-anomaly-sweep.png)`

**[FIGURE 4.3 — deliberately still a placeholder]** *Confusion matrix on RT-IOT2022 before and after the correction-rule pass, showing which class confusions the corrections repair. This is the one figure in the document that is not drawn, for the reason §4.3.1 gives: the accuracy contribution of the correction pass has not been isolated, so neither half of the before-and-after exists to plot. RT-IOT2022 is also not among the datasets the harness can load. The skip is recorded with its reason in `reproduce/figures/registry.py`.*
`![rtiot-confusion](fig/04-rtiot-confusion.png)`

## 4.5 Discussion and Contributions

The method is fast because it replaces a global search over a huge space of possible rules with a handful of local density fits keyed to the known answers, plus a closed-form consequent solve. It stays interpretable because there are only a few rules, written over named features in linguistic terms, and a person can read and edit them. And because each class rule is fit independently, as §4.3.3 noted, the construction is naturally incremental, new data updating one class's densities without disturbing the others, which suits a semi-supervised, keep-learning setting.

The anomaly rule deserves a closing word, because it is the part of this chapter I expect to matter most outside the dissertation. Taking the complement of the aggregate turns a closed-set classifier into an open-set one for the cost of one operation. At the shipped boost that decision is the maximum-membership threshold §4.3.5 works out, and I would rather say so than let the algebra imply more. What it does not give up is the property that made the model worth building: when it flags something it can say why, in the rules it already had. An unexpected condition is exactly the case where you least want an unexplainable answer.

What is and is not established. The construction is real and the timings are measured, not estimated, and the run of record now supplies the headline row's clock and its accuracy from one file, which is the re-take Table 4.5 used to owe. The speed claim is only fully persuasive against the right baselines, and I have not run the head-to-head against ANFIS and a genetic-algorithm-tuned FIS on identical splits, nor against Gaussian naive Bayes, the cheapest of the three and the one §4.2 concedes the closest kinship to. That table is the first thing I owe this chapter, a goal for completion. These numbers are also subject to the board-wide repeatability standard, with fixed hardware, multiple seeds and error bars. And the accuracy claim's scope is bounded by the naive-Bayes-like factorization: where feature interactions matter a great deal, the flat model will leave accuracy on the table, precisely the gap the hierarchical models of Chapter 6 exist to close. The bridge in the other direction, where the membership functions come from when the data has no coordinates and no Gaussian shape to fit, is Chapter 5.

---

*Draft — Chapter 4 prose, in the author's voice, opening on the Hitchhiker's Guide / consequent-first motif to match the tribble motif of Ch 1. Citations in bracketed shorthand pending the consolidated `references.bib`. Seven tables (4.1–4.7) and three figure placeholders (4.1–4.3) inline. Open items in `../ACTION_ITEMS.md`.*
