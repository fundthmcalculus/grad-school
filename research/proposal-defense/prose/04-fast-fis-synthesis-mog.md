# Chapter 4 — Fast Interpretable FIS Synthesis via Mixtures of Gaussians

## 4.1 Introduction

In *The Hitchhiker's Guide to the Galaxy*, the supercomputer Deep Thought is asked for the Answer to the Ultimate Question of Life, the Universe, and Everything. It thinks for seven and a half million years and returns 42 — at which point everyone realizes they never worked out what the Question actually was. The rest of the story is a search, backward, for a Question that fits the Answer they already have.

This chapter is built on the same inversion, and it is the reason it is fast. A fuzzy rule has an antecedent — the IF part, a question about the inputs — and a consequent — the THEN part, the answer. The conventional way to build a fuzzy model is to lay out the questions first: partition every input into fuzzy sets, form a rule for every combination, and then search for consequents that make the whole thing fit. I do the opposite. I commit to the answers first — the output classes, or a set of output values spread across the range — and only then work backward to find the antecedents that select them. Starting from the answer collapses most of the work, because the answers are few and known while the space of possible questions is enormous. And unlike Deep Thought, the method finishes in seconds rather than geological time, which is the entire point: it is fast enough that the Vogons cannot demolish the Earth to make way for a hyperspace bypass before it is done.

The problem I am solving is the one I set up in Chapter 1. A fuzzy model built by gridding the inputs has a rule count that is the product of the per-input set counts, which is exponential, and the usual way to fit it — a genetic algorithm, or gradient descent, or ANFIS — is either slow to converge or dependent on an initial guess it cannot supply for itself. My claim in this chapter is that a Mixture-of-Gaussians construction generates both the membership functions and the rules directly from the data, produces on the order of one rule per output class rather than an exponential blowup, and does so without any post-hoc genetic search or gradient descent at all. The contributions are: an answer-first (consequent-first) construction for both classification and regression; a per-feature, per-class Gaussian-mixture antecedent generator whose parameter count grows linearly rather than exponentially; an automatically synthesized *anomaly* rule — the complement of the t-conorm of every explicit rule — that gives the model open-set behavior and a handle on rare classes at no additional training cost; and a demonstration that this trains competitive models in seconds on datasets with hundreds of thousands of rows.

## 4.2 Background and Prior Art

The idea of building a model from the output side is not new, and I want to credit the ancestor directly. Sugeno and Yasukawa (1993) proposed identifying a fuzzy model by first clustering the *output* and working back to the input structure, which is the same instinct I am following. Wang and Mendel (1992) generate rules from data by a different route, and Chiu (1994) uses subtractive clustering to place rules. What I add to this lineage is a specific, cheap factorization of the antecedents and an explicit path to a compact rule base, which I lay out below.

I also owe the reader an honest caveat up front, because a sharp committee member will raise it immediately. For classification, fitting an independent Gaussian mixture per feature and per class, and combining them, is closely related to a Gaussian naive-Bayes model — a class-conditional density estimate with a feature-independence assumption. I am not going to pretend otherwise. What I am claiming is not a new density estimator; it is that this construction produces an *interpretable fuzzy inference system* — real membership functions, real linguistic rules, editable by hand — extremely fast, and that the naive-Bayes-like factorization is exactly what keeps it fast and small. Where the independence assumption costs accuracy, I recover it with a small number of correction rules rather than by abandoning the factorization.

## 4.3 Methodology

### Preparing the inputs, and why the transform is not optional

Before either construction, the inputs are transformed: features whose dynamic range spans more than a couple of orders of magnitude are log-scaled automatically, and every feature is then standardized. On Concrete the automatic detector selects two, `Slag` and `Age`. `Age` is the unsurprising one — curing time runs from one day to a year, while most features live within a factor of a few.

I used to treat this as housekeeping and leave it out of the description. That was a mistake, because it is worth more than most of the modeling choices in this chapter. Measured across ten seeds on identical splits:

**Table 4.1 — What the transform is worth, by model.** The pattern is the point, not any single row.

| Model | raw features | log + standardized | Δ |
|---|---:|---:|---:|
| flat MoG-TSK, 1st order | 0.646 ± 0.039 | 0.772 ± 0.034 | **+0.126** |
| flat MoG-TSK, 2nd order | 0.779 ± 0.036 | 0.824 ± 0.043 | +0.044 |
| flat MoG-TSK, full 2nd | 0.790 ± 0.054 | 0.859 ± 0.039 | +0.069 |
| fuzzy tree (library default) | 0.580 ± 0.067 | 0.688 ± 0.056 | +0.108 |
| CART (reference) | 0.825 ± 0.047 | 0.826 ± 0.047 | +0.001 |
| Random Forest (reference) | 0.909 ± 0.018 | 0.909 ± 0.019 | +0.000 |

Two things fall out, and the second matters more than the first.

The first is that the transform is worth nearly thirteen points of $R^2$ to the Gaussian models. A first-order model goes from 0.646 to 0.772 on nothing but a log and a rescale. Any comparison that omits it is not measuring the method.

The second is that **the transform is worth essentially nothing to CART and Random Forest** — +0.001 and +0.000, which is not a rounding artifact but the expected result. An axis-aligned decision tree splits on rank: it asks whether a feature exceeds a threshold, and a monotone transform cannot change the answer, so the tree it induces is identical. A Gaussian membership function has no such immunity, because it is defined by a location and a width in the feature's own units; skew the feature and the membership function fits the skew rather than the structure.

That asymmetry is worth stating plainly for two reasons. It explains why my pipeline requires preprocessing that the baselines do not, which would otherwise look like an inconsistency between chapters — the fuzzy tree of Chapter 6 is deliberately fit on *raw* features so its split thresholds stay physically meaningful ("cement ≥ 350", not "cement ≥ 0.42"), and it can afford to be, for exactly the rank-invariance reason above. And it forecloses the natural suspicion that the transform is quietly doing the work that I attribute to the model. It cannot be: the strongest baseline in the table is entirely unmoved by it, so whatever the transform buys, it buys specifically for the Gaussian construction rather than for the problem.

### 4.3.1 Classification: one rule per answer

For a classification problem with $N$ samples, $M$ features, and $K$ output classes, the construction runs answer-first. I segment the data by output class — the answers — and for each class I ask which features actually distinguish it, by comparing the per-feature statistics across classes; this is an $O(M^2)$ screening step that discards features carrying no discriminative signal. For each retained feature and class I fit a one-dimensional Gaussian mixture, using up to a few components, which becomes the membership function for that feature under that class. The combination then happens at two levels, and I want to be precise about which operator acts where, because an earlier draft of this paragraph had it wrong. *Within* a feature, the mixture's components are combined by a fuzzy OR — a t-conorm — so the class is recognized if this Gaussian fires or that one does. *Across* features, those per-feature results are combined by a t-norm. The rule for a class is therefore a conjunction of disjunctions rather than a single disjunction, which is what `simple_gaussian_predict` implements and what Figure 4.1 draws. Repeating over the $K$ classes gives $K$ rules, each built from at most $M \times p$ Gaussian terms. There is no grid, so there is no exponential rule base.

Finally I evaluate the confusion matrix and add a second, small pass of correction rules where two classes are being confused — the place where the feature-independence assumption is costing me. These corrections are targeted and few, and they keep the rule base readable. I should note that I have not yet isolated how much accuracy this second pass buys; the before-and-after comparison is a small experiment I owe, and Figure 4.3 is where it belongs.

**Figure 4.1 — Per-feature Gaussian mixtures, and the rule they combine into.** A real fit, built by the same three calls the harness uses: the top three features for Glass Type 1, with each feature's mixture components dashed and their t-conorm bold, and the class's own samples as a rug against everything else's. The lower panel is the rule's firing strength over every sample in the dataset, and the rule itself is written underneath in the form the construction produces. Two readings are worth having. A single rule is not a classifier — prediction is the argmax over all $K$ rules, so the overlap in the lower panel is expected rather than a failure. And the degenerate components are not an artifact of the drawing: a zero-width Gaussian on a single observation is what the automatic component count produces on a 214-sample dataset, and it is visible here because the figure fits the model rather than illustrating it.
`![mog-classification](fig/04-mog-classification.png)`

### 4.3.2 Regression: place the answers, regress the questions

For regression the answer-first construction is even more literal. I partition the output range first, and only then find the antecedents. How to place that partition is a question I have gone back and forth on, and I would rather present it as genuinely open than pretend I have settled it.

There are two natural choices, and they trade against each other. **Uniform** partitioning spreads the buckets at equal width across the output range, with one centroid pinned at each extreme. This gives the more natural function approximation — every rule owns an equal span of the output, so the consequents interpolate evenly and the extremes stay covered — but on a skewed target some buckets will contain very few samples, and a bucket with almost no data yields a badly estimated rule or none at all. **Quantile** partitioning instead cuts the output into equal-frequency buckets, so every rule is guaranteed enough data to be estimated well; the cost is that bucket boundaries crowd where the data is dense and under-resolve where it is sparse, which for a regression target is often precisely the extremes — the values one usually cares most about getting right.

So uniform optimizes for approximation geometry and quantile for statistical stability, and which wins should depend on the target's skew and on how many buckets are asked for. Rather than argue it, I measured it.

**Table 4.2 — Output partitioning on Concrete** (target skew +0.42; 10 seeds; tail RMSE over the true bottom and top deciles; "min bucket" is the smallest training-bucket occupancy).

| buckets | order | scheme | R² | tail RMSE | min bucket |
|---:|---|---|---:|---:|---:|
| 3 | 1st | uniform | **0.795 ± 0.035** | **7.90** | 132 |
| 3 | 1st | quantile | 0.773 ± 0.033 | 8.36 | 343 |
| 4 | 1st | uniform | 0.785 ± 0.034 | 8.10 | 75 |
| 4 | 1st | quantile | **0.790 ± 0.023** | 8.15 | 257 |
| 6 | 2nd | uniform | 0.840 ± 0.029 | 6.49 | 39 |
| 6 | 2nd | quantile | **0.849 ± 0.026** | **6.03** | 171 |

Three things came out of this, and none of them is the one I expected.

**On accuracy, the two schemes are indistinguishable.** An earlier three-seed run showed a clean crossover near four buckets — uniform ahead at three, quantile ahead at six — and I built an argument on it. At ten seeds it does not hold up. The largest gap anywhere in the full eighteen-configuration sweep is 0.022 in $R^2$, against seed-to-seed deviations of 0.02 to 0.04 — so even the widest separation in the sweep sits inside one standard deviation of the arms producing it. The apparent sign pattern is if anything the *reverse* of the one I claimed (uniform ahead at three buckets, quantile ahead at four and six), which is the clearest possible sign that what I was reading was noise. I am striking the crossover claim rather than keeping it with a hedge, and I am not replacing it with the mirror-image claim either. Choosing between these schemes on aggregate accuracy is choosing between two things that measure the same.

**The starvation mechanism is real even though the accuracy difference is not**, and this is the part worth keeping. Uniform's smallest bucket falls from 132 samples to 75 to 39 as the partition refines, while quantile's floor stays high by construction — 343, 257, 171. That is a genuine and predictable structural difference, and it is why uniform must eventually fail as buckets increase or skew grows: a bucket with almost no data cannot support a rule. Concrete's skew of +0.42 is simply too mild for it to bite within the range tested, which is exactly why the next experiment isolates skew directly.

**The tails do not settle it either.** Uniform holds them in two of the three pairs above, by 0.46 and 0.05 MPa, and loses the third by 0.46. The direction is consistent with the geometric argument in two cases out of three and inconsistent in the other, which is not a finding — it is the same noise the accuracy column shows, seen through a different metric.

**The hybrid was not a third option — it was a bug, and the study found it.** `partition_output` takes equal-frequency boundaries and then pins the two extreme bucket centroids to the observed min and max, which is exactly the compromise I had hoped would dominate. It produced results *identical to pure quantile* in all eighteen configurations tested, to three decimal places on every metric, which is not how two different methods behave. The reason is that the closed-form consequent solve returned its own re-derived bucket means and the prediction path used those, so the pinned values were discarded before they could influence anything.

The consequence was worse than a wasted line of code. With a target standardized to $[0,1]$, the unconstrained solve chose extreme bucket means of $-0.81$ and $-0.31$ — both outside the range the target actually occupies. Those numbers *are* the consequents of the lowest and highest rules, so the rule base was telling a reader "THEN output is $-0.81$" about a quantity that never goes below zero. For a model whose entire justification is that a person can read and edit its rules, consequents that correspond to nothing in the data are a genuine defect, and one that no accuracy metric would ever have caught.

I have fixed it upstream. The solve now accepts a `pin_extremes` flag, on by default, implemented as an exact linear equality constraint rather than a penalty: each rule block's intercept column *is* that rule's bucket mean, so the two pinned columns move to the right-hand side and the remaining coefficients are solved against the residual. The constraint holds exactly — the bucket means come back as the intended $0.0$ and $1.0$ — and accuracy is unchanged to within noise, at most about 0.003 against a seed-to-seed deviation of 0.017. The gain is not accuracy. It is that the model can now express the full observed output range by construction, and that its extreme rules say something true.

That still leaves the question the whole argument started from, because Concrete's skew is only +0.42 and the hypothesis was about skew. No collection of real datasets settles it cleanly either — they differ in dimensionality, noise, and sample size all at once, so a gap between two of them is not attributable to skew in particular. So I isolated it. A fixed linear signal is pushed through the strictly monotone map $y = \mathrm{expm1}(\lambda z)/\lambda$, which changes the *shape* of the target while leaving the information in $X$ untouched. A perfect learner would score identically at every $\lambda$; whatever degrades is precisely what the partitioning fails to absorb.

**Table 4.3 — Partitioning against target skew** (synthetic, skew isolated; 4 buckets, 2nd order, 10 seeds).

| target skew | uniform R² | quantile R² | Q − U | uniform tail RMSE | quantile tail RMSE | uniform min bucket |
|---:|---:|---:|---:|---:|---:|---:|
| +0.05 | 0.912 ± 0.009 | 0.911 ± 0.010 | −0.001 | 0.052 | **0.051** | 11 |
| +1.84 | **0.885 ± 0.018** | 0.876 ± 0.035 | −0.009 | 0.064 | **0.062** | 1 |
| +5.32 | **0.761 ± 0.070** | 0.728 ± 0.169 | −0.033 | 0.085 | **0.066** | 1 |
| +10.44 | **0.335 ± 0.124** | 0.183 ± 0.813 | −0.152 | 0.113 | **0.064** | 1 |
| +14.71 | **0.168 ± 0.073** | −2.106 ± 4.274 | −2.274 | 0.111 | **0.062** | 0 |
| +17.71 | **0.084 ± 0.016** | −12.562 ± 24.000 | −12.646 | 0.132 | **0.060** | 0 |
**The hypothesis I set out to confirm is refuted, and by my own experiment.** I predicted that quantile's advantage would grow with skew, and a three-seed run appeared to show exactly that — a clean monotone climb to +0.201 in $R^2$ by skew 11. At ten seeds the sign is the other way in every row, and the gap widens *against* quantile as skew grows: −0.009, −0.033, −0.152, −2.274, −12.646. I want to be precise about what that does and does not establish, because the temptation is to over-correct into "uniform wins," and that is not what the table says either.

**What it actually shows is that quantile becomes unstable, not that it becomes inaccurate.** Read the standard deviations rather than the means. Quantile's spread explodes — ±0.169 at skew 5, ±0.813 at skew 10, ±4.274 and ±24.000 beyond — while uniform's stays bounded and its mean degrades smoothly toward zero. A mean of −12.6 with a deviation of ±24.0 is not a measurement of typical behaviour; it is a small number of catastrophic splits dragging an otherwise reasonable distribution. Quantile on a heavily skewed target sometimes produces a usable model and sometimes produces a disaster, and the three-seed run happened to miss the disasters. Uniform fails, but it fails *predictably*, and for a component inside a larger pipeline that is often the more valuable property.

**The starvation mechanism survives, and it is still the reason uniform degrades.** The last column collapses exactly as predicted: 11 samples at symmetry, *one* by skew 1.8, and *zero* past skew 14. Equal-width buckets on a skewed target put nearly every point in the first bucket and leave the rest to estimate rules from almost nothing, which is why uniform's $R^2$ falls from 0.912 to 0.084 across the sweep. That mechanism was never in doubt; what the ten-seed run overturns is the assumption that quantile's guaranteed occupancy therefore makes it the better choice. It removes one failure mode and introduces another.

**And my tail prediction was wrong, which is the one part of the original reading that holds up.** I expected uniform to hold the *tails* better because it covers the output range evenly. The opposite happens, consistently and with small spreads: uniform's tail error grows from 0.052 to 0.132 while quantile's actually *falls*, 0.051 to 0.060. The reasoning was right about geometry and wrong about data — an evenly spaced bucket in the sparse tail is useless if nothing lands in it. Coverage of the range is worthless without coverage of the *samples*.

**A caution about the last two rows.** Past skew 14 both schemes fail — quantile's $R^2$ goes sharply negative with standard deviations of ±4.3 and ±24.0 — so the apparent uniform "win" there is noise between two broken models, not a result. At that point the target is so compressed that uniform's smallest bucket is empty outright and the problem needs a target transform, not a better partition.

Taken together, this is now an open question rather than a settled recommendation, and Chapter 7 carries it as one. The defensible summary is that neither scheme is safe on a heavily skewed target: uniform starves and decays, quantile holds its buckets but goes unstable. What the experiment has actually earned is the negative result and the diagnosis — which is worth more to a reader than the tidy monotone story it replaces.

Having placed the output partition, I use the same per-feature Gaussian-mixture approach for the antecedents and apply linear regression to obtain first-order Takagi–Sugeno–Kang consequents. This is the Deep Thought move made concrete: the answers (the output buckets) are chosen before the questions (the antecedents) are known, and the questions are then fit to them.

Two empirical observations shape this, and the first corrects something I believed for longer than I should have. Consequent order matters more than I once credited: an earlier draft of this chapter asserted that first-order consequents were enough and that higher orders barely moved the accuracy, and Table 4.4 does not support that — moving from first to second order and then to the full second-order basis is worth several points of $R^2$ apiece on Concrete. First order is a floor here, not a sufficient choice. Second, the *antecedent* refinement that Chapter 6 §6.4 measures buys progressively less as that consequent capacity grows, and almost nothing at the top of the range. That decay is the *structure before search* thesis in miniature, and it is the more interesting of the two observations: the better the structure-derived model already is, the less a subsequent search finds. The closed-form solve I use for these consequents is a single shared primitive across several model types, and I defer its full treatment to Chapter 6.

### 4.3.3 Inference

Inference is ordinary fuzzy evaluation. For classification I take the class whose rule fires most strongly, $\arg\max_k$, and for regression I defuzzify by the weighted average as usual. Nothing exotic happens at inference time; all of the leverage is in how cheaply the model was built.

One structural consequence is worth noting here, because I lean on it in the discussion. Each class rule is estimated from that class's own data and nothing else — there is no joint fit coupling the classes together. So new labeled data for one class updates that class's densities without touching any other rule, and a new class can be added by fitting one more rule rather than retraining the model. That is what makes the construction naturally incremental and semi-supervised-friendly: the model can absorb data as it arrives. I have not yet run a controlled streaming or semi-supervised benchmark, so I state this as a property of the construction rather than a measured result.

### 4.3.4 Why the parameters grow linearly

The reason the whole thing stays fast and small is the factorization. Rather than partitioning the joint input space, I condition on the output — for regression, by cutting the output into equal-frequency buckets with a quantile split — and fit an independent one-dimensional Gaussian mixture per feature within each bucket or class.

The contrast with the grid construction of Chapter 1 is the whole argument of this chapter, so it is worth writing the two side by side. Gridding the inputs gives a rule count that is a product over the inputs,

$$ N_{rules}^{\text{grid}} = \prod_{i=1}^{M} N_{\mu_i} \sim \mathcal{O}(c^{M}), $$

exponential in the number of features $M$. Conditioning on the output instead gives

$$ N_{rules}^{\text{MoG}} = K, \qquad N_{params} \sim \mathcal{O}(K \cdot M \cdot p), $$

one rule per class (or output bucket) $K$, with parameters growing *linearly* in the number of features $M$ and the components per mixture $p$. For a twelve-class problem with eighty-three features — RT-IOT2022, below — the grid form is astronomically large while the factored form is a few thousand parameters and twelve rules. That single choice is what sidesteps the rule-base explosion, and it is the same instinct as the naive-Bayes factorization, traded deliberately for speed and interpretability.

### 4.3.5 The rule for everything else: anomalies and rare classes

The construction has a consequence I did not set out to obtain, and which I then built deliberately once I noticed it was available. Because every class rule is an explicit fuzzy membership over the input space, I know not just how strongly each class fires but how strongly *anything* fires. That means I can synthesize one more rule, automatically, that says *none of the above*.

The construction is direct. Let $\mu_k(x)$ be the firing strength of the rule for class $k$. Aggregating the known classes with the t-conorm $S$ gives the degree to which the model recognizes $x$ as something it has seen; the fuzzy complement of that is the degree to which it does not:

$$ \mu_{\text{anom}}(x) = 1 - S\big(\mu_1(x) + \theta,\; \mu_2(x) + \theta,\; \ldots,\; \mu_K(x) + \theta\big). $$

The $\theta$ term is a boost applied to each known-class firing before aggregation, and it is the single knob that sets how eager the anomaly rule is. A large $\theta$ inflates the known classes, shrinks the complement, and makes the model reluctant to cry anomaly; a small $\theta$ makes it suspicious of anything less than a confident match. Inference is unchanged — I take the $\arg\max$ over the $K$ class firings *plus* this one extra — so the anomaly rule simply wins whenever no known rule fires strongly enough to beat it. The choice of t-conorm matters here too, and it is a parameter rather than a fixed decision: the drastic Hamacher conorm is aggressive about treating partial matches as recognition, while min/max is more permissive of novelty.

Three things make this worth a section rather than a footnote.

It is **free**. There is no second model, no separately trained detector, no extra pass over the data. The anomaly rule is an algebraic consequence of the rules I already have, so it costs one t-conorm and one subtraction at inference time. Contrast this with the usual practice of bolting a one-class SVM or an isolation forest alongside a classifier and reconciling two models' disagreements.

It is **interpretable in exactly the way the rest of the model is**. When the anomaly rule fires, the explanation is not an opaque outlier score; it is "none of the known rules matched this, and here is how close each one came." Because the class firings are themselves readable, so is the negative result. That is a genuinely useful property for an operator deciding whether to escalate.

It addresses **rare classes**, which is the failure mode that motivated it. A class with a handful of examples is nearly invisible to accuracy-driven training: a model that ignores it entirely still scores well. But rare and unseen events are frequently the ones that matter — a novel intrusion, an off-nominal flight condition, a failure mode absent from the training set. The complement rule catches these *as a category*, without needing examples of them, which is the one thing a supervised class rule can never do.

I should be clear about the prior art, since novelty detection and open-set recognition are well-established fields with their own literature — one-class SVMs, isolation forests, Mahalanobis-distance novelty scores, and the open-set recognition line of work all attack this problem directly, and I do not claim to beat them at it. What I claim is narrower and, I think, more interesting: in a fuzzy inference system built this way, open-set behavior is not an addition but a *consequence*, obtained by applying the same t-conorm and complement the model already uses, and it inherits the model's interpretability for free. The honest comparison — my complement rule against a dedicated one-class detector on the same data — is run in §4.4. The short version is that the mechanism behaves as described and the three methods come out indistinguishable on the testbed available, which supports parity but not superiority.

## 4.4 Results

The datasets here are public, so unlike the psychiatric set of Chapter 3 I can name them freely.

On the **PhiUSIIL phishing URL** dataset the model reaches 0.997 ± 0.001 accuracy in about seven-tenths of a second, with two rules and a handful of clauses. That is the headline: a readable, two-rule fuzzy classifier, competitive on accuracy, trained in the time it takes to describe it. I should temper the "competitive" immediately, though — CART and a random forest both score a perfect 1.000 on this dataset, so PhiUSIIL is saturated and cannot separate these methods on accuracy. What it demonstrates is the *rule count and the training time*, not superiority.

On **RT-IOT2022** — 123,000 instances, 83 features, 12 output classes — the model trains in under a minute. This is the scale point: the answer-first construction does not fall over when the data gets large and multi-class, because the work is proportional to classes times features rather than to any product over inputs.

On the **BETH** host-telemetry set I test the anomaly rule of §4.3.5 in its hardest honest configuration: the model is trained on *benign traffic only* and then shown a test set containing malicious activity it has never seen. Nothing about the malicious class is available at training time — there is no "attack" rule to fire — so detection has to come from the complement rule alone. This is open-set recognition rather than classification, and it is the setting where the construction earns its keep. Because the boost $\theta$ is a single scalar, the operating point is a one-dimensional sweep rather than a retraining problem, and the sensitivity/precision trade-off can be read straight off that curve.

On the **UCI Concrete Compressive Strength** regression set, the flat model's test $R^2$ is −0.334, 0.772, and 0.824 at TSK orders zero, one, and two, rising to 0.859 with the full second-order basis. The zeroth-order figure deserves a word rather than a quiet omission: with constant consequents the model is not merely no better than predicting the mean, it is measurably *worse* than predicting the mean — a negative $R^2$ — so first-order consequents are not a refinement here but a requirement. From first order on the result is reasonable, and it is the launch point for the hierarchical models of Chapter 6.

The reconciliation this section used to promise has now been done. Chapter 6 previously quoted a flat-model $R^2$ of 0.658 from the tree-and-mixture experiment, which read as a contradiction of the figures above; under one protocol — identical splits, seeds, and preprocessing — the flat model scores 0.861 with refinement and the discrepancy turns out to have been configuration rather than disagreement. Chapter 6 Table 6.1 and the table above are now the same measurement seen from two chapters. The one remaining trap for a reader comparing them is Chapter 6's Table 6.2, which deliberately runs everything untuned at raw features and reports the flat model at 0.650; that is a different question, not a different answer.

**Table 4.4 — What the Mixture-of-Gaussians construction achieves.** Measured on a single workstation; the baseline comparison is Table 4.5.

| Dataset (task) | Size (N × M) | Train time | Accuracy / R² | Rule base |
|---|---|---:|---:|---|
| PhiUSIIL (binary classification) | 235K × 54 | 0.72 ± 0.04 s | 0.997 ± 0.001 acc. | 2 rules (K = 2) |
| RT-IOT2022 (12-class) | 123K × 83 | < 60 s | *pending* | ~12 rules (K = 12) |
| Concrete (regression, TSK order 1) | 1,030 × 8 | seconds | R² = 0.772 ± 0.034 | 3 output buckets |
| Concrete (regression, TSK order 2) | 1,030 × 8 | seconds | R² = 0.824 ± 0.043 | 3 output buckets |
| Concrete (regression, full 2nd) | 1,030 × 8 | seconds | **R² = 0.859 ± 0.039** | 3 output buckets |

The rule-base column is the point of the table as much as the accuracy is: for classification the count is simply the number of classes, and for regression the number of output buckets — never a product over inputs. RT-IOT2022 is the sharpest case, since a grid over 83 features would be beyond enumeration while this model carries twelve rules.

**Table 4.5 — Baseline comparison** *(structure fixed; cells to be filled by the reproduction harness).* The speed claim is only persuasive against the methods it displaces, so this is the first experiment owed to the chapter. Every method runs on identical splits, multi-seed with error bars, under the Goal G4 protocol. Two notes on how to read it. The Concrete column runs **every** arm on the log-and-standardized features of §4.3, so nothing in it is comparing models fitted on different inputs — an earlier version of this table paired the MoG's best accuracy with a training time measured at raw features, which is exactly the quiet mismatch the harness exists to prevent. Applying the transform uniformly costs the baselines nothing: CART and Random Forest split on rank and are therefore invariant to any monotone transform, which Table 4.1 measures directly at +0.001 and +0.000, so their column reads the same either way and the uniformity is free. And the MoG appears on two rows because only the first is fully instrumented: the 1st-order row's accuracy and training time come from the same measured run, while the full-second-order arm's time is not separately timed yet and is marked pending rather than borrowed from the row above.

| Method | Concrete R² | Concrete train time | PhiUSIIL accuracy | PhiUSIIL train time |
|---|---:|---:|---:|---:|
| **MoG FIS (this work)**, 1st order | 0.780 ± 0.029 | **0.37 ± 0.01 s** | **0.997 ± 0.001** | **0.72 ± 0.04 s** |
| **MoG FIS**, full 2nd order | **0.859 ± 0.039** | *pending* | — | — |
| ANFIS | *pending* | *pending* | *pending* | *pending* |
| GA-tuned FIS | *pending* | *pending* | *pending* | *pending* |
| CART (reference) | 0.826 ± 0.047 | seconds | 1.000 ± 0.000 | seconds |
| Random Forest (reference) | 0.909 ± 0.019 | seconds | 1.000 ± 0.000 | seconds |

**The mechanism, measured.** The BETH files are not in the repository, so the harness runs the same protocol on public data: leave-one-class-out, where each class is withheld from training in turn and treated as unseen, averaged over held-out classes and seeds. Sweeping the boost gives the operating curve the section promised.

**Table 4.6 — The anomaly operating curve.** Detection and false alarm as functions of $\theta$, on Glass (6 classes, leave-one-class-out); Hamacher conorm, 10 seeds, archived as `reproduce/outputs/seeds10-2026-08-01/`.

| $\theta$ | detection rate | false-alarm rate | detection − false alarm |
|---:|---:|---:|---:|
| 0.50 | 0.754 | 0.532 | +0.222 |
| **0.60** | 0.708 | 0.469 | **+0.239** |
| 0.70 | 0.631 | 0.395 | +0.236 |
| 0.80 | 0.559 | 0.332 | +0.227 |
| 0.90 | 0.473 | 0.277 | +0.196 |
| 0.99 | 0.380 | 0.211 | +0.169 |
| 1.10 | 0.000 | 0.000 | 0.000 |

The curve behaves exactly as §4.3.5 says it should, which is the first thing worth confirming. Raising $\theta$ inflates the known-class firings, shrinks the complement, and monotonically reduces both detection and false alarms; past $\theta = 1.1$ the boost saturates the aggregate and the anomaly rule stops firing altogether. The knob is real and it is monotone, so an operator can trade sensitivity against nuisance alarms by turning one scalar, which was the design claim.

Two honest observations follow. The default of $\theta = 0.99$ inherited from the BETH configuration is *not* a good operating point on this data — it gives about seven-tenths the separation available, and sits well down the low-sensitivity end of the curve — which is an argument for reporting the curve rather than a single number, and for tuning $\theta$ per deployment. Note also that there is no sharp optimum to tune *to*: $J$ sits between +0.222 and +0.239 across the whole range from 0.5 to 0.8, so the choice within that band is nearly free and the knob is forgiving rather than delicate. And the absolute performance, while no longer poor, is not yet strong: at the best setting the rule detects 71% of unseen points at a 47% false-alarm rate, which is a real but noisy signal rather than a deployable detector.

I want to be careful not to over-read that in either direction. Glass has 214 samples across six classes, several with fewer than a dozen members, so withholding a class removes much of the little data there is and the remaining model is asked to be confident about a space it has barely seen. It is a stress test, not a demonstration. What it establishes is that the mechanism works as described; what it does not establish is that the complement rule is *competitive*, and that requires BETH or a comparable dataset.

**Table 4.7 — Against detectors built for the job** *(θ = 0.99, matched operating points).* The baselines' contamination is set to the complement rule's observed false-alarm rate, so all three are compared at the same point on their curves rather than at whatever default each ships with.

| Method | Detection rate | False-alarm rate | Detection − false alarm | Separate model? |
|---|---:|---:|---:|:--:|
| **Complement rule (this work)** | 0.380 ± 0.334 | 0.211 ± 0.139 | +0.169 | no |
| Isolation Forest | 0.401 ± 0.324 | 0.230 ± 0.149 | **+0.171** | yes |
| One-class SVM | 0.281 ± 0.259 | 0.206 ± 0.125 | +0.076 | yes |

Isolation forest nominally leads, by 0.002. I am not going to read anything into that, and the ordering here has now changed twice under me — an earlier draft had the complement rule on top by a comfortable-looking margin, a five-seed re-run put isolation forest ahead by 0.038, and at ten seeds the two are level to within a fifth of a percent. The standard deviations across held-out classes are roughly twice the *largest* gap in the table and forty times the smallest, so every one of those orderings was noise being read as signal. What the table supports is the cheaper claim that motivated the section: the complement rule performs *comparably to purpose-built detectors while requiring no second model*, which is the property worth having. The one-class SVM trailing at +0.076 is the only separation here that exceeds its own error bar. Establishing more than parity needs a testbed larger than 214 samples, and is a goal for completion.

> **Reproduction.** Each table in this chapter has its own generator under `reproduce/tables/`, all emitting Markdown and CSV with mean ± standard deviation across a fixed seed set: Table 4.1 from `table_hyperparam_normalization.py`, Table 4.2 from `table_g5_output_partitioning.py`, Table 4.3 from `table_g5b_skew_sweep.py`, Tables 4.4 and 4.5 from `table_4_1_mog_baselines.py`, and Tables 4.6 and 4.7 from `table_4_4_openset.py` (the operating curve requires `REPRO_THETA_SWEEP=1`). Cells marked *pending* are those whose adapter or dataset was not yet wired up; the harness prints exactly what it could not run rather than substituting a guess. Per-cell provenance and the current reconciliation status of each table are tracked in `reproduce/PROVENANCE_MAP.md`.
>
> **TODO — repeatable performance (board-wide standard):** the numbers above are single-machine point estimates. Reproduce under the fixed protocol — pinned clocks and thermals, multiple seeds, reported error bars — before citation. See `ACTION_ITEMS.md` §A and Chapter 7, Goal G4.

**Figure 4.2 — The open-set operating curve.** Table 4.6, plotted: detection and false-alarm rate against the boost $\theta$, with $J$ — their difference, and the column the table reports — as the shaded band between them. The inherited default $\theta = 0.99$ and the saturation past $\theta = 1.1$ are both marked. The figure is drawn from the table's own CSV rather than from a second run of the sweep, so the two cannot disagree; regenerate the table with `REPRO_THETA_SWEEP=1` and the figure follows. To be re-taken on BETH via `plot_anomaly_threshold_sweep` when those data are available.
`![anomaly-sweep](fig/04-anomaly-sweep.png)`

**[FIGURE 4.3 — deliberately still a placeholder]** *Confusion matrix on RT-IOT2022 before and after the correction-rule pass, showing which class confusions the corrections repair. This is the one figure in the document that is not drawn, and the reason is the one §4.3.1 already gives: the accuracy contribution of the correction pass has not been isolated, so neither half of the before-and-after exists to plot. RT-IOT2022 is also not among the datasets the harness can currently load. It is left as a placeholder rather than approximated, and the skip is recorded with its reason in `reproduce/figures/registry.py`.*
`![rtiot-confusion](fig/04-rtiot-confusion.png)`

## 4.5 Discussion and Contributions

The method is fast for a simple reason: it replaces a global search over a huge space of possible rules with a handful of local density fits keyed to the known answers, and a closed-form solve for the consequents. It stays interpretable for an equally simple reason: there are only a few rules, they are written over named features in linguistic terms, and a person can read and edit them. And because each class rule is fit independently, as §4.3.3 noted, the construction is naturally incremental — new data updates one class's densities without disturbing the others — which lends itself to a semi-supervised, keep-learning setting.

The anomaly rule deserves a closing word, because it is the part of this chapter I expect to matter most outside the dissertation. Building the model as explicit fuzzy rules over the input space means the model knows the shape of what it has seen, and therefore knows the shape of what it has not. Taking the complement of the aggregate turns a closed-set classifier into an open-set one for the cost of one operation, and it does so without giving up the property that made the model worth building — when it flags something, it can say why. For the domains where this work is aimed, that combination is the whole point: an unexpected condition is exactly the case where you least want an unexplainable answer.

I want to be clear about what is and is not established. The construction is real and the timing numbers are real, but the speed claim is only fully persuasive against the right baselines, and I have not yet run the head-to-head against ANFIS and a genetic-algorithm-tuned FIS on identical splits; that table is the first thing I owe this chapter, and it is noted as a goal for completion. The performance numbers here are also subject to the board-wide repeatability standard — fixed hardware, multiple seeds, error bars — like every other number in the dissertation. And the honest scope of the accuracy claim is bounded by the naive-Bayes-like factorization: where feature interactions matter a great deal, the flat model will leave accuracy on the table, which is precisely the gap the hierarchical models of Chapter 6 exist to close. The bridge in the other direction — where the membership functions come from when the data has no coordinates and no Gaussian shape to fit — is Chapter 5.

---

*Draft — Chapter 4 prose, in the author's voice, opening on the Hitchhiker's Guide / consequent-first motif to match the tribble motif of Ch 1. Citations in bracketed shorthand pending the consolidated `references.bib`. Seven tables (4.1–4.7) and three figure placeholders (4.1–4.3) inline. Open items in `../ACTION_ITEMS.md`.*
