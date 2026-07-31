# Chapter 4 — Fast Interpretable FIS Synthesis via Mixtures of Gaussians

## 4.1 Introduction

In *The Hitchhiker's Guide to the Galaxy*, the supercomputer Deep Thought is asked for the Answer to the Ultimate Question of Life, the Universe, and Everything. It thinks for seven and a half million years and returns 42 — at which point everyone realizes they never worked out what the Question actually was. The rest of the story is a search, backward, for a Question that fits the Answer they already have.

This chapter is built on the same inversion, and it is the reason it is fast. A fuzzy rule has an antecedent — the IF part, a question about the inputs — and a consequent — the THEN part, the answer. The conventional way to build a fuzzy model is to lay out the questions first: partition every input into fuzzy sets, form a rule for every combination, and then search for consequents that make the whole thing fit. I do the opposite. I commit to the answers first — the output classes, or a set of output values spread across the range — and only then work backward to find the antecedents that select them. Starting from the answer collapses most of the work, because the answers are few and known while the space of possible questions is enormous. And unlike Deep Thought, the method finishes in seconds rather than geological time, which is the entire point: it is fast enough that the Vogons cannot demolish the Earth to make way for a hyperspace bypass before it is done.

The problem I am solving is the one I set up in Chapter 1. A fuzzy model built by gridding the inputs has a rule count that is the product of the per-input set counts, which is exponential, and the usual way to fit it — a genetic algorithm, or gradient descent, or ANFIS — is either slow to converge or dependent on an initial guess it cannot supply for itself. My claim in this chapter is that a Mixture-of-Gaussians construction generates both the membership functions and the rules directly from the data, produces on the order of one rule per output class rather than an exponential blowup, and does so without any post-hoc genetic search or gradient descent at all. The contributions are: an answer-first (consequent-first) construction for both classification and regression; a per-feature, per-class Gaussian-mixture antecedent generator whose parameter count grows linearly rather than exponentially; an automatically synthesized *anomaly* rule — the complement of the t-conorm of every explicit rule — that gives the model open-set behavior and a handle on rare classes at no additional training cost; and a demonstration that this trains competitive models in seconds on datasets with hundreds of thousands of rows.

## 4.2 Background and Prior Art

The idea of building a model from the output side is not new, and I want to credit the ancestor directly. Sugeno and Yasukawa (1993) proposed identifying a fuzzy model by first clustering the *output* and working back to the input structure, which is the same instinct I am following. Wang and Mendel (1992) generate rules from data by a different route, and Chiu (1994) uses subtractive clustering to place rules. What I add to this lineage is a specific, cheap factorization of the antecedents and an explicit path to a compact rule base, which I lay out below.

I also owe the reader an honest caveat up front, because a sharp committee member will raise it immediately. For classification, fitting an independent Gaussian mixture per feature and per class, and combining them, is closely related to a Gaussian naive-Bayes model — a class-conditional density estimate with a feature-independence assumption. I am not going to pretend otherwise. What I am claiming is not a new density estimator; it is that this construction produces an *interpretable fuzzy inference system* — real membership functions, real linguistic rules, editable by hand — extremely fast, and that the naive-Bayes-like factorization is exactly what keeps it fast and small. Where the independence assumption costs accuracy, I recover it with a small number of correction rules rather than by abandoning the factorization.

## 4.3 Methodology

### 4.3.1 Classification: one rule per answer

For a classification problem with $N$ samples, $M$ features, and $K$ output classes, the construction runs answer-first. I segment the data by output class — the answers — and for each class I ask which features actually distinguish it, by comparing the per-feature statistics across classes; this is an $O(M^2)$ screening step that discards features carrying no discriminative signal. For each retained feature and class I fit a one-dimensional Gaussian mixture, using up to a few components, which becomes the membership function for that feature under that class. I combine the per-feature memberships for a class with a fuzzy OR — a t-conorm — and that combination *is* the rule for that class. Repeating over the $K$ classes gives $K$ rules, each a disjunction over at most $M \times p$ Gaussian terms. There is no grid, so there is no exponential rule base.

Finally I evaluate the confusion matrix and add a second, small pass of correction rules where two classes are being confused — the place where the feature-independence assumption is costing me. These corrections are targeted and few, and they keep the rule base readable. I should note that I have not yet isolated how much accuracy this second pass buys; the before-and-after comparison is a small experiment I owe, and Figure 4.3 is where it belongs.

**[FIGURE 4.1 — placeholder]** *Per-feature Gaussian-mixture membership functions for one output class, and the fuzzy-OR that forms the class rule. Show two or three features stacked, with the resulting linguistic rule written underneath.*
`![mog-classification](fig/04-mog-classification.png)`

### 4.3.2 Regression: place the answers, regress the questions

For regression the answer-first construction is even more literal. I partition the output range first, and only then find the antecedents. There are two ways to place that partition, and the choice matters enough to state explicitly. The simple one is to spread the output centroids *uniformly* across the range, with one pinned at each extreme, which is the right default when I have no reason to think the output is unevenly distributed. The better one for skewed targets is an *equal-frequency* (quantile) split, which puts more resolution where the data actually is; this is what the implementation uses by default, and it is the version I mean in §4.3.4 below. Either way the answers come first.

Having fixed the output partition, I use the same per-feature Gaussian-mixture approach for the antecedents and apply linear regression to obtain first-order Takagi–Sugeno–Kang consequents. This is the Deep Thought move made concrete: the answers (the output buckets) are chosen before the questions (the antecedents) are known, and the questions are then fit to them.

Two empirical observations shape this. First, first-order TSK consequents are enough — going to second order or higher barely moves the accuracy on the problems I have tried. Second, a round of local optimization on top of the fitted model also barely helps, which is a small but direct piece of evidence for the *structure before search* thesis: once the model is built from the data's structure, there is very little left for a search to find. The closed-form solve I use for these consequents is a single shared primitive across several model types, and I defer its full treatment to Chapter 6.

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

I should be clear about the prior art, since novelty detection and open-set recognition are well-established fields with their own literature — one-class SVMs, isolation forests, Mahalanobis-distance novelty scores, and the open-set recognition line of work all attack this problem directly, and I do not claim to beat them at it. What I claim is narrower and, I think, more interesting: in a fuzzy inference system built this way, open-set behavior is not an addition but a *consequence*, obtained by applying the same t-conorm and complement the model already uses, and it inherits the model's interpretability for free. The honest comparison — my complement rule against a dedicated one-class detector on the same data — is an experiment I owe.

## 4.4 Results

The datasets here are public, so unlike the psychiatric set of Chapter 3 I can name them freely.

On the **PhiUSIIL phishing URL** dataset the model reaches 97–99% accuracy in about six seconds, with two rules and a handful of clauses. That is the headline: a readable, two-rule fuzzy classifier, competitive on accuracy, trained in the time it takes to describe it.

On **RT-IOT2022** — 123,000 instances, 83 features, 12 output classes — the model trains in under a minute. This is the scale point: the answer-first construction does not fall over when the data gets large and multi-class, because the work is proportional to classes times features rather than to any product over inputs.

On the **BETH** host-telemetry set I test the anomaly rule of §4.3.5 in its hardest honest configuration: the model is trained on *benign traffic only* and then shown a test set containing malicious activity it has never seen. Nothing about the malicious class is available at training time — there is no "attack" rule to fire — so detection has to come from the complement rule alone. This is open-set recognition rather than classification, and it is the setting where the construction earns its keep. Because the boost $\theta$ is a single scalar, the operating point is a one-dimensional sweep rather than a retraining problem, and the sensitivity/precision trade-off can be read straight off that curve.

On the **UCI Concrete Compressive Strength** regression set, the flat model's test $R^2$ is about 0.44, 0.77, and 0.87 at TSK orders zero, one, and two respectively — which is both a reasonable result and the launch point for the hierarchical models of Chapter 6, where a tree and a mixture of experts push it further. One caution for the reader comparing chapters: Chapter 6 quotes a flat-model $R^2$ of 0.658 on the same dataset, which looks like a contradiction and is not. That figure comes from the tree-and-mixture experiment, which uses a different split, preprocessing, and order selection. Running one consistent Concrete benchmark so the flat baseline reads identically in both chapters is a reconciliation I owe.

**Table 4.1 — What the Mixture-of-Gaussians construction achieves.** Measured on a single workstation; the baseline comparison is Table 4.2.

| Dataset (task) | Size (N × M) | Train time | Accuracy / R² | Rule base |
|---|---|---:|---:|---|
| PhiUSIIL (binary classification) | 235K × 54 | ~6 s | 97–99% acc. | 2 rules (K = 2) |
| RT-IOT2022 (12-class) | 123K × 83 | < 60 s | *pending* | ~12 rules (K = 12) |
| Concrete (regression, TSK order 0) | 1,030 × 8 | seconds | R² ≈ 0.44 | 3 output buckets |
| Concrete (regression, TSK order 1) | 1,030 × 8 | seconds | R² ≈ 0.77 | 3 output buckets |
| Concrete (regression, TSK order 2) | 1,030 × 8 | seconds | R² ≈ 0.87 | 3 output buckets |

The rule-base column is the point of the table as much as the accuracy is: for classification the count is simply the number of classes, and for regression the number of output buckets — never a product over inputs. RT-IOT2022 is the sharpest case, since a grid over 83 features would be beyond enumeration while this model carries twelve rules.

**Table 4.2 — Baseline comparison** *(structure fixed; cells to be filled by the reproduction harness).* The speed claim is only persuasive against the methods it displaces, so this is the first experiment owed to the chapter. Every method runs on identical splits, multi-seed with error bars, under the Goal G4 protocol.

| Method | Concrete R² | Concrete train time | PhiUSIIL accuracy | PhiUSIIL train time |
|---|---:|---:|---:|---:|
| **MoG FIS (this work)** | ≈ 0.87 | seconds | 97–99% | ~6 s |
| ANFIS | *pending* | *pending* | *pending* | *pending* |
| GA-tuned FIS | *pending* | *pending* | *pending* | *pending* |
| CART / Random Forest (reference) | *pending* | *pending* | *pending* | *pending* |

**Table 4.3 — Open-set detection with the complement rule** *(structure fixed; cells from the harness).* Trained on benign traffic only; the malicious class is unseen at training time. The comparison that matters is against detectors built for this job. Detection and false-alarm rates are reported at a matched operating point taken from the $\theta$ sweep, since quoting either one alone would be meaningless.

| Method | Trained on | Detection rate | False-alarm rate | Extra model? |
|---|---|---:|---:|:--:|
| **Complement rule (this work)** | benign only | *pending* | *pending* | no |
| One-class SVM | benign only | *pending* | *pending* | yes |
| Isolation Forest | benign only | *pending* | *pending* | yes |

> **Reproduction.** Tables 4.1–4.3 regenerate from `reproduce/tables/table_4_1_mog_baselines.py`, which emits Markdown and CSV with mean ± standard deviation across a fixed seed set. Cells marked *pending* are those whose adapter or dataset was not yet wired up; the harness prints exactly what it could not run rather than substituting a guess.
>
> **TODO — repeatable performance (board-wide standard):** the numbers above are single-machine point estimates. Reproduce under the fixed protocol — pinned clocks and thermals, multiple seeds, reported error bars — before citation. See `ACTION_ITEMS.md` §A and Chapter 7, Goal G4.

**[FIGURE 4.2 — placeholder]** *The anomaly threshold sweep on BETH: detection rate and false-alarm rate as functions of the boost $\theta$, showing the operating curve a user picks a point on. Generated by `plot_anomaly_threshold_sweep`.*
`![anomaly-sweep](fig/04-anomaly-sweep.png)`

**[FIGURE 4.3 — placeholder]** *Confusion matrix on RT-IOT2022 before and after the correction-rule pass, showing which class confusions the corrections repair.*
`![rtiot-confusion](fig/04-rtiot-confusion.png)`

## 4.5 Discussion and Contributions

The method is fast for a simple reason: it replaces a global search over a huge space of possible rules with a handful of local density fits keyed to the known answers, and a closed-form solve for the consequents. It stays interpretable for an equally simple reason: there are only a few rules, they are written over named features in linguistic terms, and a person can read and edit them. And because each class rule is fit independently, as §4.3.3 noted, the construction is naturally incremental — new data updates one class's densities without disturbing the others — which lends itself to a semi-supervised, keep-learning setting.

The anomaly rule deserves a closing word, because it is the part of this chapter I expect to matter most outside the dissertation. Building the model as explicit fuzzy rules over the input space means the model knows the shape of what it has seen, and therefore knows the shape of what it has not. Taking the complement of the aggregate turns a closed-set classifier into an open-set one for the cost of one operation, and it does so without giving up the property that made the model worth building — when it flags something, it can say why. For the domains where this work is aimed, that combination is the whole point: an unexpected condition is exactly the case where you least want an unexplainable answer.

I want to be clear about what is and is not established. The construction is real and the timing numbers are real, but the speed claim is only fully persuasive against the right baselines, and I have not yet run the head-to-head against ANFIS and a genetic-algorithm-tuned FIS on identical splits; that table is the first thing I owe this chapter, and it is noted as a goal for completion. The performance numbers here are also subject to the board-wide repeatability standard — fixed hardware, multiple seeds, error bars — like every other number in the dissertation. And the honest scope of the accuracy claim is bounded by the naive-Bayes-like factorization: where feature interactions matter a great deal, the flat model will leave accuracy on the table, which is precisely the gap the hierarchical models of Chapter 6 exist to close. The bridge in the other direction — where the membership functions come from when the data has no coordinates and no Gaussian shape to fit — is Chapter 5.

---

*Draft — Chapter 4 prose, in the author's voice, opening on the Hitchhiker's Guide / consequent-first motif to match the tribble motif of Ch 1. Citations in bracketed shorthand pending the consolidated `references.bib`. Three tables (4.1–4.3) and three figure placeholders (4.1–4.3) inline. Source outline in `../chapters/04-fast-fis-synthesis-mog.md`; open items in `../ACTION_ITEMS.md`.*
