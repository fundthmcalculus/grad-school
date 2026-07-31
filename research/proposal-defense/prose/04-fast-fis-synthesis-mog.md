# Chapter 4 — Fast Interpretable FIS Synthesis via Mixtures of Gaussians

## 4.1 Introduction

In *The Hitchhiker's Guide to the Galaxy*, the supercomputer Deep Thought is asked for the Answer to the Ultimate Question of Life, the Universe, and Everything. It thinks for seven and a half million years and returns 42 — at which point everyone realizes they never worked out what the Question actually was. The rest of the story is a search, backward, for a Question that fits the Answer they already have.

This chapter is built on the same inversion, and it is the reason it is fast. A fuzzy rule has an antecedent — the IF part, a question about the inputs — and a consequent — the THEN part, the answer. The conventional way to build a fuzzy model is to lay out the questions first: partition every input into fuzzy sets, form a rule for every combination, and then search for consequents that make the whole thing fit. I do the opposite. I commit to the answers first — the output classes, or a set of output values spread across the range — and only then work backward to find the antecedents that select them. Starting from the answer collapses most of the work, because the answers are few and known while the space of possible questions is enormous. And unlike Deep Thought, the method finishes in seconds rather than geological time, which is the entire point: it is fast enough that the Vogons cannot demolish the Earth to make way for a hyperspace bypass before it is done.

The problem I am solving is the one I set up in Chapter 1. A fuzzy model built by gridding the inputs has a rule count that is the product of the per-input set counts, which is exponential, and the usual way to fit it — a genetic algorithm, or gradient descent, or ANFIS — is either slow to converge or dependent on an initial guess it cannot supply for itself. My claim in this chapter is that a Mixture-of-Gaussians construction generates both the membership functions and the rules directly from the data, produces on the order of one rule per output class rather than an exponential blowup, and does so without any post-hoc genetic search or gradient descent at all. The contributions are: an answer-first (consequent-first) construction for both classification and regression; a per-feature, per-class Gaussian-mixture antecedent generator whose parameter count grows linearly rather than exponentially; and a demonstration that this trains competitive models in seconds on datasets with hundreds of thousands of rows.

## 4.2 Background and Prior Art

The idea of building a model from the output side is not new, and I want to credit the ancestor directly. Sugeno and Yasukawa (1993) proposed identifying a fuzzy model by first clustering the *output* and working back to the input structure, which is the same instinct I am following. Wang and Mendel (1992) generate rules from data by a different route, and Chiu (1994) uses subtractive clustering to place rules. What I add to this lineage is a specific, cheap factorization of the antecedents and an explicit path to a compact rule base, which I lay out below.

I also owe the reader an honest caveat up front, because a sharp committee member will raise it immediately. For classification, fitting an independent Gaussian mixture per feature and per class, and combining them, is closely related to a Gaussian naive-Bayes model — a class-conditional density estimate with a feature-independence assumption. I am not going to pretend otherwise. What I am claiming is not a new density estimator; it is that this construction produces an *interpretable fuzzy inference system* — real membership functions, real linguistic rules, editable by hand — extremely fast, and that the naive-Bayes-like factorization is exactly what keeps it fast and small. Where the independence assumption costs accuracy, I recover it with a small number of correction rules rather than by abandoning the factorization.

## 4.3 Methodology

### 4.3.1 Classification: one rule per answer

For a classification problem with $N$ samples, $M$ features, and $K$ output classes, the construction runs answer-first. I segment the data by output class — the answers — and for each class I ask which features actually distinguish it, by comparing the per-feature statistics across classes; this is an $O(M^2)$ screening step that discards features carrying no discriminative signal. For each retained feature and class I fit a one-dimensional Gaussian mixture, using up to a few components, which becomes the membership function for that feature under that class. I combine the per-feature memberships for a class with a fuzzy OR (a t-conorm), and that combination *is* the rule for that class. Repeating over the $K$ classes gives $K$ rules, each a linear combination of at most $M \times p$ Gaussians. There is no grid, so there is no exponential rule base.

Finally I evaluate the confusion matrix and add a second, small pass of correction rules where two classes are being confused — the place where the feature-independence assumption is costing me. These corrections are targeted and few, and they keep the rule base readable.

**[FIGURE 4.1 — placeholder]** *Per-feature Gaussian-mixture membership functions for one output class, and the fuzzy-OR that forms the class rule. Show two or three features stacked, with the resulting linguistic rule written underneath.*
`![mog-classification](fig/04-mog-classification.png)`

### 4.3.2 Regression: place the answers, regress the questions

For regression the answer-first construction is even more literal. I place the output centroids first — spread uniformly across the output range unless I know better, with one centroid pinned at each extreme of the range — and only then find the antecedents. Having fixed the outputs, I use the same per-feature Gaussian-mixture approach for the antecedents and apply linear regression to obtain first-order Takagi–Sugeno–Kang consequents. This is the Deep Thought move made concrete: the answers (the output centroids) are chosen before the questions (the antecedents) are known, and the questions are then fit to them.

Two empirical observations shape this. First, first-order TSK consequents are enough — going to second order or higher barely moves the accuracy on the problems I have tried. Second, a round of local optimization on top of the fitted model also barely helps, which is a small but direct piece of evidence for the *structure before search* thesis: once the model is built from the data's structure, there is very little left for a search to find. The closed-form solve I use for these consequents is a single shared primitive across several model types, and I defer its full treatment to Chapter 6.

### 4.3.3 Inference

Inference is ordinary fuzzy evaluation. For classification I take the class whose rule fires most strongly, $\arg\max_k$, and for regression I defuzzify by the weighted average as usual. Nothing exotic happens at inference time; all of the leverage is in how cheaply the model was built.

### 4.3.4 Why the parameters grow linearly

The reason the whole thing stays fast and small is the factorization. Rather than partitioning the joint input space, I condition on the output — for regression, by cutting the output into equal-frequency buckets with a quantile split — and fit an independent one-dimensional Gaussian mixture per feature within each bucket or class. The number of parameters is therefore on the order of the number of buckets or classes times the number of features times the components per mixture, which is *linear* in the inputs, not exponential. That single choice is what sidesteps the rule-base explosion, and it is the same instinct as the naive-Bayes factorization — traded, deliberately, for speed and interpretability.

## 4.4 Results

The datasets here are public, so unlike the psychiatric set of Chapter 3 I can name them freely.

On the **PhiUSIIL phishing URL** dataset the model reaches 97–99% accuracy in about six seconds, with two rules and a handful of clauses. That is the headline: a readable, two-rule fuzzy classifier, competitive on accuracy, trained in the time it takes to describe it.

On **RT-IOT2022** — 123,000 instances, 83 features, 12 output classes — the model trains in under a minute. This is the scale point: the answer-first construction does not fall over when the data gets large and multi-class, because the work is proportional to classes times features rather than to any product over inputs.

On the **UCI Concrete Compressive Strength** regression set, the flat model's test $R^2$ is about 0.44, 0.77, and 0.87 at TSK orders zero, one, and two respectively — which is both a reasonable result and the launch point for the hierarchical models of Chapter 6, where a tree and a mixture of experts push it further.

**Table 4.1 — Training time and accuracy.** MoG columns are measured; the baseline columns are the experiments owed to this chapter (see the open item below and `ACTION_ITEMS.md` §C) and must be run on identical splits under the G4 protocol.

| Dataset (task) | MoG train time | MoG accuracy / R² | ANFIS | GA-tuned FIS | tree / RF ref |
|---|---:|---:|:--:|:--:|:--:|
| PhiUSIIL (classification) | ~6 s | 97–99% | _TODO_ | _TODO_ | _TODO_ |
| RT-IOT2022 (12-class) | < 60 s | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| Concrete (regression) | _TODO_ | R² ≈ 0.87 (order 2) | _TODO_ | _TODO_ | _TODO_ |

> **TODO — repeatable performance (board-wide standard):** the MoG timings/accuracies above are single-machine point estimates; reproduce under the fixed protocol (pinned clocks/thermals, multiple seeds, error bars) with the baseline columns filled before citation. See `ACTION_ITEMS.md` §A and Ch 7 Goal G4.

**[FIGURE 4.2 — placeholder]** *Confusion matrix on RT-IOT2022 before and after the correction-rule pass, showing which class confusions the corrections repair.*
`![rtiot-confusion](fig/04-rtiot-confusion.png)`

## 4.5 Discussion and Contributions

The method is fast for a simple reason: it replaces a global search over a huge space of possible rules with a handful of local density fits keyed to the known answers, and a closed-form solve for the consequents. It stays interpretable for an equally simple reason: there are only a few rules, they are written over named features in linguistic terms, and a person can read and edit them. And because the construction is incremental — new data updates the per-class densities without retraining from scratch — it lends itself to a semi-supervised, keep-learning setting.

I want to be clear about what is and is not established. The construction is real and the timing numbers are real, but the speed claim is only fully persuasive against the right baselines, and I have not yet run the head-to-head against ANFIS and a genetic-algorithm-tuned FIS on identical splits; that table is the first thing I owe this chapter, and it is noted as a goal for completion. The performance numbers here are also subject to the board-wide repeatability standard — fixed hardware, multiple seeds, error bars — like every other number in the dissertation. And the honest scope of the accuracy claim is bounded by the naive-Bayes-like factorization: where feature interactions matter a great deal, the flat model will leave accuracy on the table, which is precisely the gap the hierarchical models of Chapter 6 exist to close. The bridge in the other direction — where the membership functions come from when the data has no coordinates and no Gaussian shape to fit — is Chapter 5.

---

*Draft — Chapter 4 prose, in the author's voice, opening on the Hitchhiker's Guide / consequent-first motif to match the tribble motif of Ch 1. Citations in bracketed shorthand pending the consolidated `references.bib`. Two figures and one table placeholder marked inline. Source outline in `../chapters/04-fast-fis-synthesis-mog.md`.*
