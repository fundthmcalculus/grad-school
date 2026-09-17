# Chapter 1 — The Problem

A fuzzy inference system (FIS) is one of the few *readable* model families: a short list of IF–THEN rules an expert can read and edit.

Three things break it at scale:

- **Rule-base explosion** — grid partitioning: $N_{rules}=\prod_i N_{\mu_i}$, exponential in the features (12 inputs × 3 sets → 531,441 rules)
- **Stochastic training** — GA / gradient descent: slow, initialization-sensitive
- **Analysis tools stop at a few thousand points** — classical VAT is $O(N^3)$: 124 s at 4,096 points → ~4 days for the 58,000-point shuttle set

![](fig/01-structure-before-search.png){width=72%}
