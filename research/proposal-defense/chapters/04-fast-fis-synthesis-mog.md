# Chapter 4 — Fast Interpretable FIS Synthesis via Mixture-of-Gaussians

**Status:** Outline · Part II (COMPLETED work — quals "draft paper 3", MoG portion done)
**Repo:** `tribble-fis` (`gaussian_classifier.py`, `gaussian_regressor.py`, `gauss_math.py`; `gaussian_mixture/*`)
**One-line claim:** a Mixture-of-Gaussians procedure that generates FIS antecedents *and* rules directly from data — no rule-base explosion, no post-hoc GA/GD — training classifiers/regressors orders of magnitude faster than optimization-based FIS while staying interpretable.

---

## 4.1 Introduction

- The training bottleneck restated (from Ch 1): GA slow; GD needs a great initial guess; rule count = ∏ per-input MF counts.
- **Claim:** GMM identifies membership functions fast and accurately; logic operators picked post-facto; only the minimum number of rules (K rules for K classes). Semi-supervised-friendly (incorporate new data cheaply).
- Contributions: (1) quantile-conditioned, per-feature-factorized Gaussian-mixture antecedents (linear growth, dodges curse of dimensionality); (2) discriminant-feature selection per class; (3) closed-form ridge-TSK consequents (shared primitive — deeper in Ch 6); (4) confusion-matrix-driven second-pass correction rules.

## 4.2 Background & Prior Art

- Sugeno–Yasukawa 1993 (output-first / output-clustering modeling) — the direct ancestor; concede it, then scope the daylight (per-feature factorization + quantile conditioning + FIS export).
- Wang–Mendel 1992; Chiu 1994 (subtractive clustering); Jang–Sun 1993 (RBF↔TSK equivalence).
- **Caveat to state honestly:** for classification this factorization reduces to a Gaussian-naive-Bayes-like class-conditional density estimate — say so, and argue the value is the *interpretable FIS artifact* + speed, not a new density estimator.

## 4.3 Methodology

### 4.3.1 Classification training
1. Segment data by output class; compute statistical differences among inputs to pick **discriminant features** (O(M²)).
2. Fit a 1-D GMM per (feature, class) up to p Gaussians (KMeans + per-cluster normal).
3. **OR** the per-feature memberships together (t-conorm); repeat over selected features & classes.
4. Evaluate confusion matrix → identify second-pass correction rules (cascade of specialists w/ abstention — deeper in Ch 6).
- Result: K final rules, each a linear combination of M×p Gaussians. No rule-base explosion.

### 4.3.2 Regression training (TSK)
- Output centroids uniformly across output range (+ one at each extremum) unless prior knowledge; same antecedent approach; **linear regression → order-1 TSK consequents**.
- Finding to report: order-1 TSK suffices; higher orders and local optimization barely help.

### 4.3.3 Inference
- Standard fuzzy evaluation; classification by argmax over class rule activations.

### 4.3.4 Quantile-conditioned factorized antecedents (the scoped novelty)
- `partition_output` via `pd.qcut` (equal-frequency output buckets) → per-(feature,bucket) 1-D GMM; naive-Bayes-like factorization → **linear** parameters in (#buckets × features).

## 4.4 Results

*From `draft-paper3.md`, `gaussian_mixture/*` benchmark scripts.*

- **PhiUSIIL phishing** (UCI): 97–99% accuracy in ~6 s; 2 rules + a handful of clauses; no GD/GA.
- **RT-IOT2022:** 123K instances, 83 features, 12 classes — trains in < 60 s.
- **NASA shuttle / others:** ties into Ch 3 datasets.
- **UCI Concrete (regression):** flat MoG-TSK baseline R² ≈ 0.44/0.77/0.87 at orders 0/1/2 (also the launch point for Ch 6's tree/HME improvements).
- **Figures:** GMM membership plots per feature/class; confusion matrices; training-time-vs-accuracy comparison against ANFIS/GA-FIS baselines (**to add** — reviewers will want ANFIS & GA-tuned FIS head-to-head).
- **Table:** train time & accuracy: MoG vs ANFIS vs GA-FIS vs (decision tree / random forest reference).

## 4.5 Discussion & Contributions

- Why it's fast: replaces iterative global search with per-class local density fits + closed-form consequents.
- Interpretability: few rules, named features, linguistic terms; semi-supervised update path.
- Limits / honesty: naive-Bayes-like assumption; discriminant-feature selection is heuristic; needs the ANFIS/GA baselines to make the speed claim airtight.
- Hands off to Ch 5 (where MFs come from when there's *no* Gaussian assumption / only a dissimilarity matrix) and Ch 6 (hierarchy + the shared ridge solver + refinement).

---

### Open items
- Add ANFIS and GA-tuned-FIS baselines (train time + accuracy) — currently the comparison is implied, not tabulated.
- Confirm which datasets are cleared for publication (patient data is obfuscated/private).
- Decide split line with Ch 6 for the ridge-TSK consequent solver (introduce here, prove/complete there).
