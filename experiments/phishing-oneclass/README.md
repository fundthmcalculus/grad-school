# Experiment: one-class phishing detection on PhiUSIIL

**Status:** complete · **Started:** 2026-08-29

Findings are in [`RESULTS.md`](RESULTS.md). Short version: a one-class model
trained only on legitimate URLs catches phishing almost perfectly
(**Mahalanobis AUROC 0.999, ~99.7% of phishing caught at a 1% false-positive
rate**), and the score survives every leakage control I threw at it — but only
because the *legitimate* class in this dataset is pathologically homogeneous
(always HTTPS, never a query string, `URLSimilarityIndex` exactly 100 for all
134,850 legit rows). The headline number is real for this dataset and will not
transfer to the wild.

## Question

Train a "one-sided" model on legitimate samples alone — never showing it a
phishing sample — and measure how well it flags phishing as out-of-distribution.

## Data

`data/PhiUSIIL_Phishing_URL_Dataset.csv`, 235,795 rows. `label` is
**1 = legitimate (134,850), 0 = phishing (100,945)**. Fifty numeric features
after dropping the five string/id columns (FILENAME, URL, Domain, TLD, Title).

## Protocol

Strict one-class. Legitimate rows split 70/30; the 70% fits the scaler and every
model. Phishing is never seen in training. Test set = held-out legitimate
("normals") + all phishing ("anomalies"). Thresholds are calibrated to a target
false-positive rate **on training normals only** — no phishing label ever
touches a model or a threshold. Three one-class models: `IsolationForest`,
`OneClassSVM(rbf)`, and a Gaussian `Mahalanobis` distance under the legit
covariance.

## Leakage controls

Every model runs against three feature sets so any easy win is visible, not
hidden:

- **all** — 50 numeric features.
- **noleak** — drops the three features derived from knowledge of the
  legitimate class (`URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`).
- **hard** — drops all seven single features whose raw value alone separates the
  classes at AUC ≥ 0.95 (data-driven, not hand-picked).

## Run

```
uv run --no-sync python experiments/phishing-oneclass/run.py
```

~50 s on CPU. One script; it prints the per-feature-set table plus the two
diagnostics (near-oracle features and the legit-constant "tripwire" features)
that the findings rest on.
