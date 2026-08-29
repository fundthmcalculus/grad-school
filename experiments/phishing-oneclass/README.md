# Experiment: one-class phishing detection on PhiUSIIL

**Status:** complete · **Started:** 2026-08-29 · **Primary model:** `tribblefis.one_class.TribbleOneClassDetector`

Findings are in [`RESULTS.md`](RESULTS.md). Short version: train the tribble fuzzy
one-class detector on legitimate URLs only and it flags phishing almost perfectly
on the **leak-free standard benchmark** — **AUROC 0.9992 ± 0.0001, 99.8% of
phishing caught at a 1% false-positive rate** (mean ± std over 5 seeds) — using
its `surprisal` formulation (summed
`-log` membership), not the saturating `1 - max firing` complement rule. It ties
a whitened-Gaussian Mahalanobis baseline, which is the tell that under
`whiten=True` + one Gaussian per component the surprisal score *is* a diagonalised
Mahalanobis distance. The near-perfect number is real for this corpus but rides
on how homogeneous PhiUSIIL's "legitimate" class is, and will not transfer to
live URLs.

## Question

Train a "one-sided" model on legitimate samples alone — never showing it a
phishing sample — and measure how well it flags phishing as out-of-distribution.

## Model

`TribbleOneClassDetector` (this project's fuzzy one-class estimator) with:

```python
TribbleOneClassDetector(whiten=True, score="surprisal", cov="pca",
                        n_gaussians=1, norm_conorm="probability")
```

`score="surprisal"` sums `-log(membership_j)` in the log domain; the default
`complement` (`1 - max firing`) rounds to 1.0 for every point past ~60 features
and must not be used with whitening on wide input (see the estimator's module
docstring). `whiten=True` PCA-whitens first because the product-t-norm rule
assumes feature independence. sklearn one-class detectors (Mahalanobis,
OneClassSVM, IsolationForest) run on the same features for comparison.

## Standard benchmark feature policy — no tripwire-leakage features

The model is **not** allowed to train on:

- **Target leaks** — `URLSimilarityIndex` (similarity to a whitelist of legit
  URLs), `TLDLegitimateProb`, `URLCharProb`.
- **Tripwires** — every feature that is *exactly constant across all legitimate
  training rows* (detected data-drivenly, not hand-listed): in PhiUSIIL these are
  always-HTTPS, never a query string (`?`/`=`/`&`), never an IP-domain, never
  obfuscation. A single such feature otherwise sends any phishing row that
  differs on it to infinite Gaussian distance and carries the whole score.

Near-oracle *content* features (`LineOfCode`, `NoOfImage`, …; separable because
phishing pages here are empty crawls) are **kept** in the standard set — they are
signal, not leakage — but the `no_content` robustness set drops them too, as a
floor. See `data.py` for the exact policy.

## Data

`data/PhiUSIIL_Phishing_URL_Dataset.csv`, 235,795 rows. `label` = 1 legitimate
(134,850) / 0 phishing (100,945). Fifty numeric features before the policy above;
39 in the standard benchmark, 33 in the robustness floor.

## Run

```
uv run --no-sync python experiments/phishing-oneclass/run.py
```

~3 min on CPU (5 seeds), exits 0. Reports every (model, feature set) as
mean ± std over `data.SEEDS`. Prints the imported `tribblefis` source path and
commit (numbers depend on the tribble-fis pin), the feature policy, and the full
tables.
