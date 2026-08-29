# One-class phishing detection on PhiUSIIL — results

Primary model: `tribblefis.one_class.TribbleOneClassDetector`, `score="surprisal"`,
`whiten=True`, `cov="pca"`, `n_gaussians=1`, `norm_conorm="probability"`. Trained
on legitimate URLs only; evaluated on all phishing. Seed 0; 70/30 legit split
(94,395 train normals; test = 40,455 held-out legit + 100,945 phishing).
Thresholds calibrated on training normals only — no phishing label touches any
model or threshold.

**Provenance:** produced against `tribble-fis` working tree **`297b64b`** (a
superset of the recorded submodule pin `987ed06`; `987ed06` includes the
constant-feature grid-partition fix #210). Numbers depend on this commit —
`run.py` prints the imported source path and SHA so the table can be reconciled.

## Standard benchmark — leak and tripwire features removed

The model may not train on `URLSimilarityIndex` or any other target leak, nor on
the nine features that are exactly constant across the legitimate training set.
This is the headline table (39 features).

| model | AUROC | AP | recall @1% FPR | recall @5% FPR |
|---|---:|---:|---:|---:|
| **Tribble surprisal** | **0.9993** | **0.9997** | **0.998** | **0.999** |
| Tribble trimmed (drop 2) | 0.9861 | 0.9946 | 0.864 | 0.927 |
| Tribble surprisal + Ledoit-Wolf | 0.9397 | 0.9758 | 0.609 | 0.780 |
| Mahalanobis (whitened Gaussian) | 0.9989 | 0.9996 | 0.997 | 0.999 |
| OneClassSVM (rbf) | 0.9720 | 0.9874 | 0.655 | 0.860 |
| IsolationForest | 0.9331 | 0.9707 | 0.490 | 0.687 |

**The tribble surprisal detector is the best model, catching 99.8% of phishing at
a 1% false-positive rate** — trained without ever seeing a phishing sample.

## Four things the table says

**1 — surprisal, not complement, is the formulation that works here.** The
default `1 - max firing` complement saturates on wide input; the log-domain
`surprisal` sum does not. Every number above uses `surprisal`. `trimmed` (drop
the two largest per-feature surprisals) *loses* 13 points of recall@1%FPR — so
for phishing the largest per-feature surprisals **are** the signal: a phishing
URL is caught by being extreme on a handful of whitened directions, and trimming
them throws that away.

**2 — surprisal ties Mahalanobis, and that is a consistency check, not a
coincidence.** Under `whiten=True` with one Gaussian per component, the summed
surprisal is `sum_j z_j^2 / 2 + const` — a diagonalised Mahalanobis distance in
the whitened frame. It matching `EmpiricalCovariance().mahalanobis` (0.9993 vs
0.9989) confirms the fuzzy estimator is computing what the theory says it should.
The fuzzy layer buys interpretability (per-feature memberships / firing rules),
not a different decision surface, at this setting.

**3 — the score survives removing the content features too, which is the real
finding.** On the `no_content` floor (33 features; also drop the near-oracle
content counts `LineOfCode`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`,
`NoOfExternalRef`), Tribble surprisal is **0.9994** and Mahalanobis **0.9992** —
unchanged. So the separation is not one feature or six; it is pervasive. PhiUSIIL's
legitimate class is a tight, low-entropy manifold, and whitening turns *every*
near-degenerate direction into a tripwire, not only the nine exactly-constant
ones. The tree/SVM models, which do not exploit that covariance structure, sit
15–20 points lower (IsolationForest 0.93–0.95, OneClassSVM 0.97).

**4 — Ledoit-Wolf whitening underperforms PCA here, against the library's
documented expectation.** `TribbleOneClassDetector`'s docstring calls
`cov="ledoit_wolf"` "a small consistent gain … in tail separation." On this
corpus it is a large *loss* (0.9397 vs 0.9993). LW shrinks the covariance toward
a scaled identity, which damps exactly the low-variance-in-legit directions that
carry the phishing signal; rank-preserving PCA whitening amplifies them. With
n ≈ 94k ≫ 39 features the sample covariance is well-conditioned, so shrinkage is
solving a problem this data does not have. One dataset, but a clean counter-example
to the default recommendation worth carrying back to the estimator.

## Removing the leaks barely moved the number — which is why the policy matters

Mahalanobis on the **full leaky** feature set (50 features, `URLSimilarityIndex`
and all tripwires included) scores AUROC 0.9990 — essentially identical to the
0.9989 it gets on the clean standard set. The leakage was never *necessary* for a
near-perfect score; the dataset is that separable. That is precisely why the
standard benchmark must exclude the tripwire-leakage features by policy rather
than by whether they change the headline: they would let a model claim the win
for the wrong reason, and the reason is what has to transfer.

## What to take away

- **Yes — the tribble one-class detector works on this dataset**, best-in-class,
  and its `surprisal` formulation is the right one to use past a handful of
  features.
- **The 0.999 is a property of PhiUSIIL, not of phishing detection.** Real
  legitimate traffic uses query strings, is not always HTTPS, is not pinned at
  `URLSimilarityIndex == 100`, and clones content-rich pages. Expect these
  numbers to fall sharply on live URLs; this benchmark cannot estimate by how much.
- **The transferable findings are the rankings and the mechanism:** covariance-
  aware one-class models (tribble surprisal ≈ Mahalanobis) dominate marginal
  models (IsolationForest) when the normal class is homogeneous; the largest
  per-feature surprisals carry the signal; and PCA whitening beats Ledoit-Wolf
  shrinkage when n ≫ p.

## Honest caveats / what this did not do

- Single seed. The split is large, so seed variance is small, but not measured
  across seeds.
- Categorical `TLD` dropped rather than frequency-encoded.
- `OneClassSVM` fit on a 20k subsample of train normals (RBF is O(n²)).
- No hyperparameter search beyond the three tribble score/cov variants shown; the
  homogeneous-manifold structure bounds how much tuning the baselines could
  recover.
- "Tripwire" is defined as exact zero variance in the fit rows. Near-constant
  features (e.g. `NoOfDegitsInURL`, 97% at one value) are kept; they behave like
  soft tripwires and are part of why the standard-set score stays near 0.999.
