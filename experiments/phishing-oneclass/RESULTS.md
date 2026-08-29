# One-class phishing detection on PhiUSIIL — results

Primary model: `tribblefis.one_class.TribbleOneClassDetector`, `score="surprisal"`,
`whiten=True`, `cov="pca"`, `n_gaussians=1`, `norm_conorm="probability"`. Trained
on legitimate URLs only; evaluated on all phishing. **Mean ± std over 5 seeds**
(0–4); each seed reshuffles the 70/30 legit split (94,395 train normals; test =
40,455 held-out legit + 100,945 phishing) and reseeds the model. Thresholds
calibrated on training normals only — no phishing label touches any model or
threshold. The feature policy is a fixed benchmark definition and does not vary
by seed (see `data.py`).

**Provenance:** the numbers here reproduce **byte-identically** across three
`tribble-fis` commits — the current pin **`ef15d6a`**, the earlier `987ed06`
(the constant-feature grid-partition fix #210), and the working tree `297b64b`
they were first taken on — because the one-class code path (`one_class.py`,
`gauss_math.py`, `gauss_data.py`) is unchanged across them. Verified by running
against each. `run.py` prints the imported source path and SHA so the table can
be reconciled. The proposal-defense citation (Table 4.12) uses the pin `ef15d6a`.

## Standard benchmark — leak and tripwire features removed

The model may not train on `URLSimilarityIndex` or any other target leak, nor on
the nine features that are exactly constant across the legitimate training set.
This is the headline table (39 features).

| model | AUROC | AP | recall @1% FPR | recall @5% FPR |
|---|---:|---:|---:|---:|
| **Tribble surprisal** | **0.9992 ± 0.0001** | **0.9997 ± 0.0001** | **0.9977 ± 0.0004** | **0.9986 ± 0.0000** |
| Tribble trimmed (drop 2) | 0.9848 ± 0.0008 | 0.9940 ± 0.0004 | 0.8492 ± 0.0075 | 0.9238 ± 0.0020 |
| Tribble surprisal + Ledoit-Wolf | 0.9390 ± 0.0013 | 0.9756 ± 0.0007 | 0.6094 ± 0.0030 | 0.7772 ± 0.0036 |
| Mahalanobis (whitened Gaussian) | 0.9988 ± 0.0003 | 0.9995 ± 0.0001 | 0.9974 ± 0.0000 | 0.9986 ± 0.0000 |
| OneClassSVM (rbf) | 0.9718 ± 0.0012 | 0.9872 ± 0.0003 | 0.6463 ± 0.0079 | 0.8527 ± 0.0039 |
| IsolationForest | 0.9370 ± 0.0040 | 0.9727 ± 0.0018 | 0.5023 ± 0.0180 | 0.7026 ± 0.0121 |

**The tribble surprisal detector is the best model, catching 99.8% of phishing at
a 1% false-positive rate** — trained without ever seeing a phishing sample. The
error bars (5 seeds) are tiny because the test set is 141k points: the rankings
are real, not seed noise. IsolationForest carries the widest bars (±0.004 AUROC,
±0.018 recall@1%FPR) from tree randomness; the covariance-aware models are stable
to the fourth decimal. Tribble surprisal edges Mahalanobis on AUROC by ~0.0004
(about one Mahalanobis std) and is a statistical tie at the 1%-FPR operating
point — as it must be; see point 2.

## Four things the table says

**1 — surprisal, not complement, is the formulation that works here.** The
default `1 - max firing` complement saturates on wide input; the log-domain
`surprisal` sum does not. Every number above uses `surprisal`. `trimmed` (drop
the two largest per-feature surprisals) *loses* ~15 points of recall@1%FPR
(0.998 → 0.849) — so for phishing the largest per-feature surprisals **are** the
signal: a phishing URL is caught by being extreme on a handful of whitened
directions, and trimming them throws that away.

**2 — surprisal ties Mahalanobis, and that is a consistency check, not a
coincidence.** Under `whiten=True` with one Gaussian per component, the summed
surprisal is `sum_j z_j^2 / 2 + const` — a diagonalised Mahalanobis distance in
the whitened frame. It matching `EmpiricalCovariance().mahalanobis` (0.9992 vs
0.9988, overlapping error bars) confirms the fuzzy estimator is computing what
the theory says it should.
The fuzzy layer buys interpretability (per-feature memberships / firing rules),
not a different decision surface, at this setting.

**3 — the score survives removing the content features too, which is the real
finding.** On the `no_content` floor (33 features; also drop the near-oracle
content counts `LineOfCode`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`,
`NoOfExternalRef`), Tribble surprisal is **0.9994 ± 0.0000** and Mahalanobis
**0.9991 ± 0.0002** — unchanged. So the separation is not one feature or six; it
is pervasive. PhiUSIIL's
legitimate class is a tight, low-entropy manifold, and whitening turns *every*
near-degenerate direction into a tripwire, not only the nine exactly-constant
ones. The tree/SVM models, which do not exploit that covariance structure, sit
15–20 points lower (IsolationForest 0.93–0.95, OneClassSVM 0.97).

**4 — Ledoit-Wolf whitening underperforms PCA here, against the library's
documented expectation.** `TribbleOneClassDetector`'s docstring calls
`cov="ledoit_wolf"` "a small consistent gain … in tail separation." On this
corpus it is a large *loss* (0.9390 vs 0.9992). LW shrinks the covariance toward
a scaled identity, which damps exactly the low-variance-in-legit directions that
carry the phishing signal; rank-preserving PCA whitening amplifies them. With
n ≈ 94k ≫ 39 features the sample covariance is well-conditioned, so shrinkage is
solving a problem this data does not have. One dataset, but a clean counter-example
to the default recommendation worth carrying back to the estimator.

## Removing the leaks barely moved the number — which is why the policy matters

Mahalanobis on the **full leaky** feature set (50 features, `URLSimilarityIndex`
and all tripwires included) scores AUROC 0.9988 ± 0.0003 — identical to the
0.9988 it gets on the clean standard set. The leakage was never *necessary* for a
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
