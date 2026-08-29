# One-class phishing detection on PhiUSIIL — results

Trained on legitimate URLs only; evaluated on all phishing. Numbers from
`run.py` at seed 0 (70/30 legit split; 94,395 train normals; test = 40,455
held-out legit + 100,945 phishing). Anomaly score higher = more phishing-like.

## The headline

A **Gaussian one-class model (Mahalanobis distance under the legit covariance)
catches ~99.7% of phishing at a 1% false-positive rate**, trained without ever
seeing a phishing sample. `OneClassSVM` is close behind; `IsolationForest`
trails badly.

| feature set | model | AUROC | AP | recall @1% FPR | recall @5% FPR |
|---|---|---:|---:|---:|---:|
| all (50) | IsolationForest | 0.940 | 0.974 | 0.537 | 0.718 |
| all (50) | OneClassSVM(rbf) | 0.999 | 1.000 | 0.993 | 0.995 |
| all (50) | **Mahalanobis** | **0.999** | **1.000** | **0.997** | **0.999** |
| noleak (47) | IsolationForest | 0.930 | 0.970 | 0.507 | 0.677 |
| noleak (47) | OneClassSVM(rbf) | 0.975 | 0.989 | 0.655 | 0.864 |
| noleak (47) | **Mahalanobis** | **0.999** | **1.000** | **0.997** | **0.999** |
| hard (43) | IsolationForest | 0.948 | 0.979 | 0.600 | 0.765 |
| hard (43) | OneClassSVM(rbf) | 0.976 | 0.990 | 0.715 | 0.875 |
| hard (43) | **Mahalanobis** | **0.999** | **1.000** | **0.997** | **0.999** |

## The result that matters — the win survives every control, and that's the tell

I expected the leakage control to knock the numbers down. It didn't:

- Dropping the three **legitimacy-derived leak** features (`URLSimilarityIndex`,
  `TLDLegitimateProb`, `URLCharProb`) moves Mahalanobis from 0.9990 → 0.9989.
- Dropping **all seven** single features that alone separate the classes at
  AUC ≥ 0.95 moves it to 0.9992 — it goes *up*.

A one-class score that is invariant to removing the obvious separators is not
"robustly good," it's a symptom. So I looked for what carries it.

## Why it's easy: the legitimate class is pathologically homogeneous

Nine features are **exactly constant across all 94,395 legitimate training
rows**. Each is a tripwire: a Gaussian one-class model puts any phishing row that
differs on it at effectively infinite distance.

| tripwire (constant for every legit URL) | value | % of phishing that differ |
|---|---|---:|
| `URLSimilarityIndex` | 100 | 99.2% |
| `IsHTTPS` | 1 | 50.8% |
| `NoOfQMarkInURL` | 0 | 6.1% |
| `NoOfEqualsInURL` | 0 | 5.4% |
| `NoOfAmpersandInURL` | 0 | 0.9% |
| `IsDomainIP` | 0 | 0.6% |
| `HasObfuscation` / `NoOfObfuscatedChar` / `ObfuscationRatio` | 0 | 0.5% |

Read literally: in PhiUSIIL, a legitimate URL is *always* HTTPS, *never* contains
a query string (`?`, `=`, `&`), *never* uses an IP-address domain, and *never*
carries obfuscation. `IsHTTPS` alone — constant among legit, tripped by half of
phishing — is why the **hard** set (which keeps it; its own single-feature AUC is
only 0.75, below the 0.95 cut) still scores 0.999. That's also why
`IsolationForest`, which splits on marginal quantiles rather than exploiting a
zero-variance dimension, is the weakest model here (0.93–0.95): it cannot turn
"legit never does X" into an infinite distance the way the covariance-based
models do.

## What to take away

- **Yes, one-class detection works on this dataset**, and a plain Gaussian model
  beats the fancier ones. If the deliverable is "a one-class detector for
  PhiUSIIL," Mahalanobis at a train-calibrated 1% FPR is it.
- **The 0.999 is a property of the dataset, not of the method.** Real
  legitimate traffic uses query strings, isn't always HTTPS, and doesn't sit at
  `URLSimilarityIndex == 100`. Roughly half the phishing detection rides on a
  single binary tripwire (`IsHTTPS`) that will not hold outside this corpus.
  Expect these numbers to fall sharply on live URLs.
- **Model ranking, not the absolute score, is the transferable finding:** when a
  one-class problem has degenerate (zero-variance-in-normal) dimensions, a
  full-covariance Gaussian or an RBF `OneClassSVM` exploits them and an
  isolation forest leaves them on the table.

## Honest caveats / what this did not do

- Categorical `TLD` was dropped rather than frequency-encoded; it may add signal.
- `OneClassSVM` is fit on a 20k subsample of train normals (RBF is O(n²)); the
  full-train number could differ slightly.
- Single seed. The split is large enough that seed variance is small, but this
  was not repeated across seeds.
- No hyperparameter search — `IsolationForest` in particular might close some of
  the gap with tuning, though the tripwire structure above bounds how much.

## Natural next step

This project's thesis instrument is the tribble fuzzy classifier's "none of the
above" anomaly rule (Ch 4.3.5; see `experiments/fuzzy-lm-anomaly.md`). The honest
test is *not* whether it hits 0.999 here — the tripwires make that cheap — but
whether it holds up on the **hard** feature set against these baselines, where
the easy separators are gone.
