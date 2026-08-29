"""Shared data loading, feature policy, and one-class split for the PhiUSIIL
phishing one-class experiment. Imported by ``run.py``.

The feature policy is the load-bearing part. A one-class model fit on the
legitimate class of PhiUSIIL scores ~0.999 for reasons that are mostly artifacts
of how the dataset was built, so which features it is *allowed* to see decides
whether the number means anything.

Three tiers of feature are quarantined:

* ``LEAK`` -- features computed from knowledge of the legitimate class itself
  (``URLSimilarityIndex`` = similarity to a whitelist of legit URLs;
  ``TLDLegitimateProb`` / ``URLCharProb`` = empirical legit probabilities).
  Target leakage; excluded from the standard benchmark unconditionally.
* **tripwires** -- features that are *exactly constant* across every legitimate
  training row (zero variance). A Gaussian one-class model puts any phishing row
  that differs on one at effectively infinite distance, so a single such feature
  can carry the whole score. Detected data-drivenly (not hand-listed) and
  excluded from the standard benchmark. In PhiUSIIL these are: always HTTPS,
  never a query string (``?``/``=``/``&``), never an IP-domain, never obfuscation
  -- real properties, but constant only because this corpus's "legitimate" class
  is unusually homogeneous, so they will not hold on live URLs.
* **near-oracle content features** -- ``LineOfCode``, ``NoOfImage``, etc., whose
  raw value alone separates the classes at AUC >= 0.95 because phishing pages
  here are mostly empty crawls. These are *not* leakage and are KEPT in the
  standard benchmark, but the ``no_content`` set drops them too, as a robustness
  floor: is there one-class signal once every easy separator is gone?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CSV = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
DROP_STR = ["FILENAME", "URL", "Domain", "TLD", "Title", "label"]
LEAK = ["URLSimilarityIndex", "TLDLegitimateProb", "URLCharProb"]
NEAR_ORACLE_SEP = 0.95
SEED = 0
TRAIN_FRAC = 0.70


def load():
    """Return (X numeric-feature frame, y) with y: 1 = legit, 0 = phish."""
    df = pd.read_csv(CSV)
    y = df["label"].to_numpy(int)
    X = df.drop(columns=DROP_STR).select_dtypes(include=[np.number]).reset_index(drop=True)
    return X, y


def train_idx(y):
    """Indices of the legit rows used to fit (the model never sees phishing)."""
    rng = np.random.default_rng(SEED)
    legit = np.flatnonzero(y == 1)
    rng.shuffle(legit)
    return legit[: int(TRAIN_FRAC * len(legit))]


def feature_policy(X, y):
    """Classify every numeric feature; return the named feature sets.

    ``standard`` is the benchmark set: LEAK and tripwires removed, everything
    else kept. Tripwires are found from the fit rows only, so the policy uses no
    information the model wouldn't have.
    """
    tr = train_idx(y)
    tripwire = [c for c in X.columns if X[c].to_numpy(float)[tr].std() == 0.0]
    near_oracle = []
    for c in X.columns:
        auc = roc_auc_score(y, X[c].to_numpy(float))
        if max(auc, 1 - auc) >= NEAR_ORACLE_SEP:
            near_oracle.append(c)
    quarantined = set(LEAK) | set(tripwire)
    standard = [c for c in X.columns if c not in quarantined]
    no_content = [c for c in standard if c not in set(near_oracle)]
    return {
        "leak": list(LEAK),
        "tripwire": tripwire,
        "near_oracle": near_oracle,
        "standard": standard,       # the benchmark: leak + tripwires removed
        "no_content": no_content,   # robustness floor: also drop near-oracle content
        "all_leaky": list(X.columns),  # reference only -- DO NOT CITE
    }


def split(X, y, cols):
    """One-class split: fit on 70% of legit; test = held-out legit + all phish.

    Returns raw (unscaled) frames so each model applies its own scaling/whitening.
    ``y_test``: 0 = normal (held-out legit), 1 = anomaly (phishing).
    """
    tr = train_idx(y)
    legit = np.flatnonzero(y == 1)
    rng = np.random.default_rng(SEED)
    rng.shuffle(legit)
    legit_te = legit[int(TRAIN_FRAC * len(legit)):]
    phish = np.flatnonzero(y == 0)
    test = np.concatenate([legit_te, phish])
    y_test = np.concatenate([np.zeros(len(legit_te)), np.ones(len(phish))]).astype(int)
    Xc = X[cols]
    return (
        Xc.iloc[tr].reset_index(drop=True),
        Xc.iloc[test].reset_index(drop=True),
        y_test,
    )


def detection_at_fpr(train_scores, test_scores, y_test, fpr_targets=(0.01, 0.05)):
    """Threshold set on TRAIN normals to hit a target FPR; report phishing recall
    and the realised FPR on held-out legit. No phishing label touches the
    threshold."""
    out = {}
    legit_s = test_scores[y_test == 0]
    phish_s = test_scores[y_test == 1]
    for fpr in fpr_targets:
        thr = np.quantile(train_scores, 1 - fpr)
        out[fpr] = (float(np.mean(phish_s > thr)), float(np.mean(legit_s > thr)))
    return out
