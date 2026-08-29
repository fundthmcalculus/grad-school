"""One-class phishing detection on PhiUSIIL: train only on legitimate URLs.

Question
--------
If we fit a one-class ("one-sided") model on *legitimate* samples alone and
never show it a phishing sample, how well does it flag phishing at test time as
out-of-distribution?

Data
----
``data/PhiUSIIL_Phishing_URL_Dataset.csv`` -- 235,795 rows, ``label`` is
1 = legitimate (134,850), 0 = phishing (100,945). Fifty numeric features after
dropping the five string/id columns (FILENAME, URL, Domain, TLD, Title).

Protocol (strict one-class)
---------------------------
* Legitimate rows are split 70/30. The 70% *fits* the scaler and each model.
  Phishing rows are never seen in training.
* Test set = held-out legitimate (the "normals") + all phishing (the "anomalies").
* Each model produces an anomaly score (higher = more anomalous). We report
  threshold-free ROC-AUC / average-precision on that score, and, at a threshold
  calibrated to a fixed false-positive rate *on training normals only*, the
  phishing detection rate (recall). Calibrating the threshold on train normals
  keeps the protocol honest -- no phishing label touches model or threshold.

Leakage control
---------------
Three features are derived from knowledge of the legitimate class itself --
``URLSimilarityIndex`` (similarity to a whitelist of legit URLs, single-feature
AUC 0.996), ``TLDLegitimateProb`` and ``URLCharProb`` (empirical legit
probabilities). Including them hands the one-class model the answer it is meant
to learn. We run every model twice: ``all`` features and ``noleak`` (those three
dropped) so the difference is visible rather than hidden.

Run from the repo root:  ``uv run --no-sync python experiments/phishing-oneclass/run.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

CSV = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
DROP = ["FILENAME", "URL", "Domain", "TLD", "Title", "label"]
LEAK = ["URLSimilarityIndex", "TLDLegitimateProb", "URLCharProb"]
NEAR_ORACLE_SEP = 0.95  # a single feature at/above this AUC already separates the classes
SEED = 0
OCSVM_FIT_N = 20_000  # OneClassSVM is O(n^2); fit on a subsample of train normals


def load():
    df = pd.read_csv(CSV)
    y = df["label"].to_numpy(int)  # 1 = legit, 0 = phish
    X = df.drop(columns=DROP).select_dtypes(include=[np.number])
    return X, y


def split(X: pd.DataFrame, y: np.ndarray, cols):
    """70/30 split of legitimate rows; all phishing goes to the anomaly set."""
    rng = np.random.default_rng(SEED)
    legit = np.flatnonzero(y == 1)
    phish = np.flatnonzero(y == 0)
    rng.shuffle(legit)
    cut = int(0.70 * len(legit))
    tr_idx, legit_te_idx = legit[:cut], legit[cut:]

    Xm = X[cols].to_numpy(float)
    scaler = StandardScaler().fit(Xm[tr_idx])
    Xs = scaler.transform(Xm)
    # y_test: 1 = anomaly (phish), 0 = normal (held-out legit)
    test_idx = np.concatenate([legit_te_idx, phish])
    y_test = np.concatenate([np.zeros(len(legit_te_idx)), np.ones(len(phish))]).astype(int)
    return Xs[tr_idx], Xs[test_idx], y_test, Xs


def scorers(X_tr):
    """Return {name: (fit_predict_score)} where each yields anomaly scores.

    Higher score = more anomalous. Each model is fit on X_tr (train normals).
    """
    rng = np.random.default_rng(SEED)

    def isoforest():
        m = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1)
        m.fit(X_tr)
        return lambda X: -m.score_samples(X)  # score_samples: higher = normal

    def ocsvm():
        sub = X_tr
        if len(X_tr) > OCSVM_FIT_N:
            sub = X_tr[rng.choice(len(X_tr), OCSVM_FIT_N, replace=False)]
        m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        m.fit(sub)
        return lambda X: -m.decision_function(X)  # higher = more anomalous

    def mahalanobis():
        cov = EmpiricalCovariance().fit(X_tr)
        return lambda X: cov.mahalanobis(X)  # squared distance; higher = anomalous

    return {
        "IsolationForest": isoforest,
        "OneClassSVM(rbf)": ocsvm,
        "Mahalanobis": mahalanobis,
    }


def detection_at_fpr(train_scores, test_scores, y_test, fpr_targets=(0.01, 0.05, 0.10)):
    """Threshold set on TRAIN normals to hit a target FPR; report recall on phish
    and the realised FPR on held-out legit."""
    out = {}
    legit_scores = test_scores[y_test == 0]
    phish_scores = test_scores[y_test == 1]
    for fpr in fpr_targets:
        thr = np.quantile(train_scores, 1 - fpr)  # flag top-fpr fraction of normals
        recall = float(np.mean(phish_scores > thr))
        real_fpr = float(np.mean(legit_scores > thr))
        out[fpr] = (recall, real_fpr)
    return out


def run_featureset(X, y, cols, tag):
    X_tr, X_te, y_te, X_all = split(X, y, cols)
    print(f"\n{'='*72}\nfeature set: {tag}  ({len(cols)} features)"
          f"  train_normals={len(X_tr)}  test={len(y_te)} "
          f"(legit={int((y_te==0).sum())}, phish={int((y_te==1).sum())})")
    rows = []
    for name, build in scorers(X_tr).items():
        score = build()
        s_tr = score(X_tr)
        s_te = score(X_te)
        auc = roc_auc_score(y_te, s_te)
        ap = average_precision_score(y_te, s_te)
        det = detection_at_fpr(s_tr, s_te, y_te)
        rows.append((name, auc, ap, det))
        d1, d5 = det[0.01], det[0.05]
        print(f"  {name:18s} AUROC={auc:.4f}  AP={ap:.4f}  "
              f"recall@1%FPR={d1[0]:.3f}(fpr {d1[1]:.3f})  "
              f"recall@5%FPR={d5[0]:.3f}(fpr {d5[1]:.3f})")
    return rows


def near_oracle_features(X, y):
    """Features whose raw value alone separates the classes at AUC >= threshold.
    Data-driven, not hand-picked -- these are what a one-class model can coast on."""
    hits = []
    for c in X.columns:
        auc = roc_auc_score(y, X[c].to_numpy(float))
        if max(auc, 1 - auc) >= NEAR_ORACLE_SEP:
            hits.append(c)
    return hits


def legit_homogeneity(X, y):
    """Why one-class is easy here: features that are *constant* across every
    legitimate training row become infinite-distance tripwires -- any phishing
    row that differs on one is flagged. Report the constant-in-legit features and
    the share of phishing that trips each."""
    rng = np.random.default_rng(SEED)
    legit = np.flatnonzero(y == 1)
    rng.shuffle(legit)
    tr = legit[: int(0.70 * len(legit))]
    phish = np.flatnonzero(y == 0)
    print("\nfeatures constant across ALL legit-train rows (one-class tripwires):")
    for c in X.columns:
        v = X[c].to_numpy(float)
        if v[tr].std() == 0.0:
            const = v[tr][0]
            trips = float(np.mean(v[phish] != const))
            print(f"  {c:28s} = {const:g} for all legit;  {trips*100:5.1f}% of phishing differ")


def main():
    X, y = load()
    all_cols = list(X.columns)
    noleak_cols = [c for c in all_cols if c not in LEAK]
    oracle = near_oracle_features(X, y)
    hard_cols = [c for c in all_cols if c not in oracle]
    print(f"near-oracle single features (sep >= {NEAR_ORACLE_SEP}): {oracle}")
    legit_homogeneity(X, y)
    results = {
        "all": run_featureset(X, y, all_cols, "all"),
        "noleak": run_featureset(X, y, noleak_cols, f"noleak (dropped {', '.join(LEAK)})"),
        "hard": run_featureset(X, y, hard_cols,
                               f"hard (dropped {len(oracle)} near-oracle features)"),
    }
    return results


if __name__ == "__main__":
    main()
