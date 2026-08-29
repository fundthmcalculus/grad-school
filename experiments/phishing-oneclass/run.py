"""One-class phishing benchmark on PhiUSIIL, primary model = the tribble fuzzy
one-class detector.

Train a one-class ("one-sided") model on legitimate URLs only and measure how
well it flags phishing as out-of-distribution. The primary estimator is
:class:`tribblefis.one_class.TribbleOneClassDetector` -- this project's own fuzzy
one-class formulation, *not* the multi-class "none of the above" complement rule.
It fits Gaussian memberships on the normal class and scores a point by its summed
surprisal ``sum_j -log(membership_j)`` (the ``score="surprisal"`` formulation:
non-saturating, unlike the ``1 - max firing`` complement, which rounds to 1.0 for
every point past ~60 features -- see the estimator's module docstring). Features
are PCA-whitened first because the product-t-norm rule assumes independence.

Standard benchmark feature policy (see ``data.py``): the model is **not** allowed
to train on the tripwire-leakage features. ``URLSimilarityIndex`` (a whitelist-
similarity leak) and the other legitimacy-derived leaks are removed, and so is
every feature that is exactly constant across the legitimate training set
(always-HTTPS, never-a-query-string, ...), since a single such tripwire otherwise
carries the whole score. sklearn one-class baselines run on the same features.

Run from the repo root:
    uv run --no-sync python experiments/phishing-oneclass/run.py

Provenance: numbers depend on the tribble-fis commit. This script prints the
imported source path and commit so the table can be reconciled later.
"""

from __future__ import annotations

import os

# Cap BLAS/OpenMP threads BEFORE numpy/numba load. On many-core Windows the
# numba-bundled OpenBLAS corrupts the heap at interpreter teardown (exit
# 0xC0000374) once its precompiled NUM_THREADS is exceeded -- harmless to the
# already-printed results, but it makes the script exit nonzero. Setting these
# first avoids it and keeps thread use reproducible.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import subprocess
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for sibling data.py

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import tribblefis
from tribblefis.one_class import TribbleOneClassDetector

from data import SEED, detection_at_fpr, feature_policy, load, split

OCSVM_FIT_N = 20_000  # OneClassSVM is O(n^2); fit on a subsample of train normals


# -- models: each returns (fit_on_train_normals) -> anomaly-score function -----
# Higher score = more anomalous (phishing-like) in every case.

def m_tribble_surprisal(X_tr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = TribbleOneClassDetector(
            whiten=True, score="surprisal", cov="pca",
            n_gaussians=1, norm_conorm="probability", random_state=SEED,
        ).fit(X_tr)
    return det.anomaly_score


def m_tribble_trimmed(X_tr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = TribbleOneClassDetector(
            whiten=True, score="trimmed", trim=2, cov="pca",
            n_gaussians=1, norm_conorm="probability", random_state=SEED,
        ).fit(X_tr)
    return det.anomaly_score


def m_tribble_ledoit(X_tr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = TribbleOneClassDetector(
            whiten=True, score="surprisal", cov="ledoit_wolf",
            n_gaussians=1, norm_conorm="probability", random_state=SEED,
        ).fit(X_tr)
    return det.anomaly_score


def _sklearn_prep(X_tr):
    sc = StandardScaler().fit(X_tr.to_numpy(float))
    return sc, sc.transform(X_tr.to_numpy(float))


def m_mahalanobis(X_tr):
    sc, Z = _sklearn_prep(X_tr)
    cov = EmpiricalCovariance().fit(Z)
    return lambda X: cov.mahalanobis(sc.transform(np.asarray(X, float)))


def m_ocsvm(X_tr):
    sc, Z = _sklearn_prep(X_tr)
    rng = np.random.default_rng(SEED)
    sub = Z if len(Z) <= OCSVM_FIT_N else Z[rng.choice(len(Z), OCSVM_FIT_N, replace=False)]
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(sub)
    return lambda X: -m.decision_function(sc.transform(np.asarray(X, float)))


def m_isoforest(X_tr):
    sc, Z = _sklearn_prep(X_tr)
    m = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Z)
    return lambda X: -m.score_samples(sc.transform(np.asarray(X, float)))


PRIMARY = {
    "Tribble surprisal": m_tribble_surprisal,
    "Tribble trimmed": m_tribble_trimmed,
    "Tribble surp+LW": m_tribble_ledoit,
}
BASELINES = {
    "Mahalanobis": m_mahalanobis,
    "OneClassSVM(rbf)": m_ocsvm,
    "IsolationForest": m_isoforest,
}


def evaluate(X, y, cols, models, tag):
    X_tr, X_te, y_te = split(X, y, cols)
    print(f"\n{'='*78}\n{tag}  ({len(cols)} features)  "
          f"train_normals={len(X_tr)}  test={len(y_te)} "
          f"(legit={int((y_te==0).sum())}, phish={int((y_te==1).sum())})")
    rows = []
    for name, build in models.items():
        score = build(X_tr)
        s_tr, s_te = np.asarray(score(X_tr)), np.asarray(score(X_te))
        auc = roc_auc_score(y_te, s_te)
        ap = average_precision_score(y_te, s_te)
        det = detection_at_fpr(s_tr, s_te, y_te)
        (r1, f1), (r5, f5) = det[0.01], det[0.05]
        rows.append((name, auc, ap, r1, r5))
        print(f"  {name:20s} AUROC={auc:.4f}  AP={ap:.4f}  "
              f"recall@1%FPR={r1:.3f}(fpr {f1:.3f})  recall@5%FPR={r5:.3f}(fpr {f5:.3f})")
    return rows


def main():
    print(f"tribblefis source: {tribblefis.__file__}")
    try:
        sha = subprocess.check_output(
            ["git", "-C", "tribble-fis", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        print(f"tribble-fis commit (working tree): {sha}")
    except Exception:
        pass

    X, y = load()
    fp = feature_policy(X, y)
    print(f"\nquarantined from the standard benchmark:")
    print(f"  leak (target-derived): {fp['leak']}")
    print(f"  tripwire (constant across all legit-train): {fp['tripwire']}")
    print(f"  kept but near-oracle content (dropped only in no_content): {fp['near_oracle']}")

    models = {**PRIMARY, **BASELINES}
    evaluate(X, y, fp["standard"], models, "STANDARD BENCHMARK (leak + tripwires removed)")
    evaluate(X, y, fp["no_content"], models, "ROBUSTNESS FLOOR (also drop near-oracle content)")

    # Reference only: the inflation the standard policy exists to prevent.
    print("\n" + "-" * 78)
    print("REFERENCE (DO NOT CITE): Mahalanobis on the full leaky feature set --")
    print("shows the score the tripwire-leakage policy is designed to reject.")
    evaluate(X, y, fp["all_leaky"], {"Mahalanobis": m_mahalanobis},
             "leaky reference")


if __name__ == "__main__":
    main()
