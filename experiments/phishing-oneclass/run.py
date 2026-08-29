"""One-class phishing benchmark on PhiUSIIL, primary model = the tribble fuzzy
one-class detector. Multi-seed, with error bars.

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
every feature that is exactly constant across the legitimate class
(always-HTTPS, never-a-query-string, ...), since a single such tripwire otherwise
carries the whole score. sklearn one-class baselines run on the same features.

Every (model, feature set) is run across ``data.SEEDS`` -- each seed reshuffles
the legit train/test split and reseeds the model -- and reported as mean +/- std.

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

from data import SEEDS, detection_at_fpr, feature_policy, load, split

OCSVM_FIT_N = 20_000  # OneClassSVM is O(n^2); fit on a subsample of train normals


# -- models: each (X_tr, seed) -> anomaly-score function -----------------------
# Higher score = more anomalous (phishing-like) in every case.

def _tribble(X_tr, seed, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = TribbleOneClassDetector(
            whiten=True, n_gaussians=1, norm_conorm="probability",
            random_state=seed, **kw,
        ).fit(X_tr)
    return det.anomaly_score


def m_tribble_surprisal(X_tr, seed):
    return _tribble(X_tr, seed, score="surprisal", cov="pca")


def m_tribble_trimmed(X_tr, seed):
    return _tribble(X_tr, seed, score="trimmed", trim=2, cov="pca")


def m_tribble_ledoit(X_tr, seed):
    return _tribble(X_tr, seed, score="surprisal", cov="ledoit_wolf")


def _sklearn_prep(X_tr):
    sc = StandardScaler().fit(X_tr.to_numpy(float))
    return sc, sc.transform(X_tr.to_numpy(float))


def m_mahalanobis(X_tr, seed):
    sc, Z = _sklearn_prep(X_tr)
    cov = EmpiricalCovariance().fit(Z)
    return lambda X: cov.mahalanobis(sc.transform(np.asarray(X, float)))


def m_ocsvm(X_tr, seed):
    sc, Z = _sklearn_prep(X_tr)
    rng = np.random.default_rng(seed)
    sub = Z if len(Z) <= OCSVM_FIT_N else Z[rng.choice(len(Z), OCSVM_FIT_N, replace=False)]
    m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(sub)
    return lambda X: -m.decision_function(sc.transform(np.asarray(X, float)))


def m_isoforest(X_tr, seed):
    sc, Z = _sklearn_prep(X_tr)
    m = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(Z)
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
METRICS = ("AUROC", "AP", "recall@1%FPR", "recall@5%FPR")


def _one_seed(X, y, cols, build, seed):
    X_tr, X_te, y_te = split(X, y, cols, seed=seed)
    score = build(X_tr, seed)
    s_tr, s_te = np.asarray(score(X_tr)), np.asarray(score(X_te))
    det = detection_at_fpr(s_tr, s_te, y_te)
    return {
        "AUROC": roc_auc_score(y_te, s_te),
        "AP": average_precision_score(y_te, s_te),
        "recall@1%FPR": det[0.01][0],
        "recall@5%FPR": det[0.05][0],
    }


def evaluate(X, y, cols, models, tag, seeds=SEEDS):
    X_tr0, _, y_te0 = split(X, y, cols, seed=seeds[0])
    print(f"\n{'='*92}\n{tag}  ({len(cols)} features)  train_normals={len(X_tr0)}  "
          f"test={len(y_te0)} (legit={int((y_te0==0).sum())}, phish={int((y_te0==1).sum())})  "
          f"seeds={list(seeds)}")
    print(f"  {'model':20s} " + "  ".join(f"{m:>16s}" for m in METRICS))
    rows = {}
    for name, build in models.items():
        per = [_one_seed(X, y, cols, build, s) for s in seeds]
        agg = {m: (float(np.mean([p[m] for p in per])),
                   float(np.std([p[m] for p in per]))) for m in METRICS}
        rows[name] = agg
        cells = "  ".join(f"{agg[m][0]:.4f}±{agg[m][1]:.4f}" for m in METRICS)
        print(f"  {name:20s} {cells}")
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
    print("\nquarantined from the standard benchmark:")
    print(f"  leak (target-derived): {fp['leak']}")
    print(f"  tripwire (constant across all legit): {fp['tripwire']}")
    print(f"  kept but near-oracle content (dropped only in no_content): {fp['near_oracle']}")

    models = {**PRIMARY, **BASELINES}
    evaluate(X, y, fp["standard"], models, "STANDARD BENCHMARK (leak + tripwires removed)")
    evaluate(X, y, fp["no_content"], models, "ROBUSTNESS FLOOR (also drop near-oracle content)")

    # Reference only: the inflation the standard policy exists to prevent.
    print("\n" + "-" * 92)
    print("REFERENCE (DO NOT CITE): Mahalanobis on the full leaky feature set --")
    print("shows the score the tripwire-leakage policy is designed to reject.")
    evaluate(X, y, fp["all_leaky"], {"Mahalanobis": m_mahalanobis}, "leaky reference")


if __name__ == "__main__":
    main()
