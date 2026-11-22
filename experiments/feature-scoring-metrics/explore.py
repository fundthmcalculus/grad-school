"""Feature-scoring metrics: accuracy, cost and selection for each candidate.

Protocol (see README.md): scores use the training split only; every metric
selects the top-k features by its own score with no redundancy filter; the same
classifier is fit on the selected columns. Predictions are registered in README.md.

Run (from repo root):  uv run --project tribble-fis python experiments/feature-scoring-metrics/explore.py
Writes results.csv and results.md next to this file.

Knobs (unset, the run is the 2026-11-22 one: ten metrics, results.csv):
    FSM_METRICS   e.g. "lib:ks,lib:hellinger,lib:jensen_shannon,lib:mutual_info"
        A `lib:` prefix ranks with the library's own scorer
        (`calculate_gaussian_correlation(method=...)`, tribble-fis #253) instead of
        this file's reference implementation of the same metric.
    FSM_OUT       output stem, e.g. "results_library" (writes <stem>.csv, <stem>.md)
"""

from __future__ import annotations

import csv
import re
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "reproduce", "tables"))
sys.path.insert(0, os.path.join(REPO, "reproduce"))

from tribblefis.gauss_math import calculate_gaussian_correlation  # noqa: E402
from tribblefis.gaussian_classifier import TribbleClassifier  # noqa: E402

import _fuzzy_models as _fm  # noqa: E402

warnings.filterwarnings("ignore")

SEEDS = range(10)
KS = (2, 4, 5, 8, 10)
BUILTIN = ("wasserstein", "bhattacharyya", "composite", "higher_order")
NEW = ("anova_f", "mutual_info", "ks", "energy", "hellinger", "jensen_shannon")
METRICS = tuple(
    m.strip()
    for m in os.environ.get("FSM_METRICS", ",".join(BUILTIN + NEW)).split(",")
    if m.strip()
)
OUT_STEM = os.environ.get("FSM_OUT", "results")
BINS = 20
EPS = 1e-12


# ---- new candidate scores ------------------------------------------------------
def _cdf_gap_integral(u, v, power):
    """Exact integral over x of |F_u - F_v|**power, on the union of support points."""
    u = np.sort(u)
    v = np.sort(v)
    grid = np.union1d(u, v)
    if len(grid) < 2:
        return 0.0
    fu = np.searchsorted(u, grid[:-1], side="right") / len(u)
    fv = np.searchsorted(v, grid[:-1], side="right") / len(v)
    return float(np.sum(np.abs(fu - fv) ** power * np.diff(grid)))


def _binned(a, b, edges):
    p = np.histogram(a, bins=edges)[0].astype(float)
    q = np.histogram(b, bins=edges)[0].astype(float)
    return p / max(p.sum(), 1.0), q / max(q.sum(), 1.0)


def _hellinger(p, q):
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def _jensen_shannon(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2((a[mask] + EPS) / (b[mask] + EPS))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def one_vs_rest(x, y, pair_fn):
    """Mean over classes of pair_fn(samples_of_class, samples_of_rest)."""
    vals = []
    for c in np.unique(y):
        a = x[y == c]
        b = x[y != c]
        if len(a) and len(b):
            vals.append(pair_fn(a, b))
    return float(np.mean(vals)) if vals else 0.0


def scores_for(metric, Xtr, ytr, seed):
    """Per-feature score, higher = more useful, computed on the training split only."""
    n_features = Xtr.shape[1]
    if metric == "anova_f":
        f, _ = f_classif(Xtr, ytr)
        return np.nan_to_num(f)
    if metric == "mutual_info":
        return mutual_info_classif(Xtr, ytr, n_neighbors=3, random_state=seed)
    out = np.zeros(n_features)
    for j in range(n_features):
        x = Xtr[:, j]
        if metric == "ks":
            out[j] = one_vs_rest(x, ytr, lambda a, b: ks_2samp(a, b).statistic)
        elif metric == "energy":
            out[j] = one_vs_rest(x, ytr, lambda a, b: _cdf_gap_integral(a, b, 2))
        elif metric in ("hellinger", "jensen_shannon"):
            edges = np.unique(np.quantile(x, np.linspace(0, 1, BINS + 1)))
            if len(edges) < 2:
                continue
            fn = _hellinger if metric == "hellinger" else _jensen_shannon
            out[j] = one_vs_rest(x, ytr, lambda a, b: fn(*_binned(a, b, edges)))
    return out


def builtin_scores(method, Xtr_df, ytr):
    ranked = calculate_gaussian_correlation(
        Xtr_df.reset_index(drop=True),
        pd.Series(ytr).reset_index(drop=True),
        method=method,
        top_n=-1,
    )
    return {name: float(score) for name, score in ranked}


# ---- experiment ------------------------------------------------------------------
def load_datasets():
    out = {}
    X, y = _fm.load_phiusiil()
    out["PhiUSIIL"] = (pd.DataFrame(np.asarray(X)), np.asarray(y))
    Xb, yb = load_breast_cancer(return_X_y=True, as_frame=True)
    out["Breast Cancer"] = (Xb.reset_index(drop=True), np.asarray(yb))
    Xw, yw = load_wine(return_X_y=True, as_frame=True)
    out["Wine"] = (Xw.reset_index(drop=True), np.asarray(yw))
    return out


def run():
    rows = []
    for ds_name, (Xdf, y) in load_datasets().items():
        names = list(Xdf.columns)
        for seed in SEEDS:
            Xtr_df, Xte_df, ytr, yte = train_test_split(
                Xdf, y, test_size=0.2, random_state=seed
            )
            Xtr = np.asarray(Xtr_df, dtype=float)
            Xte = np.asarray(Xte_df, dtype=float)
            for metric in METRICS:
                t0 = time.perf_counter()
                if metric in BUILTIN or metric.startswith("lib:"):
                    sc = builtin_scores(metric.removeprefix("lib:"), Xtr_df, ytr)
                    order = [
                        names.index(n)
                        for n, _ in sorted(sc.items(), key=lambda kv: -kv[1])
                    ]
                else:
                    s = scores_for(metric, Xtr, ytr, seed)
                    order = list(np.argsort(-s, kind="stable"))
                score_s = time.perf_counter() - t0
                for k in KS:
                    if k > len(names):
                        continue
                    sel = order[:k]
                    t1 = time.perf_counter()
                    # correlation_threshold=1.0 disables the redundancy filter
                    # (gauss_math._finalise_ranking filters only for 0 < t < 1).
                    clf = TribbleClassifier(
                        top_n=k,
                        scoring_method="wasserstein",
                        random_state=seed,
                        correlation_threshold=1.0,
                    )
                    clf.fit(Xtr[:, sel], ytr)
                    fit_s = time.perf_counter() - t1
                    acc = accuracy_score(yte, np.asarray(clf.predict(Xte[:, sel])))
                    kept_list = (
                        list(clf.top_features_) if clf.top_features_ is not None else []
                    )
                    kept = len(kept_list)
                    # The classifier names its inputs by position within the subset it was
                    # given (feature_j is column sel[j]), so map back to the original column.
                    kept_names = "|".join(
                        str(names[sel[int(re.sub(r"\D", "", str(f)))]])
                        for f in kept_list
                    )
                    rows.append(
                        dict(
                            dataset=ds_name,
                            metric=metric,
                            k=k,
                            seed=seed,
                            acc=acc,
                            score_s=score_s,
                            fit_s=fit_s,
                            kept=kept,
                            features=kept_names,
                        )
                    )
                print(f"{ds_name} seed={seed} {metric}: done", flush=True)
    return rows


def write(rows):
    csv_path = os.path.join(HERE, OUT_STEM + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["dataset", "k", "metric"])
        .agg(
            acc_mean=("acc", "mean"),
            acc_sd=("acc", "std"),
            score_s=("score_s", "mean"),
            fit_s=("fit_s", "mean"),
            kept=("kept", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# Feature-scoring metrics: results",
        "",
        "Mean ± s.d. over 10 seeds. `score_s` is the time to score all features on the training split; "
        "`fit_s` is the classifier fit on the selected features; `kept` is the mean number of features the "
        "classifier kept after its redundancy filter.",
        "",
    ]
    for ds in agg["dataset"].unique():
        lines += [
            f"## {ds}",
            "",
            "| k | metric | accuracy | score s | fit s | kept |",
            "|---|---|---|---|---|---|",
        ]
        sub = agg[agg["dataset"] == ds].sort_values(
            ["k", "acc_mean"], ascending=[True, False]
        )
        for _, r in sub.iterrows():
            lines.append(
                f"| {int(r['k'])} | {r['metric']} | {r['acc_mean']:.3f} ± {r['acc_sd']:.3f} | "
                f"{r['score_s']:.3f} | {r['fit_s']:.3f} | {r['kept']:.1f} |"
            )
        lines.append("")
    with open(os.path.join(HERE, OUT_STEM + ".md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("wrote", csv_path)


if __name__ == "__main__":
    write(run())
