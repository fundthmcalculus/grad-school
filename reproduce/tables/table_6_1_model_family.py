"""Table 6.1 -- flat FIS / fuzzy tree / hierarchical mixture vs. baselines.

Regression on UCI Concrete (R2, RMSE); classification on PhiUSIIL (accuracy).
Model family = tribblefis + fuzzytree; baselines = sklearn CART & Random Forest,
plus an optional M5 model tree if `m5py` is installed. Every number is
mean +/- std across `common.SEEDS`.

Run (from repo root):  uv run --project tribble-fis python reproduce/tables/table_6_1_model_family.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))  # reproduce/  -> import common
sys.path.insert(0, _TABLES)  # reproduce/tables -> import _fuzzy_models
import common as C  # noqa: E402
import _fuzzy_models as _fm  # noqa: E402

(M5Prime,) = C.optional_import("m5py", ["M5Prime"])


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def run_regression():
    data = _fm.load_concrete()
    if data is None:
        print("  [concrete] dataset unavailable; regression rows -> N/A")
        na = [C.NA] * 6
        return [["Concrete", "R2", *na], ["Concrete", "RMSE (MPa)", *na]]
    X, y = data
    acc = {
        k: {"r2": [], "rmse": []} for k in ["flat", "tree", "hme", "cart", "rf", "m5"]
    }
    for seed in C.SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        preds = {
            "flat": _fm.fit_predict(_fm.mog_regressor(seed), Xtr, ytr, Xte),
            "tree": _fm.fit_predict(_fm.tree_regressor(seed), Xtr, ytr, Xte),
            "hme": _fm.fit_predict(_fm.hme_regressor(seed), Xtr, ytr, Xte),
            "cart": DecisionTreeRegressor(random_state=seed).fit(Xtr, ytr).predict(Xte),
            "rf": RandomForestRegressor(n_estimators=200, random_state=seed)
            .fit(Xtr, ytr)
            .predict(Xte),
            "m5": (
                M5Prime().fit(np.asarray(Xtr), ytr).predict(np.asarray(Xte))
                if M5Prime
                else None
            ),
        }
        for k, p in preds.items():
            if p is not None:
                acc[k]["r2"].append(r2_score(yte, p))
                acc[k]["rmse"].append(_rmse(yte, p))
    order = ["flat", "tree", "hme", "cart", "rf", "m5"]
    r2_row = ["Concrete", "R2", *[C.cell(acc[k]["r2"]) for k in order]]
    rmse_row = ["Concrete", "RMSE (MPa)", *[C.cell(acc[k]["rmse"]) for k in order]]
    return [r2_row, rmse_row]


def run_classification():
    data = _fm.load_phiusiil()
    if data is None:
        return [["PhiUSIIL", "accuracy", *([C.NA] * 6)]]
    X, y = data
    acc = {k: [] for k in ["flat", "tree", "hme", "cart", "rf", "m5"]}
    for seed in C.SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        preds = {
            "flat": _fm.fit_predict(_fm.mog_classifier(seed), Xtr, ytr, Xte),
            "tree": _fm.fit_predict(_fm.tree_classifier(seed), Xtr, ytr, Xte),
            "hme": _fm.fit_predict(_fm.hme_classifier(seed), Xtr, ytr, Xte),
            "cart": DecisionTreeClassifier(random_state=seed)
            .fit(Xtr, ytr)
            .predict(Xte),
            "rf": RandomForestClassifier(n_estimators=200, random_state=seed)
            .fit(Xtr, ytr)
            .predict(Xte),
            "m5": None,  # M5 is a regression model tree; N/A for classification
        }
        for k, p in preds.items():
            if p is not None:
                acc[k].append(accuracy_score(yte, p))
    order = ["flat", "tree", "hme", "cart", "rf", "m5"]
    return [["PhiUSIIL", "accuracy", *[C.cell(acc[k]) for k in order]]]


def main():
    print("Table 6.1 -- model family vs. baselines")
    rows = run_regression() + run_classification()
    header = [
        "Dataset",
        "metric",
        "flat",
        "fuzzy tree",
        "mixture (HME)",
        "CART",
        "Random Forest",
        "M5",
    ]
    C.emit(
        "table_6_1",
        "Table 6.1 -- Model family on Concrete and PhiUSIIL",
        header,
        rows,
        note="Model-family columns from tribblefis/fuzzytree; baselines from "
        "scikit-learn (M5 via m5py if installed). Higher R2/accuracy and "
        "lower RMSE are better.",
    )


if __name__ == "__main__":
    main()
