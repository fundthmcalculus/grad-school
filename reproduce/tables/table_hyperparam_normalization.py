"""Hyperparameters x normalization: what actually moves the Concrete numbers.

Two confounds were left open by the reconciliation, and they are entangled:

  NORMALIZATION   Does the auto log-transform (of high-dynamic-range features --
                  it selects Age) plus feature standardization help, and does it
                  help every model equally? The flat MoG pipeline uses it; the
                  tree/HME demo deliberately does NOT, to keep split thresholds
                  physically meaningful.

  HYPERPARAMETERS Do the tree and mixture underperform because the hierarchy is
                  genuinely weaker, or because the reconciliation ran them at
                  library defaults rather than the settings
                  `tribble-tree/demo_concrete.py` chose for them?

This script crosses the two so each effect can be read separately. Every cell
shares splits and seeds with every other cell.

Demo settings, taken verbatim from demo_concrete.py rather than invented here:
  tree  FuzzyRegressionTree(tsk_order="1st", criterion="variance", max_depth=3,
                            n_terms=2, top_n=4, min_soft_count=20)
  HME   HierarchicalFuzzyExpertsRegressor(criterion="variance", max_depth=2,
                            n_gate_terms=2, top_n=4, min_soft_count=40,
                            min_expert_samples=60,
                            expert_kwargs={"n_output_buckets":3,"tsk_order":"1st"})
Note the demo passes no random_state, so the "demo" arms are constructed exactly
as written; any seed dependence there comes from the split alone.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_hyperparam_normalization.py

Knobs:
    REPRO_SEEDS="0,1,2"
    REPRO_ORDERS="1st,2nd,full-2nd"    flat-MoG orders to include
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C            # noqa: E402
import _fuzzy_models as F     # noqa: E402
# One definition, shared with table_4_1: a second copy that drifted would make
# the two tables quietly incomparable, which is the failure this harness exists
# to catch.
from _fuzzy_models import normalize  # noqa: E402

ORDERS = [o.strip() for o in os.environ.get("REPRO_ORDERS", "1st,2nd,full-2nd").split(",")]
N_BUCKETS = 3
L2 = 1e-2


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# --------------------------------------------------------------------------- #
# flat MoG, with normalization as a switch
# --------------------------------------------------------------------------- #
def mog(X, y_raw, seed, order, norm):
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation, create_gaussian_membership_dict,
        standard_transform, take_top_features)
    from tribblefis.regression import partition_output, predict_tsk, solve_tsk_consequents

    yt = standard_transform(y_raw)
    y, ybm = partition_output(N_BUCKETS, yt)
    Xt = normalize(X)[0] if norm else X.copy()

    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.2, random_state=seed)
    diffs = calculate_gaussian_correlation(Xtr, ytr["y_bucket"])
    _, top_vars = take_top_features(diffs, top_n=len(Xt.columns))
    memb = create_gaussian_membership_dict(Xtr, ytr["y_bucket"],
                                           top_n_var_names=top_vars, n_gaussians=-1)
    corr, ybm2 = solve_tsk_consequents(Xtr, memb, top_vars, ybm, ytr,
                                       n_output_buckets=N_BUCKETS, order=order,
                                       l2_reg=L2, basis="raw", cross_pairs=None)
    pred = predict_tsk(Xte, memb, top_vars, ybm2, corr,
                       order=order, basis="raw", cross_pairs=None)

    truth = np.asarray(yte["y_value"] if "y_value" in getattr(yte, "columns", []) else yte,
                       dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    span = float(np.asarray(y_raw, float).max() - np.asarray(y_raw, float).min())
    return r2_score(truth, pred), _rmse(truth, pred) * span


# --------------------------------------------------------------------------- #
# tree / HME at default vs demo settings
# --------------------------------------------------------------------------- #
def tree_hme_builders():
    """{(label, setting): factory} -- factories take no args and return an estimator."""
    import importlib
    ft = importlib.import_module("fuzzytree")
    T = getattr(ft, "FuzzyRegressionTree", None)
    H = getattr(ft, "HierarchicalFuzzyExpertsRegressor", None)
    b = {}
    if T is not None:
        b[("fuzzy tree", "library default")] = lambda: T()
        b[("fuzzy tree", "demo-tuned")] = lambda: T(
            tsk_order="1st", criterion="variance", max_depth=3,
            n_terms=2, top_n=4, min_soft_count=20)
    if H is not None:
        b[("mixture of experts", "library default")] = lambda: H()
        b[("mixture of experts", "demo-tuned")] = lambda: H(
            criterion="variance", max_depth=2, n_gate_terms=2, top_n=4,
            min_soft_count=40, min_expert_samples=60,
            expert_kwargs={"n_output_buckets": 3, "tsk_order": "1st"})
    return b


def main():
    print("Hyperparameters x normalization on Concrete")
    data = F.load_concrete()
    if data is None:
        print("  dataset unavailable")
        return
    X, y = data
    logged = normalize(X)[1]
    print(f"  N={len(X)} M={X.shape[1]} seeds={C.SEEDS} log-transformed: {logged}")

    store: dict = {}

    def add(key, r2, rmse):
        store.setdefault(key, {"r2": [], "rmse": []})
        store[key]["r2"].append(r2)
        store[key]["rmse"].append(rmse)

    # flat MoG: both normalization settings
    for norm in (False, True):
        tag = "log+standardized" if norm else "raw"
        for order in ORDERS:
            for seed in C.SEEDS:
                try:
                    r2, rmse = mog(X, y, seed, order, norm)
                    add((f"flat MoG-TSK {order}", "pipeline default", tag), r2, rmse)
                except Exception as exc:  # noqa: BLE001
                    print(f"    [MoG {order}/{tag}] seed {seed}: {exc.__class__.__name__}")
            print(f"  done: MoG {order:<9} {tag}")

    # tree / HME / references: both normalization settings, both hyperparameter settings
    builders = tree_hme_builders()
    for norm in (False, True):
        tag = "log+standardized" if norm else "raw"
        Xu = normalize(X)[0] if norm else X
        for seed in C.SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(Xu, y, test_size=0.2, random_state=seed)
            for (label, setting), make in builders.items():
                try:
                    p = np.asarray(make().fit(Xtr, ytr).predict(Xte), dtype=float).ravel()
                    add((label, setting, tag), r2_score(yte, p), _rmse(yte, p))
                except Exception as exc:  # noqa: BLE001
                    print(f"    [{label}/{setting}/{tag}] seed {seed}: {exc.__class__.__name__}")
            for label, est in (("CART (reference)", DecisionTreeRegressor(random_state=seed)),
                               ("Random Forest (reference)",
                                RandomForestRegressor(n_estimators=200, random_state=seed))):
                p = est.fit(Xtr, ytr).predict(Xte)
                add((label, "sklearn default", tag), r2_score(yte, p), _rmse(yte, p))
        print(f"  done: tree/HME/references {tag}")

    # ---- emit: one row per (model, setting), columns for each normalization ----
    combos = sorted({(k[0], k[1]) for k in store})
    rows = []
    for model, setting in combos:
        raw = store.get((model, setting, "raw"))
        nrm = store.get((model, setting, "log+standardized"))
        r_raw = C.cell(raw["r2"]) if raw else C.NA
        r_nrm = C.cell(nrm["r2"]) if nrm else C.NA
        e_raw = C.cell(raw["rmse"]) if raw else C.NA
        e_nrm = C.cell(nrm["rmse"]) if nrm else C.NA
        delta = C.NA
        if raw and nrm:
            a, _ = C.agg(raw["r2"])
            b, _ = C.agg(nrm["r2"])
            if a is not None and b is not None:
                delta = f"{b - a:+.3f}"
        rows.append([model, setting, r_raw, r_nrm, delta, e_raw, e_nrm])

    C.emit("table_hyperparam_normalization",
           "Concrete — hyperparameters × normalization (R² and RMSE, mean ± std over seeds)",
           ["Model", "Hyperparameters", "raw features", "log + standardized", "Δ from normalizing",
            "RMSE raw (MPa)", "RMSE log+std (MPa)"],
           rows,
           note=("Normalization = auto log-transform of high-dynamic-range features (%s) "
                 "followed by feature standardization, exactly as `concrete.py` applies it. "
                 "Demo-tuned settings are taken verbatim from `tribble-tree/demo_concrete.py` "
                 "(tree: max_depth=3, n_terms=2, top_n=4, min_soft_count=20; HME: max_depth=2, "
                 "n_gate_terms=2, top_n=4, min_soft_count=40, min_expert_samples=60, 1st-order "
                 "experts) — that script passes no random_state, so those arms are built as "
                 "written and vary only with the split. Identical splits and seeds throughout."
                 % (", ".join(map(str, logged)) or "none")))


if __name__ == "__main__":
    main()
