"""Hyperparameters x normalization: what actually moves the Concrete numbers.

Two confounds were left open by the reconciliation, and they are entangled:

  NORMALIZATION   Does the auto log-transform (of high-dynamic-range features --
                  it selects Slag and Age) plus feature scaling help, and does it
                  help every model equally? The flat MoG pipeline uses it; the
                  tree/HME demo deliberately does NOT, to keep split thresholds
                  physically meaningful. Three levels: raw, log + min-max to
                  [0,1] (`UnitScalar`), and log + z-score (`StandardScalar`).
                  The middle one is what every run before `tribble-fis` a385a1a
                  reported as "log+std" -- see the ARMS comment below.

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

# The normalization axis, now with all three levels.
#
# Until `tribble-fis` a385a1a there were only two, and the second was mislabelled:
# the transform behind every "log+std" number this repository has ever emitted was
# `gauss_math.standard_transform`, which min-max scaled to [0,1] despite its name.
# It never z-scored, so **log + z-score had never been measured at all**. Upstream
# renamed the two honestly (`UnitScalar` / `StandardScalar`), which makes the third
# level cheap to add and impossible to keep conflating.
#
# `"log + min-max"` is therefore not a new arm -- it is the old "log + standardized"
# column under its true name, and its numbers are unchanged (verified byte-identical
# against `outputs/full-14900hx-r2/`). `"log + z-score"` is the new measurement.
#
# Both transforms are strictly monotone per feature, so CART and Random Forest --
# which split on rank -- must be invariant across all three arms. That is this
# table's CONTROL, not a side observation: if a reference row moves under the
# z-score arm, the plumbing is wrong, because the world cannot have changed.
ARMS = (("raw", None), ("log + min-max", "unit"), ("log + z-score", "standard"))
RAW, MINMAX, ZSCORE = (a[0] for a in ARMS)


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# --------------------------------------------------------------------------- #
# flat MoG, with normalization as a switch
# --------------------------------------------------------------------------- #
def mog(X, y_raw, seed, order, scaler):
    """One flat MoG-TSK measurement. `scaler` is None (raw), "unit", or "standard".

    The TARGET transform is min-max in every arm and deliberately does not vary:
    the axis under study is the feature transform, and `partition_output` plus the
    pinned extreme bucket means both assume a target on [0,1] (see Ch 4 §4.3's
    pin_extremes discussion). Moving the target scaling too would confound the
    one variable this table exists to isolate.
    """
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation, create_gaussian_membership_dict,
        take_top_features)
    from tribblefis.regression import partition_output, predict_tsk, solve_tsk_consequents

    yt = F.unit_scale(y_raw)
    y, ybm = partition_output(N_BUCKETS, yt)
    Xt = normalize(X, scaler=scaler)[0] if scaler else X.copy()

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

    # flat MoG: all three normalization settings
    for tag, scaler in ARMS:
        for order in ORDERS:
            for seed in C.SEEDS:
                try:
                    r2, rmse = mog(X, y, seed, order, scaler)
                    add((f"flat MoG-TSK {order}", "pipeline default", tag), r2, rmse)
                except Exception as exc:  # noqa: BLE001
                    print(f"    [MoG {order}/{tag}] seed {seed}: {exc.__class__.__name__}")
            print(f"  done: MoG {order:<9} {tag}")

    # tree / HME / references: all three normalizations, both hyperparameter settings
    builders = tree_hme_builders()
    for tag, scaler in ARMS:
        Xu = normalize(X, scaler=scaler)[0] if scaler else X
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
    def _mean(bucket):
        if not bucket:
            return None
        m, _ = C.agg(bucket["r2"])
        return m

    def _delta(a, b):
        """b - a, or N/A if either side is missing."""
        return f"{b - a:+.3f}" if (a is not None and b is not None) else C.NA

    combos = sorted({(k[0], k[1]) for k in store})
    rows = []
    for model, setting in combos:
        got = {tag: store.get((model, setting, tag)) for tag, _ in ARMS}
        r2c = {tag: (C.cell(got[tag]["r2"]) if got[tag] else C.NA) for tag, _ in ARMS}
        rmc = {tag: (C.cell(got[tag]["rmse"]) if got[tag] else C.NA) for tag, _ in ARMS}
        m = {tag: _mean(got[tag]) for tag, _ in ARMS}
        rows.append([model, setting,
                     r2c[RAW], r2c[MINMAX], r2c[ZSCORE],
                     _delta(m[RAW], m[MINMAX]),      # Δ from log+min-max
                     _delta(m[RAW], m[ZSCORE]),      # Δ from log+z-score
                     _delta(m[MINMAX], m[ZSCORE]),   # z-score vs min-max: which normalization
                     rmc[RAW], rmc[MINMAX], rmc[ZSCORE]])

    C.emit("table_hyperparam_normalization",
           "Concrete — hyperparameters × normalization, three arms "
           "(R² and RMSE, mean ± std over seeds)",
           ["Model", "Hyperparameters",
            "raw features", "log + min-max", "log + z-score",
            "Δ min-max − raw", "Δ z-score − raw", "Δ z-score − min-max",
            "RMSE raw (MPa)", "RMSE log+min-max (MPa)", "RMSE log+z-score (MPa)"],
           rows,
           note=("**The normalization axis now has three levels, and the middle one has been "
                 "renamed rather than changed.** Every prior run of this table labelled its "
                 "second column *log + standardized*; the transform behind it was "
                 "`gauss_math.standard_transform`, which min-max scaled to [0,1] **despite its "
                 "name** — it never z-scored. That column is now labelled **log + min-max** and "
                 "its numbers are unchanged (verified byte-identical against "
                 "`outputs/full-14900hx-r2/`). **log + z-score** is a genuinely new "
                 "measurement: `tribblefis.scaling.StandardScalar` (μ=0, σ=1), which had never "
                 "been run against this pipeline. Both arms log-transform the same "
                 "high-dynamic-range features first (%s), detected at "
                 "`log_dynamic_range=2`.\n\n"
                 "**CART and Random Forest are the control.** Both split on rank, and both "
                 "transforms are strictly monotone per feature, so both reference rows must "
                 "read ≈0.000 in the *Δ z-score − min-max* column. A reference row that moves "
                 "there indicts the plumbing, not the transform.\n\n"
                 "The **target** is min-max scaled in all three arms and deliberately does not "
                 "vary — `partition_output` and the pinned extreme bucket means both assume a "
                 "target on [0,1], so varying it too would confound the feature transform this "
                 "table isolates. Demo-tuned settings are taken verbatim from "
                 "`tribble-tree/demo_concrete.py` (tree: max_depth=3, n_terms=2, top_n=4, "
                 "min_soft_count=20; HME: max_depth=2, n_gate_terms=2, top_n=4, "
                 "min_soft_count=40, min_expert_samples=60, 1st-order experts) — that script "
                 "passes no random_state, so those arms are built as written and vary only with "
                 "the split. Identical splits and seeds throughout."
                 % (", ".join(map(str, logged)) or "none")))


if __name__ == "__main__":
    main()
