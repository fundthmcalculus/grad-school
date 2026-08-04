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
import pandas as pd
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
# `REPRO_LEAKY=1` restores the pre-2026-08-04 behaviour, where the target scale, the
# output partition and the feature scaler were all fit on all 1,030 rows before the
# split. Kept so an archive taken before that date can be reproduced, and it announces
# itself in the log because a run that quietly used it would look identical in the table.
LEAKY = os.environ.get("REPRO_LEAKY", "").strip() not in ("", "0", "false")


def _split_first(X, y_raw, seed, scaler):
    """Split first, then fit every data-dependent transform on the training fold.

    Returns `(Xtr, Xte, ytr, yte, y_bucket_mean)` in the shape `mog` consumes.

    The test rows keep their scaled target value and are given bucket labels derived from
    the training fold's edges. Those labels are cosmetic here -- `mog` reads only
    `yte["y_value"]`, for scoring -- but deriving them from the train edges is the honest
    way to fill the column, and it keeps the frame shape identical to the leaky path.

    `span`, used to put RMSE back on the MPa scale, still comes from the full raw target.
    It is a unit conversion applied after scoring, not a model input, and holding it fixed
    keeps the RMSE columns comparable across arms and across both code paths.
    """
    from tribblefis.regression import partition_output

    idx = np.arange(len(X))
    itr, _ite = train_test_split(idx, test_size=0.2, random_state=seed)
    train = np.zeros(len(X), dtype=bool)
    train[itr] = True

    yr = pd.Series(np.asarray(y_raw, dtype=float).ravel(), index=X.index,
                   name=getattr(y_raw, "name", "y_value"))
    lo, hi = float(yr[train].min()), float(yr[train].max())
    yt = F.unit_scale_with(lo, hi, yr)

    y_train_part, ybm = partition_output(N_BUCKETS, yt[train])
    buckets = pd.Series(np.nan, index=yt.index, name="y_bucket")
    buckets[train] = y_train_part["y_bucket"].values
    # Uniform cuts: equal-width edges over the TRAINING fold's observed range.
    edges = np.linspace(float(yt[train].min()), float(yt[train].max()), N_BUCKETS + 1)[1:-1]
    buckets[~train] = np.digitize(yt[~train].values, edges)
    y = pd.concat([buckets.astype(int), yt.rename("y_value")], axis=1)

    if scaler:
        sc, _logged = F.fit_scaler(X[train], scaler=scaler, log_dynamic_range=2)
        Xt = F.apply_scaler(sc, X)
    else:
        Xt = X.copy()

    return Xt[train], Xt[~train], y[train], y[~train], ybm


def mog(X, y_raw, seed, order, scaler):
    """One flat MoG-TSK measurement. `scaler` is None (raw), "unit", or "standard".

    The TARGET transform is min-max in every arm and deliberately does not vary:
    the axis under study is the feature transform, and `partition_output` plus the
    pinned extreme bucket means both assume a target on [0,1] (see Ch 4 §4.3's
    pin_extremes discussion). Moving the target scaling too would confound the
    one variable this table exists to isolate.

    Leak-free since 2026-08-04: the target scale, the output partition and the feature
    scaler are all fit on the training fold. Every one of them used to be fit on all
    1,030 rows before the split, so each arm saw a transform derived partly from the rows
    it was then scored on -- and the output partition's per-bucket means reach the
    prediction path through `solve_tsk_consequents`, which makes that half a real leak
    rather than merely transductive. `REPRO_LEAKY=1` restores the old path.
    """
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation, create_gaussian_membership_dict,
        take_top_features)
    from tribblefis.regression import partition_output, predict_tsk, solve_tsk_consequents

    if LEAKY:
        yt = F.unit_scale(y_raw)
        y, ybm = partition_output(N_BUCKETS, yt)
        Xt = normalize(X, scaler=scaler)[0] if scaler else X.copy()
        Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.2, random_state=seed)
    else:
        Xtr, Xte, ytr, yte, ybm = _split_first(X, y_raw, seed, scaler)

    diffs = calculate_gaussian_correlation(Xtr, ytr["y_bucket"])
    _, top_vars = take_top_features(diffs, top_n=len(Xtr.columns))
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
    # Leak-free since 2026-08-04 here too: the feature scaler is fit per training fold.
    # It used to be fit once on the full frame outside the seed loop. That half is milder
    # than the MoG path's -- no target information is involved, and for CART, Random Forest
    # and the fuzzy trees a monotone per-feature map is rank-invariant, which is the control
    # this table's own argument rests on -- but "milder" is not "absent", and leaving one
    # arm transductive while fixing another is how a table stops being internally
    # comparable. The hoist out of the seed loop goes with it; that costs one scaler fit
    # per seed on 1,030 rows.
    builders = tree_hme_builders()
    for tag, scaler in ARMS:
        Xu = normalize(X, scaler=scaler)[0] if (scaler and LEAKY) else X
        for seed in C.SEEDS:
            if LEAKY or not scaler:
                Xtr, Xte, ytr, yte = train_test_split(Xu, y, test_size=0.2, random_state=seed)
            else:
                idx = np.arange(len(X))
                itr, _ite = train_test_split(idx, test_size=0.2, random_state=seed)
                tr = np.zeros(len(X), dtype=bool); tr[itr] = True
                sc, _lg = F.fit_scaler(X[tr], scaler=scaler, log_dynamic_range=2)
                Xs = F.apply_scaler(sc, X)
                Xtr, Xte, ytr, yte = Xs[tr], Xs[~tr], y[tr], y[~tr]
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
