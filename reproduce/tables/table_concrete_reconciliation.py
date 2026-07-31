"""The Concrete reconciliation -- one protocol, with the real pipeline.

Three different R^2 figures for "the flat model" on UCI Concrete appear in the
proposal and are not comparable:

  Chapter 4      0.44 / 0.77 / 0.87   flat MoG-TSK at orders 0 / 1 / 2
  Chapter 6      0.658                flat baseline in the tree/mixture experiment
  Chapter 6      0.88 -> 0.92         antecedent refinement, its own baseline

A first pass at reconciling these used the sklearn-style wrapper on raw features
and reproduced none of them, which was informative but not conclusive: it left
open whether the gap was preprocessing, refinement, or something else. This
version settles that by replicating `gaussian_mixture/concrete.py` -- the actual
pipeline behind Chapter 4's numbers -- and making the two candidate explanations
into separable arms:

  PREPROCESSING   raw features           (what the tree/HME demo uses, on purpose,
                                          so split thresholds stay physical)
                  log + standardized     (what concrete.py uses: auto log-transform
                                          of high-dynamic-range features, then
                                          standardization, with a quantile-
                                          partitioned standardized target)

  REFINEMENT      off                    closed-form consequents only
                  on                     per-order antecedent refinement via
                                          `refine_antecedents_coordinate`
                                          (concrete.py's default -- the "optimal
                                          setup")

Every arm shares splits and seeds, so the numbers may be read against each other.
The tree, mixture, and sklearn references are run under both preprocessing
regimes for the same reason.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_concrete_reconciliation.py

Knobs:
    REPRO_SEEDS="0,1,2,3,4"    seeds
    REPRO_ORDERS="0th,1st,2nd" which TSK orders to sweep (add full-2nd,3rd for the full set)
    REPRO_REFINE="both"        off | on | both
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

ORDERS = [o.strip() for o in os.environ.get("REPRO_ORDERS", "0th,1st,2nd").split(",")]
REFINE_MODE = os.environ.get("REPRO_REFINE", "both").lower()
N_BUCKETS = 3
REFINE_L2 = 1e-2


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# --------------------------------------------------------------------------- #
# the concrete.py pipeline, faithfully
# --------------------------------------------------------------------------- #
def mog_arm(X, y_raw, seed, order, refine):
    """One flat MoG-TSK measurement, replicating gaussian_mixture/concrete.py.

    Returns (r2, rmse) or None. The transform sequence, bucket count, ridge
    strength, basis, and refinement call are all taken from that script rather
    than chosen here.
    """
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        detect_and_apply_log_transform,
        standard_transform,
        take_top_features,
    )
    from tribblefis.regression import partition_output, predict_tsk, solve_tsk_consequents

    yt = standard_transform(y_raw)                      # affine: R^2-invariant
    y, y_bucket_mean = partition_output(N_BUCKETS, yt)

    Xt, logged = detect_and_apply_log_transform(X.copy(), min_dynamic_range=2)
    Xt = standard_transform(Xt, column=Xt.columns)

    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.2, random_state=seed)

    diffs = calculate_gaussian_correlation(Xtr, ytr["y_bucket"])
    _, top_vars = take_top_features(diffs, top_n=len(Xt.columns))
    memb = create_gaussian_membership_dict(Xtr, ytr["y_bucket"],
                                           top_n_var_names=top_vars, n_gaussians=-1)

    model = memb
    if refine:
        from tribblefis.refine import refine_antecedents_coordinate
        model, _ = refine_antecedents_coordinate(
            memb, Xtr, ytr, top_vars, n_output_buckets=N_BUCKETS, order=order,
            l2_reg=REFINE_L2, basis="raw", cross_pairs=None)

    corr, ybm = solve_tsk_consequents(
        Xtr, model, top_vars, y_bucket_mean, ytr,
        n_output_buckets=N_BUCKETS, order=order, l2_reg=REFINE_L2,
        basis="raw", cross_pairs=None)
    pred = predict_tsk(Xte, model, top_vars, ybm, corr,
                       order=order, basis="raw", cross_pairs=None)

    yt_true = yte["y_value"] if hasattr(yte, "columns") and "y_value" in yte else yte
    yt_true = np.asarray(yt_true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()

    # The MoG pipeline scores on the TRANSFORMED target. R^2 is invariant under
    # that affine map, but RMSE is not -- reporting it as-is would put this arm on
    # a different scale from every other row. Rescale to the original MPa span so
    # the RMSE column means one thing throughout.
    span = float(np.asarray(y_raw, dtype=float).max() - np.asarray(y_raw, dtype=float).min())
    return r2_score(yt_true, pred), _rmse(yt_true, pred) * span, logged


def preprocess_for_others(X, y, seed, style):
    """Split for the tree/mixture/sklearn arms under a chosen preprocessing style."""
    if style == "raw":
        return train_test_split(X, y, test_size=0.2, random_state=seed)
    from tribblefis.gauss_math import detect_and_apply_log_transform, standard_transform
    Xt, _ = detect_and_apply_log_transform(X.copy(), min_dynamic_range=2)
    Xt = standard_transform(Xt, column=Xt.columns)
    return train_test_split(Xt, y, test_size=0.2, random_state=seed)


def other_arms(seed):
    """Tree, mixture, and sklearn references -- all take (Xtr,ytr,Xte)->pred."""
    import importlib
    out = {}
    for label, attr in (("fuzzy tree", "FuzzyRegressionTree"),
                        ("mixture of experts (HME)", "HierarchicalFuzzyExpertsRegressor")):
        try:
            cls = getattr(importlib.import_module("fuzzytree"), attr, None)
            if cls is None:
                continue

            def mk(cls=cls):
                def run(Xtr, ytr, Xte):
                    try:
                        m = cls(random_state=seed)
                    except TypeError:
                        m = cls()
                    return np.asarray(m.fit(Xtr, ytr).predict(Xte))
                return run
            out[label] = mk()
        except Exception as exc:  # noqa: BLE001
            print(f"    [{attr}] unavailable ({exc.__class__.__name__})")
    out["CART (reference)"] = lambda a, b, c: DecisionTreeRegressor(
        random_state=seed).fit(a, b).predict(c)
    out["Random Forest (reference)"] = lambda a, b, c: RandomForestRegressor(
        n_estimators=200, random_state=seed).fit(a, b).predict(c)
    return out


def main():
    print("Concrete reconciliation -- replicating gaussian_mixture/concrete.py")
    data = F.load_concrete()
    if data is None:
        C.emit("table_concrete_reconciliation",
               "Concrete reconciliation — ONE protocol",
               ["Arm", "Preprocessing", "Refinement", "R²", "RMSE"],
               [["(dataset unavailable)", C.NA, C.NA, C.NA, C.NA]])
        return
    X, y = data
    print(f"  N={len(X)}  M={X.shape[1]}  seeds={C.SEEDS}  orders={ORDERS}")

    refine_flags = {"off": [False], "on": [True], "both": [False, True]}[REFINE_MODE]
    store: dict = {}
    logged_note = ""

    # --- flat MoG arms, concrete.py pipeline ---
    for refine in refine_flags:
        for order in ORDERS:
            key = (f"flat MoG-TSK {order}", "log+standardized",
                   "refined" if refine else "closed-form only")
            for seed in C.SEEDS:
                try:
                    r2, rmse, logged = mog_arm(X, y, seed, order, refine)
                    store.setdefault(key, {"r2": [], "rmse": []})
                    store[key]["r2"].append(r2)
                    store[key]["rmse"].append(rmse)
                    if logged and not logged_note:
                        logged_note = ", ".join(map(str, logged))
                except Exception as exc:  # noqa: BLE001
                    print(f"    [{key[0]} {key[2]}] seed {seed} failed: "
                          f"{exc.__class__.__name__}: {exc}")
            print(f"  done: {key[0]:<22} {key[2]}")

    # --- tree / mixture / references, under BOTH preprocessing styles ---
    for style, style_label in (("raw", "raw"), ("transformed", "log+standardized")):
        for seed in C.SEEDS:
            Xtr, Xte, ytr, yte = preprocess_for_others(X, y, seed, style)
            for label, run in other_arms(seed).items():
                key = (label, style_label, "n/a")
                try:
                    p = np.asarray(run(Xtr, ytr, Xte), dtype=float).ravel()
                    store.setdefault(key, {"r2": [], "rmse": []})
                    store[key]["r2"].append(r2_score(yte, p))
                    store[key]["rmse"].append(_rmse(yte, p))
                except Exception as exc:  # noqa: BLE001
                    print(f"    [{label}/{style}] seed {seed} failed: {exc.__class__.__name__}")
        print(f"  done: other arms under {style_label}")

    rows = [[k[0], k[1], k[2], C.cell(v["r2"]), C.cell(v["rmse"], fmt="{:.2f}")]
            for k, v in store.items()]

    C.emit("table_concrete_reconciliation",
           "Concrete reconciliation — one protocol, preprocessing and refinement separated",
           ["Model", "Preprocessing", "Refinement", "R²", "RMSE"], rows,
           note=("Flat MoG arms replicate `gaussian_mixture/concrete.py`: standardized "
                 "target, quantile output partition (%d buckets), auto log-transform of "
                 "high-dynamic-range features%s, feature standardization, closed-form ridge "
                 "consequents (basis=raw, l2=%g), and — where marked — per-order antecedent "
                 "refinement via `refine_antecedents_coordinate`. Tree/mixture/reference arms "
                 "are run under BOTH preprocessing styles, because the tree demo deliberately "
                 "uses raw features to keep split thresholds physically meaningful. Identical "
                 "splits and seeds throughout; mean ± std across seeds."
                 % (N_BUCKETS, f" ({logged_note})" if logged_note else "", REFINE_L2)))


if __name__ == "__main__":
    main()
