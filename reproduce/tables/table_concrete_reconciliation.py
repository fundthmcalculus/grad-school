"""The Concrete reconciliation -- the highest-priority housekeeping experiment.

Three different R^2 figures for "the flat model" on UCI Concrete appear in the
proposal, and they are not comparable:

  Chapter 4      0.44 / 0.77 / 0.87   flat MoG-TSK at orders 0 / 1 / 2
  Chapter 6      0.658                flat baseline in the tree/mixture experiment
  Chapter 6      0.88 -> 0.92         antecedent refinement, its own baseline

All three are real measurements of different configurations (split, preprocessing,
consequent order, objective). The worst symptom is that refinement's 0.92 appears
to beat the hierarchical mixture's 0.791, which would make Chapter 6 pointless --
it does not, because they are different setups, but a reader cannot know that from
the tables.

This script fixes it the only way that actually works: run EVERY model on ONE
identical protocol -- same splits, same seeds, same preprocessing -- so the
numbers can be read against each other. It replaces the flat-baseline cells in
both chapters rather than adding a third convention.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_concrete_reconciliation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C            # noqa: E402
import _fuzzy_models as F     # noqa: E402


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def _score(name, y, p, store):
    """Record R2/RMSE for one model on one split, or note the failure."""
    if p is None:
        return
    store.setdefault(name, {"r2": [], "rmse": []})
    store[name]["r2"].append(r2_score(y, p))
    store[name]["rmse"].append(_rmse(y, p))


def build_models(seed):
    """Every model that reports a Concrete number anywhere in the proposal.

    Returns {label: callable(X_tr, y_tr, X_te) -> predictions or None}.
    The MoG orders are what Chapter 4 quotes; the tree/mixture are Chapter 6's.
    """
    import importlib

    models = {}

    # --- Chapter 4: flat MoG-TSK at each consequent order ---
    try:
        gr = importlib.import_module("tribblefis.gaussian_regressor")
        MoG = gr.MixtureOfGaussiansFuzzyRegressor
        for label, order in (("flat MoG-TSK (order 0)", "0th"),
                             ("flat MoG-TSK (order 1)", "1st"),
                             ("flat MoG-TSK (order 2)", "2nd")):
            def mk(order=order):
                def run(Xtr, ytr, Xte):
                    m = MoG(n_output_buckets=3, tsk_order=order,
                            top_n=-1, random_state=seed)
                    return np.asarray(m.fit(Xtr, ytr).predict(Xte))
                return run
            models[label] = mk()
    except Exception as exc:  # noqa: BLE001
        print(f"  [MoG] unavailable ({exc.__class__.__name__})")

    # --- Chapter 6: fuzzy tree and hierarchical mixture ---
    for label, attr in (("fuzzy tree (1st-order leaves)", "FuzzyRegressionTree"),
                        ("mixture of experts (HME)", "HierarchicalFuzzyExpertsRegressor")):
        try:
            ft = importlib.import_module("fuzzytree")
            cls = getattr(ft, attr, None)
            if cls is None:
                print(f"  [{attr}] not found in fuzzytree")
                continue

            def mk(cls=cls):
                def run(Xtr, ytr, Xte):
                    try:
                        m = cls(random_state=seed)
                    except TypeError:
                        m = cls()
                    return np.asarray(m.fit(Xtr, ytr).predict(Xte))
                return run
            models[label] = mk()
        except Exception as exc:  # noqa: BLE001
            print(f"  [{attr}] unavailable ({exc.__class__.__name__})")

    # --- reference baselines, on the same protocol ---
    models["CART (reference)"] = lambda Xtr, ytr, Xte: (
        DecisionTreeRegressor(random_state=seed).fit(Xtr, ytr).predict(Xte))
    models["Random Forest (reference)"] = lambda Xtr, ytr, Xte: (
        RandomForestRegressor(n_estimators=200, random_state=seed).fit(Xtr, ytr).predict(Xte))
    return models


def main():
    print("Concrete reconciliation -- one protocol for every model")
    data = F.load_concrete()
    if data is None:
        print("  [concrete] dataset unavailable; cannot reconcile")
        C.emit("table_concrete_reconciliation",
               "Concrete reconciliation — ONE protocol for all models",
               ["Model", "R²", "RMSE (MPa)"],
               [["(dataset unavailable)", C.NA, C.NA]],
               note="Concrete_Data could not be loaded; see reproduce/tables/README.md.")
        return
    X, y = data
    print(f"  N={len(X)}  M={X.shape[1]}  seeds={C.SEEDS}")

    store: dict = {}
    for seed in C.SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        for label, run in build_models(seed).items():
            try:
                _score(label, yte, run(Xtr, ytr, Xte), store)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{label}] failed on seed {seed}: {exc.__class__.__name__}")

    order = [
        "flat MoG-TSK (order 0)", "flat MoG-TSK (order 1)", "flat MoG-TSK (order 2)",
        "fuzzy tree (1st-order leaves)", "mixture of experts (HME)",
        "CART (reference)", "Random Forest (reference)",
    ]
    rows = []
    for label in order:
        if label not in store:
            rows.append([label, C.NA, C.NA])
            continue
        rows.append([label,
                     C.cell(store[label]["r2"]),
                     C.cell(store[label]["rmse"], fmt="{:.2f}")])

    C.emit("table_concrete_reconciliation",
           "Concrete reconciliation — ONE protocol for all models",
           ["Model", "R²", "RMSE (MPa)"], rows,
           note=("Identical splits, seeds, and preprocessing for every row, which is the "
                 "point: these numbers may be compared with each other, and the figures "
                 "currently in Chapters 4 and 6 may not. 80/20 split, mean ± std across "
                 "seeds. Replace the flat-baseline cells in both chapters with these "
                 "rather than adding a third convention."))


if __name__ == "__main__":
    main()
