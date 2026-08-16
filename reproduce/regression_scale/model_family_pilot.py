#!/usr/bin/env python3
"""Pilot: does Table 6.1's model-family comparison (flat MoG, fuzzy tree, HME,
CART, Random Forest, M5) generalize past Concrete's 1,030 rows? Same models,
same convention as `reproduce/tables/table_6_1_model_family.py` -- RAW
features, no normalize/decorrelate, because that table fits everything on raw
features on purpose so fuzzy-tree splits stay physically meaningful. Same
one-seed-first pattern as `mog_top_p_sweep.py`; see RESULTS_2026-08-05.md.

The "flat" (MoG) row here is the UNTUNED library default (top_p=0.95,
top_n=-1) -- NOT `mog_top_p_sweep.py`'s decorrelated + tuned result. Mixing a
tuned arm into an otherwise-untuned comparison would be exactly the kind of
apples-to-oranges table Table 4.1 was written to avoid.

Run from repo root (needs `tribble-fis` *and* `tribble-tree` importable):
    PILOT_TRIBBLE_FIS=/path/to/tribble-fis \\
        uv run --project tribble-fis python reproduce/regression_scale/model_family_pilot.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import _datasets as D  # noqa: E402

FIS_ROOT = os.environ.get("PILOT_TRIBBLE_FIS", os.path.join(REPO_ROOT, "tribble-fis"))
sys.path.insert(0, os.path.join(FIS_ROOT, "src"))  # tribblefis
sys.path.insert(0, os.path.join(FIS_ROOT, "tribble-tree"))  # fuzzytree

sys.path.insert(0, os.path.join(REPO_ROOT, "reproduce"))
sys.path.insert(0, os.path.join(REPO_ROOT, "reproduce", "tables"))
import common as C  # noqa: E402
import _fuzzy_models as _fm  # noqa: E402

(M5Prime,) = C.optional_import("m5py", ["M5Prime"])

SEED = 0


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def run(name, loader):
    print(f"\n=== {name} ===")
    X, y = loader()
    print(f"  shape: {X.shape}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "flat (MoG, untuned)": lambda: _fm.mog_regressor(SEED),
        "fuzzy tree": lambda: _fm.tree_regressor(SEED),
        "HME": lambda: _fm.hme_regressor(SEED),
        "CART": lambda: DecisionTreeRegressor(random_state=SEED),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, random_state=SEED
        ),
    }
    if M5Prime:
        models["M5"] = lambda: M5Prime()

    for label, factory in models.items():
        t0 = time.perf_counter()
        try:
            model = factory()
            if model is None:
                print(f"  {label:<22} unavailable (factory returned None)")
                continue
            if label == "M5":
                p = model.fit(np.asarray(Xtr), ytr).predict(np.asarray(Xte))
            else:
                p = model.fit(Xtr, ytr).predict(Xte)
            fit_s = time.perf_counter() - t0
            r2 = r2_score(yte, p)
            rmse = _rmse(yte, p)
            print(f"  {label:<22} fit={fit_s:6.2f}s  R2={r2:8.4f}  RMSE={rmse:10.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<22} FAILED: {exc.__class__.__name__}: {exc}")


if __name__ == "__main__":
    print(
        f"tribble-fis commit: {os.popen(f'git -C {FIS_ROOT} rev-parse HEAD').read().strip()}"
    )
    print(f"m5py available: {M5Prime is not None}")
    run("California Housing, raw features, model family", D.load_housing)
    run("Superconductivity, raw features, model family", D.load_superconduct)
