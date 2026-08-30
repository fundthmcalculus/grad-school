#!/usr/bin/env python3
"""Feature-range sweep: MinMaxScaler [0,1] vs [-1,1] across regression datasets.

E9 tested ``UnitScalar(feature_range=(-1, 1))`` as one diagnostic arm on
Concrete only.  This script sweeps the range across every available regression
dataset so the effect (if any) can be read as a *systematic* finding rather
than a single-dataset anecdote.

Each dataset × range × seed cell fits a flat MoG-TSK regressor at 1st order
(the same pipeline the proposal tables use) and reports test R² on a shared
80/20 split.  The log pre-step uses ``log_dynamic_range=2`` throughout — the
standard the shipped tables use.

Run (from repo root):

    uv run --project tribble-fis python \
        reproduce/experiments/feature_range_sweep.py

Knobs:
    REPRO_SEEDS="0,1,2"         quick smoke run
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPRO = os.path.dirname(_HERE)
sys.path.insert(0, _REPRO)
sys.path.insert(0, os.path.join(_REPRO, "tables"))

import common as C  # noqa: E402
import _fuzzy_models as F  # noqa: E402

LOG_DR = 2
N_BUCKETS = 3
TSK_ORDER = "1st"

RANGES = [
    ("[ 0, 1]", (0.0, 1.0)),
    ("[-1, 1]", (-1.0, 1.0)),
    ("[-0.5, 1.5]", (-0.5, 1.5)),
]

REGRESSION_DATASETS = [
    ("Concrete", F.load_concrete),
    ("Body Fat", F.load_bodyfat),
    ("Bike Sharing", lambda: F.load_bikeshare(sample_size=5000)),
    ("WEC Sydney", lambda: F.load_wec(site="Sydney", n_wecs=100)),
]


def _make_scaler(feature_range):
    from tribblefis.scaling import MinMaxScaler
    return MinMaxScaler(feature_range=feature_range, log_dynamic_range=LOG_DR)


def _regressor(seed):
    from tribblefis.gaussian_regressor import TribbleRegressor
    return TribbleRegressor(
        n_output_buckets=N_BUCKETS,
        tsk_order=TSK_ORDER,
        top_n=-1,
        random_state=seed,
    )


def run_one(X, y, seed, feature_range):
    """Fit a flat MoG-TSK regressor and return test R²."""
    import pandas as pd

    scaler = _make_scaler(feature_range)
    Xt = pd.DataFrame(
        scaler.fit_transform(X.copy()), index=X.index, columns=X.columns
    )
    from tribblefis.scaling import MinMaxScaler as MMS
    y_sc = MMS(log_dynamic_range=None)
    yt = y_sc.fit_transform(np.asarray(y, dtype=float).reshape(-1, 1)).ravel()

    Xtr, Xte, ytr, yte = train_test_split(Xt, yt, test_size=0.2, random_state=seed)
    model = _regressor(seed)
    try:
        model.fit(Xtr, ytr)
        p = np.asarray(model.predict(Xte), dtype=float).ravel()
    except Exception as exc:  # noqa: BLE001
        print(f"    fit/predict failed: {exc.__class__.__name__}")
        return None
    if not np.all(np.isfinite(p)):
        return None
    return float(r2_score(yte, p))


def main() -> int:
    rows = []
    for ds_name, loader in REGRESSION_DATASETS:
        data = loader()
        if data is None:
            print(f"  {ds_name}: unavailable, skipping")
            for label, _ in RANGES:
                rows.append([ds_name, label, C.NA])
            continue

        X, y = data
        print(f"  {ds_name}: N={len(X)}  M={X.shape[1]}")

        for label, frange in RANGES:
            r2s = []
            for seed in C.SEEDS:
                r2 = run_one(X, y, seed, frange)
                if r2 is not None:
                    r2s.append(r2)
            rows.append([ds_name, label, C.cell(r2s) if r2s else C.NA])
            if r2s:
                print(f"    {label}  R² = {np.mean(r2s):+.3f} ± {np.std(r2s):.3f}")
            else:
                print(f"    {label}  FAILED")

    C.emit(
        "feature_range_sweep",
        title="Feature-range sweep: MinMaxScaler [0,1] vs [-1,1] (flat MoG-TSK 1st order)",
        header=["Dataset", "feature_range", "test R²"],
        rows=rows,
        note=(
            f"Seeds = {C.SEEDS}. Flat MoG-TSK 1st order, n_output_buckets={N_BUCKETS}, "
            f"top_n=-1, log_dynamic_range={LOG_DR}. Target min-max scaled to [0,1] "
            "in all arms (the target transform is held constant). 80/20 split. "
            "The question: does shifting the feature domain to [-1,1] systematically "
            "help, hurt, or make no difference?"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
