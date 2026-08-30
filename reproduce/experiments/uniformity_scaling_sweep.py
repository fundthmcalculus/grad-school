#!/usr/bin/env python3
"""Uniformity scaling sweep: do uniformity-preserving transforms improve FIS accuracy?

The hypothesis: ``log1p`` is one way to tame skewed features, but it only helps
features whose dynamic range spans decades.  A **uniformity transform** —
mapping each feature's marginal to approximate uniform — should help *any*
non-uniform feature by spreading MFs evenly across the probability mass.

Arms:

  * **raw**           — no feature transform (floor).
  * **log + min-max** — the shipped default (``UnitScalar`` / ``MinMaxScaler``).
  * **quantile**      — ``QuantileUniformScaler`` (sklearn QuantileTransformer
                        + affine to ``feature_range``).
  * **empirical CDF** — ``EmpiricalCDFScaler`` (rank / n, + affine).
  * **PL-CDF k=5**    — ``PiecewiseLinearCDFScaler(n_pieces=5)``.
  * **PL-CDF k=10**   — ``PiecewiseLinearCDFScaler(n_pieces=10)``.
  * **PL-CDF k=20**   — ``PiecewiseLinearCDFScaler(n_pieces=20)``.

Each arm is tested on [0, 1] and [-1, 1] feature ranges so the range and
transform effects can be read independently.

Each cell: flat MoG-TSK 1st order, ten seeds, shared 80/20 splits.

Run (from repo root):

    uv run --project tribble-fis python \
        reproduce/experiments/uniformity_scaling_sweep.py

Knobs:
    REPRO_SEEDS="0,1,2"         quick smoke run
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPRO = os.path.dirname(_HERE)
sys.path.insert(0, _REPRO)
sys.path.insert(0, os.path.join(_REPRO, "tables"))
sys.path.insert(0, _HERE)

import common as C  # noqa: E402
import _fuzzy_models as F  # noqa: E402
from uniformity_transforms import (  # noqa: E402
    EmpiricalCDFScaler,
    PiecewiseLinearCDFScaler,
    QuantileUniformScaler,
)

LOG_DR = 2
N_BUCKETS = 3
TSK_ORDER = "1st"

FEATURE_RANGES = [
    ("[0,1]", (0.0, 1.0)),
    ("[-1,1]", (-1.0, 1.0)),
]


def _make_arms(feature_range):
    """Return [(label, scaler_factory)] for one feature_range setting."""
    from tribblefis.scaling import MinMaxScaler

    fr = feature_range
    return [
        ("raw", lambda _fr=fr: None),
        ("log+minmax", lambda _fr=fr: MinMaxScaler(feature_range=_fr, log_dynamic_range=LOG_DR)),
        ("quantile", lambda _fr=fr: QuantileUniformScaler(feature_range=_fr, n_quantiles=200)),
        ("ecdf", lambda _fr=fr: EmpiricalCDFScaler(feature_range=_fr)),
        ("PL-CDF k=5", lambda _fr=fr: PiecewiseLinearCDFScaler(feature_range=_fr, n_pieces=5)),
        ("PL-CDF k=10", lambda _fr=fr: PiecewiseLinearCDFScaler(feature_range=_fr, n_pieces=10)),
        ("PL-CDF k=20", lambda _fr=fr: PiecewiseLinearCDFScaler(feature_range=_fr, n_pieces=20)),
    ]


def _regressor(seed):
    from tribblefis.gaussian_regressor import TribbleRegressor
    return TribbleRegressor(
        n_output_buckets=N_BUCKETS,
        tsk_order=TSK_ORDER,
        top_n=-1,
        random_state=seed,
    )


REGRESSION_DATASETS = [
    ("Concrete", F.load_concrete),
    ("Body Fat", F.load_bodyfat),
    ("Bike Sharing", lambda: F.load_bikeshare(sample_size=5000)),
    ("WEC Sydney", lambda: F.load_wec(site="Sydney", n_wecs=100)),
]


def run_one(X, y, seed, scaler):
    """Fit a flat MoG-TSK regressor and return test R²."""
    from tribblefis.scaling import MinMaxScaler

    if scaler is not None:
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        scaler.fit(Xtr_raw)
        Xtr = pd.DataFrame(
            scaler.transform(Xtr_raw), index=Xtr_raw.index, columns=X.columns
        )
        Xte = pd.DataFrame(
            scaler.transform(Xte_raw), index=Xte_raw.index, columns=X.columns
        )
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

    y_sc = MinMaxScaler(log_dynamic_range=None)
    ytr_arr = np.asarray(ytr, dtype=float).reshape(-1, 1)
    yte_arr = np.asarray(yte, dtype=float).reshape(-1, 1)
    y_sc.fit(ytr_arr)
    ytr_s = y_sc.transform(ytr_arr).ravel()
    yte_s = y_sc.transform(yte_arr).ravel()

    model = _regressor(seed)
    try:
        model.fit(Xtr, ytr_s)
        p = np.asarray(model.predict(Xte), dtype=float).ravel()
    except Exception as exc:  # noqa: BLE001
        print(f"      fit/predict failed: {exc.__class__.__name__}: {exc}")
        return None
    if not np.all(np.isfinite(p)):
        return None
    return float(r2_score(yte_s, p))


def main() -> int:
    rows = []
    for ds_name, loader in REGRESSION_DATASETS:
        data = loader()
        if data is None:
            print(f"  {ds_name}: unavailable, skipping")
            continue
        X, y = data
        print(f"\n  {ds_name}: N={len(X)}  M={X.shape[1]}")

        for fr_label, feature_range in FEATURE_RANGES:
            arms = _make_arms(feature_range)
            for arm_label, make_scaler in arms:
                r2s = []
                for seed in C.SEEDS:
                    scaler = make_scaler()
                    r2 = run_one(X, y, seed, scaler)
                    if r2 is not None:
                        r2s.append(r2)
                cell = C.cell(r2s) if r2s else C.NA
                rows.append([ds_name, fr_label, arm_label, cell])
                if r2s:
                    print(
                        f"    {fr_label:6s} {arm_label:14s}  "
                        f"R² = {np.mean(r2s):+.3f} ± {np.std(r2s):.3f}"
                    )
                else:
                    print(f"    {fr_label:6s} {arm_label:14s}  FAILED")

    C.emit(
        "uniformity_scaling_sweep",
        title="Uniformity scaling sweep: transform × feature_range × dataset (flat MoG-TSK 1st order)",
        header=["Dataset", "Range", "Transform", "test R²"],
        rows=rows,
        note=(
            f"Seeds = {C.SEEDS}. Flat MoG-TSK 1st order, n_output_buckets={N_BUCKETS}, "
            f"top_n=-1. Target min-max scaled to [0,1] in all arms. 80/20 split, "
            "scaler fit on training fold only (no transduction). "
            "Transforms: raw = no scaling; log+minmax = tribblefis MinMaxScaler "
            f"(log_dynamic_range={LOG_DR}); quantile = sklearn QuantileTransformer "
            "to uniform then affine; ecdf = empirical CDF (rank/n); "
            "PL-CDF k=N = piecewise-linear CDF with N equal-probability affine segments."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
