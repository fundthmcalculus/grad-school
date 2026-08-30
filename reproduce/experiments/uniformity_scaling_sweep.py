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
                        + affine to [0, 1]).
  * **empirical CDF** — ``EmpiricalCDFScaler`` (rank / n, + affine to [0, 1]).

Feature range is fixed at [0, 1] — a separate experiment
(``feature_range_sweep.py``) already showed range makes no difference.

Regression datasets report test R²; classification datasets report test
accuracy.  All cells: flat MoG-TSK / TribbleClassifier, ten seeds, shared
80/20 splits, scaler fit on training fold only (no transduction).

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
from sklearn.metrics import accuracy_score, r2_score
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
    QuantileUniformScaler,
)

LOG_DR = 2
N_BUCKETS = 3
TSK_ORDER = "1st"
FEATURE_RANGE = (0.0, 1.0)


def _make_arms():
    """Return [(label, scaler_factory)] for the fixed feature_range."""
    from tribblefis.scaling import MinMaxScaler

    return [
        ("raw", lambda: None),
        ("log+minmax", lambda: MinMaxScaler(feature_range=FEATURE_RANGE, log_dynamic_range=LOG_DR)),
        ("quantile", lambda: QuantileUniformScaler(feature_range=FEATURE_RANGE, n_quantiles=200)),
        ("ecdf", lambda: EmpiricalCDFScaler(feature_range=FEATURE_RANGE)),
    ]


def _regressor(seed):
    from tribblefis.gaussian_regressor import TribbleRegressor
    return TribbleRegressor(
        n_output_buckets=N_BUCKETS,
        tsk_order=TSK_ORDER,
        top_n=-1,
        random_state=seed,
    )


def _classifier(seed):
    from tribblefis.gaussian_classifier import TribbleClassifier
    return TribbleClassifier(top_n=5, random_state=seed)


# --- Datasets ----------------------------------------------------------------

REGRESSION_DATASETS = [
    ("Concrete", F.load_concrete),
    ("Body Fat", F.load_bodyfat),
    ("Bike Sharing", lambda: F.load_bikeshare(sample_size=5000)),
]

CLASSIFICATION_DATASETS = [
    ("Glass", F.load_glass),
    ("Shuttle", lambda: F.load_shuttle(sample_size=10000)),
    ("PhiUSIIL", lambda: F.load_phiusiil(sample_size=10000)),
]


# --- Runners -----------------------------------------------------------------

def run_regression(X, y, seed, scaler):
    """Fit a flat MoG-TSK regressor and return test R²."""
    from tribblefis.scaling import MinMaxScaler

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    if scaler is not None:
        scaler.fit(Xtr_raw)
        Xtr = pd.DataFrame(
            scaler.transform(Xtr_raw), index=Xtr_raw.index, columns=X.columns
        )
        Xte = pd.DataFrame(
            scaler.transform(Xte_raw), index=Xte_raw.index, columns=X.columns
        )
    else:
        Xtr, Xte = Xtr_raw, Xte_raw

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


def run_classification(X, y, seed, scaler):
    """Fit a TribbleClassifier and return test accuracy."""
    # Drop classes with fewer than 2 members (stratified split requires ≥2)
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    rare = set(classes[counts < 2])
    if rare:
        keep_idx = np.where(~np.isin(y_arr, list(rare)))[0]
        X = X.iloc[keep_idx].reset_index(drop=True) if hasattr(X, 'iloc') else X[keep_idx]
        y = y_arr[keep_idx]

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    if scaler is not None:
        scaler.fit(Xtr_raw)
        Xtr = pd.DataFrame(
            scaler.transform(Xtr_raw), index=Xtr_raw.index, columns=X.columns
        )
        Xte = pd.DataFrame(
            scaler.transform(Xte_raw), index=Xte_raw.index, columns=X.columns
        )
    else:
        Xtr, Xte = Xtr_raw, Xte_raw

    model = _classifier(seed)
    try:
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
    except Exception as exc:  # noqa: BLE001
        print(f"      fit/predict failed: {exc.__class__.__name__}: {exc}")
        return None
    return float(accuracy_score(yte, p))


def sweep_datasets(datasets, runner, metric_name, rows):
    """Run all arms across all datasets for one task type."""
    arms = _make_arms()
    for ds_name, loader in datasets:
        data = loader()
        if data is None:
            print(f"  {ds_name}: unavailable, skipping")
            for label, _ in arms:
                rows.append([ds_name, metric_name, label, C.NA])
            continue
        X, y = data
        print(f"\n  {ds_name}: N={len(X)}  M={X.shape[1]}")

        for arm_label, make_scaler in arms:
            scores = []
            for seed in C.SEEDS:
                scaler = make_scaler()
                score = runner(X, y, seed, scaler)
                if score is not None:
                    scores.append(score)
            cell = C.cell(scores) if scores else C.NA
            rows.append([ds_name, metric_name, arm_label, cell])
            if scores:
                print(
                    f"    {arm_label:14s}  {metric_name} = "
                    f"{np.mean(scores):+.3f} ± {np.std(scores):.3f}"
                )
            else:
                print(f"    {arm_label:14s}  FAILED")


def main() -> int:
    rows = []
    print("=== Regression datasets ===")
    sweep_datasets(REGRESSION_DATASETS, run_regression, "R²", rows)
    print("\n=== Classification datasets ===")
    sweep_datasets(CLASSIFICATION_DATASETS, run_classification, "accuracy", rows)

    C.emit(
        "uniformity_scaling_sweep",
        title="Uniformity scaling sweep: transform × dataset (flat MoG-TSK / TribbleClassifier)",
        header=["Dataset", "Metric", "Transform", "Score"],
        rows=rows,
        note=(
            f"Seeds = {C.SEEDS}. Regression: flat MoG-TSK 1st order, "
            f"n_output_buckets={N_BUCKETS}, top_n=-1, target min-max to [0,1]. "
            "Classification: TribbleClassifier, top_n=5. "
            "All arms: 80/20 split, scaler fit on training fold only. "
            "Feature range fixed at [0,1] (range was shown to make no difference). "
            "Transforms: raw = no scaling; log+minmax = tribblefis MinMaxScaler "
            f"(log_dynamic_range={LOG_DR}); quantile = sklearn QuantileTransformer "
            "to uniform then affine; ecdf = empirical CDF (rank/n)."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
