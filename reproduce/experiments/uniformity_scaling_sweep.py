#!/usr/bin/env python3
"""Uniformity scaling sweep: do uniformity-preserving transforms improve FIS accuracy?

The hypothesis: ``log1p`` is one way to tame skewed features, but it only helps
features whose dynamic range spans decades.  A **uniformity transform** —
mapping each feature's marginal to approximate uniform — should help *any*
non-uniform feature by spreading MFs evenly across the probability mass.

Arms (every scaler is the library's, from ``tribblefis.scaling`` -- grad-school
#249: an experiment-local copy of the same names drifted from what the library
ships, and nothing checked the two agreed):

  * **raw**           — no feature transform (floor).
  * **log+minmax**    — the shipped default (``MinMaxScaler``, log pre-step).
  * **quantile**      — ``QuantileUniformScaler`` (sklearn QuantileTransformer
                        + affine to [0, 1]).
  * **piecewise**     — ``PiecewiseLinearCDFScaler`` (10-piece monotone CDF).
  * **ecdf**          — ``EmpiricalCDFScaler`` (midpoint ECDF, + affine).

Regression arms run at TWO consequent orders, ``1st`` and ``auto``.  #244
(``experiments/tsk-order-auto``) measured ``1st`` 0.21 R² below ``auto`` on
Concrete and 0.11 on Bike Sharing; a transform's gain at ``1st`` alone could be
it compensating for an under-powered consequent, so both are reported and the
reader can see whether the gain survives at the stronger order.

Every regression row also reports the PAIRED difference against log+minmax at
the same seed and order (mean ± sample std of the per-seed deltas): each arm
sees the identical split at a given seed, so the paired ± is the honest
uncertainty on "transform X beats the default", not the two marginal ± added.

Warnings are NOT suppressed wholesale: library ``UserWarning``s (e.g. the
rows-per-coefficient guard that fires on 252-row Body Fat) are counted per
cell and printed once each, so a degenerate arm is visible in the log.

Feature range is fixed at [0, 1] -- ``feature_range_sweep.py`` showed range
makes no difference.

All cells: flat MoG-TSK / TribbleClassifier, ten seeds, shared 80/20 splits,
scaler fit on training fold only (no transduction).  Regression reports test R²
on the min-max-scaled target; classification reports test accuracy.

Run (from repo root):

    uv run --project tribble-fis python \
        reproduce/experiments/uniformity_scaling_sweep.py

Knobs:
    REPRO_SEEDS="0,1,2"         quick smoke run

Results of record: ``reproduce/experiments/UNIFORMITY_SCALING_RESULTS.md``.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPRO = os.path.dirname(_HERE)
sys.path.insert(0, _REPRO)
sys.path.insert(0, os.path.join(_REPRO, "tables"))

import common as C  # noqa: E402
import _fuzzy_models as F  # noqa: E402

# _fuzzy_models installs a blanket warnings.filterwarnings("ignore") at import.
# Undo it: this sweep's Body Fat cells are exactly where the library's
# small-sample diagnostics fire, and #249 asks for them to be visible.
warnings.resetwarnings()
warnings.simplefilter("default")
for _cat in (FutureWarning, DeprecationWarning, PendingDeprecationWarning):
    warnings.simplefilter("ignore", _cat)

from tribblefis.scaling import (  # noqa: E402
    EmpiricalCDFScaler,
    MinMaxScaler,
    PiecewiseLinearCDFScaler,
    QuantileUniformScaler,
)

LOG_DR = 2
N_BUCKETS = 3
TSK_ORDERS = ("1st", "auto")
FEATURE_RANGE = (0.0, 1.0)
BASELINE_ARM = "log+minmax"


def _make_arms():
    """Return [(label, scaler_factory)] for the fixed feature_range."""
    return [
        ("raw", lambda: None),
        (
            BASELINE_ARM,
            lambda: MinMaxScaler(feature_range=FEATURE_RANGE, log_dynamic_range=LOG_DR),
        ),
        (
            "quantile",
            lambda: QuantileUniformScaler(n_quantiles=200, feature_range=FEATURE_RANGE),
        ),
        (
            "piecewise",
            lambda: PiecewiseLinearCDFScaler(n_pieces=10, feature_range=FEATURE_RANGE),
        ),
        ("ecdf", lambda: EmpiricalCDFScaler(feature_range=FEATURE_RANGE)),
    ]


def _regressor(seed, tsk_order):
    from tribblefis.gaussian_regressor import TribbleRegressor

    return TribbleRegressor(
        n_output_buckets=N_BUCKETS,
        tsk_order=tsk_order,
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


def _scale(scaler, Xtr_raw, Xte_raw, columns):
    if scaler is None:
        return Xtr_raw, Xte_raw
    scaler.fit(Xtr_raw)
    Xtr = pd.DataFrame(scaler.transform(Xtr_raw), index=Xtr_raw.index, columns=columns)
    Xte = pd.DataFrame(scaler.transform(Xte_raw), index=Xte_raw.index, columns=columns)
    return Xtr, Xte


def run_regression(X, y, seed, scaler, tsk_order):
    """Fit a flat MoG-TSK regressor; return (test R², resolved order) or None."""
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    Xtr, Xte = _scale(scaler, Xtr_raw, Xte_raw, X.columns)

    y_sc = MinMaxScaler(log_dynamic_range=None)
    ytr_arr = np.asarray(ytr, dtype=float).reshape(-1, 1)
    yte_arr = np.asarray(yte, dtype=float).reshape(-1, 1)
    y_sc.fit(ytr_arr)
    ytr_s = y_sc.transform(ytr_arr).ravel()
    yte_s = y_sc.transform(yte_arr).ravel()

    model = _regressor(seed, tsk_order)
    try:
        model.fit(Xtr, ytr_s)
        p = np.asarray(model.predict(Xte), dtype=float).ravel()
    except Exception as exc:  # noqa: BLE001
        print(f"      fit/predict failed: {exc.__class__.__name__}: {exc}")
        return None
    if not np.all(np.isfinite(p)):
        return None
    return float(r2_score(yte_s, p)), getattr(model, "tsk_order_", tsk_order)


def run_classification(X, y, seed, scaler):
    """Fit a TribbleClassifier; return (test accuracy, None) or None."""
    # Drop classes with fewer than 2 members (stratified split requires ≥2)
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    rare = set(classes[counts < 2])
    if rare:
        keep_idx = np.where(~np.isin(y_arr, list(rare)))[0]
        X = (
            X.iloc[keep_idx].reset_index(drop=True)
            if hasattr(X, "iloc")
            else X[keep_idx]
        )
        y = y_arr[keep_idx]

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    Xtr, Xte = _scale(scaler, Xtr_raw, Xte_raw, X.columns)

    model = _classifier(seed)
    try:
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
    except Exception as exc:  # noqa: BLE001
        print(f"      fit/predict failed: {exc.__class__.__name__}: {exc}")
        return None
    return float(accuracy_score(yte, p)), None


def _run_cell(runner, X, y, make_scaler, **kw):
    """Run one arm over every seed; return ({seed: score}, Counter(orders), n_warn)."""
    scores, orders, msgs = {}, Counter(), Counter()
    for seed in C.SEEDS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            warnings.simplefilter("always", RuntimeWarning)
            out = runner(X, y, seed, make_scaler(), **kw)
        for w in caught:
            if issubclass(w.category, (UserWarning, RuntimeWarning)):
                msgs[f"{w.category.__name__}: {str(w.message).splitlines()[0]}"] += 1
        if out is not None:
            scores[seed], order = out
            if order is not None:
                orders[order] += 1
    for m, n in msgs.items():
        print(f"      [{n}x] {m[:160]}")
    return scores, orders, sum(msgs.values())


def _fmt_delta(deltas):
    """Paired delta as '+0.032 ± 0.021' (sign on the mean only)."""
    m, sd = C.agg(deltas)
    if m is None:
        return C.NA
    return f"{m:+.3f} ± {sd:.3f}" if sd else f"{m:+.3f}"


def run_linear_reference(X, y, seed):
    """OLS on raw features, same split and target scaling: a ceiling check.

    Not an arm of the sweep. It is here so a reader can tell a real transform
    gain from a degenerate baseline: if log+minmax sits far below plain OLS on
    the same split, the gap a transform "closes" is the baseline's defect.
    """
    from sklearn.linear_model import LinearRegression

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    lo, hi = float(np.min(ytr)), float(np.max(ytr))
    ytr_s = (np.asarray(ytr, dtype=float) - lo) / (hi - lo)
    yte_s = (np.asarray(yte, dtype=float) - lo) / (hi - lo)
    p = LinearRegression().fit(Xtr, ytr_s).predict(Xte)
    return float(r2_score(yte_s, p))


def _fmt_orders(orders):
    return ", ".join(f"{o}×{n}" for o, n in sorted(orders.items())) or "—"


def sweep_regression(rows, perseed):
    arms = _make_arms()
    for ds_name, loader in REGRESSION_DATASETS:
        data = loader()
        if data is None:
            print(f"  {ds_name}: unavailable, skipping")
            for order in TSK_ORDERS:
                for label, _ in arms:
                    rows.append([ds_name, order, label, C.NA, C.NA, "—", C.NA])
            continue
        X, y = data
        n_test = len(X) - len(train_test_split(X, test_size=0.2, random_state=0)[0])
        print(f"\n  {ds_name}: N={len(X)}  M={X.shape[1]}  test rows={n_test}")
        lin = [run_linear_reference(X, y, s) for s in C.SEEDS]
        lin_m, lin_sd = C.agg(lin)
        print(f"   OLS reference (raw features): R² = {lin_m:+.3f} ± {lin_sd:.3f}")
        rows.append([ds_name, "—", "OLS reference", C.cell(lin), "—", "—", 0])
        for order in TSK_ORDERS:
            print(f"   tsk_order={order}")
            cells = {}
            for label, make in arms:
                cells[label] = _run_cell(run_regression, X, y, make, tsk_order=order)
            base = cells[BASELINE_ARM][0]
            for label, _ in arms:
                scores, orders, n_warn = cells[label]
                vals = list(scores.values())
                common = [s for s in scores if s in base]
                deltas = [scores[s] - base[s] for s in common]
                delta = "—" if label == BASELINE_ARM else _fmt_delta(deltas)
                rows.append(
                    [
                        ds_name,
                        order,
                        label,
                        C.cell(vals) if vals else C.NA,
                        delta,
                        _fmt_orders(orders),
                        n_warn,
                    ]
                )
                for s, v in scores.items():
                    perseed.append([ds_name, order, label, s, f"{v:.6f}"])
                if vals:
                    m, sd = C.agg(vals)
                    print(
                        f"    {label:11s} R² = {m:+.3f} ± {sd:.3f}   "
                        f"Δ = {delta:>16s}   orders: {_fmt_orders(orders)}"
                    )
                else:
                    print(f"    {label:11s} FAILED")


def sweep_classification(rows):
    arms = _make_arms()
    for ds_name, loader in CLASSIFICATION_DATASETS:
        data = loader()
        if data is None:
            print(f"  {ds_name}: unavailable, skipping")
            for label, _ in arms:
                rows.append([ds_name, label, C.NA, C.NA])
            continue
        X, y = data
        print(f"\n  {ds_name}: N={len(X)}  M={X.shape[1]}")
        cells = {
            label: _run_cell(run_classification, X, y, make) for label, make in arms
        }
        base = cells[BASELINE_ARM][0]
        for label, _ in arms:
            scores = cells[label][0]
            vals = list(scores.values())
            deltas = [scores[s] - base[s] for s in scores if s in base]
            delta = "—" if label == BASELINE_ARM else _fmt_delta(deltas)
            rows.append([ds_name, label, C.cell(vals) if vals else C.NA, delta])
            if vals:
                m, sd = C.agg(vals)
                print(f"    {label:11s} acc = {m:.3f} ± {sd:.3f}   Δ = {delta}")
            else:
                print(f"    {label:11s} FAILED")


def main() -> int:
    reg_rows, perseed, cls_rows = [], [], []
    print("=== Regression datasets ===")
    sweep_regression(reg_rows, perseed)
    print("\n=== Classification datasets ===")
    sweep_classification(cls_rows)

    common_note = (
        "All arms: 80/20 split, scaler fit on training fold only, feature_range "
        "[0,1]. Every scaler is tribblefis.scaling's: log+minmax = MinMaxScaler "
        f"(log_dynamic_range={LOG_DR}); quantile = QuantileUniformScaler "
        "(n_quantiles=200); piecewise = PiecewiseLinearCDFScaler (n_pieces=10); "
        "ecdf = EmpiricalCDFScaler. Δ = paired per-seed difference vs log+minmax "
        "(mean ± sample std of the deltas)."
    )
    C.emit(
        "uniformity_scaling_sweep",
        title="Uniformity scaling sweep, regression: transform × order × dataset "
        "(flat MoG-TSK)",
        header=[
            "Dataset",
            "tsk_order",
            "Transform",
            "test R²",
            "Δ vs log+minmax",
            "resolved order",
            "library warnings",
        ],
        rows=reg_rows,
        note=(
            f"Regression: flat MoG-TSK, n_output_buckets={N_BUCKETS}, top_n=-1, "
            "target min-max to [0,1]; tsk_order in {1st, auto}, 'resolved order' "
            "is what auto picked per seed. 'OLS reference' is sklearn "
            "LinearRegression on raw features, same splits -- not an arm, a check "
            "on whether the log+minmax baseline is degenerate. 'library warnings' counts UserWarning/"
            "RuntimeWarning raised across all seeds of the cell. " + common_note
        ),
    )
    C.emit(
        "uniformity_scaling_sweep_classification",
        title="Uniformity scaling sweep, classification: transform × dataset "
        "(TribbleClassifier)",
        header=["Dataset", "Transform", "test accuracy", "Δ vs log+minmax"],
        rows=cls_rows,
        note="Classification: TribbleClassifier, top_n=5, stratified split. "
        + common_note,
    )
    C.emit(
        "uniformity_scaling_sweep_perseed",
        title="Uniformity scaling sweep, regression: per-seed test R²",
        header=["Dataset", "tsk_order", "Transform", "seed", "test R²"],
        rows=perseed,
        note="Raw per-seed values behind uniformity_scaling_sweep.md.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
