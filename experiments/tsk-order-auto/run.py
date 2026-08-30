"""Does ``tsk_order="auto"`` remove the full-2nd overfitting foot-gun? (issue #120)

A full-2nd TSK consequent fits ``1 + 2*n_features + C(n_features, 2)``
coefficients per rule. Where the training rows comfortably outnumber those
coefficients it buys accuracy; where they do not it overfits, and grad-school
issue #120 measured that cliff on diabetes-scale data. tribble-fis #200 (in the
``ae0ef13`` pin) answered it with ``tsk_order="auto"``: k-fold CV over the
candidate orders at fit time, with the pick exposed as ``tsk_order_``.

This script measures whether ``auto`` actually discharges #120's acceptance
criterion -- *"usable as a default with no negative test R2 on diabetes-scale
data, without hand-applying the rows/coeff check"* -- by running every fixed
order and ``auto`` over the same paired splits.

Run::

    source reproduce/hostenv.sh            # Windows hosts without MSVC
    uv run --project tribble-fis python experiments/tsk-order-auto/run.py

Options come from the environment so the script stays argument-free:

``SEEDS``     comma-separated split seeds (default ``0,...,9``)
``ORDERS``    comma-separated orders (default every candidate, plus ``auto``)
``DATASETS``  comma-separated dataset keys (default all three)

Diabetes ships with scikit-learn and always runs. Concrete and Bike Sharing come
through ``repro_data`` and read from ``data/``; if that file is absent the
dataset is reported as unavailable rather than aborting the run -- the same
convention the ``reproduce/`` generators use for an N/A cell.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import time
import warnings
from collections import Counter

# Cap the thread pools BEFORE numpy is imported. On this project's many-core
# Windows host tribblefis's numba/OpenBLAS stack corrupts the heap at process
# teardown (0xC0000374) when the pools are left at their default size.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
from sklearn.datasets import load_diabetes  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_SIZE = 0.25
DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# Every order tribble-fis's `auto` considers, plus `auto` itself. Keeping the
# fixed orders in the table is the point: `auto` is only interesting relative to
# what a caller would otherwise have had to pick by hand.
DEFAULT_ORDERS = ("0th", "1st", "2nd", "full-2nd", "3rd", "auto")


def _diabetes():
    d = load_diabetes()
    return np.asarray(d.data, float), np.asarray(d.target, float)


def _concrete():
    from repro_data import load_concrete

    got = load_concrete()
    if got is None:
        return None
    X, y = got
    return X.to_numpy(float), np.asarray(y, float)


def _bikeshare():
    from repro_data import load_bikeshare

    # 4,000 rows: the size at which issue #120 measured the bikeshare row, kept
    # so the two tables are comparable.
    got = load_bikeshare(sample_size=4000)
    if got is None:
        return None
    X, y = got
    return np.asarray(X, float), np.asarray(y, float)


DATASETS = {
    "diabetes": ("Diabetes (sklearn)", _diabetes),
    "concrete": ("Concrete", _concrete),
    "bikeshare": ("Bike Sharing (n=4000)", _bikeshare),
}


def _quietly(fn, *args, **kwargs):
    """Run `fn`, swallowing the loaders'/fitters' progress chatter.

    The feature-ranking banner TribbleRegressor prints on every fit would
    otherwise be 180 screens of output for one table.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        # The rows/coeff UserWarning is exactly what this script is measuring the
        # replacement for; firing it 180 times says nothing new.
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _agg(values):
    """(mean, sample std) -- ddof=1, matching reproduce/common.py::agg."""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def run_dataset(key, seeds, orders):
    label, loader = DATASETS[key]
    data = _quietly(loader)
    if data is None:
        print(f"\n== {label}: data/ file absent -- skipped")
        return None
    X, y = data
    n_rows, n_feat = X.shape
    # The coefficient count is the whole reason this issue exists, so it is
    # reported next to the accuracies rather than left for the reader to derive.
    coeffs = 1 + 2 * n_feat + n_feat * (n_feat - 1) // 2
    print(
        f"\n== {label}: {n_rows} rows x {n_feat} features; "
        f"full-2nd = {coeffs} coeffs/rule; rows/coeff = {n_rows / coeffs:.1f}"
    )
    out = {
        "label": label,
        "rows": n_rows,
        "features": n_feat,
        "full_2nd_coeffs": coeffs,
        "rows_per_coeff": n_rows / coeffs,
        "orders": {},
    }
    for order in orders:
        r2s, secs, picks = [], [], []
        for seed in seeds:
            # Paired splits: every order sees the identical train/test partition
            # at a given seed, so the per-seed differences are the model's.
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed
            )

            def _fit():
                model = TribbleRegressor(tsk_order=order, random_state=seed)
                model.fit(Xtr, ytr)
                return model, model.predict(Xte)

            t0 = time.perf_counter()
            model, pred = _quietly(_fit)
            secs.append(time.perf_counter() - t0)
            r2s.append(float(r2_score(yte, pred)))
            picks.append(str(getattr(model, "tsk_order_", order)))
        mean, std = _agg(r2s)
        tally = dict(Counter(picks))
        print(
            f"  {order:<9} R2 = {mean:+.4f} +- {std:.4f}   "
            f"fit {sum(secs) / len(secs):6.2f}s   picks={tally}"
        )
        out["orders"][order] = {
            "r2": r2s,
            "seconds": secs,
            "picks": picks,
            "mean": mean,
            "std": std,
            "mean_seconds": sum(secs) / len(secs),
        }
    return out


def main():
    seeds = [
        int(s)
        for s in os.environ.get("SEEDS", ",".join(str(s) for s in DEFAULT_SEEDS)).split(
            ","
        )
    ]
    orders = os.environ.get("ORDERS", ",".join(DEFAULT_ORDERS)).split(",")
    keys = os.environ.get("DATASETS", ",".join(DATASETS)).split(",")

    print(f"tsk_order study -- seeds={seeds}")
    print(f"orders={orders}")
    results = {}
    for key in keys:
        got = run_dataset(key, seeds, orders)
        if got is not None:
            results[key] = got

    path = os.path.join(HERE, "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"seeds": seeds, "orders": orders, "datasets": results}, fh, indent=1)
    print(f"\nwrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
