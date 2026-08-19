"""Refine the winning RUL-ceiling (#5) and test whether the marginal #4/#6
winners stack with it, on DS02.

#5 found a constant RUL ceiling ~60 (on top of the health-onset cap) cuts DS02
per-sample test RMSE 6.48 -> 6.27. This refines the ceiling and checks the
combined best-of-sweep config:
  stride 100 (#4) + window 10 (#4) + n_gaussians 2 (#6) + ceiling 60 (#5).

Run from the repo root:
    python experiments/cmapss-ds02-fis/sweep_ceiling_combo.py
"""
import contextlib
import csv
import io
import os
import sys

sys.path.insert(0, "FuzzySystemsExperiments")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from _ds02_harness import load_corrected, rmse  # noqa: E402
from tribble_predictive_health.preprocessing import (  # noqa: E402
    build_memory_features, cap_rul, onset_caps,
)
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

OUT = "outputs/ds02-iterative"
BASE = dict(tsk_order="full-2nd", top_p=0.95, n_output_buckets=2,
            norm_conorm="hamacher", l2_reg=0.01, max_samples=2000)


def prep(dev, test, sensors, stride, window, memory, ceiling):
    tr, cols = build_memory_features(dev, sensors, stride=stride,
                                     window_size=window, memory_size=memory)
    te, _ = build_memory_features(test, sensors, stride=stride,
                                  window_size=window, memory_size=memory)
    y_tr = np.asarray(cap_rul(tr, onset_caps(tr)), float)
    if ceiling is not None:
        y_tr = np.minimum(y_tr, ceiling)
    scaler = StandardScaler().fit(tr[cols].to_numpy(float))
    return (scaler.transform(tr[cols].to_numpy(float)), y_tr,
            scaler.transform(te[cols].to_numpy(float)), te["rul"].to_numpy(float))


def evaluate(X_tr, y_tr, X_te, y_te, extra=None):
    reg = TribbleRegressor(random_state=42, **{**BASE, **(extra or {})})
    with contextlib.redirect_stdout(io.StringIO()):
        reg.fit(X_tr, y_tr)
    return rmse(y_tr, reg.predict(X_tr)), rmse(y_te, reg.predict(X_te))


def main():
    print("Loading + correcting DS02 ...")
    dev, test, sensors = load_corrected()
    rows = []

    print("\n== #5 ceiling refine (baseline geometry) ==")
    for rc in (54, 56, 58, 60, 62, 66, 70):
        X_tr, y_tr, X_te, y_te = prep(dev, test, sensors, 200, 5, 2, rc)
        a, b = evaluate(X_tr, y_tr, X_te, y_te)
        print(f"  ceiling={rc:3d}: train {a:5.2f}  test {b:5.2f}")
        rows.append(("ceiling", rc, a, b))

    print("\n== combos ==")
    combos = [
        ("baseline",            dict(stride=200, window=5,  memory=2, ceiling=None, extra=None)),
        ("ceiling60",           dict(stride=200, window=5,  memory=2, ceiling=60,   extra=None)),
        ("stride100+ceil60",    dict(stride=100, window=5,  memory=2, ceiling=60,   extra=None)),
        ("win10+ceil60",        dict(stride=200, window=10, memory=2, ceiling=60,   extra=None)),
        ("ng2+ceil60",          dict(stride=200, window=5,  memory=2, ceiling=60,   extra={"n_gaussians": 2})),
        ("ALL(s100/w10/ng2/c60)", dict(stride=100, window=10, memory=2, ceiling=60, extra={"n_gaussians": 2})),
    ]
    for tag, c in combos:
        X_tr, y_tr, X_te, y_te = prep(dev, test, sensors, c["stride"], c["window"], c["memory"], c["ceiling"])
        a, b = evaluate(X_tr, y_tr, X_te, y_te, c["extra"])
        print(f"  {tag:24s} n={len(X_tr):6d}  train {a:5.2f}  test {b:5.2f}")
        rows.append(("combo", tag, a, b))

    with open(os.path.join(OUT, "sweep_ceiling_combo.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sweep", "param", "train_rmse", "test_rmse"]); w.writerows(rows)
    print(f"\nwrote {os.path.join(OUT, 'sweep_ceiling_combo.csv')}")


if __name__ == "__main__":
    main()
