"""#4 memory-feature geometry and #5 RUL-cap sweeps on N-CMAPSS DS02.

Both act *upstream* of the fuzzy system -- the regressor is held at the shipped
DS02 default (full-2nd, 2 output buckets, top_p 0.95, hamacher; baseline
per-sample test RMSE ~6.48) so each sweep is a clean one-factor test.

  #4  build_memory_features(stride, window_size, memory_size): how densely the
      stream is subsampled and how long the short/long rolling-average memories
      are -- i.e. how much degradation context each sample carries.
  #5  the RUL target cap: the pipeline caps RUL at each engine's health-onset
      cycle (onset_caps). Here we additionally test a constant ceiling Rc (the
      classic C-MAPSS piecewise-linear-RUL lever) on top of / instead of it.

Condition correction is fit once; only featurisation / target change per row.
Writes CSV to outputs/ds02-iterative/. Run from the repo root:

    python experiments/cmapss-ds02-fis/sweep_features_target.py
"""

import contextlib
import csv
import io
import os

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from _ds02_harness import bootstrap, load_corrected, rmse  # noqa: E402

bootstrap("FuzzySystemsExperiments", os.path.dirname(__file__))
from tribble_predictive_health.preprocessing import (  # noqa: E402
    build_memory_features,
    cap_rul,
    onset_caps,
)
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

OUT = "outputs/ds02-iterative"
os.makedirs(OUT, exist_ok=True)
REG = dict(
    tsk_order="full-2nd",
    top_p=0.95,
    n_output_buckets=2,
    norm_conorm="hamacher",
    l2_reg=0.01,
    max_samples=2000,
)


def fit_eval(X_tr, y_tr, X_te, y_te):
    reg = TribbleRegressor(random_state=42, **REG)
    with contextlib.redirect_stdout(io.StringIO()):
        reg.fit(X_tr, y_tr)
    return rmse(y_tr, reg.predict(X_tr)), rmse(y_te, reg.predict(X_te))


def featurize(dev, test, sensors, stride, window, memory):
    tr, cols = build_memory_features(
        dev, sensors, stride=stride, window_size=window, memory_size=memory
    )
    te, _ = build_memory_features(
        test, sensors, stride=stride, window_size=window, memory_size=memory
    )
    return tr, te, cols


def scale_and_cap(tr, te, cols, ceiling=None):
    caps = onset_caps(tr)
    y_tr = np.asarray(cap_rul(tr, caps), float)
    y_te = te["rul"].to_numpy(float)
    if ceiling is not None:
        y_tr = np.minimum(y_tr, ceiling)  # test target is left as scored (uncapped)
    scaler = StandardScaler().fit(tr[cols].to_numpy(float))
    return (
        scaler.transform(tr[cols].to_numpy(float)),
        y_tr,
        scaler.transform(te[cols].to_numpy(float)),
        y_te,
    )


def main():
    print("Loading + correcting DS02 ...")
    dev, test, sensors = load_corrected()
    rows = []

    def run(tag, param, tr, te, cols, ceiling=None):
        X_tr, y_tr, X_te, y_te = scale_and_cap(tr, te, cols, ceiling)
        a, b = fit_eval(X_tr, y_tr, X_te, y_te)
        print(
            f"  {tag:16s} {str(param):22s} n={len(X_tr):6d} f={len(cols):2d} "
            f"cap<={y_tr.max():5.1f}  train {a:5.2f}  test {b:5.2f}"
        )
        rows.append((tag, str(param), len(X_tr), len(cols), float(y_tr.max()), a, b))

    # baseline (default geometry, onset cap only)
    print("\n== baseline ==")
    tr0, te0, cols0 = featurize(dev, test, sensors, 200, 5, 2)
    run("baseline", "stride200/w5/m2", tr0, te0, cols0)

    # #4a stride
    print("\n== #4 stride ==")
    for s in (50, 100, 400, 800):
        tr, te, cols = featurize(dev, test, sensors, s, 5, 2)
        run("stride", s, tr, te, cols)
    # #4b window_size
    print("\n== #4 window_size ==")
    for w in (3, 10, 20, 40):
        tr, te, cols = featurize(dev, test, sensors, 200, w, 2)
        run("window", w, tr, te, cols)
    # #4c memory_size
    print("\n== #4 memory_size ==")
    for m in (1, 3, 4):  # extractor requires memory_size < window_size (=5 here)
        tr, te, cols = featurize(dev, test, sensors, 200, 5, m)
        run("memory", m, tr, te, cols)

    # #5 RUL cap ceiling (reuse baseline geometry)
    print("\n== #5 RUL ceiling (on top of onset cap; baseline max ~72) ==")
    for rc in (30, 40, 50, 60, 90):
        run("ceiling", rc, tr0, te0, cols0, ceiling=rc)

    with open(os.path.join(OUT, "sweep_features_target.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sweep",
                "param",
                "n_train",
                "n_feat",
                "cap_max",
                "train_rmse",
                "test_rmse",
            ]
        )
        w.writerows(rows)
    print(f"\nwrote {os.path.join(OUT, 'sweep_features_target.csv')}")


if __name__ == "__main__":
    main()
