"""Shared DS02 featurisation for the iterative-training-residual experiments.

Reproduces exactly what `TribblePredictiveHealth` does for the DS02 default
(`raw_memory`) config -- condition correction (fit on dev), memory features,
onset RUL cap, StandardScaler -- but stops short of the fuzzy system so the
experiment scripts can drop in their own (boosted / staged) regressors and read
off the *training* residual directly. Run from the repo root.
"""

import sys as _sys


def bootstrap(*paths):
    """Insert repo-root-relative sys.path entries needed by this experiments
    directory's scripts (e.g. "FuzzySystemsExperiments",
    "gated-minimax-selection", or a script's own ``os.path.dirname(__file__)``
    to reach sibling modules like this one). Paths are inserted at position 0
    in the given order -- same paths, same order, same relative-vs-absolute
    semantics as the inline ``sys.path.insert(0, ...)`` calls this replaces.
    """
    for p in paths:
        _sys.path.insert(0, p)


bootstrap("FuzzySystemsExperiments")
import numpy as np
from sklearn.preprocessing import StandardScaler

from tribble_predictive_health import load_ncmapss
from tribble_predictive_health.preprocessing import (
    apply_condition_correction,
    build_memory_features,
    cap_rul,
    fit_condition_correction,
    onset_caps,
)

H5 = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"


def load(h5=H5):
    dev, cond, sensors = load_ncmapss(h5, "dev")
    test, _, _ = load_ncmapss(h5, "test")
    models = fit_condition_correction(dev, sensors, cond)
    dev = apply_condition_correction(dev, sensors, cond, models)
    test = apply_condition_correction(test, sensors, cond, models)

    tr, cols = build_memory_features(dev, sensors)
    te, _ = build_memory_features(test, sensors)

    caps = onset_caps(tr)
    y_tr = cap_rul(tr, caps)  # capped train target
    y_te = te["rul"].to_numpy(float)  # test target is uncapped (as scored)

    scaler = StandardScaler().fit(tr[cols].to_numpy(float))
    X_tr = scaler.transform(tr[cols].to_numpy(float))
    X_te = scaler.transform(te[cols].to_numpy(float))
    return dict(
        X_tr=X_tr,
        y_tr=np.asarray(y_tr, float),
        X_te=X_te,
        y_te=y_te,
        unit_te=te["unit"].to_numpy(),
        cycle_te=te["cycle"].to_numpy(),
        cols=cols,
    )


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_corrected(h5=H5):
    """The condition-corrected dev/test frames + sensor list, before any
    featurisation -- so the feature-geometry / RUL-cap sweeps can re-featurise
    with their own knobs. Returns (dev, test, sensors)."""
    dev, cond, sensors = load_ncmapss(h5, "dev")
    test, _, _ = load_ncmapss(h5, "test")
    models = fit_condition_correction(dev, sensors, cond)
    dev = apply_condition_correction(dev, sensors, cond, models)
    test = apply_condition_correction(test, sensors, cond, models)
    return dev, test, sensors
