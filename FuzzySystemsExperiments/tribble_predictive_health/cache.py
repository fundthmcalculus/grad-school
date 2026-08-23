"""Disk cache for the expensive front of the N-CMAPSS pipeline.

Loading a raw N-CMAPSS file (2.4 GB for DS02) and turning it into condition-
corrected feature tables costs seconds to a minute-plus and gigabytes of RAM;
the *result* is tiny (tens of thousands of rows). So for fast model-tuning
iteration this module runs `load_ncmapss -> condition correction -> feature
aggregation` once and pickles the small tables, keyed on the source file's
identity and the preprocessing parameters. Everything downstream reads the
cache and goes straight to the fuzzy-system fit.

The cached tables are exactly what `TribblePredictiveHealth.fit` builds
internally before it caps/scales/fits -- already condition-corrected and
featurised -- so a cached run feeds them through the estimator's `fit_featurized`
/ `score_featurized` / `predict_samples_featurized` entry points (with
`condition_correction=False`, since the cache has already done it) and produces
byte-identical numbers to the uncached `fit` / `score` path. See
`test_cache.py`.

Caches live under `outputs/` (git-ignored). Bump `CACHE_VERSION` if the
preprocessing logic changes in a way that should invalidate every cache.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass

from .data import load_ncmapss
from .preprocessing import (
    apply_condition_correction,
    build_memory_features,
    build_whole_cycle_features,
    fit_condition_correction,
)

# Bump when the meaning of a cached table changes (a preprocessing-logic edit).
CACHE_VERSION = 1

# repo/FuzzySystemsExperiments/tribble_predictive_health/cache.py -> repo
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE_DIR = os.path.join(_REPO, "outputs", "cmapss-cache")


@dataclass
class Bundle:
    """Condition-corrected, featurised train/test tables for one file+aggregation."""

    dev: object  # pandas DataFrame
    test: object  # pandas DataFrame
    feature_cols: list


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------
def _feature_params(aggregation, stride, window_size, memory_size):
    """Only the parameters that change the featurised output, per aggregation."""
    if aggregation == "raw_memory":
        return dict(stride=stride, window_size=window_size, memory_size=memory_size)
    return {}  # whole_cycle has no tunable featurisation params (AGG_FUNCS fixed)


def _key(h5_path, aggregation, condition_correction, baseline_cycles, feat_params):
    """A dict fully identifying a cached table: source-file identity (so a
    changed .h5 invalidates) plus every preprocessing knob and the version."""
    st = os.stat(h5_path)
    return {
        "version": CACHE_VERSION,
        "file": os.path.basename(h5_path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "aggregation": aggregation,
        "condition_correction": bool(condition_correction),
        "baseline_cycles": baseline_cycles,
        "feat_params": feat_params,
    }


def _key_digest(key):
    blob = repr(sorted(key.items())).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _cache_file(cache_dir, key):
    tag = key["file"].replace("N-CMAPSS_", "").replace(".h5", "")
    return os.path.join(
        cache_dir, f"{tag}__{key['aggregation']}__{_key_digest(key)}.pkl"
    )


# ---------------------------------------------------------------------------
# Build (the expensive path) and the cached wrappers
# ---------------------------------------------------------------------------
def _featurize(aggregation, df, sensors, feat_params):
    if aggregation == "whole_cycle":
        return build_whole_cycle_features(df, sensors)
    if aggregation == "raw_memory":
        return build_memory_features(df, sensors, **feat_params)
    raise ValueError(f"unknown aggregation {aggregation!r}")


def build_many(
    h5_path,
    aggregations,
    *,
    condition_correction=True,
    baseline_cycles=15,
    stride=200,
    window_size=5,
    memory_size=2,
):
    """Load `h5_path` once, condition-correct once, and featurise it for each
    aggregation in `aggregations`. Returns {aggregation: Bundle}. This is the
    expensive step the cache exists to avoid repeating."""
    dev, cond, sensors = load_ncmapss(h5_path, "dev")
    test, _, _ = load_ncmapss(h5_path, "test")
    if condition_correction:
        models = fit_condition_correction(
            dev, sensors, cond, baseline_cycles=baseline_cycles
        )
        dev = apply_condition_correction(dev, sensors, cond, models)
        test = apply_condition_correction(test, sensors, cond, models)

    out = {}
    for agg in aggregations:
        fp = _feature_params(agg, stride, window_size, memory_size)
        tr, cols = _featurize(agg, dev, sensors, fp)
        te, _ = _featurize(agg, test, sensors, fp)
        out[agg] = Bundle(dev=tr, test=te, feature_cols=list(cols))
    return out


def _read(path, expected_key):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None  # unreadable/partial cache -> rebuild
    if payload.get("key") != expected_key:  # hash collision / stale -> rebuild
        return None
    return payload["bundle"]


def _write(path, key, bundle):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"key": key, "bundle": bundle}, f)
    os.replace(tmp, path)  # atomic: never leave a half-written cache


def load_or_build_many(
    h5_path,
    aggregations,
    *,
    condition_correction=True,
    baseline_cycles=15,
    stride=200,
    window_size=5,
    memory_size=2,
    cache_dir=DEFAULT_CACHE_DIR,
    rebuild=False,
    verbose=True,
):
    """{aggregation: Bundle} for `h5_path`, served from cache where possible.

    Any aggregation whose cache is missing/stale is (re)built; when several miss,
    they are built from a single file load, so a cold run reads the .h5 once, not
    once per aggregation."""
    params = dict(
        condition_correction=condition_correction,
        baseline_cycles=baseline_cycles,
        stride=stride,
        window_size=window_size,
        memory_size=memory_size,
    )
    keys, paths, result, missing = {}, {}, {}, []
    for agg in aggregations:
        fp = _feature_params(agg, stride, window_size, memory_size)
        keys[agg] = _key(h5_path, agg, condition_correction, baseline_cycles, fp)
        paths[agg] = _cache_file(cache_dir, keys[agg])
        hit = None if rebuild else _read(paths[agg], keys[agg])
        if hit is not None:
            result[agg] = hit
            if verbose:
                print(f"  (cached: {os.path.relpath(paths[agg], _REPO)})")
        else:
            missing.append(agg)

    if missing:
        if verbose:
            print(
                f"  building {', '.join(missing)} from {os.path.basename(h5_path)} ..."
            )
        built = build_many(h5_path, missing, **params)
        for agg in missing:
            _write(paths[agg], keys[agg], built[agg])
            result[agg] = built[agg]
    return {agg: result[agg] for agg in aggregations}


def load_or_build(h5_path, aggregation, **kwargs):
    """Single-aggregation convenience wrapper around `load_or_build_many`."""
    return load_or_build_many(h5_path, [aggregation], **kwargs)[aggregation]
