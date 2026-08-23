"""Tests for the preprocessing cache.

Plumbing tests (key derivation, write/read/invalidation) are pure and always
run. The correctness test -- that a cached run gives byte-identical numbers to
the uncached `fit`/`score` path -- needs tribblefis and the DS02 file, so it is
skipped where either is missing.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# Import the package the way the drivers do, independent of pytest's cwd.
_FSE = pathlib.Path(__file__).resolve().parents[2]
if str(_FSE) not in sys.path:
    sys.path.insert(0, str(_FSE))

from tribble_predictive_health import cache  # noqa: E402

DS02 = _FSE.parent / "NASA-CMAPSS" / "N-CMAPSS_DS02-006.h5"


# --- pure plumbing (no h5, no tribblefis) ----------------------------------
def _dummy_bundle():
    df = pd.DataFrame(
        {"unit": [1, 1], "cycle": [1, 2], "rul": [9.0, 8.0], "f0": [0.1, 0.2]}
    )
    return cache.Bundle(dev=df, test=df.copy(), feature_cols=["f0"])


def _key_for(tmp_path, **over):
    # a self-consistent key without touching a real file (stat is faked in)
    base = dict(
        version=cache.CACHE_VERSION,
        file="fake.h5",
        size=123,
        mtime_ns=456,
        aggregation="raw_memory",
        condition_correction=True,
        baseline_cycles=15,
        feat_params={"stride": 200, "window_size": 5, "memory_size": 2},
    )
    base.update(over)
    return base


def test_write_read_roundtrip(tmp_path):
    key = _key_for(tmp_path)
    path = cache._cache_file(str(tmp_path), key)
    assert cache._read(path, key) is None  # nothing cached yet
    cache._write(path, key, _dummy_bundle())
    got = cache._read(path, key)
    assert got is not None
    pd.testing.assert_frame_equal(got.dev, _dummy_bundle().dev)
    assert got.feature_cols == ["f0"]


def test_read_rejects_stale_key(tmp_path):
    key = _key_for(tmp_path)
    path = cache._cache_file(str(tmp_path), key)
    cache._write(path, key, _dummy_bundle())
    # a file whose mtime changed must not be served from the old cache
    stale = _key_for(tmp_path, mtime_ns=999)
    assert cache._read(path, stale) is None


def test_key_and_path_track_params():
    k1 = _key_for(None)
    k2 = _key_for(None, aggregation="whole_cycle")
    k3 = _key_for(None, feat_params={"stride": 100, "window_size": 5, "memory_size": 2})
    assert cache._key_digest(k1) != cache._key_digest(k2)
    assert cache._key_digest(k1) != cache._key_digest(k3)
    # whole_cycle carries no featurisation params
    assert cache._feature_params("whole_cycle", 200, 5, 2) == {}
    assert cache._feature_params("raw_memory", 200, 5, 2) == {
        "stride": 200,
        "window_size": 5,
        "memory_size": 2,
    }


# --- correctness: cached numbers == uncached numbers -----------------------
needs_ds02 = pytest.mark.skipif(not DS02.exists(), reason="N-CMAPSS DS02 not present")


@needs_ds02
def test_cached_path_matches_uncached(tmp_path):
    pytest.importorskip("tribblefis")
    from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss

    # Uncached reference: fit(X, y)/score do their own correction + featurise.
    dev, _, _ = load_ncmapss(str(DS02), "dev")
    test, _, _ = load_ncmapss(str(DS02), "test")
    ref = TribblePredictiveHealth().fit(dev, dev["rul"]).score(test)

    # Cached: the bundle is already corrected+featurised, so fit_featurized with
    # condition_correction=False must reproduce ref exactly.
    b = cache.load_or_build(
        str(DS02), "raw_memory", cache_dir=str(tmp_path), rebuild=True, verbose=False
    )
    eng = TribblePredictiveHealth(condition_correction=False)
    eng.fit_featurized(b.dev, b.feature_cols)
    got = eng.score_featurized(b.test)

    assert set(got) == set(ref)
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-9)


@needs_ds02
def test_load_or_build_serves_second_call_from_cache(tmp_path):
    pytest.importorskip("tribblefis")
    cold = cache.load_or_build(
        str(DS02), "raw_memory", cache_dir=str(tmp_path), rebuild=True, verbose=False
    )
    warm = cache.load_or_build(
        str(DS02), "raw_memory", cache_dir=str(tmp_path), verbose=False
    )
    pd.testing.assert_frame_equal(cold.dev, warm.dev)
    pd.testing.assert_frame_equal(cold.test, warm.test)
    assert cold.feature_cols == warm.feature_cols
    assert np.isfinite(warm.dev[warm.feature_cols].to_numpy(float)).all()
