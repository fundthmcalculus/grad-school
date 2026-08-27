"""Unit tests for the processing-engine steps. The pure steps use synthetic
data; the memory-feature builder is skipped where tribblefis is unavailable."""

import numpy as np
import pandas as pd
import pytest

from tribble_predictive_health import preprocessing as pp


def _stream(n_units=3, n_cycles=6, samples_per_cycle=4, seed=0):
    """A tiny tidy run-to-failure stream: sensor = 3*condition + degradation."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for c in range(1, n_cycles + 1):
            for _ in range(samples_per_cycle):
                w = rng.uniform(0.0, 1.0)
                s = 3.0 * w + 0.5 * c + rng.normal(scale=0.01)
                rows.append((u, c, 1 if c <= 3 else 0, float(n_cycles - c), w, s))
    return pd.DataFrame(rows, columns=["unit", "cycle", "health", "rul", "W_x", "Xs_s"])


def test_condition_correction_removes_condition_dependence():
    df = _stream()
    models = pp.fit_condition_correction(df, ["Xs_s"], ["W_x"], baseline_cycles=1000)
    out = pp.apply_condition_correction(df, ["Xs_s"], ["W_x"], models)
    assert models["Xs_s"].coef_[0] == pytest.approx(3.0, abs=0.3)
    # residual is (almost) uncorrelated with the operating condition
    assert abs(np.corrcoef(out["W_x"], out["Xs_s"])[0, 1]) < 0.2


def test_onset_caps_is_rul_at_first_unhealthy_cycle():
    caps = pp.onset_caps(_stream())
    # health flips at cycle 4; RUL there is n_cycles - 4 = 2
    assert all(caps[u] == pytest.approx(2.0) for u in range(3))


def test_cap_rul_caps_known_units_and_passes_unknown():
    df = _stream(n_units=2)
    capped = np.asarray(pp.cap_rul(df, {0: 2.0}))  # only unit 0 known
    is_u0 = df["unit"].to_numpy() == 0
    assert capped[is_u0].max() <= 2.0 + 1e-9
    assert np.allclose(capped[~is_u0], df["rul"].to_numpy()[~is_u0])


def test_per_cycle_averages_duplicate_samples():
    pc = pp.per_cycle(
        np.array([0, 0, 0, 0]),
        np.array([1, 1, 2, 2]),
        np.array([10.0, 12.0, 5.0, 7.0]),
    )
    assert pc["cycle"].tolist() == [1, 2]
    assert pc["pred"].tolist() == [11.0, 6.0]


def test_clamp_monotone_never_rises():
    pc = pd.DataFrame(
        {"unit": [0] * 4, "cycle": [1, 2, 3, 4], "pred": [10.0, 12.0, 8.0, 9.0]}
    )
    out = pp.clamp_monotone(pc).sort_values("cycle")
    assert out["pred"].tolist() == [10.0, 10.0, 8.0, 8.0]
    assert np.all(np.diff(out["pred"].to_numpy()) <= 0)


def test_build_whole_cycle_features_shape_and_columns():
    df = _stream(n_units=2, n_cycles=4, samples_per_cycle=5)
    out, cols = pp.build_whole_cycle_features(df, ["Xs_s"])
    assert len(out) == 2 * 4  # one row per (unit, cycle)
    assert cols == ["Xs_s_mean", "Xs_s_std", "Xs_s_min", "Xs_s_max", "Xs_s_last"]
    assert {"unit", "cycle", "rul", "health"}.issubset(out.columns)


def test_build_memory_features_carries_labels_and_fills_warmup():
    pytest.importorskip("tribblefis")
    df = _stream(n_units=2, n_cycles=12, samples_per_cycle=1)
    out, cols = pp.build_memory_features(
        df, ["Xs_s"], stride=1, window_size=3, memory_size=2
    )
    assert cols and all(c in out.columns for c in ("unit", "cycle", "health", "rul"))
    assert not out[cols].isna().any().any()  # warmup NaNs were bfilled/ffilled
