"""Unit tests for the RUL scoring conventions. Pure numpy/pandas -- no fuzzy
system, no data file."""

import numpy as np
import pandas as pd
import pytest

from tribble_predictive_health import metrics


def test_rmse_matches_manual():
    y = np.array([0.0, 2.0, 4.0])
    p = np.array([1.0, 2.0, 2.0])  # errors 1, 0, 2
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((1 + 0 + 4) / 3))


def test_rmse_zero_on_perfect():
    y = np.array([3.0, 1.0, 4.0])
    assert metrics.rmse(y, y) == 0.0


def test_nasa_score_penalizes_late_harder_than_early():
    # "late" = overestimated RUL (pred > true): the dangerous direction.
    true = np.array([50.0])
    late = metrics.nasa_score(true, np.array([60.0]))
    early = metrics.nasa_score(true, np.array([40.0]))
    assert late > early
    assert late == pytest.approx(np.exp(10 / 10.0))
    assert early == pytest.approx(np.exp(10 / 13.0))


def test_nasa_score_zero_error_equals_count():
    true = np.array([1.0, 2.0, 3.0])
    assert metrics.nasa_score(true, true) == pytest.approx(3.0)  # exp(0) * 3


def _pc(units, cycles, pred, true=None):
    cols = {"unit": units, "cycle": cycles, "pred": pred}
    if true is not None:
        cols["true"] = true
    return pd.DataFrame(cols)


def test_rising_fraction_counts_upsteps():
    # preds 5, 4, 6, 3 -> diffs -1, +2, -3 -> 1 of 3 steps rose
    df = _pc([0, 0, 0, 0], [1, 2, 3, 4], [5.0, 4.0, 6.0, 3.0])
    assert metrics.rising_fraction(df) == pytest.approx(1 / 3)


def test_rising_fraction_no_steps_is_zero():
    assert metrics.rising_fraction(_pc([0], [1], [5.0])) == 0.0


def test_per_engine_canonical_uses_last_cycle():
    # engine 0 last cycle 3: true 10 pred 12 (late by 2); engine 1 last: exact
    df = _pc(
        [0, 0, 1, 1],
        [1, 3, 1, 2],
        [99.0, 12.0, 99.0, 20.0],
        true=[99.0, 10.0, 99.0, 20.0],
    )
    r, n = metrics.per_engine_canonical(df)
    assert r == pytest.approx(np.sqrt((4 + 0) / 2))
    assert n == pytest.approx(np.exp(2 / 10.0) + 1.0)


def test_score_reports_every_convention_and_monotone_is_flat():
    rng = np.random.default_rng(0)
    units = np.repeat([0, 1], 5)
    cycles = np.tile(np.arange(5), 2)
    true = np.concatenate([np.arange(5, 0, -1)] * 2).astype(float)
    pred = true + rng.normal(scale=0.5, size=true.size)
    s = metrics.score(units, cycles, true, pred)
    assert set(s) == {
        "per_sample_rmse",
        "raw_cycle_rmse",
        "raw_rising",
        "monotone_cycle_rmse",
        "monotone_rising",
        "per_engine_rmse",
        "per_engine_nasa",
    }
    assert s["monotone_rising"] == 0.0  # the clamp guarantees this
