"""Estimator-level tests for TribblePredictiveHealth.

A fast synthetic smoke test exercises the fit/predict/score plumbing without a
data file; a gated integration test reproduces the DS02 result when the
N-CMAPSS DS02 HDF5 file is present (it is skipped in CI, where the file is not).

Both need tribblefis (the fuzzy system), so the whole module is skipped when it
is unavailable.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tribblefis")

from tribble_predictive_health import TribblePredictiveHealth  # noqa: E402

# tests/ -> tribble_predictive_health/ -> FuzzySystemsExperiments/ -> repo root
DS02 = (
    pathlib.Path(__file__).resolve().parents[3]
    / "data"
    / "nasa-cmapps2"
    / "N-CMAPSS_DS02-006.h5"
)


def _run_to_failure(n_units=6, n_cycles=30, samples_per_cycle=8, seed=0):
    """Synthetic tidy stream whose sensors drift monotonically toward failure,
    with an operating-condition channel the correction step should remove."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        life = n_cycles + int(rng.integers(-3, 4))
        for c in range(1, n_cycles + 1):
            frac = c / life  # 0 (new) -> ~1 (worn)
            for _ in range(samples_per_cycle):
                w1, w2 = rng.uniform(0, 1), rng.uniform(0, 1)
                base = 2.0 * w1 - 1.5 * w2  # condition dependence
                rows.append(
                    (
                        u,
                        c,
                        1 if frac < 0.5 else 0,
                        float(max(life - c, 0)),
                        w1,
                        w2,
                        base + 4.0 * frac + rng.normal(scale=0.05),
                        base - 3.0 * frac + rng.normal(scale=0.05),
                        base + 2.0 * frac**2 + rng.normal(scale=0.05),
                    )
                )
    return pd.DataFrame(
        rows,
        columns=[
            "unit",
            "cycle",
            "health",
            "rul",
            "W_1",
            "W_2",
            "Xs_a",
            "Xs_b",
            "Xs_c",
        ],
    )


@pytest.mark.parametrize("aggregation", ["whole_cycle", "raw_memory"])
def test_synthetic_fit_predict_score_plumbing(aggregation):
    df = _run_to_failure()
    kw = dict(aggregation=aggregation, random_state=0)
    if aggregation == "raw_memory":
        kw["stride"] = 2  # the default 200 would empty a 30-cycle engine
    engine = TribblePredictiveHealth(**kw).fit(df, df["rul"])

    frame = engine.predict_frame(df, include_true=True)
    assert set(frame.columns) == {"unit", "cycle", "rul", "true"}
    # predict() is per cycle: one value per (unit, cycle) it emits
    assert len(engine.predict(df)) == len(frame)
    # the deployable trajectory only ever falls
    for _, sub in frame.groupby("unit"):
        assert np.all(np.diff(sub.sort_values("cycle")["rul"].to_numpy()) <= 0)

    s = engine.score(df)
    assert s["monotone_rising"] == 0.0
    assert np.isfinite(s["per_sample_rmse"])
    assert engine.n_rules_ >= 1


def test_condition_correction_toggle_changes_the_fit():
    df = _run_to_failure(seed=1)
    kw = dict(aggregation="whole_cycle", random_state=0)
    on = TribblePredictiveHealth(**kw).fit(df, df["rul"])
    off = TribblePredictiveHealth(condition_correction=False, **kw).fit(df, df["rul"])
    assert on.condition_models_  # models were learned
    assert not off.condition_models_  # and skipped when disabled
    # the two configurations do not produce an identical prediction
    assert not np.allclose(on.predict(df), off.predict(df))


@pytest.mark.skipif(not DS02.exists(), reason="N-CMAPSS DS02 HDF5 not present")
def test_ds02_end_to_end_on_real_data():
    """End-to-end smoke test on the real DS02 split: the engine fits, scores in
    the right ballpark, and emits a valid monotone RUL trajectory.

    The RMSE bound is deliberately LOOSE. DS02 per-sample RMSE is subject to a
    known Linux<->Windows reproducibility gap in the tribblefis pipeline (~6.5
    on the canonical Windows runs the dissertation reports, ~7.2 on Linux);
    both tribble-fis pins give the same per-platform figure, so this is
    platform, not staleness. A tight numeric pin would fail on the other OS, so
    assert only a sane band plus the structural invariants that must hold
    everywhere."""
    from tribble_predictive_health import load_ncmapss

    dev, _, _ = load_ncmapss(str(DS02), "dev")
    test, _, _ = load_ncmapss(str(DS02), "test")
    engine = TribblePredictiveHealth().fit(dev, dev["rul"])

    m = engine.score(test)
    assert 5.0 < m["per_sample_rmse"] < 9.0  # comfortably beats MLP (8.34)
    assert m["monotone_cycle_rmse"] <= m["raw_cycle_rmse"] + 1e-9  # clamp helps
    assert m["monotone_rising"] == 0.0  # clamp guarantees this

    frame = engine.predict_frame(test, include_true=True)
    assert len(engine.predict(test)) == len(frame)
    for _, sub in frame.groupby("unit"):
        assert np.all(np.diff(sub.sort_values("cycle")["rul"].to_numpy()) <= 0)


def test_firing_exponent_param_roundtrips():
    """The new knob is a plain, clonable estimator parameter (default 1.0)."""
    assert TribblePredictiveHealth().firing_exponent == 1.0
    eng = TribblePredictiveHealth(firing_exponent=0.5)
    assert eng.get_params()["firing_exponent"] == 0.5


def test_firing_exponent_is_forwarded_to_the_regressor():
    """It reaches the underlying TribbleRegressor. Skipped on a tribblefis that
    predates the knob (grad-school pins one that has it)."""
    import inspect

    from tribblefis.gaussian_regressor import TribbleRegressor

    if "firing_exponent" not in inspect.signature(TribbleRegressor).parameters:
        pytest.skip("installed tribblefis predates firing_exponent")
    df = _run_to_failure()
    eng = TribblePredictiveHealth(
        aggregation="whole_cycle", firing_exponent=0.5, random_state=0
    ).fit(df, df["rul"])
    assert eng.regressor_.firing_exponent == 0.5
