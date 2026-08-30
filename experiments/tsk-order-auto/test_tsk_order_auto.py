"""Regression test for grad-school issue #120's acceptance criterion.

    "full-2nd (or an ``auto`` order) usable as a default with no negative test
    R2 on diabetes-scale data, without hand-applying the rows/coeff check."

``run.py`` is the full study (three datasets, six orders, ten seeds, ~36 s).
This file is the part CI can afford to run on every PR: diabetes only, three
seeds, and assertions on *relations between arms on paired splits* rather than
on absolute R2 values, so a different BLAS or platform cannot make it flake
while still catching a real regression in ``tsk_order="auto"``.

Diabetes is the right case to pin. At 442 rows and 10 features a full-2nd
consequent fits 66 coefficients per rule -- 6.7 rows per coefficient -- which is
below the ~5x-and-up band where the interaction basis pays for itself, and it is
where issue #120 measured the overfit cliff.

Run: ``python -m pytest experiments/tsk-order-auto/test_tsk_order_auto.py``
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.datasets import load_diabetes  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

SEEDS = (0, 1, 2)
FIXED_ORDERS = ("0th", "1st", "2nd", "full-2nd", "3rd")
# Margin the `auto` arm is allowed to give up against the best FIXED order.
# Measured at ten seeds: auto 0.4428 +- 0.0807 vs the best fixed (1st)
# 0.4474 +- 0.0791, a gap of 0.005. 0.05 is ten times that.
AUTO_SLACK = 0.05
# Margin by which `auto` must BEAT full-2nd, i.e. the size of the foot-gun this
# feature exists to remove. Measured at ten seeds: 0.4428 vs 0.1573 = 0.286.
FOOTGUN_MARGIN = 0.10


def _fit_score(order, seed):
    """Test R2 and the resolved order, on the seed's paired split."""
    data = load_diabetes()
    X, y = np.asarray(data.data, float), np.asarray(data.target, float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        # The rows/coeff UserWarning is the thing `auto` replaces; it firing is
        # expected on the full-2nd arm and says nothing about the assertions.
        warnings.simplefilter("ignore")
        model = TribbleRegressor(tsk_order=order, random_state=seed)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
    return float(r2_score(yte, pred)), str(getattr(model, "tsk_order_", order))


@pytest.fixture(scope="module")
def scores():
    """{order: [r2 per seed]} plus {order: [resolved pick per seed]}."""
    r2, picks = {}, {}
    for order in FIXED_ORDERS + ("auto",):
        pairs = [_fit_score(order, s) for s in SEEDS]
        r2[order] = [p[0] for p in pairs]
        picks[order] = [p[1] for p in pairs]
    return r2, picks


def _mean(xs):
    return sum(xs) / len(xs)


def test_auto_is_never_negative(scores):
    """#120's literal acceptance criterion, on diabetes-scale data."""
    r2, _ = scores
    negative = [(s, v) for s, v in zip(SEEDS, r2["auto"]) if v <= 0.0]
    assert not negative, f"tsk_order='auto' produced a non-positive test R2: {negative}"


def test_full_2nd_still_needs_the_guard(scores):
    """The hazard is real, so the guard is doing work rather than nothing.

    If this ever fails, `auto` has stopped being *necessary* on this dataset --
    which is a finding worth looking at, not a broken test to loosen.
    """
    r2, _ = scores
    gap = _mean(r2["auto"]) - _mean(r2["full-2nd"])
    assert gap > FOOTGUN_MARGIN, (
        f"full-2nd no longer underperforms 'auto' on diabetes (gap {gap:+.3f} "
        f"<= {FOOTGUN_MARGIN}); auto={_mean(r2['auto']):+.3f}, "
        f"full-2nd={_mean(r2['full-2nd']):+.3f}"
    )


def test_auto_matches_the_best_hand_picked_order(scores):
    """Safety must not cost accuracy: `auto` has to be a usable default.

    A guard that avoided the overfit by always collapsing to `0th` would pass
    the test above and be useless, so this pins `auto` against the best fixed
    order.

    That baseline is an ORACLE: it is chosen by TEST R2, which no honest model
    selection could do. That is deliberate and in the safe direction -- it makes
    the assertion strictly harder to pass than any achievable baseline would --
    but it is not a number `auto` is being unfairly compared against by
    accident.
    """
    r2, _ = scores
    best_order = max(FIXED_ORDERS, key=lambda o: _mean(r2[o]))
    best, auto = _mean(r2[best_order]), _mean(r2["auto"])
    assert auto >= best - AUTO_SLACK, (
        f"'auto' ({auto:+.3f}) fell more than {AUTO_SLACK} behind the best fixed "
        f"order '{best_order}' ({best:+.3f})"
    )


def test_auto_reports_the_order_it_chose(scores):
    """`tsk_order_` is the audit trail: a run must be able to say what it fit."""
    _, picks = scores
    for seed, pick in zip(SEEDS, picks["auto"]):
        assert pick in FIXED_ORDERS, (
            f"seed {seed}: tsk_order_ is {pick!r}, not one of the concrete "
            f"orders {FIXED_ORDERS}"
        )
    for order in FIXED_ORDERS:
        assert set(picks[order]) == {order}, (
            f"a fixed tsk_order={order!r} must resolve to itself, got "
            f"{sorted(set(picks[order]))}"
        )
