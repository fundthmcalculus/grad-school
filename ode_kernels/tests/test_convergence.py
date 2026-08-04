"""Empirical order-of-convergence check -- the test that actually proves a
tableau is correct, as opposed to merely self-consistent.

Runs each method at *fixed* step size (bypassing the adaptive controller
entirely, straight through the compiled ``step_generic_py`` kernel) across a
range of step sizes on ``y' = -y`` (known exact solution ``exp(-t)``), and
checks that the empirical log-log slope of global error vs. h is at least
the method's claimed order. A tableau with a wrong coefficient will not
produce a smooth-looking-but-wrong-order curve; it fails to converge at the
claimed rate at all, which this comfortably catches.

The bound is one-sided (>= order - slack, not == order) deliberately: on a
scalar linear autonomous test problem, a method's error is exactly
``R(h) - exp(h)`` where ``R`` is its stability polynomial, and several of
these tableaus happen to match a term or two of ``exp`` beyond what their
classical (nonlinear, B-series) order guarantees -- observed here as ode56
tracking almost like an order-7 method and ode78 almost like order-9 on
this specific problem. That is a known, harmless property of the linear
test, not a bug; asserting equality against it would be asserting the wrong
thing. What must never happen is the slope coming in *below* the claimed
order, which is what a transcription error in a high-order coefficient
actually looks like.
"""

import numpy as np
import pytest

from ode_kernels import _rk_kernels, tableaus


def _fixed_step_integrate(method, f, t0, tf, y0, h):
    tab = tableaus.TABLEAUS[method]
    A, b, e, c = tab.as_arrays()
    n_stages = tab.n_stages
    y = np.asarray(y0, dtype=np.float64).copy()
    t = t0
    n_steps = int(round((tf - t0) / h))
    k1_carry = None
    for _ in range(n_steps):
        y_new, err, K = _rk_kernels.step_generic_py(
            f, t, y, h, A, b, e, c, n_stages, (), k1_carry
        )
        k1_carry = K[-1] if tab.fsal else None
        y = y_new
        t += h
    return t, y


def _decay(t, y):
    return -y


# Step sizes hand-picked per method to sit in the asymptotic truncation-
# error-dominated regime: large enough to stay well clear of the float64
# roundoff floor (~1e-16), small enough to be past the pre-asymptotic
# transient at h~O(1). Higher-order methods converge so fast that their
# usable window is narrower and centered on larger h.
_STEP_SIZES = {
    "ode12": [0.125, 0.0625, 0.03125, 0.015625],
    "ode23": [0.125, 0.0625, 0.03125, 0.015625],
    "ode45": [0.25, 0.125, 0.0625, 0.03125],
    "ode56": [0.5, 0.25, 0.125],
    "ode67": [1.0, 0.5, 0.25, 0.125],
    "ode78": [1.0, 0.5, 0.25],
}

_ROUNDOFF_FLOOR = 1e-13


@pytest.mark.parametrize("method", list(tableaus.TABLEAUS))
def test_empirical_order_at_least_claimed_order(method):
    tab = tableaus.TABLEAUS[method]
    t0, tf = 0.0, 1.0
    exact = np.exp(-tf)

    hs = np.array(_STEP_SIZES[method])
    errors = np.array([
        abs(_fixed_step_integrate(method, _decay, t0, tf, [1.0], h)[1][0] - exact)
        for h in hs
    ])

    usable = errors > _ROUNDOFF_FLOOR
    hs_fit, err_fit = hs[usable], errors[usable]
    assert len(hs_fit) >= 2, (
        f"{method}: errors hit the roundoff floor too fast to fit a slope: {errors}"
    )

    slope = np.polyfit(np.log(hs_fit), np.log(err_fit), 1)[0]
    assert slope >= tab.order - 0.5, (
        f"{method}: empirical order {slope:.2f} is below the claimed order "
        f"{tab.order} (errors={errors})"
    )
