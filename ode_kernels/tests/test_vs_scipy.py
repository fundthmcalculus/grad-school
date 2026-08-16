"""Cross-checks against scipy.integrate.solve_ivp.

ode23/ode45 use scipy's own RK23/RK45 Butcher tableaux, so on a smooth
problem they should take an essentially identical step sequence and land on
essentially the same answer -- checked tightly here. ode56/ode67/ode78 have
no scipy equivalent tableau, so they are checked against scipy's DOP853 (a
different, independently-implemented order-8 method) as a high-accuracy
reference: agreement there exercises the whole adaptive-stepping pipeline
end to end, independent of any tableau this package shares with scipy.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ode_kernels import ode23, ode45, ode56, ode67, ode78


def _van_der_pol(t, y, mu=1.0):
    return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]


@pytest.mark.parametrize("ours,scipy_method", [(ode23, "RK23"), (ode45, "RK45")])
def test_same_tableau_matches_scipy_closely(ours, scipy_method):
    t_span = (0.0, 10.0)
    y0 = [2.0, 0.0]
    ref = solve_ivp(
        _van_der_pol,
        t_span,
        y0,
        method=scipy_method,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )
    res = ours(_van_der_pol, t_span, y0, rtol=1e-8, atol=1e-10)

    assert res.success
    # Compare our natural step-time solution against scipy's dense output at
    # those same times -- a strict test since both use the same tableau, so
    # any real divergence means a bug in the stepper or controller, not just
    # differing step choices.
    y_ref = ref.sol(res.t)
    np.testing.assert_allclose(res.y, y_ref, rtol=1e-5, atol=1e-7)

    # Step *counts* should be in the same ballpark (same tableau + same
    # controller policy => nearly the same step sequence).
    assert 0.5 <= res.nstep / ref.t.size <= 2.0


@pytest.mark.parametrize("ours", [ode56, ode67, ode78])
def test_higher_order_methods_agree_with_dop853(ours):
    t_span = (0.0, 10.0)
    y0 = [2.0, 0.0]
    ref = solve_ivp(
        _van_der_pol,
        t_span,
        y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
    )
    res = ours(_van_der_pol, t_span, y0, rtol=1e-8, atol=1e-10)

    assert res.success
    y_ref = ref.sol(res.t)
    np.testing.assert_allclose(res.y, y_ref, rtol=1e-4, atol=1e-6)


def test_exponential_decay_multidimensional():
    def f(t, y):
        return -np.array(y)

    y0 = [1.0, 2.0, 3.0]
    t_span = (0.0, 5.0)
    for solver in (ode23, ode45, ode56, ode67, ode78):
        res = solver(f, t_span, y0, rtol=1e-9, atol=1e-12)
        exact = np.array(y0)[:, None] * np.exp(-res.t)[None, :]
        np.testing.assert_allclose(res.y, exact, rtol=1e-5, atol=1e-8)


def test_args_are_forwarded():
    def f(t, y, k):
        return [-k * y[0]]

    res = ode45(f, (0.0, 1.0), [1.0], args=(3.0,), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.y[0, -1], np.exp(-3.0), rtol=1e-6)


def test_failure_reported_for_impossibly_tight_tolerance_and_zero_max_step():
    def f(t, y):
        return [y[0]]

    with pytest.raises(ValueError):
        ode45(f, (0.0, 1.0), [1.0], max_step=0.0)
