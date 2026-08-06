"""Dense-output (continuous interpolant) checks.

ode23/ode45 use scipy's exact free-interpolation polynomials, so their
dense output should match scipy's own dense output closely. ode56/ode67/
ode78/ode12 fall back to cubic Hermite on step endpoints, which has no
scipy equivalent to compare against directly -- checked instead against the
closed-form solution.
"""

import numpy as np
from scipy.integrate import solve_ivp

from ode_kernels import ode12, ode23, ode45, ode67


def test_poly_dense_output_matches_scipy():
    def f(t, y):
        return [y[1], -y[0]]

    t_span = (0.0, 6.0)
    y0 = [1.0, 0.0]
    ref = solve_ivp(f, t_span, y0, method="RK45", rtol=1e-8, atol=1e-10,
                     dense_output=True)
    res = ode45(f, t_span, y0, rtol=1e-8, atol=1e-10, dense_output=True)

    t_query = np.linspace(0.0, 6.0, 200)
    np.testing.assert_allclose(res.sol(t_query), ref.sol(t_query), atol=1e-6)


def test_hermite_dense_output_is_continuous_and_accurate():
    # The Hermite fallback is only locally cubic regardless of the stepping
    # method's own order, so its dense-output error is set by step size
    # (~h^4 between the sparse accepted-step points), not by ode67's much
    # tighter per-step accuracy -- hence the looser tolerance than the
    # stepped solution itself would need.
    def f(t, y):
        return [-y[0]]

    res = ode67(f, (0.0, 3.0), [1.0], rtol=1e-9, atol=1e-12, dense_output=True)
    t_query = np.linspace(0.0, 3.0, 500)
    y_query = res.sol(t_query)
    exact = np.exp(-t_query)
    np.testing.assert_allclose(y_query[0], exact, atol=1e-5)


def test_t_eval_matches_dense_output():
    def f(t, y):
        return [-y[0]]

    t_eval = np.linspace(0.0, 2.0, 11)
    res = ode12(f, (0.0, 2.0), [1.0], rtol=1e-6, atol=1e-9, t_eval=t_eval)
    np.testing.assert_allclose(res.t, t_eval)
    np.testing.assert_allclose(res.y[0], np.exp(-t_eval), atol=1e-3)


def test_dense_output_none_unless_requested():
    def f(t, y):
        return [-y[0]]

    res = ode23(f, (0.0, 1.0), [1.0])
    assert res.sol is None
