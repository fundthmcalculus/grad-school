"""odeexp: exponential Rosenbrock-Euler integrator with step-doubling control."""

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp

from ode_kernels import odeexp


def test_exact_on_linear_autonomous_system():
    """An exponential integrator should reproduce a linear autonomous system
    to machine precision in a single step -- it isn't approximating the
    linear part at all, it's solving it exactly via the matrix exponential."""
    A = np.array([[-2.0, 1.0], [0.5, -3.0]])

    def f(t, y):
        return A.dot(y)

    y0 = np.array([1.0, -0.5])
    res = odeexp(f, (0.0, 1.5), y0, jac=lambda t, y, fy: A)
    exact = scipy.linalg.expm(A * res.t[-1]).dot(y0)
    np.testing.assert_allclose(res.y[:, -1], exact, atol=1e-9)


def test_matches_scipy_on_forced_linear_system():
    """A time-dependent forcing term exercises the non-autonomous correction
    (the augmented df/dt term) -- without it this drifts off scipy's answer."""

    def f(t, y):
        return np.array([-50.0 * (y[0] - np.cos(t))])

    t_span = (0.0, 2.0)
    y0 = [0.0]
    ref = solve_ivp(f, t_span, y0, method="Radau", rtol=1e-10, atol=1e-12)
    res = odeexp(f, t_span, y0, rtol=1e-6, atol=1e-9)

    assert res.success
    np.testing.assert_allclose(res.y[0, -1], ref.y[0, -1], atol=1e-4)


def test_finite_difference_jacobian_matches_analytic():
    A = np.diag([-1.0, -10.0, -100.0])

    def f(t, y):
        return A.dot(y)

    y0 = [1.0, 1.0, 1.0]
    t_span = (0.0, 0.5)
    res_fd = odeexp(f, t_span, y0, rtol=1e-8, atol=1e-11)
    res_an = odeexp(f, t_span, y0, jac=lambda t, y, fy: A, rtol=1e-8, atol=1e-11)

    np.testing.assert_allclose(res_fd.y[:, -1], res_an.y[:, -1], rtol=1e-4)


def test_adaptivity_grows_step_on_easy_problem():
    def f(t, y):
        return np.array([-1.0 * y[0]])

    res = odeexp(f, (0.0, 20.0), [1.0], rtol=1e-6, atol=1e-9)
    assert res.success
    steps = np.diff(res.t)
    assert steps[-1] > steps[0] * 2  # step size should have grown substantially
