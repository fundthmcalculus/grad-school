"""Regression guard for a real bug found during development: ode78's error
*estimator* (as opposed to its propagated solution) was wrong in a way the
other test files would never have caught.

The original ``tableaus.py`` built ode78's error vector as ``b8 - b7`` using
a second weight vector transcribed alongside ``b8`` from the same NASA Trick
source, exactly the pattern that works correctly for ode23/45/56/67. It
looked right, passed the row-sum/weight-sum consistency check, and even
passed the fixed-step empirical-order test -- because that test only
exercises ``b`` (the propagated solution), which was fine. What was
actually broken only showed up in *adaptive* integration: the b8-b7
difference for this particular tableau scales like h**2, not h**8 (Fehlberg
b7 alone is only ~2nd-order accurate against this A matrix; the classical
estimator uses a different, 4-stage combination -- see tableaus.py's
comment on `_RKF78_E`). A wildly oversized error estimate made the
controller reject almost everything, so ode78 needed ~200x more function
evaluations than ode56/ode67 to solve a trivial problem -- one order
*higher* while costing two orders of magnitude *more*.

This test would have caught it: a correctly-calibrated higher-order method
should need comparable or fewer accepted steps than a lower-order one on an
easy problem at the same tolerance, not orders of magnitude more.
"""

import numpy as np

from ode_kernels import ode23, ode45, ode56, ode67, ode78


def _decay(t, y):
    return [-y[0]]


def test_higher_order_methods_are_not_pathologically_conservative():
    rtol, atol = 1e-8, 1e-11
    naccept = {}
    for name, solver in [("ode23", ode23), ("ode45", ode45), ("ode56", ode56),
                          ("ode67", ode67), ("ode78", ode78)]:
        res = solver(_decay, (0.0, 5.0), [1.0], rtol=rtol, atol=atol)
        assert res.success
        naccept[name] = res.naccept

    # A smooth, easy problem: every higher-order method in the family should
    # need at most a small multiple of the step count of any lower-order one
    # at the same tolerance -- never 10x-100x more, which is what a broken
    # error estimator (rejecting almost every step) looks like.
    baseline = naccept["ode23"]
    for name in ("ode45", "ode56", "ode67", "ode78"):
        assert naccept[name] <= 3 * baseline, (
            f"{name} took {naccept[name]} accepted steps vs ode23's "
            f"{baseline} -- looks like an oversized error estimate is "
            f"forcing pathologically small steps (naccept={naccept})"
        )


def test_ode78_error_estimator_scales_as_h_pow_8():
    """Directly re-derive the property that broke: halving h should shrink
    the error *estimate* (not just the true error) by roughly 2**8=256."""
    from ode_kernels import _rk_kernels, tableaus

    tab = tableaus.TABLEAUS["ode78"]
    A, b, e, c = tab.as_arrays()
    y = np.array([1.0])

    def one_step_error(h):
        _, err, _ = _rk_kernels.step_generic_py(
            _decay, 0.0, y, h, A, b, e, c, tab.n_stages, (), None
        )
        return abs(err[0])

    e1 = one_step_error(0.1)
    e2 = one_step_error(0.05)
    ratio = e1 / e2
    assert 150 < ratio < 400, (
        f"ode78 error estimate ratio for h -> h/2 was {ratio:.1f}, expected "
        f"~256 (2**8) for an order-7 error estimator; got a value near 4 "
        f"the last time this was ~2nd order instead"
    )
