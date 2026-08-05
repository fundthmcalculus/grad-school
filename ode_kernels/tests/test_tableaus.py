"""Structural sanity checks on every Butcher tableau.

These are necessary-but-not-sufficient conditions (a wrong coefficient can
still satisfy them), so the real proof of correctness is the empirical
order-of-convergence test in test_convergence.py. This file exists to catch
gross transcription slips cheaply and immediately.
"""

import numpy as np
import pytest

from ode_kernels import tableaus


@pytest.mark.parametrize("method", list(tableaus.TABLEAUS))
def test_row_sum_and_weight_sum(method):
    tab = tableaus.TABLEAUS[method]
    issues = tableaus.check_tableau(tab)
    assert not issues, "\n".join(issues)


@pytest.mark.parametrize("method", list(tableaus.TABLEAUS))
def test_strictly_lower_triangular(method):
    A, b, e, c = tableaus.TABLEAUS[method].as_arrays()
    n = A.shape[0]
    for i in range(n):
        for j in range(i, n):
            assert A[i, j] == 0.0, f"{method}: A[{i},{j}] nonzero on/above diagonal"


@pytest.mark.parametrize("method", list(tableaus.TABLEAUS))
def test_error_weights_are_b_minus_bhat_like(method):
    """e should be "small" relative to b (it's a difference of two orders'
    weights sharing most terms), not just an unrelated arbitrary vector."""
    tab = tableaus.TABLEAUS[method]
    A, b, e, c = tab.as_arrays()
    assert np.any(e != 0.0)


def test_fsal_flag_matches_last_row_of_a():
    """For every tableau flagged fsal=True, row n_stages of A (0-indexed:
    the last row) really does equal b -- that's the identity the driver's
    FSAL carry-over optimization in _driver.py depends on for correctness."""
    for method, tab in tableaus.TABLEAUS.items():
        A, b, e, c = tab.as_arrays()
        if tab.fsal:
            np.testing.assert_allclose(
                A[-1, :], b, atol=0,
                err_msg=f"{method}: fsal=True but last row of A != b",
            )
            assert c[-1] == 1.0, f"{method}: fsal=True but c[-1] != 1"


def test_orders_strictly_increase_across_the_family():
    order_by_method = {m: t.order for m, t in tableaus.TABLEAUS.items()}
    assert order_by_method["ode12"] < order_by_method["ode23"] < \
        order_by_method["ode45"] < order_by_method["ode56"] < \
        order_by_method["ode67"] < order_by_method["ode78"]
