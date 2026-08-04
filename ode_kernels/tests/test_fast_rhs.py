"""The nogil fast path (compiled RHS via a raw function pointer) must give
the same trajectory as the generic Python-callable path -- it's purely a
performance path, not a different algorithm."""

import numpy as np
import pytest

numba = pytest.importorskip("numba")
from numba import cfunc, types  # noqa: E402

from ode_kernels import ode45


@cfunc(types.void(types.double, types.CPointer(types.double),
                   types.CPointer(types.double), types.intc))
def _decay_cfunc(t, y, dy, n):
    dy[0] = -y[0]


def _decay_py(t, y):
    return [-y[0]]


def test_fast_path_matches_python_path():
    slow = ode45(_decay_py, (0.0, 4.0), [1.0], rtol=1e-9, atol=1e-12)
    fast = ode45(_decay_cfunc, (0.0, 4.0), [1.0], rtol=1e-9, atol=1e-12)

    assert fast.success
    np.testing.assert_allclose(fast.t, slow.t)
    np.testing.assert_allclose(fast.y, slow.y, rtol=1e-10)


def test_fast_path_matches_closed_form():
    fast = ode45(_decay_cfunc, (0.0, 4.0), [1.0], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(fast.y[0], np.exp(-fast.t), atol=1e-7)
