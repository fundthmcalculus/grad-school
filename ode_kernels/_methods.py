"""Public solve_ivp-style entry points: ode12, ode23, ode45, ode56, ode67, ode78."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ._common import OdeResult
from ._driver import integrate

_COMMON_DOC = """
    Parameters
    ----------
    fun : callable
        Right-hand side: ``fun(t, y) -> array_like``, or ``fun(t, y, *args)``
        if `args` is given. For a nogil fast path, `fun` may instead be a
        compiled callable exposing an ``.address`` attribute (e.g. a
        ``numba.cfunc``) with C signature
        ``void(double t, double* y, double* dy, int n)``; `args` is not
        supported together with the fast path.
    t_span : (float, float)
        Interval of integration ``(t0, tf)``.
    y0 : array_like
        Initial state.
    rtol, atol : float or array_like, optional
        Relative/absolute tolerances, combined as
        ``atol + rtol * abs(y)`` exactly as in
        ``scipy.integrate.solve_ivp``. Defaults 1e-3 / 1e-6.
    max_step : float, optional
        Maximum allowed step size.
    first_step : float, optional
        Initial step size; chosen automatically (Hairer-Norsett-Wanner
        heuristic) if omitted.
    dense_output : bool, optional
        If True, `result.sol` is a callable continuous interpolant.
    t_eval : array_like, optional
        Times at which to store the solution, interpolated from the dense
        output. If omitted, the natural (adaptive) step times are returned.
    args : tuple, optional
        Extra positional arguments passed to `fun`.

    Returns
    -------
    OdeResult
        Same fields as :class:`scipy.integrate.OdeResult` plus `nstep`,
        `naccept`, `nreject`.
"""


def _wrap(method: str):
    def solver(
        fun: Callable,
        t_span: Sequence[float],
        y0: Sequence[float],
        *,
        rtol=1e-3,
        atol=1e-6,
        max_step: float = np.inf,
        first_step: Optional[float] = None,
        dense_output: bool = False,
        t_eval: Optional[Sequence[float]] = None,
        args: Optional[Sequence] = None,
    ) -> OdeResult:
        return integrate(
            method, fun, t_span, y0, rtol=rtol, atol=atol, max_step=max_step,
            first_step=first_step, dense_output=dense_output, t_eval=t_eval,
            args=args,
        )

    return solver


ode12 = _wrap("ode12")
ode12.__doc__ = (
    "Heun-Euler embedded Runge-Kutta pair, order 2(1).\n"
    "Cheapest, lowest-order member of the family; mainly useful as a "
    "worked reference / for very smooth, loose-tolerance problems.\n"
    + _COMMON_DOC
)

ode23 = _wrap("ode23")
ode23.__doc__ = (
    "Bogacki-Shampine embedded Runge-Kutta pair, order 3(2).\n"
    "Same Butcher tableau scipy.integrate.solve_ivp(method='RK23') uses.\n"
    + _COMMON_DOC
)

ode45 = _wrap("ode45")
ode45.__doc__ = (
    "Dormand-Prince embedded Runge-Kutta pair, order 5(4).\n"
    "Same Butcher tableau as scipy.integrate.solve_ivp's default "
    "method='RK45'.\n"
    + _COMMON_DOC
)

ode56 = _wrap("ode56")
ode56.__doc__ = (
    "Verner's efficient embedded Runge-Kutta pair, order 6(5) ('Vern6').\n"
    + _COMMON_DOC
)

ode67 = _wrap("ode67")
ode67.__doc__ = (
    "Verner's efficient embedded Runge-Kutta pair, order 7(6) ('Vern7').\n"
    + _COMMON_DOC
)

ode78 = _wrap("ode78")
ode78.__doc__ = (
    "Fehlberg's classical embedded Runge-Kutta pair, order 8(7) ('RKF78').\n"
    + _COMMON_DOC
)
