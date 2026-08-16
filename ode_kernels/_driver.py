"""Adaptive-step driver shared by ode12/23/45/56/67/78.

This is deliberately plain Python: it runs once per *step*, not once per
*stage*, so the only things that matter here are correctness and matching
scipy's ``solve_ivp`` control policy -- there is nothing on this path worth
paying Cython's complexity for. The per-stage arithmetic that actually
dominates runtime lives in ``_rk_kernels.pyx``.

The step-size controller (safety factor, min/max growth factor, the
after-a-rejection damping to <=1) and the initial-step heuristic are
line-for-line the algorithm in ``scipy/integrate/_ivp/rk.py`` and
``common.py``, so a given (fun, tolerances, initial condition) integrates
along essentially the same step sequence scipy's own explicit RK solvers
would -- which is the sense in which these kernels are built to "match
python's normal integrator" rather than merely resemble it.
"""

from __future__ import annotations

import ctypes
from typing import Callable, Optional, Sequence

import numpy as np

from . import _rk_kernels
from . import tableaus as _tableaus
from ._common import (
    EPS,
    MAX_FACTOR,
    MIN_FACTOR,
    SAFETY,
    DenseOutput,
    HermiteSegment,
    OdeResult,
    PolySegment,
    rms_norm,
    select_initial_step,
)

_FAST_RHS_SIGNATURE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
)


def _fast_rhs_address(fun) -> Optional[int]:
    """Return a raw C function pointer address for ``fun`` if it exposes one.

    Recognizes the convention used by ``numba.cfunc``-compiled functions
    (an ``.address`` int attribute) with the required nogil-compatible
    signature ``void(double t, double* y, double* dy, int n)``. Anything
    else falls back to the generic Python-callable path.
    """
    addr = getattr(fun, "address", None)
    if isinstance(addr, int) and addr != 0:
        return addr
    return None


def _make_callable_from_address(address: int, n: int) -> Callable:
    cfun = _FAST_RHS_SIGNATURE(address)

    def _f(t, y):
        y = np.ascontiguousarray(y, dtype=np.float64)
        dy = np.empty(n, dtype=np.float64)
        cfun(
            t,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
        )
        return dy

    return _f


def _broadcast_tol(tol, n: int, name: str) -> np.ndarray:
    arr = np.asarray(tol, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"`{name}` has wrong shape {arr.shape}, expected ({n},)")
    return arr.copy()


def integrate(
    method: str,
    fun: Callable,
    t_span: Sequence[float],
    y0: Sequence[float],
    rtol=1e-3,
    atol=1e-6,
    max_step: float = np.inf,
    first_step: Optional[float] = None,
    dense_output: bool = False,
    t_eval: Optional[Sequence[float]] = None,
    args: Optional[Sequence] = None,
) -> OdeResult:
    tab = _tableaus.TABLEAUS[method]
    A, b, e, c = tab.as_arrays()
    P = tab.P_array()
    n_stages = tab.n_stages
    error_exponent = -1.0 / (tab.error_order + 1)

    t0, tf = float(t_span[0]), float(t_span[1])
    direction = 1.0 if tf >= t0 else -1.0
    y0 = np.atleast_1d(np.asarray(y0, dtype=np.float64))
    n = y0.size

    args = tuple(args) if args else ()
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")

    rtol_arr = _broadcast_tol(rtol, n, "rtol")
    rtol_arr = np.maximum(rtol_arr, 100 * EPS)
    atol_arr = _broadcast_tol(atol, n, "atol")
    if np.any(atol_arr < 0):
        raise ValueError("`atol` must be non-negative.")

    fast_addr = None if args else _fast_rhs_address(fun)
    if fast_addr is not None:
        py_fun = _make_callable_from_address(fast_addr, n)
    else:
        if args:
            py_fun = lambda t, y: np.asarray(fun(t, y, *args), dtype=np.float64)
        else:
            py_fun = lambda t, y: np.asarray(fun(t, y), dtype=np.float64)

    def eval_rhs(t, y):
        if fast_addr is not None:
            return _rk_kernels.eval_fast(fast_addr, t, np.ascontiguousarray(y))
        return py_fun(t, y)

    y = y0.copy()
    t = t0
    f_current = eval_rhs(t0, y)
    nfev = 1

    if t0 == tf:
        h_abs = 0.0
    elif first_step is None:
        h_abs = select_initial_step(
            py_fun,
            t0,
            y,
            tf,
            max_step,
            f_current,
            direction,
            tab.error_order,
            rtol_arr,
            atol_arr,
        )
        nfev += 1
    else:
        h_abs = abs(first_step)
        if h_abs > abs(tf - t0):
            raise ValueError("`first_step` exceeds the integration bounds.")

    ts = [t0]
    ys = [y.copy()]
    segments: list = []
    seg_ts: list = [t0]
    nstep = naccept = nreject = 0
    status = 0
    message = "The solver successfully reached the end of the integration interval."

    while status == 0:
        if direction * (t - tf) >= 0:
            break

        min_step = 10 * abs(np.nextafter(t, direction * np.inf) - t)
        if h_abs > max_step:
            h_abs = max_step
        elif h_abs < min_step:
            h_abs = min_step

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                status = -1
                message = "Required step size became too small."
                break

            h = h_abs * direction
            t_new = t + h
            if direction * (t_new - tf) > 0:
                t_new = tf
            h = t_new - t
            h_abs = abs(h)

            if fast_addr is not None:
                y_new, err, K = _rk_kernels.step_generic_fast(
                    fast_addr, t, y, h, A, b, e, c, n_stages, f_current
                )
            else:
                # py_fun already closes over `args`; step_generic_py must not
                # apply them a second time.
                y_new, err, K = _rk_kernels.step_generic_py(
                    py_fun, t, y, h, A, b, e, c, n_stages, (), f_current
                )
            nfev += n_stages - 1
            nstep += 1

            scale = atol_arr + np.maximum(np.abs(y), np.abs(y_new)) * rtol_arr
            error_norm = rms_norm(err / scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm**error_exponent)
                if step_rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
                step_accepted = True
                naccept += 1
            else:
                h_abs *= max(MIN_FACTOR, SAFETY * error_norm**error_exponent)
                step_rejected = True
                nreject += 1

        if status != 0:
            break

        f_new = K[-1] if tab.fsal else None
        if dense_output or t_eval is not None:
            if P is not None:
                Q = K.T.dot(P)
                segments.append(PolySegment(t, h, y, Q))
            else:
                if f_new is None:
                    f_new = eval_rhs(t_new, y_new)
                    nfev += 1
                segments.append(HermiteSegment(t, h, y, y_new, K[0], f_new))
            seg_ts.append(t_new)

        if tab.fsal:
            f_current = K[-1] if f_new is None else f_new
        elif f_new is not None:
            # Already evaluated for the dense-output segment above; reuse it.
            f_current = f_new
        else:
            f_current = eval_rhs(t_new, y_new)
            nfev += 1

        t = t_new
        y = y_new
        ts.append(t)
        ys.append(y.copy())

    success = status >= 0
    t_arr = np.array(ts)
    y_arr = np.array(ys).T

    sol = None
    if dense_output:
        sol = DenseOutput(np.array(seg_ts), segments)

    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=np.float64)
        dense = DenseOutput(np.array(seg_ts), segments)
        y_eval = dense(t_eval)
        t_arr, y_arr = t_eval, y_eval

    return OdeResult(
        t=t_arr,
        y=y_arr,
        success=success,
        message=message,
        nfev=nfev,
        nstep=nstep,
        naccept=naccept,
        nreject=nreject,
        sol=sol,
        method=method,
    )
