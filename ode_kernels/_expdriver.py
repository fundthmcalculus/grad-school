"""Adaptive driver for odeexp: exponential Rosenbrock-Euler + step doubling.

Unlike the RK family, there is no natural embedded lower-order companion for
a single exponential Rosenbrock-Euler step, so error control here uses
classical Richardson step doubling instead: one step of size h is compared
against two steps of size h/2, and the difference -- scaled by 1/(2^p - 1)
for the method's order p=2 -- estimates the local error. This costs 3x the
linear algebra of a bare step per attempt, which is the honest price of
adaptivity for a method whose whole point is to take large steps through
stiff linear behavior; it is not paying for something a proper embedded
pair would get for free, because no such pair is being left on the table.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from . import _expint
from ._common import (
    EPS,
    MAX_FACTOR,
    MIN_FACTOR,
    SAFETY,
    DenseOutput,
    HermiteSegment,
    OdeResult,
    rms_norm,
)

_ORDER = 2
_ERROR_EXPONENT = -1.0 / (_ORDER + 1)
_RICHARDSON_DENOM = 2.0 ** _ORDER - 1.0


def _finite_diff_jacobian(f, t, y, fy):
    n = y.size
    J = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        eps = np.sqrt(EPS) * max(1.0, abs(y[j]))
        y_pert = y.copy()
        y_pert[j] += eps
        J[:, j] = (f(t, y_pert) - fy) / eps
    return J


def _finite_diff_dfdt(f, t, y, fy):
    eps = np.sqrt(EPS) * max(1.0, abs(t))
    return (f(t + eps, y) - fy) / eps


def _exprb2_step(f, jac, t, y, h):
    """One exponential Rosenbrock-Euler step, correctly handling explicit
    t-dependence by autonomizing: treat time as an extra state ``dt/ds = 1``
    and linearize the augmented system, rather than freezing ``f`` at ``t``
    for the whole step. Skipping this (i.e. using the textbook autonomous
    formula ``y + h*phi1(hJ)f`` directly on a non-autonomous ``f``) is a
    common shortcut that quietly degrades accuracy whenever the right-hand
    side varies on a timescale comparable to the step -- the step-size
    controller compensates by shrinking ``h``, which is what first exposed
    this while integrating a forced linear decay in testing.

    Builds one (n+2) x (n+2) augmented matrix
    ``[[hJ, h df/dt, h f], [0, 0, h], [0, 0, 0]]`` and reads the update off
    its matrix exponential's last column, via the same augmented-matrix
    action trick as the pure-autonomous case (now extended by one extra
    row/column to also carry ``d/dt`` exactly).
    """
    n = y.size
    fy = f(t, y)
    J = jac(t, y, fy) if jac is not None else _finite_diff_jacobian(f, t, y, fy)
    ft = _finite_diff_dfdt(f, t, y, fy)

    m = n + 2
    M = np.zeros((m, m), dtype=np.float64)
    M[:n, :n] = h * J
    M[:n, n] = h * ft
    M[:n, n + 1] = h * fy
    M[n, n + 1] = h
    E = _expint.matrix_exponential(np.ascontiguousarray(M))
    y_new = y + E[:n, n + 1]
    return y_new, fy


def odeexp(
    fun: Callable,
    t_span: Sequence[float],
    y0: Sequence[float],
    *,
    jac: Optional[Callable] = None,
    rtol=1e-3,
    atol=1e-6,
    max_step: float = np.inf,
    first_step: Optional[float] = None,
    dense_output: bool = False,
    t_eval: Optional[Sequence[float]] = None,
    args: Optional[Sequence] = None,
) -> OdeResult:
    """Exponential Rosenbrock-Euler integrator with step-doubling error control.

    Linearizes ``y' = f(t, y)`` about the current state via the Jacobian
    ``J`` (analytic, if `jac(t, y, fy) -> ndarray (n, n)` is given, else
    forward finite differences) and integrates the linear part exactly using
    the matrix ``phi1`` function, treating the nonlinear remainder to first
    order. Well suited to mildly-nonlinear stiff systems dominated by fast
    linear decay, where an explicit Runge-Kutta method would need very small
    steps to stay stable. Not intended for large systems -- it forms a dense
    Jacobian and its matrix exponential every step.

    Parameters mirror :func:`ode_kernels.ode45`; see its docstring for the
    common ones. `jac`, if given, is called as ``jac(t, y, fy)`` (`fy` is
    ``fun(t, y)``, passed in case the Jacobian shares work with it) and must
    return the dense Jacobian.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    direction = 1.0 if tf >= t0 else -1.0
    y0 = np.atleast_1d(np.asarray(y0, dtype=np.float64))
    n = y0.size
    args = tuple(args) if args else ()

    if args:
        f = lambda t, y: np.asarray(fun(t, y, *args), dtype=np.float64)
    else:
        f = lambda t, y: np.asarray(fun(t, y), dtype=np.float64)

    rtol_arr = np.maximum(np.broadcast_to(np.asarray(rtol, dtype=float), (n,)), 100 * EPS)
    atol_arr = np.broadcast_to(np.asarray(atol, dtype=float), (n,)).astype(float)

    y = y0.copy()
    t = t0
    nfev = 0
    njev = 0

    if first_step is not None:
        h_abs = abs(first_step)
    elif t0 == tf:
        h_abs = 0.0
    else:
        h_abs = abs(tf - t0) * 1e-3 or 1e-6

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

            y_big, f0 = _exprb2_step(f, jac, t, y, h)
            y_half, _ = _exprb2_step(f, jac, t, y, 0.5 * h)
            y_new, _ = _exprb2_step(f, jac, t + 0.5 * h, y_half, 0.5 * h)
            nfev += 3 * (n + 2) if jac is None else 3 * 2
            njev += 3

            err = (y_new - y_big) / _RICHARDSON_DENOM
            nstep += 1

            scale = atol_arr + np.maximum(np.abs(y), np.abs(y_new)) * rtol_arr
            error_norm = rms_norm(err / scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm ** _ERROR_EXPONENT)
                if step_rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
                step_accepted = True
                naccept += 1
            else:
                h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** _ERROR_EXPONENT)
                step_rejected = True
                nreject += 1

        if status != 0:
            break

        if dense_output or t_eval is not None:
            f_new = f(t_new, y_new)
            nfev += 1
            segments.append(HermiteSegment(t, h, y, y_new, f0, f_new))
            seg_ts.append(t_new)

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
        t_eval_arr = np.asarray(t_eval, dtype=np.float64)
        dense = DenseOutput(np.array(seg_ts), segments)
        y_eval = dense(t_eval_arr)
        t_arr, y_arr = t_eval_arr, y_eval

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
        method="odeexp",
    )
