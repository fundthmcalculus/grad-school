"""Shared, pure-Python plumbing used by every ode_kernels solver.

Kept out of Cython deliberately: this code runs once per integration (initial
step selection) or once per call (result packaging), never in the hot
per-stage loop, so there is nothing here worth paying compilation complexity
for. The hot loop lives in ``_rk_kernels.pyx``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

EPS = np.finfo(float).eps
SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0


def rms_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x) / x.size**0.5)


def select_initial_step(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    y0: np.ndarray,
    t_bound: float,
    max_step: float,
    f0: np.ndarray,
    direction: float,
    order: int,
    rtol: np.ndarray,
    atol: np.ndarray,
) -> float:
    """Hairer-Norsett-Wanner initial step heuristic.

    Identical algorithm to ``scipy.integrate._ivp.common.select_initial_step``
    (Hairer, Norsett & Wanner, "Solving ODEs I", Sec. II.4) so that ode_kernels
    solvers start from the same first step scipy's would, given the same
    problem and tolerances.
    """
    if y0.size == 0:
        return np.inf
    interval_length = abs(t_bound - t0)
    if interval_length == 0.0:
        return 0.0

    scale = atol + np.abs(y0) * rtol
    d0 = rms_norm(y0 / scale)
    d1 = rms_norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, interval_length)

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = rms_norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))

    return min(100 * h0, h1, interval_length, max_step)


@dataclass
class OdeResult:
    """Mirrors the shape of :class:`scipy.integrate.OdeResult`."""

    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    nfev: int
    nstep: int
    naccept: int
    nreject: int
    sol: Optional["DenseOutput"] = None
    t_events: Optional[list] = None
    y_events: Optional[list] = None
    method: str = field(default="")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OdeResult(method={self.method!r}, success={self.success}, "
            f"nfev={self.nfev}, naccept={self.naccept}, nreject={self.nreject}, "
            f"t[-1]={self.t[-1] if len(self.t) else None})"
        )


class DenseOutput:
    """Piecewise interpolant over accepted steps, evaluated like scipy's ``sol``.

    Each segment is either a scipy-style free interpolation polynomial
    (exact match for ``ode23``/``ode45``, which ship a validated ``P`` matrix)
    or, for the higher-order methods that don't have one hand-derived, a
    cubic Hermite fit using the endpoint values and derivatives -- which is
    always available (2 states + 2 derivatives at the segment ends) and at
    least matches the continuous solution to 4th order locally, independent
    of the stepping method's order.
    """

    def __init__(self, ts: np.ndarray, segments: list):
        self.ts = ts
        self.segments = segments

    def __call__(self, t):
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t).astype(float)
        out = np.empty((self.segments[0].n, t.size))
        idx = np.searchsorted(self.ts, t, side="left")
        idx = np.clip(idx, 1, len(self.ts) - 1)
        for j, (ti, seg_idx) in enumerate(zip(t, idx)):
            out[:, j] = self.segments[seg_idx - 1](ti)
        return out[:, 0] if scalar_input else out


class HermiteSegment:
    """Cubic Hermite interpolant on [t0, t0+h] from (y0, f0) and (y1, f1)."""

    __slots__ = ("t0", "h", "y0", "y1", "f0", "f1", "n")

    def __init__(self, t0: float, h: float, y0, y1, f0, f1):
        self.t0 = t0
        self.h = h
        self.y0 = y0
        self.y1 = y1
        self.f0 = f0
        self.f1 = f1
        self.n = y0.size

    def __call__(self, t: float) -> np.ndarray:
        h = self.h
        s = (t - self.t0) / h if h != 0 else 0.0
        s2 = s * s
        s3 = s2 * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        return h00 * self.y0 + h10 * h * self.f0 + h01 * self.y1 + h11 * h * self.f1


class PolySegment:
    """scipy-style free interpolant: y(t0 + s*h) = y0 + h * P^T(s) @ (Q^T @ K)."""

    __slots__ = ("t0", "h", "y0", "Q", "order", "n")

    def __init__(self, t0: float, h: float, y0, Q):
        self.t0 = t0
        self.h = h
        self.y0 = y0
        self.Q = Q  # shape (n, order)
        self.order = Q.shape[1]
        self.n = y0.size

    def __call__(self, t: float) -> np.ndarray:
        s = (t - self.t0) / self.h if self.h != 0 else 0.0
        p = np.array([s ** (k + 1) for k in range(self.order)])
        return self.y0 + self.h * self.Q.dot(p)
