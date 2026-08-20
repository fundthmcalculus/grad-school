"""Provably optimal / provably near-optimal fuzzy control.

Companion to `fis_action.py` and README §8.

The organizing result is that for a control-affine plant the *exact* suboptimality
of any admissible controller is a weighted L2 norm of its deviation from the
optimal law:

    J(x0) - V*(x0)  =  INT_0^inf (u - u*)^T R (u - u*) dt          (Theorem C2)

the integral being taken along the closed loop driven by `u` itself.  So fitting
a fuzzy controller is not a surrogate for optimal control -- minimizing the right
weighted approximation error *is* minimizing the true excess cost, exactly.  Two
consequences drive everything here:

1.  The correct weight is the closed-loop occupation measure, not Lebesgue
    measure on a box.  A fit that is uniformly good over a domain is the wrong
    objective; a fit that is good where the trajectories actually go is the right
    one.
2.  Because the identity is exact and the right-hand side is non-negative, every
    fitted controller comes with a computable certificate of its own
    suboptimality -- no bounding constants, no conservatism.

Ground truth comes from *inverse* optimal control: choose V, f, g, R first and
define the state cost q by the HJB equation.  Then V is the exact value function
and u* the exact optimal law by construction, so approximation quality can be
measured against something known rather than against another approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

af64 = NDArray[np.float64]


# --------------------------------------------------------------------------
# Linear-quadratic benchmark
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LqrProblem:
    """xdot = A x + B u,  J = INT x'Qx + u'Ru dt."""

    a_mat: af64
    b_mat: af64
    q_mat: af64
    r_mat: af64

    def solve(self) -> tuple[af64, af64]:
        """Return (P, K) with u* = -K x and V*(x) = x' P x."""
        p = solve_continuous_are(self.a_mat, self.b_mat, self.q_mat, self.r_mat)
        k = np.linalg.solve(self.r_mat, self.b_mat.T @ p)
        return p, k

    def value(self, x: af64) -> float:
        p, _ = self.solve()
        return float(x @ p @ x)


def tsk_represents_affine_exactly(gain: af64, phi_weights: af64, states: af64) -> float:
    """Max error of a partition-of-unity TSK model reproducing u(x) = -K x.

    Setting every consequent to the same affine map makes the blend collapse:
    sum_i phi_i(x) (-K x) = -K x, because sum_i phi_i = 1.  The membership
    functions cancel out entirely and never enter the result -- which is exactly
    why fuzzy structure buys nothing on an LTI/LQR problem.
    """
    blended = np.einsum("ix,dsx->dx", phi_weights, -gain[:, :, None] * states[None])
    exact = -gain @ states
    return float(np.max(np.abs(blended - exact)))


# --------------------------------------------------------------------------
# Inverse-optimal nonlinear benchmark
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class InverseOptimalScalar:
    """Scalar control-affine plant with an exactly known optimal control law.

    Given xdot = f(x) + g(x) u, a chosen value function V and weight R, the HJB

        0 = q + V_x f - (1/4) V_x^2 g^2 / R

    is *solved for q* rather than for V.  V is then the exact value function and

        u*(x) = -(1/2) R^-1 g(x) V_x(x)

    the exact optimal control, with no approximation anywhere.  The only thing to
    check is that the induced q is non-negative, i.e. that the problem posed is a
    legitimate one.
    """

    f: Callable[[af64], af64]
    g: Callable[[af64], af64]
    v: Callable[[af64], af64]
    v_x: Callable[[af64], af64]
    r: float = 1.0

    def q(self, x: af64) -> af64:
        vx = self.v_x(x)
        return -vx * self.f(x) + 0.25 * vx**2 * self.g(x) ** 2 / self.r

    def u_star(self, x: af64) -> af64:
        return -0.5 * self.g(x) * self.v_x(x) / self.r

    def q_is_valid(self, x: af64) -> bool:
        """A negative state cost would make the 'optimal' problem meaningless."""
        return bool(np.all(self.q(x) >= -1e-12))


def cubic_benchmark() -> InverseOptimalScalar:
    """xdot = -x + u,  V* = x^2 + x^4/2,  u* = -(x + x^3),  q = 3x^2 + 4x^4 + x^6.

    Chosen so the optimal law is genuinely nonlinear -- a linear controller cannot
    represent it, so the fuzzy structure has something to do.
    """
    return InverseOptimalScalar(
        f=lambda x: -x,
        g=lambda x: np.ones_like(x),
        v=lambda x: x**2 + 0.5 * x**4,
        v_x=lambda x: 2.0 * x + 2.0 * x**3,
        r=1.0,
    )


# --------------------------------------------------------------------------
# Closed-loop simulation, cost, and the exact suboptimality certificate
# --------------------------------------------------------------------------
@dataclass
class ClosedLoopResult:
    x0: float
    cost: float
    """Achieved J(x0) = INT q(x) + R u^2 dt."""
    optimal_cost: float
    """V*(x0), known exactly for an inverse-optimal benchmark."""
    control_error_integral: float
    """INT R (u - u*)^2 dt along this trajectory -- the Theorem C2 right-hand side."""
    final_state: float
    stable: bool

    @property
    def gap(self) -> float:
        """Certified excess cost.  Non-negative for any admissible controller."""
        return self.cost - self.optimal_cost

    @property
    def relative_gap(self) -> float:
        return self.gap / max(abs(self.optimal_cost), 1e-15)

    @property
    def identity_residual(self) -> float:
        """|gap - INT R (u-u*)^2 dt|.  Zero verifies Theorem C2 numerically."""
        return abs(self.gap - self.control_error_integral)


def simulate(
    prob: InverseOptimalScalar,
    u_fn: Callable[[af64], af64],
    x0: float,
    t_end: float = 60.0,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> ClosedLoopResult:
    """Integrate the closed loop and accumulate cost, both alongside the state.

    The running cost and the control-error integral are carried as extra ODE
    states rather than quadratured afterwards, so they inherit the integrator's
    error control instead of being limited by the output sampling.
    """

    def rhs(_t: float, z: af64) -> list[float]:
        x = np.array([z[0]])
        u = u_fn(x)
        us = prob.u_star(x)
        xdot = prob.f(x) + prob.g(x) * u
        run = prob.q(x) + prob.r * u**2
        err = prob.r * (u - us) ** 2
        return [float(xdot[0]), float(run[0]), float(err[0])]

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        [x0, 0.0, 0.0],
        rtol=rtol,
        atol=atol,
        method="LSODA",
        dense_output=False,
    )
    xf = float(sol.y[0, -1])
    stable = bool(sol.success and abs(xf) < 1e-4 * max(1.0, abs(x0)))
    return ClosedLoopResult(
        x0=x0,
        cost=float(sol.y[1, -1]),
        optimal_cost=float(prob.v(np.array([x0]))[0]),
        control_error_integral=float(sol.y[2, -1]),
        final_state=xf,
        stable=stable,
    )


# --------------------------------------------------------------------------
# Occupation measure -- the correct fitting weight
# --------------------------------------------------------------------------
def occupation_density(
    prob: InverseOptimalScalar,
    u_fn: Callable[[af64], af64],
    x0_samples: af64,
    grid: af64,
    t_end: float = 40.0,
    bandwidth: float | None = None,
) -> af64:
    """Closed-loop occupation density rho(x) on `grid`, in time units.

    rho(x) dx is the expected time the closed loop spends in dx, averaged over
    the given initial conditions.  Theorem C2 says this -- not Lebesgue measure --
    is the weight under which control-law approximation error converts into
    excess cost.
    """
    if bandwidth is None:
        bandwidth = 0.05 * (grid.max() - grid.min())
    dens = np.zeros_like(grid)
    for x0 in x0_samples:
        sol = solve_ivp(
            lambda _t, z: [
                float(
                    (
                        prob.f(np.array([z[0]]))
                        + prob.g(np.array([z[0]])) * u_fn(np.array([z[0]]))
                    )[0]
                )
            ],
            (0.0, t_end),
            [float(x0)],
            rtol=1e-9,
            atol=1e-11,
            dense_output=True,
            method="LSODA",
        )
        ts = np.linspace(0.0, t_end, 4000)
        xs = sol.sol(ts)[0]
        dt = ts[1] - ts[0]
        # Gaussian-smoothed time histogram; the kernel keeps the weight usable as
        # a quadrature weight rather than a spiky empirical measure.
        dens += (
            dt
            * np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / bandwidth) ** 2).sum(
                axis=1
            )
            / (bandwidth * np.sqrt(2 * np.pi))
        )
    return dens / len(x0_samples)


# --------------------------------------------------------------------------
# Lyapunov / region-of-attraction certificate
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StabilityCertificate:
    verified: bool
    roa_radius: float
    """Largest r with Vdot < 0 for all 0 < |x| <= r; a certified inner estimate."""
    worst_vdot: float
    level_set: float
    """V* value on the ROA boundary -- the certified sublevel set."""


def stability_certificate(
    prob: InverseOptimalScalar,
    u_fn: Callable[[af64], af64],
    x_max: float = 5.0,
    n: int = 20001,
    eps: float = 1e-9,
) -> StabilityCertificate:
    """Certify closed-loop stability by testing Vdot < 0 using V* as a candidate.

    V* is positive definite by construction, so any region where its derivative
    along the *fuzzy* closed loop is negative is a genuine region of attraction.
    The radius reported is the largest symmetric interval on which the test
    passes everywhere, which makes it an inner estimate -- conservative, never
    optimistic.
    """
    x = np.linspace(-x_max, x_max, n)
    nz = np.abs(x) > eps
    xg = x[nz]
    vdot = prob.v_x(xg) * (prob.f(xg) + prob.g(xg) * u_fn(xg))
    ok = vdot < 0.0
    # Walk outwards from the origin and stop at the first failure.  If the very
    # first (innermost) test already fails there is no certified region at all --
    # radius 0, not the full domain.  A controller with u(0) != 0 does not even
    # hold the origin, and must report exactly that.
    order = np.argsort(np.abs(xg))
    radii = np.abs(xg[order])
    failed = np.where(~ok[order])[0]
    if failed.size == 0:
        radius = float(x_max)
    elif failed[0] == 0:
        radius = 0.0
    else:
        radius = float(radii[failed[0] - 1])
    inside = np.abs(xg) <= radius
    return StabilityCertificate(
        verified=bool(radius > 0.0),
        roa_radius=radius,
        worst_vdot=(
            float(np.max(vdot[inside])) if radius > 0.0 and inside.any() else 0.0
        ),
        level_set=float(prob.v(np.array([radius]))[0]),
    )


# --------------------------------------------------------------------------
# Beyond quadratic cost: the Bregman form of Theorem C2
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class InverseOptimalConvex:
    """Scalar plant with a general control cost c(u), convex and differentiable.

    Theorem C2 is usually stated for c(u) = R u^2, but nothing in the derivation
    needs the quadratic -- only that the HJB stationarity condition can be
    inverted.  Writing u* for the minimizer, for ANY convex differentiable c

        ell(x,u) + V_x (f + g u)  =  c(u) - c(u*) - c'(u*)(u - u*)  =  D_c(u || u*)

    the Bregman divergence of c at u*.  Integrating along the closed loop gives
    the general certificate

        J(x0) - V*(x0)  =  INT D_c(u || u*) dt                      (Theorem C2')

    which is still exact, still requires no constants, and is still non-negative
    for free -- D_c >= 0 is precisely convexity of c.  The quadratic case
    D_c = R (u - u*)^2 is one instance.
    """

    f: Callable[[af64], af64]
    g: Callable[[af64], af64]
    v: Callable[[af64], af64]
    v_x: Callable[[af64], af64]
    c: Callable[[af64], af64]
    """Control cost c(u), convex and differentiable."""
    c_prime: Callable[[af64], af64]
    c_prime_inv: Callable[[af64], af64]
    """Inverse of c', used to solve the HJB stationarity condition for u*."""

    def u_star(self, x: af64) -> af64:
        return self.c_prime_inv(-self.g(x) * self.v_x(x))

    def q(self, x: af64) -> af64:
        us = self.u_star(x)
        return -self.c(us) - self.v_x(x) * (self.f(x) + self.g(x) * us)

    def bregman(self, u: af64, us: af64) -> af64:
        return self.c(u) - self.c(us) - self.c_prime(us) * (u - us)

    def q_is_valid(self, x: af64) -> bool:
        return bool(np.all(self.q(x) >= -1e-9))


def quartic_benchmark() -> InverseOptimalConvex:
    """c(u) = u^4: convex, differentiable, and emphatically not quadratic."""
    return InverseOptimalConvex(
        f=lambda x: -x,
        g=lambda x: np.ones_like(x),
        v=lambda x: x**2,
        v_x=lambda x: 2.0 * x,
        c=lambda u: u**4,
        c_prime=lambda u: 4.0 * u**3,
        c_prime_inv=lambda y: np.sign(y) * np.abs(y / 4.0) ** (1.0 / 3.0),
    )


def cosh_benchmark() -> InverseOptimalConvex:
    """c(u) = cosh(u) - 1: convex, and not a polynomial at all."""
    return InverseOptimalConvex(
        f=lambda x: -x,
        g=lambda x: np.ones_like(x),
        v=lambda x: x**2,
        v_x=lambda x: 2.0 * x,
        c=lambda u: np.cosh(u) - 1.0,
        c_prime=lambda u: np.sinh(u),
        c_prime_inv=lambda y: np.arcsinh(y),
    )


def simulate_convex(
    prob: InverseOptimalConvex,
    u_fn: Callable[[af64], af64],
    x0: float,
    t_end: float = 60.0,
) -> dict[str, float]:
    """Closed loop with cost and Bregman integral carried as ODE states."""

    def rhs(_t: float, z: af64) -> list[float]:
        x = np.array([z[0]])
        u = u_fn(x)
        us = prob.u_star(x)
        return [
            float((prob.f(x) + prob.g(x) * u)[0]),
            float((prob.q(x) + prob.c(u))[0]),
            float(prob.bregman(u, us)[0]),
        ]

    sol = solve_ivp(
        rhs, (0.0, t_end), [x0, 0.0, 0.0], rtol=1e-11, atol=1e-13, method="LSODA"
    )
    j = float(sol.y[1, -1])
    v0 = float(prob.v(np.array([x0]))[0])
    breg = float(sol.y[2, -1])
    return {
        "cost": j,
        "optimal_cost": v0,
        "gap": j - v0,
        "bregman_integral": breg,
        "residual": abs((j - v0) - breg),
        "final_state": float(sol.y[0, -1]),
    }


def _invert_monotone(
    c_prime: Callable[[af64], af64], y: af64, u0: af64 | None = None, iters: int = 60
) -> af64:
    """Newton inversion of a strictly increasing c'.

    Used when the stationarity condition c'(u*) = -g V_x has no closed-form
    inverse.  Strict monotonicity (i.e. strict convexity of c) is what makes the
    root unique, so this never has to choose between branches.
    """
    u = np.zeros_like(y) if u0 is None else u0.copy()
    for _ in range(iters):
        h = 1e-7 * np.maximum(np.abs(u), 1.0)
        d = (c_prime(u + h) - c_prime(u - h)) / (2 * h)
        u = u - (c_prime(u) - y) / np.where(np.abs(d) < 1e-12, 1e-12, d)
    return u


def quadratic_quartic_benchmark() -> InverseOptimalConvex:
    """c(u) = u^2 + u^4: the familiar 'penalize large effort harder' cost.

    Unlike a pure quartic this has c''(0) > 0, so u* has bounded slope at the
    origin and the closed loop is not stiff there -- worth noting, because the
    pure quartic's cube-root optimal law is numerically nasty for exactly that
    reason without being any more instructive.
    """
    return InverseOptimalConvex(
        f=lambda x: -x,
        g=lambda x: np.ones_like(x),
        v=lambda x: x**2,
        v_x=lambda x: 2.0 * x,
        c=lambda u: u**2 + u**4,
        c_prime=lambda u: 2.0 * u + 4.0 * u**3,
        c_prime_inv=lambda y: _invert_monotone(
            lambda u: 2.0 * u + 4.0 * u**3, y, u0=y / 2.0
        ),
    )
