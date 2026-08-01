"""Two-cart (two-mass-spring) benchmark with a fuzzy least-action controller.

Companion to `fis_control.py` and README §9.

The plant is the classic non-collocated flexible-mode benchmark

    m1 xdd1 = k_lin (x2 - x1) + k_nl (x2 - x1)^3 - c v1 + u
    m2 xdd2 = -k_lin (x2 - x1) - k_nl (x2 - x1)^3 - c v2

with the force applied to cart 1 while cart 2 is the one that must be brought to
rest -- the actuator and the hard-to-control mode are on opposite sides of a
spring, which is what makes it a benchmark rather than an exercise.

Why this problem is a fair test of the framework rather than a rigged one: with
k_nl = 0 and a quadratic cost, Theorem C1 (README §8a) says a partition-of-unity
TSK with affine consequents reproduces the LQR law *exactly* and the membership
functions cancel. Any apparent fuzzy advantage there would be an artifact. What
breaks the degeneracy here is the objective, not the plant: settling time and
peak force are not quadratic, and force saturation makes the optimal law
non-smooth. So the comparison below is only meaningful because the cost is the
one actually asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from math import comb
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize

af64 = NDArray[np.float64]


# --------------------------------------------------------------------------
# Plant
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TwoCart:
    m1: float = 1.0
    m2: float = 1.0
    k_lin: float = 1.0
    k_nl: float = 0.0
    """Cubic spring stiffness. Non-zero makes the plant genuinely nonlinear, so
    Theorem C1's exact-LQR-representation argument no longer applies."""
    damping: float = 0.0
    u_max: float = 1.0

    def rhs(self, z: af64, u: float) -> af64:
        x1, x2, v1, v2 = z
        # Clamp the deflection before cubing.  A diverging closed loop otherwise
        # overflows in d**3 and returns NaN, which silently poisons the ODE
        # solver instead of being reported as the instability it is.
        d = float(np.clip(x2 - x1, -1e3, 1e3))
        spring = self.k_lin * d + self.k_nl * d**3
        u_sat = float(np.clip(u, -self.u_max, self.u_max)) if np.isfinite(u) else 0.0
        return np.array([
            v1,
            v2,
            (spring - self.damping * v1 + u_sat) / self.m1,
            (-spring - self.damping * v2) / self.m2,
        ])

    def linearization(self) -> tuple[af64, af64]:
        """(A, B) about the origin; exact when k_nl = 0."""
        k, c = self.k_lin, self.damping
        a = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-k / self.m1, k / self.m1, -c / self.m1, 0.0],
            [k / self.m2, -k / self.m2, 0.0, -c / self.m2],
        ])
        b = np.array([[0.0], [0.0], [1.0 / self.m1], [0.0]])
        return a, b

    def lqr(self, q_mat: af64, r_val: float) -> af64:
        a, b = self.linearization()
        p = solve_continuous_are(a, b, q_mat, np.array([[r_val]]))
        return (np.array([[1.0 / r_val]]) @ b.T @ p)[0]


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------
@dataclass
class Metrics:
    settling_time: float
    peak_force: float
    energy: float
    """INT u^2 dt."""
    settled: bool

    def score(self, ref: "Metrics") -> float:
        """Equal-weight normalized scalarization against a reference."""
        if not self.settled:
            return float("inf")
        return (
            self.settling_time / ref.settling_time
            + self.peak_force / ref.peak_force
            + self.energy / ref.energy
        ) / 3.0


def simulate(
    plant: TwoCart,
    u_fn: Callable[[af64], float],
    z0: af64,
    t_end: float = 60.0,
    tol: float = 0.02,
) -> tuple[Metrics, af64, af64, af64]:
    """Closed-loop run returning (metrics, times, states, controls).

    Settling time uses the standard 2% criterion on the full state norm, taken as
    the LAST time the trajectory exits the tolerance ball -- not the first time it
    enters. A controller that enters early and then drifts back out has not
    settled, and the first-entry definition would score it as if it had.
    """
    ts = np.linspace(0.0, t_end, 3001)

    # Hard evaluation budget.  A marginally-stable, fast-oscillating closed loop
    # never trips the divergence event but drives the stiff solver to
    # ever-smaller steps, so the run neither fails nor returns.  Exhausting the
    # budget is itself the verdict: a controller that cannot be integrated in a
    # bounded number of steps is not a usable controller.
    budget = {"n": 0}

    class _Exhausted(RuntimeError):
        pass

    def f(_t, z):
        budget["n"] += 1
        if budget["n"] > 400_000:
            raise _Exhausted
        return plant.rhs(np.asarray(z), u_fn(np.asarray(z)))

    # Terminate on divergence.  Without this a destabilizing controller sends the
    # stiff solver into an effectively unbounded step-refinement loop: the run
    # does not fail, it just never returns.  Cutting out at a large bound turns
    # "hangs forever" into "reports not settled", which is the actual answer.
    def diverged(_t, z):
        return float(np.linalg.norm(z)) - 1e4

    diverged.terminal = True
    diverged.direction = 1.0

    try:
        sol = solve_ivp(f, (0.0, t_end), z0, t_eval=ts, rtol=1e-8, atol=1e-10,
                        method="LSODA", events=diverged)
    except _Exhausted:
        return Metrics(float("inf"), float("inf"), float("inf"), False), \
            ts, np.zeros((len(ts), len(z0))), np.zeros(len(ts))
    zs = sol.y.T
    if sol.t_events is not None and len(sol.t_events[0]) > 0:
        return Metrics(float("inf"), float("inf"), float("inf"), False), \
            sol.t, zs, np.zeros(len(zs))
    if len(zs) < len(ts) or not np.all(np.isfinite(zs)):
        return Metrics(float("inf"), float("inf"), float("inf"), False), ts, zs, \
            np.zeros(len(zs))
    us = np.array([
        float(np.clip(u_fn(z), -plant.u_max, plant.u_max)) for z in zs
    ])
    norms = np.linalg.norm(zs, axis=1)
    thresh = tol * max(np.linalg.norm(z0), 1e-12)
    outside = np.where(norms > thresh)[0]
    settled = bool(sol.success and norms[-1] <= thresh)
    t_settle = float(ts[outside[-1]]) if outside.size and settled else (
        0.0 if settled else float("inf")
    )
    energy = float(np.trapezoid(us**2, ts))
    return (
        Metrics(t_settle, float(np.max(np.abs(us))), energy, settled),
        ts, zs, us,
    )


# --------------------------------------------------------------------------
# Open-loop trajectory optimization: the performance ceiling
# --------------------------------------------------------------------------
def optimal_trajectory(
    plant: TwoCart,
    z0: af64,
    n_knots: int = 40,
    t_end: float = 30.0,
    w_time: float = 1.0,
    w_peak: float = 1.0,
    w_energy: float = 1.0,
    u_init: af64 | None = None,
    n_restarts: int = 3,
    maxiter: int = 600,
    seed: int = 0,
) -> tuple[af64, af64, af64, af64]:
    """Direct-transcription optimum for one initial condition.

    Returns (times, states, controls, knot_values).  This is an *open-loop*
    optimum computed with full knowledge of z0, so it is a genuine lower bound on
    what any feedback law can achieve from that z0 -- the right ceiling to measure
    a controller against, and deliberately not attainable by feedback in general.

    Non-smooth terms are replaced by smooth surrogates for the optimizer only
    (log-sum-exp for the peak, a state-norm integral for settling time); the
    metrics reported everywhere else are the true non-smooth ones.
    """
    ts = np.linspace(0.0, t_end, n_knots + 1)
    dt = ts[1] - ts[0]

    def rollout(knots: af64) -> tuple[af64, af64]:
        z = z0.astype(float).copy()
        zs = [z.copy()]
        for i in range(n_knots):
            u = float(knots[i])
            # RK4 with the control held constant across the interval.
            k1 = plant.rhs(z, u)
            k2 = plant.rhs(z + 0.5 * dt * k1, u)
            k3 = plant.rhs(z + 0.5 * dt * k2, u)
            k4 = plant.rhs(z + dt * k3, u)
            z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            zs.append(z.copy())
        return np.array(zs), knots

    def objective(knots: af64) -> float:
        zs, _ = rollout(knots)
        norms = np.linalg.norm(zs, axis=1)
        # Settling surrogate: time-weighted state norm punishes late decay.
        settle = float(np.trapezoid(norms * ts, ts))
        peak = float(np.log(np.sum(np.exp(20.0 * np.abs(knots)))) / 20.0)
        energy = float(np.trapezoid(knots**2, ts[:-1]))
        terminal = 50.0 * float(norms[-1] ** 2)
        return w_time * settle + w_peak * peak + w_energy * energy + terminal

    rng = np.random.default_rng(seed)
    if u_init is None:
        u_init = np.zeros(n_knots)
    u_init = np.asarray(u_init, dtype=float)
    if len(u_init) != n_knots:
        u_init = np.interp(np.linspace(0, 1, n_knots),
                           np.linspace(0, 1, len(u_init)), u_init)
    best = None
    for trial in range(n_restarts):
        start = u_init if trial == 0 else u_init + rng.normal(0, 0.2, n_knots)
        res = minimize(
            objective, np.clip(start, -plant.u_max, plant.u_max), method="L-BFGS-B",
            bounds=[(-plant.u_max, plant.u_max)] * n_knots,
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    zs, knots = rollout(best.x)
    return ts, zs, np.append(knots, knots[-1]), knots


# --------------------------------------------------------------------------
# Multi-input TSK controller with variable-projected consequents
# --------------------------------------------------------------------------
def poly_basis(z: af64, order: int) -> af64:
    """Monomials in the state up to `order`, constant term first.

    Sizes for a 4-state plant: 1, 5, 15, 35 for orders 0..3.  Whatever the order,
    the consequents stay *linear in the parameters*, so variable projection
    (README §3a) still returns the global optimum -- raising the order enlarges
    the model class without costing the one theorem the fit relies on.
    """
    terms = [1.0]
    if order >= 1:
        terms.extend(z)
    for deg in range(2, order + 1):
        for combo in combinations_with_replacement(range(len(z)), deg):
            terms.append(float(np.prod([z[i] for i in combo])))
    return np.array(terms)


def basis_size(n_state: int, order: int) -> int:
    return sum(comb(n_state + d - 1, d) for d in range(order + 1))


@dataclass
class TskController:
    centers: af64
    """(N, 4) rule centres in state space."""
    widths: af64
    """(N, 4) per-rule, per-axis widths."""
    theta: af64 = field(default_factory=lambda: np.zeros(0))
    """(N * basis_size,) consequent coefficients, rule-major."""
    order: int = 1
    """Consequent polynomial order. 1 = affine (classic TSK)."""

    def phi(self, z: af64) -> af64:
        d = (z[None, :] - self.centers) / self.widths
        mu = np.exp(-np.sum(d**2, axis=1))
        s = mu.sum()
        return mu / s if s > 1e-300 else np.full(len(mu), 1.0 / len(mu))

    def regressors(self, z: af64) -> af64:
        return np.outer(self.phi(z), poly_basis(z, self.order)).ravel()

    def n_params(self) -> int:
        return len(self.centers) * basis_size(self.centers.shape[1], self.order)

    def __call__(self, z: af64) -> float:
        return float(self.regressors(np.asarray(z)) @ self.theta)


def place_rules(samples: af64, weights: af64, n_rules: int, seed: int = 0
                ) -> tuple[af64, af64]:
    """Weighted k-means rule placement on the occupation-weighted state samples.

    Placing rules where the closed loop actually spends time is the multi-input
    form of the §8d prescription; placing them on a uniform grid over a box would
    optimize the wrong measure and, in 4 dimensions, waste nearly all of them.
    """
    rng = np.random.default_rng(seed)
    w = weights / weights.sum()
    idx = rng.choice(len(samples), size=n_rules, replace=False, p=w)
    cen = samples[idx].copy()
    for _ in range(60):
        d = ((samples[:, None, :] - cen[None, :, :]) ** 2).sum(axis=2)
        lab = np.argmin(d, axis=1)
        for i in range(n_rules):
            m = lab == i
            if w[m].sum() > 0:
                cen[i] = (samples[m] * w[m, None]).sum(axis=0) / w[m].sum()
    # Widths from the spread of each cluster, floored so no rule collapses to a
    # point (the identifiability condition C5, in its multi-input form).
    wid = np.empty_like(cen)
    d = ((samples[:, None, :] - cen[None, :, :]) ** 2).sum(axis=2)
    lab = np.argmin(d, axis=1)
    global_scale = samples.std(axis=0) + 1e-9
    for i in range(n_rules):
        m = lab == i
        wid[i] = samples[m].std(axis=0) if m.sum() > 4 else global_scale
    wid = np.maximum(wid, 0.15 * global_scale[None, :])
    return cen, wid


def label_state(
    plant: TwoCart,
    z: af64,
    n_knots: int = 15,
    t_end: float = 12.0,
    u_init: af64 | None = None,
) -> float:
    """Expert label u*(z): first control of a short-horizon re-solve from z.

    This is the receding-horizon expert that off-trajectory augmentation needs.
    Only the first control is kept, so the horizon can be much shorter and the
    optimizer much lazier than for the full reference trajectories -- the tail of
    a labelling solve is discarded anyway.  Warm-starting from the parent
    trajectory's controls is what keeps this affordable at hundreds of labels.
    """
    _, _, us, _ = optimal_trajectory(
        plant, np.asarray(z, dtype=float), n_knots=n_knots, t_end=t_end,
        u_init=u_init, n_restarts=1, maxiter=120,
    )
    return float(us[0])


def augment_tube(
    plant: TwoCart,
    samples: af64,
    weights: af64,
    sigma: float = 0.15,
    n_per: int = 1,
    seed: int = 0,
) -> tuple[af64, af64, af64]:
    """Perturb states around the optimal trajectories and re-label them.

    The cheap form of off-trajectory augmentation: it widens the support of the
    training set without knowing where the fitted controller will actually go.
    `sigma` is relative to the per-axis spread of the existing samples.
    """
    rng = np.random.default_rng(seed)
    scale = samples.std(axis=0) + 1e-9
    new_z, new_u, new_w = [], [], []
    for _ in range(n_per):
        for z, w in zip(samples, weights):
            zp = z + rng.normal(0.0, sigma, size=z.shape) * scale
            new_z.append(zp)
            new_u.append(label_state(plant, zp))
            new_w.append(w)
    return np.array(new_z), np.array(new_u), np.array(new_w)


def dagger_states(
    plant: TwoCart,
    u_fn: Callable[[af64], float],
    z0s: list[af64],
    n_per_traj: int = 12,
    t_end: float = 30.0,
) -> tuple[af64, af64]:
    """States actually visited by the CURRENT controller, with occupation weights.

    This is the principled form of augmentation for this framework rather than a
    generic trick.  README §8d requires fitting under the occupation measure of
    the controller being deployed; training on the optimal controller's
    trajectories uses the wrong measure, and iterating fit -> roll out -> relabel
    is the fixed-point iteration that removes the mismatch.
    """
    zs_all, w_all = [], []
    for z0 in z0s:
        _, ts, zs, _ = simulate(plant, u_fn, z0, t_end=t_end)
        if not np.all(np.isfinite(zs)):
            continue
        # Subsample uniformly in time so the weights stay proportional to
        # occupation rather than to the ODE solver's step density.
        idx = np.linspace(0, len(zs) - 1, n_per_traj).astype(int)
        zs_all.append(zs[idx])
        w_all.append(np.full(len(idx), t_end / n_per_traj))
    if not zs_all:
        return np.zeros((0, 4)), np.zeros(0)
    return np.vstack(zs_all), np.concatenate(w_all)


def distribution_shift(closed_loop_states: af64, training_samples: af64) -> float:
    """How far the closed loop wanders from where the controller was fitted.

    Reported as the max over closed-loop states of the distance to the nearest
    training sample, in units of the training set's own per-axis spread.  A
    controller fitted only on optimal trajectories has no data off them, so its
    consequents extrapolate once the closed loop deviates -- and a TSK with more
    rules extrapolates harder, because each rule is supported by fewer samples.
    This is the standard imitation-learning failure, and it is the reason rule
    count and closed-loop quality stop being monotonically related.
    """
    scale = training_samples.std(axis=0) + 1e-9
    a = closed_loop_states / scale
    b = training_samples / scale
    d = np.sqrt(np.maximum(
        ((a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None, :]
         - 2.0 * a @ b.T), 0.0
    ))
    return float(np.max(np.min(d, axis=1)))


def fit_consequents(
    ctrl: TskController, samples: af64, targets: af64, weights: af64,
    ridge: float = 1e-8,
) -> af64:
    """Occupation-weighted variable projection for the consequents.

    Linear in theta, so this is a single weighted least-squares solve with a
    positive-semidefinite Gram -- the global optimum for these rule positions, by
    the same argument as README §3a. No iteration, no local minima.
    """
    phi_mat = np.array([ctrl.regressors(z) for z in samples])
    w = weights / weights.sum()
    gram = (phi_mat * w[:, None]).T @ phi_mat
    rhs = (phi_mat * w[:, None]).T @ targets
    scale = np.trace(gram) / max(gram.shape[0], 1)
    ctrl.theta = np.linalg.solve(gram + ridge * scale * np.eye(gram.shape[0]), rhs)
    return ctrl.theta
