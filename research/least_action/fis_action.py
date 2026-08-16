"""Least-action (Sobolev/H1) formulation of a continuous, differentiable TSK FIS.

Companion code for `research/least_action/README.md`.

The model is an order-`p` TSK system on a scalar input universe X = [x_lo, x_hi]:

    y_c(x; a, b, B) = sum_i phi_i(x) * f_i(x),   f_i(x) = sum_k B[i, k] x^k
    phi_i(x)        = mu_i(x) / sum_j mu_j(x)             (partition of unity)

and the objective is the H1 (Sobolev) action

    S = int_X [ (y_d - y_c)^2 + lam * (y_d' - y_c')^2 ] dx,   lam = ell^2 > 0.

Two facts drive everything here:

1.  For fixed antecedents the action is a convex quadratic in the consequent
    coefficients, so they can be eliminated in closed form (variable
    projection).  All non-convexity lives in the antecedent parameters.
2.  Stationarity is exactly Galerkin orthogonality of the error against the
    rule regressors psi_{i,k} = phi_i * x^k in the H1 inner product.  Sequential
    (residual) rule fitting is therefore exact iff the H1 Gram matrix is block
    diagonal across rules -- which for non-negative mu is the same thing as the
    rules having disjoint support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize

f64 = np.float64
af64 = NDArray[np.float64]

MFKind = Literal["gaussian", "cauchy", "bump"]


# --------------------------------------------------------------------------
# Quadrature
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Quadrature:
    """Gauss-Legendre rule on [lo, hi]; exact for polynomials of degree < 2n."""

    nodes: af64
    weights: af64
    lo: float
    hi: float

    @staticmethod
    def legendre(lo: float, hi: float, n: int = 400) -> "Quadrature":
        t, w = np.polynomial.legendre.leggauss(n)
        half = 0.5 * (hi - lo)
        return Quadrature(
            nodes=half * t + 0.5 * (hi + lo), weights=half * w, lo=lo, hi=hi
        )

    def integrate(self, values: af64) -> f64:
        return f64(np.dot(self.weights, values))


# --------------------------------------------------------------------------
# Membership functions
#
# Every membership function here has the form mu(x) = F(z), z = (x - a) / b, so
# a single shape triple (F, F', F'') generates every derivative needed anywhere
# in this module:
#
#     d mu / dx      =  F'(z) / b
#     d mu / da      = -F'(z) / b
#     d mu / db      = -z F'(z) / b
#     d2 mu / dx da  = -F''(z) / b^2
#     d2 mu / dx db  = -(F'(z) + z F''(z)) / b^2
#
# Adding a membership function therefore means supplying three scalar functions
# of z and nothing else -- in particular no hand-differentiated parameter
# gradients, which is where sign errors would otherwise breed.
# --------------------------------------------------------------------------
def _shape_gaussian(z: af64) -> tuple[af64, af64, af64]:
    e = np.exp(-(z**2))
    return e, -2.0 * z * e, (4.0 * z**2 - 2.0) * e


def _shape_cauchy(z: af64) -> tuple[af64, af64, af64]:
    d = 1.0 + z**2
    return 1.0 / d, -2.0 * z / d**2, (6.0 * z**2 - 2.0) / d**3


def _shape_bump(z: af64) -> tuple[af64, af64, af64]:
    inside = np.abs(z) < 1.0
    zc = np.where(inside, z, 0.0)
    # Floor u away from zero: F = exp(-1/u) underflows to zero long before the
    # 1/u^4 factors overflow, but without a floor u -> 0 produces inf * 0 = nan.
    u = np.maximum(1.0 - zc**2, 1e-8)
    fv = np.where(inside, np.exp(-1.0 / u), 0.0)
    fp = fv * (-2.0 * zc / u**2)
    fpp = fv * (4.0 * zc**2 / u**4 - 2.0 / u**2 - 8.0 * zc**2 / u**3)
    return fv, np.where(inside, fp, 0.0), np.where(inside, fpp, 0.0)


MF_SHAPES: dict[str, Callable[[af64], tuple[af64, af64, af64]]] = {
    "gaussian": _shape_gaussian,
    "cauchy": _shape_cauchy,
    "bump": _shape_bump,
}


def mf_eval(x: af64, a: f64, b: f64, kind: MFKind) -> tuple[af64, af64]:
    """(mu, d mu / dx)."""
    fv, fp, _ = MF_SHAPES[kind]((x - a) / b)
    return fv, fp / b


def mf_param_derivatives(
    x: af64, a: f64, b: f64, kind: MFKind
) -> tuple[af64, af64, af64, af64, af64, af64]:
    """(mu, dmu/dx, dmu/da, dmu/db, d2mu/dxda, d2mu/dxdb)."""
    z = (x - a) / b
    fv, fp, fpp = MF_SHAPES[kind](z)
    return (
        fv,
        fp / b,
        -fp / b,
        -z * fp / b,
        -fpp / b**2,
        -(fp + z * fpp) / b**2,
    )


def mf_gaussian(x: af64, a: f64, b: f64) -> tuple[af64, af64]:
    """exp(-((x-a)/b)^2). C^infinity, strictly positive: never exactly disjoint."""
    return mf_eval(x, a, b, "gaussian")


def mf_cauchy(x: af64, a: f64, b: f64) -> tuple[af64, af64]:
    """Yen & Langari pi-shaped MF 1/(1+((x-a)/b)^2): mu(a)=1, mu(a +- b)=0.5."""
    return mf_eval(x, a, b, "cauchy")


def mf_bump(x: af64, a: f64, b: f64) -> tuple[af64, af64]:
    """Compactly supported C^infinity bump on (a-b, a+b).

    The only kind here that can be made *exactly* disjoint while staying
    differentiable, which is what the orthogonality argument needs.
    """
    return mf_eval(x, a, b, "bump")


MF_TABLE: dict[str, Callable[[af64, f64, f64], tuple[af64, af64]]] = {
    "gaussian": mf_gaussian,
    "cauchy": mf_cauchy,
    "bump": mf_bump,
}


# --------------------------------------------------------------------------
# Rule regressors
# --------------------------------------------------------------------------
def normalized_weights(
    x: af64,
    centers: af64,
    widths: af64,
    kind: MFKind = "gaussian",
    floor: float = 1e-12,
) -> tuple[af64, af64, af64]:
    """Return (phi, dphi, mu_sum) with phi[i] = mu_i / sum_j mu_j.

    `mu_sum` is returned so callers can check the delta-coverage condition
    inf_x sum_j mu_j(x) >= delta > 0, which is what makes y_c well defined and
    keeps the Jacobian bounded.
    """
    mf = MF_TABLE[kind]
    n = len(centers)
    mu = np.empty((n, x.size))
    dmu = np.empty((n, x.size))
    for i in range(n):
        mu[i], dmu[i] = mf(x, f64(centers[i]), f64(widths[i]))
    s = mu.sum(axis=0)
    ds = dmu.sum(axis=0)
    s_safe = np.maximum(s, floor)
    phi = mu / s_safe
    dphi = (dmu * s_safe - mu * ds) / s_safe**2
    return phi, dphi, s


def rule_regressors(
    x: af64,
    centers: af64,
    widths: af64,
    order: int = 1,
    kind: MFKind = "gaussian",
    x_ref: tuple[float, float] | None = None,
) -> tuple[af64, af64, af64]:
    """psi_{i,k}(x) = phi_i(x) * t^k, flattened to rows in rule-major order.

    `t = (x - mid) / half` is the affinely rescaled input.  Rescaling spans the
    identical function space (so the model is unchanged) but keeps the H1 Gram
    matrix well conditioned; with raw x^k on [-15, 15] the Gram condition number
    runs to 1e7+ and the orthogonalization in `sequential_fit` loses most of its
    significant digits.  `x_ref` defaults to the span of `x`.

    Returns (psi, dpsi, mu_sum) with psi.shape == (n_rules * (order + 1), len(x)).
    """
    phi, dphi, s = normalized_weights(x, centers, widths, kind)
    if x_ref is None:
        lo, hi = float(np.min(x)), float(np.max(x))
        x_ref = (0.5 * (lo + hi), max(0.5 * (hi - lo), 1e-12))
    mid, half = x_ref
    t = (x - mid) / half
    n = phi.shape[0]
    rows = n * (order + 1)
    psi = np.empty((rows, x.size))
    dpsi = np.empty((rows, x.size))
    for i in range(n):
        for k in range(order + 1):
            tk = t**k
            dtk = (k * t ** (k - 1) / half) if k > 0 else np.zeros_like(t)
            psi[i * (order + 1) + k] = phi[i] * tk
            dpsi[i * (order + 1) + k] = dphi[i] * tk + phi[i] * dtk
    return psi, dpsi, s


def regressor_jacobian(
    x: af64,
    centers: af64,
    widths: af64,
    order: int = 1,
    kind: MFKind = "gaussian",
    x_ref: tuple[float, float] | None = None,
) -> tuple[af64, af64]:
    """d psi / d p and d psi' / d p for every antecedent parameter p.

    Returns (jac, djac) each of shape (2N, N*(order+1), len(x)), with the
    parameter axis ordered [a_0..a_{N-1}, b_0..b_{N-1}] to match `fit`.

    The chain is: for a parameter p belonging to rule j, write g = d mu_j / dp
    and h = d2 mu_j / dx dp.  Then with Sigma = sum_k mu_k,

        d phi_i / dp  = g (delta_ij - phi_i) / Sigma
        d phi_i'/ dp  = d/dx of the above
                      = [h (delta_ij - phi_i) - g phi_i'] / Sigma
                        - g (delta_ij - phi_i) Sigma' / Sigma^2

    using the fact that x and p are independent, so the x- and p-derivatives
    commute.  The regressor derivatives then follow from psi_ik = phi_i t^k.
    """
    if x_ref is None:
        lo, hi = float(np.min(x)), float(np.max(x))
        x_ref = (0.5 * (lo + hi), max(0.5 * (hi - lo), 1e-12))
    mid, half = x_ref
    t = (x - mid) / half
    n = len(centers)
    w = order + 1

    mu = np.empty((n, x.size))
    dmu = np.empty((n, x.size))
    g_a = np.empty((n, x.size))
    g_b = np.empty((n, x.size))
    h_a = np.empty((n, x.size))
    h_b = np.empty((n, x.size))
    for i in range(n):
        mu[i], dmu[i], g_a[i], g_b[i], h_a[i], h_b[i] = mf_param_derivatives(
            x, f64(centers[i]), f64(widths[i]), kind
        )
    sigma = np.maximum(mu.sum(axis=0), 1e-12)
    dsigma = dmu.sum(axis=0)
    phi = mu / sigma
    dphi = (dmu * sigma - mu * dsigma) / sigma**2

    jac = np.zeros((2 * n, n * w, x.size))
    djac = np.zeros((2 * n, n * w, x.size))
    for slot, (g_all, h_all) in enumerate(((g_a, h_a), (g_b, h_b))):
        for j in range(n):
            g, h = g_all[j], h_all[j]
            p_idx = slot * n + j
            for i in range(n):
                delta = 1.0 if i == j else 0.0
                dphi_i = g * (delta - phi[i]) / sigma
                ddphi_i = (h * (delta - phi[i]) - g * dphi[i]) / sigma - (
                    g * (delta - phi[i]) * dsigma / sigma**2
                )
                for k in range(w):
                    tk = t**k
                    dtk = (k * t ** (k - 1) / half) if k > 0 else np.zeros_like(t)
                    jac[p_idx, i * w + k] = dphi_i * tk
                    djac[p_idx, i * w + k] = ddphi_i * tk + dphi_i * dtk
    return jac, djac


# --------------------------------------------------------------------------
# H1 inner product, Gram matrix, decoupling diagnostics
# --------------------------------------------------------------------------
def h1_gram(psi: af64, dpsi: af64, quad: Quadrature, lam: float) -> af64:
    """G[m, n] = int (psi_m psi_n + lam psi_m' psi_n') dx."""
    w = quad.weights
    return (psi * w) @ psi.T + lam * ((dpsi * w) @ dpsi.T)


def h1_project(
    psi: af64, dpsi: af64, yd: af64, dyd: af64, quad: Quadrature, lam: float
) -> af64:
    """r[m] = <y_d, psi_m>_lam."""
    w = quad.weights
    return psi @ (w * yd) + lam * (dpsi @ (w * dyd))


@dataclass(frozen=True)
class DecouplingReport:
    """How far the rule basis is from being block-orthogonal in H1."""

    coherence: float
    """max normalized |G_mn| between regressors belonging to *different* rules."""
    off_block_energy: float
    """||off-block(G_hat)||_F / ||G_hat||_F, in [0, 1]. Zero == exact decoupling."""
    condition_number: float
    min_coverage: float
    """inf_x sum_j mu_j(x); must be > 0 for the model to be well posed."""

    def sequential_fitting_is_exact(self, tol: float = 1e-10) -> bool:
        return self.off_block_energy < tol


def decoupling_report(
    gram: af64, n_rules: int, order: int, mu_sum: af64 | None = None
) -> DecouplingReport:
    d = np.sqrt(np.clip(np.diag(gram), 1e-300, None))
    ghat = gram / np.outer(d, d)
    block = np.zeros_like(ghat, dtype=bool)
    w = order + 1
    for i in range(n_rules):
        block[i * w : (i + 1) * w, i * w : (i + 1) * w] = True
    off = ghat[~block]
    return DecouplingReport(
        coherence=float(np.max(np.abs(off))) if off.size else 0.0,
        off_block_energy=float(np.linalg.norm(off) / max(np.linalg.norm(ghat), 1e-300)),
        condition_number=float(np.linalg.cond(gram)),
        min_coverage=float(np.min(mu_sum)) if mu_sum is not None else float("nan"),
    )


# --------------------------------------------------------------------------
# Variable projection: eliminate the (linear) consequents
# --------------------------------------------------------------------------
@dataclass
class FisFit:
    centers: af64
    widths: af64
    coeffs: af64
    """shape (n_rules, order + 1); coeffs[i, k] multiplies t^k, t = (x-mid)/half."""
    order: int
    kind: MFKind
    lam: float
    x_ref: tuple[float, float]
    action: float
    l2_error: float
    h1_seminorm_error: float
    report: DecouplingReport
    center_bounds: tuple[float, float] = (-np.inf, np.inf)
    width_bounds: tuple[float, float] = (0.0, np.inf)
    min_gap: float = 0.0

    def __call__(self, x: af64) -> af64:
        psi, _, _ = rule_regressors(
            x, self.centers, self.widths, self.order, self.kind, self.x_ref
        )
        return psi.T @ self.coeffs.ravel()

    def derivative(self, x: af64) -> af64:
        _, dpsi, _ = rule_regressors(
            x, self.centers, self.widths, self.order, self.kind, self.x_ref
        )
        return dpsi.T @ self.coeffs.ravel()


def solve_consequents(
    centers: af64,
    widths: af64,
    yd: af64,
    dyd: af64,
    quad: Quadrature,
    lam: float,
    order: int = 1,
    kind: MFKind = "gaussian",
    x_ref: tuple[float, float] | None = None,
    ridge: float = 1e-9,
) -> tuple[af64, float, af64]:
    """Closed-form H1-optimal consequents for fixed antecedents.

    Returns (theta, reduced_action, gram).  The action is convex in theta, so
    this solve is *global* -- the returned value is the exact minimum of S over
    all consequents for these membership functions.

    `ridge` is relative to the mean Gram eigenvalue.  A ridge is used rather than
    a rank-truncating pseudo-inverse because it makes the map (antecedents ->
    consequents) smooth everywhere; a truncated SVD flips rank as the search
    moves and leaves the reduced action non-differentiable, which corrupts the
    finite-difference Hessian used for the optimality certificate.
    """
    if x_ref is None:
        x_ref = (0.5 * (quad.lo + quad.hi), 0.5 * (quad.hi - quad.lo))
    psi, dpsi, _ = rule_regressors(quad.nodes, centers, widths, order, kind, x_ref)
    gram = h1_gram(psi, dpsi, quad, lam)
    rhs = h1_project(psi, dpsi, yd, dyd, quad, lam)
    scale = float(np.trace(gram)) / max(gram.shape[0], 1)
    theta = np.linalg.solve(gram + ridge * scale * np.eye(gram.shape[0]), rhs)
    yy = quad.integrate(yd**2) + lam * quad.integrate(dyd**2)
    return theta, float(yy - rhs @ theta), gram


def reduced_action_and_gradient(
    centers: af64,
    widths: af64,
    yd: af64,
    dyd: af64,
    quad: Quadrature,
    lam: float,
    order: int = 1,
    kind: MFKind = "gaussian",
    x_ref: tuple[float, float] | None = None,
    ridge: float = 1e-9,
) -> tuple[af64, float, af64]:
    """Reduced action and its exact gradient w.r.t. (centers, widths).

    Returns (theta, action, grad) with grad ordered [a_0..a_{N-1}, b_0..b_{N-1}].

    Derivation.  With M = G + eps I, theta = M^{-1} r and S = <y_d,y_d> - r^T theta,

        dS = -2 (dr)^T theta + theta^T (dM) theta.

    Writing a = (d Psi / dp)^T theta -- which is exactly d y_c / dp holding the
    consequents fixed -- the first term is -2 <y_d, a> and the second contributes
    2 <a, y_c>, so they fuse into

        dS/dp = -2 <e, d y_c / dp>_H1 ,   e = y_d - y_c.

    This is the envelope theorem: because theta is already optimal, its implicit
    dependence on p contributes nothing.  It is also the same H1 pairing that
    expresses stationarity in `galerkin_residual` -- there against the regressors
    themselves (where it vanishes by construction), here against the directions
    the antecedents can move the model.  The trailing term is the derivative of
    the trace-scaled ridge, which is tiny but included so the gradient is exact
    for the objective actually being minimized rather than for an idealized one.
    """
    if x_ref is None:
        x_ref = (0.5 * (quad.lo + quad.hi), 0.5 * (quad.hi - quad.lo))
    psi, dpsi, _ = rule_regressors(quad.nodes, centers, widths, order, kind, x_ref)
    gram = h1_gram(psi, dpsi, quad, lam)
    rhs = h1_project(psi, dpsi, yd, dyd, quad, lam)
    m = gram.shape[0]
    scale = float(np.trace(gram)) / max(m, 1)
    theta = np.linalg.solve(gram + ridge * scale * np.eye(m), rhs)
    yy = quad.integrate(yd**2) + lam * quad.integrate(dyd**2)
    action = float(yy - rhs @ theta)

    jac, djac = regressor_jacobian(quad.nodes, centers, widths, order, kind, x_ref)
    yc = psi.T @ theta
    dyc = dpsi.T @ theta
    e = yd - yc
    de = dyd - dyc

    # a[p] = d y_c / d p at fixed consequents; ad[p] its x-derivative.
    a = np.einsum("pmq,m->pq", jac, theta)
    ad = np.einsum("pmq,m->pq", djac, theta)
    grad = -2.0 * (a @ (quad.weights * e) + lam * (ad @ (quad.weights * de)))

    # d/dp of the trace-scaled ridge: eps = ridge * tr(G) / m.
    tr_dg = 2.0 * (
        np.einsum("pmq,mq,q->p", jac, psi, quad.weights)
        + lam * np.einsum("pmq,mq,q->p", djac, dpsi, quad.weights)
    )
    grad += (ridge / m) * tr_dg * float(theta @ theta)
    return theta, action, grad


def fit(
    yd_fn: Callable[[af64], af64],
    dyd_fn: Callable[[af64], af64],
    n_rules: int,
    x_lo: float,
    x_hi: float,
    lam: float = 1.0,
    order: int = 1,
    kind: MFKind = "gaussian",
    n_quad: int = 400,
    min_gap: float | None = None,
    width_bounds: tuple[float, float] | None = None,
    n_restarts: int = 24,
    polish_rounds: int = 6,
    seed: int | None = 0,
) -> FisFit:
    """Fit a TSK FIS by minimizing the H1 action, with consequents projected out.

    Only the 2 * n_rules antecedent parameters are searched; the (order + 1) *
    n_rules consequent coefficients are recovered exactly at every evaluation.

    `min_gap` enforces a_{i+1} - a_i >= min_gap and `width_bounds` caps b_i.
    Together these are the identifiability constraints: without them the search
    happily drives membership functions on top of one another, the rule basis
    goes rank deficient, and the stationary point it reports has an indefinite
    reduced Hessian -- i.e. it cannot be certified as a local minimum at all.
    Pass `min_gap=0.0` and a wide `width_bounds` to reproduce that failure.
    """
    quad = Quadrature.legendre(x_lo, x_hi, n_quad)
    yd = yd_fn(quad.nodes)
    dyd = dyd_fn(quad.nodes)
    span = x_hi - x_lo
    x_ref = (0.5 * (x_lo + x_hi), 0.5 * span)
    # Regular-fuzzy-partition defaults, keyed to the natural rule pitch.  Centers
    # must stay roughly a pitch apart and widths comparable to the pitch:
    # adjacent rules then overlap enough for delta-coverage but remain
    # distinguishable, which is exactly the condition for the rule basis to have
    # full rank and the reduced Hessian to be meaningful.
    pitch = span / n_rules
    if min_gap is None:
        min_gap = 0.6 * pitch
    if width_bounds is None:
        width_bounds = (0.25 * pitch, 1.5 * pitch)

    # Optimize the centers directly, with the ordering condition
    # a_{i+1} - a_i >= min_gap as an explicit linear inequality.  (Parameterizing
    # by non-negative increments turns ordering into a box constraint but then
    # cannot also bound the largest center, which lets rules walk outside X.)
    def objective(z: af64) -> float:
        _, red, _ = solve_consequents(
            z[:n_rules], z[n_rules:], yd, dyd, quad, lam, order, kind, x_ref
        )
        return red

    bounds = [(x_lo, x_hi)] * n_rules + [width_bounds] * n_rules
    constraints = []
    if n_rules > 1 and min_gap > 0.0:
        a_mat = np.zeros((n_rules - 1, 2 * n_rules))
        for i in range(n_rules - 1):
            a_mat[i, i] = -1.0
            a_mat[i, i + 1] = 1.0
        constraints = [LinearConstraint(a_mat, lb=min_gap, ub=np.inf)]

    def objective_grad(z: af64) -> af64:
        """Exact gradient of the reduced action (envelope theorem, see
        `reduced_action_and_gradient`).

        Supplying this explicitly is not optional.  SciPy's default
        finite-difference step is round-off dominated for this objective, so
        L-BFGS-B exits after zero iterations reporting success, and SLSQP
        returns points worse than it was handed, also reporting success.
        """
        return reduced_action_and_gradient(
            z[:n_rules], z[n_rules:], yd, dyd, quad, lam, order, kind, x_ref
        )[2]

    usable = max(span - min_gap * (n_rules - 1), 0.0)
    z0 = np.concatenate(
        [
            x_lo
            + 0.5 * usable / n_rules
            + np.arange(n_rules) * (usable / n_rules + min_gap),
            np.full(n_rules, np.clip(0.6 * pitch, *width_bounds)),
        ]
    )

    best: tuple[float, af64] | None = None
    rng = np.random.default_rng(seed)
    for trial in range(n_restarts):
        if trial == 0:
            z_try = z0
        else:
            jitter = rng.uniform(-0.3, 0.3, n_rules) * pitch
            z_try = np.concatenate(
                [
                    np.clip(np.sort(z0[:n_rules] + jitter), x_lo, x_hi),
                    rng.uniform(width_bounds[0], width_bounds[1], n_rules),
                ]
            )
            # Re-impose the gap so the start point is feasible.
            for i in range(1, n_rules):
                z_try[i] = max(z_try[i], z_try[i - 1] + min_gap)
            z_try[:n_rules] = np.clip(z_try[:n_rules], x_lo, x_hi)
        res = minimize(
            objective,
            z_try,
            method="SLSQP",
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-12},
        )
        if res.x is None:
            continue
        feasible = np.all(res.x[:n_rules] >= x_lo - 1e-6) and np.all(
            res.x[:n_rules] <= x_hi + 1e-6
        )
        if feasible and (best is None or res.fun < best[0]):
            best = (float(res.fun), res.x)
    if best is None:
        best = (objective(z0), z0)

    # Second-order polish.  A first-order method on a noisy variable-projected
    # objective reliably stalls at saddles: the reduced gradient is small but the
    # projected Hessian still has a negative eigenvalue.  Escape along that
    # direction and re-solve, until no descent direction survives.
    def clamp(z: af64) -> af64:
        z = z.copy()
        z[:n_rules] = np.clip(z[:n_rules], x_lo, x_hi)
        z[n_rules:] = np.clip(z[n_rules:], width_bounds[0], width_bounds[1])
        for i in range(1, n_rules):
            z[i] = min(max(z[i], z[i - 1] + min_gap), x_hi)
        return z

    def refine(z: af64) -> af64:
        """Alternate SLSQP and L-BFGS-B until neither makes progress.

        The two stall in different places on this objective -- SLSQP handles the
        gap constraint but gives up early on the noisy reduced action, L-BFGS-B
        drives the gradient down harder but only sees box bounds -- so
        alternating them gets the reduced gradient far smaller than either alone.
        """
        for _ in range(8):
            before = objective(z)
            for method, kwargs in (
                (
                    "SLSQP",
                    {
                        "constraints": constraints,
                        "options": {"maxiter": 500, "ftol": 1e-14},
                    },
                ),
                (
                    "L-BFGS-B",
                    {"options": {"maxiter": 1000, "ftol": 1e-18, "gtol": 1e-14}},
                ),
            ):
                # Accept only genuine improvements: SLSQP will happily return a
                # worse point than it was handed and report success.
                cand = clamp(
                    minimize(
                        objective,
                        z,
                        method=method,
                        jac=objective_grad,
                        bounds=bounds,
                        **kwargs,
                    ).x
                )
                if objective(cand) < objective(z):
                    z = cand
            if before - objective(z) < 1e-13 * max(abs(before), 1.0):
                break
        return z

    z_best = refine(best[1])
    for _ in range(polish_rounds):
        v = _negative_curvature_direction(z_best, objective, n_rules, span)
        if v is None:
            break
        improved = False
        for t in (0.5, 0.25, 0.1, 0.05):
            for sgn in (+1.0, -1.0):
                z_try = refine(clamp(z_best + sgn * t * span * v))
                if objective(z_try) < objective(z_best) - 1e-12:
                    z_best, improved = z_try, True
                    break
            if improved:
                break
        if not improved:
            break

    centers, widths = z_best[:n_rules], z_best[n_rules:]
    theta, action, gram = solve_consequents(
        centers, widths, yd, dyd, quad, lam, order, kind, x_ref
    )
    psi, dpsi, mu_sum = rule_regressors(quad.nodes, centers, widths, order, kind, x_ref)
    err = yd - psi.T @ theta
    derr = dyd - dpsi.T @ theta
    return FisFit(
        centers=centers,
        widths=widths,
        coeffs=theta.reshape(n_rules, order + 1),
        order=order,
        kind=kind,
        lam=lam,
        x_ref=x_ref,
        action=action,
        l2_error=float(np.sqrt(quad.integrate(err**2))),
        h1_seminorm_error=float(np.sqrt(quad.integrate(derr**2))),
        report=decoupling_report(gram, n_rules, order, mu_sum),
        center_bounds=(x_lo, x_hi),
        width_bounds=width_bounds,
        min_gap=min_gap,
    )


def _negative_curvature_direction(
    z: af64, objective, n_rules: int, span: float, rel_step: float = 1e-2
) -> af64 | None:
    """Unit eigenvector of the most negative curvature, or None if none is found."""
    n = len(z)
    step = rel_step * np.maximum(np.abs(z), 0.1 * span)
    hess = np.empty((n, n))
    try:
        for i in range(n):
            for j in range(i, n):
                ei = np.zeros(n)
                ej = np.zeros(n)
                ei[i], ej[j] = step[i], step[j]
                hess[i, j] = hess[j, i] = (
                    objective(z + ei + ej)
                    - objective(z + ei - ej)
                    - objective(z - ei + ej)
                    + objective(z - ei - ej)
                ) / (4 * step[i] * step[j])
    except (ValueError, np.linalg.LinAlgError):
        return None
    d = np.diag(step)
    eigs, evecs = np.linalg.eigh(0.5 * (d @ hess @ d + (d @ hess @ d).T))
    if eigs[0] >= -1e-8 * max(abs(eigs).max(), 1.0):
        return None
    v = d @ evecs[:, 0]
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else None


# --------------------------------------------------------------------------
# Galerkin / Euler-Lagrange certificate
# --------------------------------------------------------------------------
def galerkin_residual(f: FisFit, yd_fn, dyd_fn, quad: Quadrature) -> af64:
    """<y_d - y_c, psi_m>_lam for every regressor m.

    Stationarity of the action on the FIS manifold is exactly the statement that
    this vector vanishes -- the weak form of `e - lam e'' = 0` tested against the
    tangent space.  It is the computable stand-in for the Euler-Lagrange
    equation, which the FIS cannot satisfy pointwise.
    """
    psi, dpsi, _ = rule_regressors(
        quad.nodes, f.centers, f.widths, f.order, f.kind, f.x_ref
    )
    e = yd_fn(quad.nodes) - psi.T @ f.coeffs.ravel()
    de = dyd_fn(quad.nodes) - dpsi.T @ f.coeffs.ravel()
    return h1_project(psi, dpsi, e, de, quad, f.lam)


def _active_constraint_normals(f: FisFit, tol: float) -> af64:
    """Rows spanning the gradients of the constraints active at the solution."""
    n = len(f.centers)
    rows: list[af64] = []
    for i in range(n):
        for bound in f.center_bounds:
            if abs(f.centers[i] - bound) <= tol * max(1.0, abs(bound)):
                r = np.zeros(2 * n)
                r[i] = 1.0
                rows.append(r)
    for i in range(n):
        for bound in f.width_bounds:
            if abs(f.widths[i] - bound) <= tol * max(1.0, abs(bound)):
                r = np.zeros(2 * n)
                r[n + i] = 1.0
                rows.append(r)
    if f.min_gap > 0.0:
        for i in range(n - 1):
            if abs((f.centers[i + 1] - f.centers[i]) - f.min_gap) <= tol * max(
                1.0, f.min_gap
            ):
                r = np.zeros(2 * n)
                r[i] = -1.0
                r[i + 1] = 1.0
                rows.append(r)
    return np.array(rows) if rows else np.zeros((0, 2 * n))


def optimality_certificate(
    f: FisFit, yd_fn, dyd_fn, quad: Quadrature, rel_step: float = 1e-2
) -> dict[str, float | bool]:
    """Second-order check on the *reduced* (antecedent-only) action.

    Three things matter here and each is easy to get wrong:

    1.  The full-parameter Hessian is never definite -- consequent rescaling and
        rule relabeling see to that -- so the test runs on the variable-projected
        objective, where the consequents have already been globally minimized.
    2.  Fitted solutions routinely sit on the identifiability constraints (width
        caps, the rule-separation gap).  At an active constraint the correct
        second-order condition is definiteness on the *critical cone*, not on the
        whole parameter space, so the Hessian is projected onto the null space of
        the active constraint normals before its eigenvalues are taken.
    3.  A ridge-regularized inner solve differentiated by finite differences has
        a noise floor of roughly eps_f / h^2.  Curvature below that floor is not
        measurable, so the verdict is reported against an estimated floor rather
        than against exact zero -- otherwise numerical dust reads as negative
        curvature and every solution looks like a saddle.
    """
    yd = yd_fn(quad.nodes)
    dyd = dyd_fn(quad.nodes)
    n_rules = len(f.centers)

    def reduced(p: af64) -> float:
        _, red, _ = solve_consequents(
            p[:n_rules], p[n_rules:], yd, dyd, quad, f.lam, f.order, f.kind, f.x_ref
        )
        return red

    p0 = np.concatenate([f.centers, f.widths])
    n = len(p0)
    step = rel_step * np.maximum(np.abs(p0), 0.1 * (quad.hi - quad.lo))

    # First-order condition for the *antecedents*.  Galerkin orthogonality only
    # certifies the consequent block -- variable projection makes that hold by
    # construction, so it says nothing about (a_i, b_i).  Without this the
    # Hessian test is meaningless: curvature is only informative at a
    # stationary point.
    reduced_grad = reduced_action_and_gradient(
        f.centers, f.widths, yd, dyd, quad, f.lam, f.order, f.kind, f.x_ref
    )[2]
    hess = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = step[i]
            ej[j] = step[j]
            hess[i, j] = hess[j, i] = (
                reduced(p0 + ei + ej)
                - reduced(p0 + ei - ej)
                - reduced(p0 - ei + ej)
                + reduced(p0 - ei - ej)
            ) / (4 * step[i] * step[j])
    # Report eigenvalues of the *scaled* Hessian D H D (D = diag(step)); this is
    # the dimensionless curvature and is what definiteness should be judged on,
    # since centers and widths do not share units of sensitivity.
    d = np.diag(step)
    scaled = 0.5 * (d @ hess @ d + (d @ hess @ d).T)

    # Project onto the critical cone: the subspace orthogonal to the normals of
    # the constraints that are active at this solution.
    # The tolerance must be loose enough to catch bounds the outer SLSQP solve
    # only approached to its own convergence tolerance; a bound missed here is
    # reported as spurious negative curvature in an infeasible direction.
    normals = _active_constraint_normals(f, tol=1e-3)
    n_active = normals.shape[0]
    if n_active:
        _, _, vt = np.linalg.svd(normals @ d)
        rank = int(np.sum(np.linalg.svd(normals @ d, compute_uv=False) > 1e-10))
        null_basis = vt[rank:].T
    else:
        null_basis = np.eye(n)
    # KKT first-order residual: the part of the gradient that the active
    # constraints cannot absorb.  At a boundary solution the raw gradient is
    # legitimately non-zero -- it is balanced by the multipliers -- so only the
    # component inside the critical cone measures failure to converge.
    scaled_grad = reduced_grad * step
    kkt_residual = (
        float(np.linalg.norm(null_basis.T @ scaled_grad)) if null_basis.size else 0.0
    )

    projected = (
        null_basis.T @ scaled @ null_basis if null_basis.size else np.zeros((0, 0))
    )
    if projected.size:
        eigs, evecs = np.linalg.eigh(projected)
    else:
        eigs, evecs = np.array([np.inf]), np.zeros((0, 0))

    # Falsification test.  An eigenvalue estimate is only evidence; walking along
    # the most-negative direction and seeing whether the action actually drops is
    # proof.  The probe stays inside a few finite-difference steps on purpose --
    # a long probe that lands in a lower basin says the landscape is multimodal,
    # which is true but says nothing about local optimality here.
    best_descent = 0.0
    if evecs.size:
        direction = d @ (null_basis @ evecs[:, 0])
        nrm = np.linalg.norm(direction)
        if nrm > 0:
            direction = direction / nrm
            base = reduced(p0)
            for t in (0.25, 0.5, 1.0, 2.0):
                for sgn in (+1.0, -1.0):
                    delta = sgn * t * direction * np.linalg.norm(step)
                    trial = p0 + delta
                    if np.all(trial[n_rules:] > 0):
                        # Subtract the first-order term: only curvature-driven
                        # descent falsifies a claimed local minimum.  A drop
                        # explained by a non-zero gradient just means the solve
                        # has not converged.
                        best_descent = max(
                            best_descent, base + reduced_grad @ delta - reduced(trial)
                        )

    # Noise floor: the inner solve is accurate to ~ridge in relative terms, and
    # the second-order mixed difference divides that by 4 h_i h_j.
    f0 = abs(reduced(p0))
    noise = 1e-9 * max(f0, 1e-12) + 1e-14
    floor = noise / (np.min(step) ** 2) * np.max(step) ** 2

    grad = galerkin_residual(f, yd_fn, dyd_fn, quad)
    psi, dpsi, _ = rule_regressors(
        quad.nodes, f.centers, f.widths, f.order, f.kind, f.x_ref
    )
    gram_eigs = np.linalg.eigvalsh(h1_gram(psi, dpsi, quad, f.lam))
    return {
        "min_eigenvalue": float(eigs.min()),
        "max_eigenvalue": (
            float(eigs.max()) if np.isfinite(eigs).all() else float("inf")
        ),
        "n_active_constraints": int(n_active),
        "critical_cone_dim": int(null_basis.shape[1]) if null_basis.size else 0,
        "noise_floor": float(floor),
        "n_negative_directions": int(np.sum(eigs < -floor)),
        "verified_descent": float(best_descent),
        "is_saddle": bool(best_descent > 10.0 * noise),
        "positive_definite": bool(eigs.min() > floor),
        "positive_semidefinite": bool(eigs.min() > -floor),
        "gram_min_eigenvalue": float(gram_eigs.min()),
        "gram_condition": float(gram_eigs.max() / max(gram_eigs.min(), 1e-300)),
        "galerkin_residual_inf_norm": float(np.max(np.abs(grad))),
        "reduced_gradient_norm": float(np.linalg.norm(reduced_grad * step)),
        "kkt_residual": kkt_residual,
        "stationary": bool(kkt_residual < 1e-4 * max(abs(reduced(p0)), 1.0)),
        "residual_h1_norm": float(
            np.sqrt(f.l2_error**2 + f.lam * f.h1_seminorm_error**2)
        ),
    }


# --------------------------------------------------------------------------
# Sequential (residual) rule identification
# --------------------------------------------------------------------------
def sequential_fit(
    centers: af64,
    widths: af64,
    yd: af64,
    dyd: af64,
    quad: Quadrature,
    lam: float,
    order: int = 1,
    kind: MFKind = "gaussian",
    x_ref: tuple[float, float] | None = None,
    orthogonalize: bool = False,
) -> tuple[af64, float]:
    """Fit rules one at a time against the running residual.

    With `orthogonalize=False` this is the naive greedy scheme: each rule sees
    only what earlier rules left behind.  It reproduces the joint solution
    exactly iff the H1 Gram is block diagonal.  With `orthogonalize=True` the
    regressor blocks are H1-Gram-Schmidt'd first (orthogonal least squares), and
    the greedy scheme is exact for *any* overlap -- at the cost of consequents
    that are no longer individually interpretable until back-substituted.
    """
    if x_ref is None:
        x_ref = (0.5 * (quad.lo + quad.hi), 0.5 * (quad.hi - quad.lo))
    psi, dpsi, _ = rule_regressors(quad.nodes, centers, widths, order, kind, x_ref)
    w = order + 1
    n_rules = len(centers)
    theta = np.zeros(psi.shape[0])
    res, dres = yd.copy(), dyd.copy()

    basis, dbasis = psi.copy(), dpsi.copy()
    transform = np.eye(psi.shape[0])
    if orthogonalize:
        for i in range(n_rules):
            sl = slice(i * w, (i + 1) * w)
            for j in range(i):
                sj = slice(j * w, (j + 1) * w)
                gjj = h1_gram(basis[sj], dbasis[sj], quad, lam)
                gji = (basis[sj] * quad.weights) @ basis[sl].T + lam * (
                    (dbasis[sj] * quad.weights) @ dbasis[sl].T
                )
                c = np.linalg.solve(gjj + 1e-9 * np.trace(gjj) / w * np.eye(w), gji)
                basis[sl] -= c.T @ basis[sj]
                dbasis[sl] -= c.T @ dbasis[sj]
                transform[sj, sl] -= c

    for i in range(n_rules):
        sl = slice(i * w, (i + 1) * w)
        g = h1_gram(basis[sl], dbasis[sl], quad, lam)
        r = h1_project(basis[sl], dbasis[sl], res, dres, quad, lam)
        t = np.linalg.solve(g + 1e-9 * np.trace(g) / w * np.eye(w), r)
        theta[sl] = t
        res = res - basis[sl].T @ t
        dres = dres - dbasis[sl].T @ t

    if orthogonalize:
        theta = transform @ theta
    fitted = psi.T @ theta
    dfitted = dpsi.T @ theta
    action = quad.integrate((yd - fitted) ** 2) + lam * quad.integrate(
        (dyd - dfitted) ** 2
    )
    return theta, float(action)


# --------------------------------------------------------------------------
# Classifier side: disjoint output sets, max == sum, annealed defuzzification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OutputPartition:
    """Consequent fuzzy sets on Y = [0, 1] with pairwise disjoint supports."""

    cores: af64
    half_widths: af64

    @staticmethod
    def uniform(n: int, gap: float = 0.0) -> "OutputPartition":
        edges = np.linspace(0.0, 1.0, n + 1)
        cores = 0.5 * (edges[:-1] + edges[1:])
        hw = np.full(n, 0.5 * (edges[1] - edges[0]) * (1.0 - gap))
        return OutputPartition(cores=cores, half_widths=hw)

    def membership(self, y: af64) -> af64:
        """Triangular B_i(y), normal (peak 1) and pairwise disjoint."""
        z = (y[None, :] - self.cores[:, None]) / self.half_widths[:, None]
        return np.clip(1.0 - np.abs(z), 0.0, None)

    def supports_are_disjoint(self, tol: float = 1e-12) -> bool:
        lo = self.cores - self.half_widths
        hi = self.cores + self.half_widths
        idx = np.argsort(lo)
        return bool(np.all(hi[idx][:-1] <= lo[idx][1:] + tol))


def aggregate_max(alpha: af64, part: OutputPartition, y: af64) -> af64:
    """Mamdani aggregation B'(y) = max_i alpha_i ^ B_i(y) (Goedel t-norm)."""
    return np.max(np.minimum(alpha[:, None], part.membership(y)), axis=0)


def aggregate_sum(alpha: af64, part: OutputPartition, y: af64) -> af64:
    """Summation surrogate sum_i alpha_i ^ B_i(y); equals `aggregate_max` exactly
    when the supports are disjoint, and is smooth in alpha either way."""
    return np.sum(np.minimum(alpha[:, None], part.membership(y)), axis=0)


def defuzz_mom(alpha: af64, part: OutputPartition) -> float:
    """Mean of maxima over disjoint output sets: the core centre of argmax_i alpha_i.

    Piecewise constant in alpha, hence discontinuous at ties -- this is exactly
    what the annealed surrogate below repairs.
    """
    return float(part.cores[int(np.argmax(alpha))])


def defuzz_annealed(alpha: af64, part: OutputPartition, beta: float = 1.0) -> float:
    """sum alpha_i^beta c_i / sum alpha_i^beta.

    beta = 1 reproduces height/centroid defuzzification of the summed aggregate;
    beta -> infinity reproduces MOM.  Smooth in alpha for every finite beta.
    """
    w = np.power(np.clip(alpha, 0.0, None), beta)
    tot = w.sum()
    if tot <= 0.0:
        return float(np.mean(part.cores))
    return float(w @ part.cores / tot)


def mom_gap_bound(alpha: af64, part: OutputPartition, beta: float) -> float:
    """Bound on |defuzz_annealed - defuzz_mom| from the firing-strength margin.

    With r = alpha_(2) / alpha_(1) the runner-up ratio and D the output diameter,
    the error is at most (N - 1) * D * r^beta / (1 + (N - 1) r^beta).
    """
    s = np.sort(np.clip(alpha, 0.0, None))[::-1]
    if s[0] <= 0.0:
        return float(part.cores.max() - part.cores.min())
    r = float(s[1] / s[0]) if len(s) > 1 else 0.0
    d = float(part.cores.max() - part.cores.min())
    k = (len(s) - 1) * r**beta
    return d * k / (1.0 + k)
