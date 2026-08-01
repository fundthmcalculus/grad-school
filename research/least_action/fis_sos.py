"""Sum-of-squares Lyapunov certificates for a rational (pi-MF) TSK closed loop.

Companion to README §9e and §12.

Why this file can exist at all: with pi-shaped membership functions
mu_i = 1/D_i, D_i = 1 + sum_j ((z_j - c_ij)/w_ij)^2 a degree-2 polynomial, the
normalized weights are rational,

    phi_i = (prod_{k != i} D_k) / (sum_k prod_{m != k} D_m) = N_i / Q,

so with polynomial consequents the whole control law is a ratio of polynomials

    u(z) = P(z) / Q(z),     P = sum_i N_i f_i,     Q = sum_k prod_{m != k} D_m.

Q is a sum of products of polynomials each >= 1, so Q >= N > 0 everywhere -- the
delta-coverage condition (constraint C2) holds automatically and, more to the
point, multiplying an inequality through by Q preserves its sign. The Lyapunov
condition on the closed loop xdot = f + g u is therefore

    grad V . f + (grad V . g) P / Q  <  0
    <=>  Q (grad V . f) + (grad V . g) P  <  0                       (polynomial)

a polynomial inequality, which is a sum-of-squares feasibility problem. A
Gaussian membership function is transcendental and admits none of this.

The certificate is only claimed where the input is unsaturated, since the plant
is nonlinear (clipped) outside that region; `unsaturated_radius` reports where
that assumption holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import cvxpy as cp
import numpy as np
import sympy as sp
from numpy.typing import NDArray

af64 = NDArray[np.float64]

# Optional side-channel so `verify_sos_exact` can retrieve the solver's Gram
# matrices and the exact polynomial data without duplicating the assembly.
_CAPTURE: dict | None = None


def capture(store: dict | None) -> None:
    """Enable/disable capture of the next certify_roa call's internals."""
    global _CAPTURE
    _CAPTURE = store


# --------------------------------------------------------------------------
# Exact rational form of the controller
# --------------------------------------------------------------------------
@dataclass
class RationalController:
    """u(z) = P(z) / Q(z), both sympy polynomials in the state symbols."""

    symbols: list[sp.Symbol]
    numerator: sp.Expr
    denominator: sp.Expr

    def evaluate(self, z: af64) -> float:
        sub = dict(zip(self.symbols, z))
        return float(self.numerator.subs(sub) / self.denominator.subs(sub))

    def degrees(self) -> tuple[int, int]:
        return (sp.Poly(self.numerator, *self.symbols).total_degree(),
                sp.Poly(self.denominator, *self.symbols).total_degree())


def to_rational(ctrl, n_state: int = 4) -> RationalController:
    """Build the exact P/Q form of a pi-MF TSK controller.

    Mirrors `TskController.regressors` symbolically rather than re-deriving it,
    so any divergence between the two is a bug in one of them and shows up
    immediately in the numerical agreement check.
    """
    if ctrl.mf != "pi":
        raise ValueError("rational form requires pi-shaped membership functions")
    z = sp.symbols(f"z0:{n_state}", real=True)
    n_rules = len(ctrl.centers)

    dens = []
    for i in range(n_rules):
        d = sp.Integer(1)
        for j in range(n_state):
            d += ((z[j] - sp.Float(ctrl.centers[i][j]))
                  / sp.Float(ctrl.widths[i][j])) ** 2
        dens.append(sp.expand(d))

    # N_i = prod_{k != i} D_k  and  Q = sum_i N_i
    n_terms = []
    for i in range(n_rules):
        prod = sp.Integer(1)
        for k in range(n_rules):
            if k != i:
                prod *= dens[k]
        n_terms.append(sp.expand(prod))
    q_poly = sp.expand(sum(n_terms))

    # Consequent monomials, in the same order as fis_twocart.poly_basis.
    def basis(order: int) -> list[sp.Expr]:
        terms: list[sp.Expr] = [sp.Integer(1)]
        if order >= 1:
            terms.extend(z)
        for deg in range(2, order + 1):
            for combo in combinations_with_replacement(range(n_state), deg):
                term = sp.Integer(1)
                for idx in combo:
                    term *= z[idx]
                terms.append(term)
        return terms

    b = basis(ctrl.order)
    theta = np.asarray(ctrl.theta).reshape(n_rules, len(b))
    p_poly = sp.Integer(0)
    for i in range(n_rules):
        f_i = sum(sp.Float(theta[i][m]) * b[m] for m in range(len(b)))
        p_poly += n_terms[i] * f_i
    p_poly = sp.expand(p_poly)

    # Zero P's constant term.  fit_consequents enforces u(0) = 0 as an exact
    # linear constraint, so P(0) = Q(0) u(0) is zero by construction -- but in
    # floating point it comes out at ~1e-18, and sympy faithfully carries that
    # as a genuine constant coefficient.  It then propagates into a degree-1
    # term in the Lyapunov polynomial, which no SOS basis starting at degree 1
    # can represent, and the SDP is reported infeasible for a quantity that is
    # numerically zero.  Dropping it makes the symbolic object represent the
    # constrained controller rather than its float residue.
    const = p_poly.subs({zi: 0 for zi in z})
    if const != 0:
        p_poly = sp.expand(p_poly - const)
    return RationalController(list(z), p_poly, q_poly)


# --------------------------------------------------------------------------
# SOS feasibility
# --------------------------------------------------------------------------
def _monomials(z: list[sp.Symbol], degree: int, min_degree: int = 0) -> list[sp.Expr]:
    """Monomials of total degree in [min_degree, degree].

    `min_degree=1` drops the constant term.  That is not an optimization: the
    S-procedure target W - sigma*(rho - V) evaluates at the origin to
    -sigma(0)*rho, which is strictly negative whenever sigma(0) > 0, so no such
    polynomial is ever SOS.  The multiplier must vanish at the origin, and the
    SOS basis must then also start at degree 1 -- W itself has no constant or
    linear term because grad V(0) = 0.  Allowing the constant lets the SDP
    satisfy the constraint only by driving sigma(0) to zero, which it does, and
    the Gram comes back sitting exactly on the PSD boundary as a result.
    """
    out: list[sp.Expr] = [sp.Integer(1)] if min_degree == 0 else []
    for d in range(max(1, min_degree), degree + 1):
        for combo in combinations_with_replacement(range(len(z)), d):
            term = sp.Integer(1)
            for i in combo:
                term *= z[i]
            out.append(term)
    return out


def _coeff_map(expr: sp.Expr, z: list[sp.Symbol]) -> dict:
    poly = sp.Poly(sp.expand(expr), *z)
    return {m: float(c) for m, c in zip(poly.monoms(), poly.coeffs())}


@dataclass(frozen=True)
class SosCertificate:
    feasible: bool
    v_matrix: af64 | None
    """V(z) = z' V_matrix z, positive definite if feasible."""
    min_eig_v: float
    min_eig_gram: float
    solver_status: str
    numerator_degree: int
    denominator_degree: int
    sdp_size: int

    def summary(self) -> str:
        if not self.feasible:
            return f"INFEASIBLE ({self.solver_status})"
        return (f"CERTIFIED (V min eig {self.min_eig_v:.3e}, "
                f"SOS Gram min eig {self.min_eig_gram:.3e})")


def certify_quadratic_lyapunov(
    rat: RationalController,
    a_mat: af64,
    b_vec: af64,
    epsilon: float = 1e-4,
    solver: str = "CLARABEL",
) -> SosCertificate:
    """Search for a quadratic V with Q (grad V . f) + (grad V . g) P SOS-negative.

    For a linear plant f = A z, g = B, both the constraint and V are linear in
    the unknown entries of the V matrix, so this is a genuine SDP rather than a
    bilinear search -- no alternation, no local minima, and infeasibility is a
    proof that no quadratic V certifies this closed loop, not a failure to find
    one.

    The strictness margin is -epsilon * ||z||^2 * Q rather than -epsilon * ||z||^2
    so that the two sides have matching degree; Q > 0 makes this a valid
    tightening.
    """
    z = rat.symbols
    n = len(z)
    v_sym = cp.Variable((n, n), symmetric=True)

    zv = sp.Matrix(z)
    # Symbolic V with unknown coefficients handled by linearity: build the
    # constraint polynomial once per basis matrix E_ab and superpose.
    grad_terms = []
    for a in range(n):
        for b_ in range(a, n):
            e = sp.zeros(n, n)
            e[a, b_] += 1
            e[b_, a] += 1
            v_e = (zv.T @ e @ zv)[0, 0]
            grad = sp.Matrix([sp.diff(v_e, zi) for zi in z])
            f_expr = (sp.Matrix(a_mat.tolist()) @ zv)
            gv_f = (grad.T @ f_expr)[0, 0]
            gv_g = (grad.T @ sp.Matrix(b_vec.reshape(-1, 1).tolist()))[0, 0]
            expr = sp.expand(rat.denominator * gv_f + gv_g * rat.numerator)
            grad_terms.append(((a, b_), _coeff_map(expr, z)))

    eps_expr = _coeff_map(
        sp.expand(-epsilon * sum(zi**2 for zi in z) * rat.denominator), z
    )

    total_deg = max(
        max((sum(m) for cm in [t[1] for t in grad_terms] for m in cm), default=0),
        max((sum(m) for m in eps_expr), default=0),
    )
    half = (total_deg + 1) // 2
    mons = _monomials(z, half)
    mon_exp = [tuple(sp.Poly(m, *z).monoms()[0]) for m in mons]
    k = len(mons)
    gram = cp.Variable((k, k), symmetric=True)

    # Match coefficients: -(constraint) - eps*||z||^2*Q  ==  mons' Gram mons
    target: dict = {}
    for mono, c in eps_expr.items():
        target[mono] = target.get(mono, 0.0) + c

    lhs_terms: dict = {}
    for (a, b_), cm in grad_terms:
        for mono, c in cm.items():
            lhs_terms.setdefault(mono, []).append((a, b_, -c))

    gram_mono: dict = {}
    for i in range(k):
        for j in range(k):
            mono = tuple(x + y for x, y in zip(mon_exp[i], mon_exp[j]))
            gram_mono.setdefault(mono, []).append((i, j))

    all_monos = set(lhs_terms) | set(target) | set(gram_mono)

    # Maximize the SOS margin rather than minimizing trace(Gram).  Minimizing the
    # trace drives every variable toward zero, which fights the strictness
    # constraints and returns a Gram that is only marginally PSD -- an invalid
    # decomposition that still reports "optimal".  V >= I fixes the scale from
    # below so the margin is meaningful rather than absorbable by rescaling.
    margin = cp.Variable()
    constraints = [
        gram >> margin * np.eye(k),
        v_sym >> np.eye(n),
        cp.trace(v_sym) <= 1e4 * n,
        margin <= 1e3,
    ]
    for mono in all_monos:
        lhs = sum(coef * v_sym[a, b_] for a, b_, coef in lhs_terms.get(mono, []))
        lhs = lhs + target.get(mono, 0.0)
        rhs = sum(gram[i, j] for i, j in gram_mono.get(mono, []))
        constraints.append(lhs == rhs)
    prob = cp.Problem(cp.Maximize(margin), constraints)
    try:
        prob.solve(solver=solver, verbose=False, max_iters=20000)
    except Exception as exc:  # solver failure is not a proof of infeasibility
        return SosCertificate(False, None, 0.0, 0.0, f"solver error: {exc}",
                              *rat.degrees(), k)

    if prob.status not in ("optimal", "optimal_inaccurate") or v_sym.value is None:
        return SosCertificate(False, None, 0.0, 0.0, str(prob.status),
                              *rat.degrees(), k)
    v_val = np.array(v_sym.value)
    g_val = 0.5 * (np.array(gram.value) + np.array(gram.value).T)
    eig_v = float(np.linalg.eigvalsh(0.5 * (v_val + v_val.T)).min())
    eig_g = float(np.linalg.eigvalsh(g_val).min())
    # A certificate is only a certificate if the Gram is ACTUALLY positive
    # semidefinite.  The solver reporting "optimal" is not evidence of that:
    # an interior-point method routinely returns a marginally indefinite Gram
    # and still calls it optimal, and accepting that would be claiming a proof
    # from a decomposition that does not exist.
    scale = max(float(np.max(np.abs(g_val))), 1e-12)
    feasible = bool(eig_g > -1e-9 * scale and eig_v > 0.0)
    return SosCertificate(
        feasible=feasible,
        v_matrix=v_val if feasible else None,
        min_eig_v=eig_v,
        min_eig_gram=eig_g,
        solver_status=f"{prob.status}, margin={float(margin.value):.3e}"
                     if margin.value is not None else str(prob.status),
        numerator_degree=rat.degrees()[0],
        denominator_degree=rat.degrees()[1],
        sdp_size=k,
    )


def unsaturated_radius(ctrl, u_max: float, x_max: float = 4.0,
                       n: int = 60000, seed: int = 0) -> float:
    """Largest r with |u(z)| <= u_max for all sampled ||z|| <= r.

    The SOS certificate uses the rational vector field, which is the true plant
    only where the input is unsaturated.  This reports the ball on which that
    assumption is sampled to hold, so the certified region must be intersected
    with it.  Sampling makes this an estimate, not a bound -- it is the one
    non-rigorous step in the pipeline and is flagged as such.
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, len(ctrl.centers[0])))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    radii = rng.uniform(0.0, x_max, size=n)
    pts = d * radii[:, None]
    bad = [r for p, r in zip(pts, radii) if abs(ctrl(p)) > u_max]
    return float(min(bad)) if bad else float(x_max)


def certify_roa(
    rat: RationalController,
    a_mat: af64,
    b_vec: af64,
    p_lyap: af64,
    rho: float,
    epsilon: float = 1e-6,
    sigma_degree: int = 2,
    solver: str = "CLARABEL",
) -> tuple[bool, float, float]:
    """Certify Vdot < 0 on the sublevel set {z' P z <= rho}, V fixed.

    Returns (certified, min_eig_gram, relative_violation) where the last is
    |min_eig| / max|Gram|.  The relative figure is the one that matters when
    judging a near-miss: SCS is a first-order solver and does not reliably reach
    better than ~1e-6 relative on a problem this size, so a violation below that
    is evidence about the solver rather than about the polynomial.

    A *global* SOS condition is the wrong question here: the rational field is
    only the true plant where the input is unsaturated, and no quadratic V
    certifies the whole of R^4 anyway (verified -- `certify_quadratic_lyapunov`
    returns infeasible with margin -4.8e-3). The right object is a sublevel set,
    obtained by the S-procedure:

        W(z) - sigma(z) (rho - V(z))  is SOS,   sigma is SOS

    where W = -[Q (grad V . f) + (grad V . g) P].  On {V <= rho} the factor
    (rho - V) is non-negative, so the first condition forces W >= 0 there.

    Fixing V (from the Riccati solution) rather than searching for it keeps the
    problem *linear* in the multiplier sigma -- searching over V and sigma jointly
    is bilinear and would need alternation, with the local minima that brings.
    The price is that a failure here means "this V does not certify rho", not
    "no V does".
    """
    z = rat.symbols
    n = len(z)
    zv = sp.Matrix(z)
    v_expr = sp.expand((zv.T @ sp.Matrix(p_lyap.tolist()) @ zv)[0, 0])
    grad = sp.Matrix([sp.diff(v_expr, zi) for zi in z])
    gv_f = (grad.T @ (sp.Matrix(a_mat.tolist()) @ zv))[0, 0]
    gv_g = (grad.T @ sp.Matrix(b_vec.reshape(-1, 1).tolist()))[0, 0]
    w_expr = sp.expand(-(rat.denominator * gv_f + gv_g * rat.numerator))

    sig_mons = _monomials(z, sigma_degree // 2, min_degree=1)
    ks = len(sig_mons)
    sig_gram = cp.Variable((ks, ks), symmetric=True)
    sig_exp = [tuple(sp.Poly(m, *z).monoms()[0]) for m in sig_mons]
    sigma_coeffs: dict = {}
    for i in range(ks):
        for j in range(ks):
            mono = tuple(x + y for x, y in zip(sig_exp[i], sig_exp[j]))
            sigma_coeffs.setdefault(mono, []).append((i, j))

    slack = sp.expand(rho - v_expr)
    slack_c = _coeff_map(slack, z)
    w_c = _coeff_map(w_expr, z)
    # Normalize the constraint.  The rule widths are O(0.1), so raw coefficients
    # span ~7 orders of magnitude and the interior-point solver returns garbage
    # that it still labels optimal.  Dividing through by max|coeff| is exact --
    # SOS-ness is invariant to positive scaling, and the multiplier sigma is a
    # free variable that absorbs the same factor.
    w_scale = max(max((abs(c) for c in w_c.values()), default=1.0), 1e-30)
    w_c = {m: c / w_scale for m, c in w_c.items()}
    slack_scale = max(max((abs(c) for c in slack_c.values()), default=1.0), 1e-30)
    slack_c = {m: c / slack_scale for m, c in slack_c.items()}
    eps_c = _coeff_map(
        sp.expand(-(epsilon / w_scale) * sum(zi**2 for zi in z)), z
    )

    deg = max(max((sum(m) for m in w_c), default=0),
              max((sum(m) for m in slack_c), default=0) + sigma_degree)
    half = (deg + 1) // 2
    mons = _monomials(z, half, min_degree=1)
    mon_exp = [tuple(sp.Poly(m, *z).monoms()[0]) for m in mons]
    k = len(mons)
    gram = cp.Variable((k, k), symmetric=True)
    gram_mono: dict = {}
    for i in range(k):
        for j in range(k):
            mono = tuple(x + y for x, y in zip(mon_exp[i], mon_exp[j]))
            gram_mono.setdefault(mono, []).append((i, j))

    # sigma * (rho - V): product of the sigma Gram entries with slack coeffs.
    prod_terms: dict = {}
    for smono, idxs in sigma_coeffs.items():
        for lmono, lc in slack_c.items():
            mono = tuple(x + y for x, y in zip(smono, lmono))
            prod_terms.setdefault(mono, []).extend((i, j, lc) for i, j in idxs)

    margin = cp.Variable()
    cons = [gram >> margin * np.eye(k), sig_gram >> 0, margin <= 1e3,
            cp.trace(sig_gram) <= 1e6]
    for mono in set(w_c) | set(prod_terms) | set(gram_mono) | set(eps_c):
        lhs = w_c.get(mono, 0.0) + eps_c.get(mono, 0.0)
        lhs = lhs - sum(lc * sig_gram[i, j] for i, j, lc in prod_terms.get(mono, []))
        rhs = sum(gram[i, j] for i, j in gram_mono.get(mono, []))
        cons.append(lhs == rhs)

    prob = cp.Problem(cp.Maximize(margin), cons)
    # Solver options are not portable: max_iters/eps_abs/eps_rel are SCS's and
    # the interior-point solvers reject them, which surfaces as a bare failure
    # rather than an informative error.
    opts = ({"max_iters": 200000, "eps_abs": 1e-9, "eps_rel": 1e-9}
            if solver == "SCS" else {})
    try:
        prob.solve(solver=solver, verbose=False, **opts)
    except Exception:
        return False, -np.inf, np.inf
    if prob.status not in ("optimal", "optimal_inaccurate") or gram.value is None:
        return False, -np.inf, np.inf
    g = 0.5 * (np.array(gram.value) + np.array(gram.value).T)
    sg = 0.5 * (np.array(sig_gram.value) + np.array(sig_gram.value).T)
    scale = max(float(np.max(np.abs(g))), 1e-12)
    eg = float(np.linalg.eigvalsh(g).min())
    es = float(np.linalg.eigvalsh(sg).min())
    ok = eg > -1e-8 * scale and es > -1e-8 * max(float(np.max(np.abs(sg))), 1e-12)
    if _CAPTURE is not None:
        _CAPTURE.update(gram=g, sigma_gram=sg, monomials=mons,
                        sigma_monomials=sig_mons, w_scale=w_scale,
                        slack_scale=slack_scale, w_expr=w_expr, v_expr=v_expr,
                        epsilon=epsilon, rho=rho)
    return ok, eg, abs(min(eg, 0.0)) / scale


def max_certified_rho(
    rat: RationalController, a_mat: af64, b_vec: af64, p_lyap: af64,
    lo: float = 1e-4, hi: float = 1e3, iters: int = 18,
    sigma_degree: int = 4, solver: str = "CLARABEL",
) -> float:
    """Largest sublevel value rho for which `certify_roa` succeeds (bisection).

    Solver choice is not a detail here.  SCS, a first-order method, reports
    violations of 1e-5 relative on this problem and certifies nothing; CLARABEL,
    an interior-point method, closes the same SDP to 1e-11 relative in under two
    seconds.  The obstruction was never the relaxation.
    """
    if not certify_roa(rat, a_mat, b_vec, p_lyap, lo,
                       sigma_degree=sigma_degree, solver=solver)[0]:
        return 0.0
    best = lo
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if certify_roa(rat, a_mat, b_vec, p_lyap, mid,
                       sigma_degree=sigma_degree, solver=solver)[0]:
            best, lo = mid, mid
        else:
            hi = mid
    return best


def lyapunov_from_linearization(
    ctrl, a_mat: af64, b_vec: af64, q_mat: af64 | None = None, h: float = 1e-6
) -> af64 | None:
    """Lyapunov matrix from the controller's OWN closed-loop linearization.

    Reusing the LQR Riccati matrix works only while the fitted controller's
    linearization stays close to the LQR gain.  A directly-optimized controller
    has no reason to, so the candidate is built from its own Jacobian at the
    origin: solve A_cl' P + P A_cl = -Q with A_cl = A + B (du/dz)(0).  Returns
    None if A_cl is not Hurwitz, in which case no local certificate exists and
    the SOS step should not be attempted.
    """
    n = len(b_vec)
    k = np.array([
        (ctrl(h * np.eye(n)[i]) - ctrl(-h * np.eye(n)[i])) / (2 * h)
        for i in range(n)
    ])
    a_cl = a_mat + np.outer(b_vec, k)
    if np.max(np.linalg.eigvals(a_cl).real) >= -1e-12:
        return None
    from scipy.linalg import solve_lyapunov
    return solve_lyapunov(a_cl.T, -(np.eye(n) if q_mat is None else q_mat))
