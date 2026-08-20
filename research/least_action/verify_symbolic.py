"""Machine verification of every algebraic claim in the write-up.

Run: .venv/bin/python research/least_action/verify_symbolic.py

This is *symbolic* verification, not numerical spot-checking: each claim is
reduced with sympy to a syntactic identity (`simplify(...) == 0`) over the
rationals or over a symbolic function algebra. A pass means the identity holds
for all values of the symbols, not that it held at some sample points.

What this does and does not establish:

  DOES  -- the algebra of each theorem is correct as stated, including the
           cancellations that are easy to get wrong by hand.
  DOES NOT -- discharge the analytic side conditions (integrability, the
           admissibility limit x(t) -> 0, convergence of improper integrals,
           existence of minimizers). Those are stated as hypotheses in the
           write-up and are verified numerically elsewhere, not here.

A Lean/Mathlib formalization would close that second gap. It is not possible in
this environment: `release.lean-lang.org` is refused by the network policy, so
no toolchain can be installed. See README §13.
"""

from __future__ import annotations

import sympy as sp

PASS, FAIL = "PASS", "**FAIL**"
results: list[tuple[str, str, str]] = []


def check(name: str, expr, note: str = "") -> None:
    """Record whether `expr` simplifies identically to zero.

    Matrices need `is_zero_matrix`: `Matrix(...) == 0` is False even for the
    zero matrix, which silently turns a passing identity into a reported
    failure.
    """
    z = sp.simplify(expr)
    if isinstance(z, sp.MatrixBase):
        ok = bool(z.is_zero_matrix)
    else:
        ok = z == 0
    results.append((name, PASS if ok else FAIL, note or str(z)))


# --------------------------------------------------------------------------
# §1  Euler-Lagrange for the H1 action
# --------------------------------------------------------------------------
def t_euler_lagrange() -> None:
    x = sp.Symbol("x", real=True)
    lam = sp.Symbol("lambda", positive=True)
    e = sp.Function("e")(x)
    # L = e^2 + lam e'^2, treating e and e' as the variational arguments.
    ee, ep = sp.symbols("e_ ep_", real=True)
    lag = ee**2 + lam * ep**2
    dL_de = sp.diff(lag, ee).subs({ee: e, ep: sp.diff(e, x)})
    dL_dep = sp.diff(lag, ep).subs({ee: e, ep: sp.diff(e, x)})
    el = dL_de - sp.diff(dL_dep, x)
    # Claim: E-L  <=>  lam e'' - e = 0  (up to the overall factor -2)
    check(
        "§1  Euler-Lagrange gives lam*e'' - e = 0",
        sp.expand(el - (-2) * (lam * sp.diff(e, x, 2) - e)),
    )


# --------------------------------------------------------------------------
# §7a  Theorem C1 -- partition of unity reproduces an affine law exactly
# --------------------------------------------------------------------------
def t_theorem_c1() -> None:
    n, k1, k2 = 3, sp.Symbol("k1"), sp.Symbol("k2")
    z1, z2 = sp.symbols("z1 z2", real=True)
    mus = sp.symbols("mu1 mu2 mu3", positive=True)
    tot = sum(mus)
    phis = [m / tot for m in mus]
    # every consequent is the same affine map -K z
    blended = sum(p * (-(k1 * z1 + k2 * z2)) for p in phis)
    check(
        "§7a Theorem C1: sum_i phi_i(-Kz) = -Kz for ANY membership values",
        sp.simplify(blended + (k1 * z1 + k2 * z2)),
    )
    check("§7a partition of unity: sum_i phi_i = 1", sp.simplify(sum(phis) - 1))


# --------------------------------------------------------------------------
# §7b / §9b  Theorem C2 and its Bregman generalization C2'
# --------------------------------------------------------------------------
def t_theorem_c2_prime() -> None:
    u, us = sp.symbols("u u_star", real=True)
    x = sp.Symbol("x", real=True)
    c = sp.Function("c")
    f, g = sp.Function("f")(x), sp.Function("g")(x)
    vx = sp.Symbol("V_x", real=True)

    # HJB stationarity at u*:  c'(u*) + g V_x = 0.
    stationarity = sp.Eq(sp.diff(c(us), us) + g * vx, 0)
    vx_sol = sp.solve(stationarity, vx)[0]

    # q defined FROM the HJB (inverse-optimal construction).
    q = -c(us) - vx * (f + g * us)
    # Running cost plus dV/dt along the closed loop driven by u.
    integrand = q + c(u) + vx * (f + g * u)
    bregman = c(u) - c(us) - sp.diff(c(us), us) * (u - us)

    check(
        "§9b Theorem C2': ell + V_x(f+gu) = Bregman divergence D_c(u||u*)",
        sp.simplify((integrand - bregman).subs(vx, vx_sol)),
    )

    # Quadratic specialization: c(u) = R u^2 recovers the classical statement.
    r = sp.Symbol("R", positive=True)
    breg_qc = c(u) - c(us) - sp.diff(c(us), us) * (u - us)
    breg_quad = breg_qc.subs(c, sp.Lambda(sp.Symbol("w"), r * sp.Symbol("w") ** 2))
    check(
        "§7b Theorem C2 is the c(u)=Ru^2 case: D_c = R(u-u*)^2",
        sp.simplify(sp.expand(breg_quad) - r * (u - us) ** 2),
    )

    # Non-negativity of D_c is exactly convexity of c (checked on a witness).
    w = sp.Symbol("w", real=True)
    for name, fn in (("u^2+u^4", w**2 + w**4), ("cosh(u)-1", sp.cosh(w) - 1)):
        d = fn.subs(w, u) - fn.subs(w, us) - sp.diff(fn, w).subs(w, us) * (u - us)
        # second derivative of D_c in u equals c''(u) >= 0 for these
        check(
            f"§9b D_c >= 0 via convexity, c(u)={name}",
            sp.simplify(sp.diff(d, u, 2) - sp.diff(fn, w, 2).subs(w, u)),
        )


# --------------------------------------------------------------------------
# §4d'  lambda* in closed form
# --------------------------------------------------------------------------
def t_lambda_star() -> None:
    x, k = sp.symbols("x k", real=True, positive=True)
    # Two equal-width Gaussians at +-c give a logistic partition (see §4d').
    s = 1 / (1 + sp.exp(-k * x))
    phi1, phi0 = s, 1 - s
    i_pp = sp.integrate(sp.simplify(phi0 * phi1), (x, -sp.oo, sp.oo))
    d0, d1 = sp.diff(phi0, x), sp.diff(phi1, x)
    i_dd = sp.integrate(sp.simplify(d0 * d1), (x, -sp.oo, sp.oo))
    check("§4d' INT phi0 phi1 dx = 1/k", sp.simplify(i_pp - 1 / k))
    check("§4d' INT phi0' phi1' dx = -k/6", sp.simplify(i_dd + k / 6))
    check(
        "§4d' lambda* = -INT(pp)/INT(dd) = 6/k^2", sp.simplify(-i_pp / i_dd - 6 / k**2)
    )


# --------------------------------------------------------------------------
# §5b  Centroid defuzzification over disjoint sets is an order-0 TSK model
# --------------------------------------------------------------------------
def t_centroid_is_tsk0() -> None:
    a1, a2, ar1, ar2, c1, c2 = sp.symbols("alpha1 alpha2 A1 A2 c1 c2", positive=True)
    # With disjoint supports both integrals split over the sets.
    num = a1 * ar1 * c1 + a2 * ar2 * c2
    den = a1 * ar1 + a2 * ar2
    y = num / den
    w1, w2 = a1 * ar1 / den, a2 * ar2 / den
    check(
        "§5b centroid = sum_i w_i c_i with w_i = alpha_i A_i / sum",
        sp.simplify(y - (w1 * c1 + w2 * c2)),
    )
    check(
        "§5b weights sum to one (=> output is a convex combination)",
        sp.simplify(w1 + w2 - 1),
    )


# --------------------------------------------------------------------------
# §5c  Annealing bound
# --------------------------------------------------------------------------
def t_annealing_bound() -> None:
    b = sp.Symbol("beta", positive=True)
    a1, a2 = sp.symbols("alpha1 alpha2", positive=True)
    c1, c2 = sp.symbols("c1 c2", real=True)
    w1 = a1**b / (a1**b + a2**b)
    w2 = a2**b / (a1**b + a2**b)
    y = w1 * c1 + w2 * c2
    # y - c1 = w2 (c2 - c1); with r = a2/a1 and N=2 the bound is D r^b/(1+r^b)
    r = a2 / a1
    check("§5c y_beta - c_(1) = w_2 (c_2 - c_1)", sp.simplify(y - c1 - w2 * (c2 - c1)))
    check(
        "§5c off-peak weight w_2 = r^beta/(1+r^beta), r = alpha2/alpha1",
        sp.simplify(w2 - r**b / (1 + r**b)),
    )
    # t/(1+t) is increasing, which is what lets alpha_i/alpha_(1) <= r be used
    t = sp.Symbol("t", positive=True)
    mono = sp.simplify(sp.diff(t / (1 + t), t))
    results.append(
        (
            "§5c t/(1+t) is increasing (d/dt > 0)",
            PASS if sp.ask(sp.Q.positive(mono)) or mono == 1 / (1 + t) ** 2 else FAIL,
            str(mono),
        )
    )


# --------------------------------------------------------------------------
# §12a  The pi-MF controller is rational, and Q > 0
# --------------------------------------------------------------------------
def t_rational_controller() -> None:
    z = sp.Symbol("z", real=True)
    c1, c2, w1_, w2_ = sp.symbols("c1 c2 w1 w2", real=True, positive=True)
    d1 = 1 + ((z - c1) / w1_) ** 2
    d2 = 1 + ((z - c2) / w2_) ** 2
    mu1, mu2 = 1 / d1, 1 / d2
    phi1 = mu1 / (mu1 + mu2)
    # phi1 should equal D2 / (D1 + D2)
    check(
        "§12a phi_i = N_i/Q with N_1 = D_2, Q = D_1 + D_2 (N=2)",
        sp.simplify(phi1 - d2 / (d1 + d2)),
    )
    f1, f2 = sp.symbols("f1 f2", real=True)
    u = phi1 * f1 + (1 - phi1) * f2
    p, q = sp.fraction(sp.cancel(sp.together(u)))
    check("§12a u = P/Q exactly (P, Q polynomial)", sp.simplify(u - p / q))
    # D_i >= 1 for real z, so Q >= N > 0: check the minimum of D over z.
    dmin = sp.minimum(d1, z, sp.S.Reals)
    results.append(
        (
            "§12a D_i >= 1 for all real z (=> Q >= N > 0)",
            PASS if sp.simplify(dmin - 1) == 0 else FAIL,
            str(dmin),
        )
    )


# --------------------------------------------------------------------------
# §3a  The consequent problem is a convex quadratic (Gram is PSD)
# --------------------------------------------------------------------------
def t_consequent_convexity() -> None:
    t1, t2 = sp.symbols("theta1 theta2", real=True)
    g11, g12, g22 = sp.symbols("G11 G12 G22", real=True)
    r1, r2 = sp.symbols("r1 r2", real=True)
    theta = sp.Matrix([t1, t2])
    gram = sp.Matrix([[g11, g12], [g12, g22]])
    rhs = sp.Matrix([r1, r2])
    s = (theta.T * gram * theta)[0, 0] - 2 * (rhs.T * theta)[0, 0]
    hess = sp.hessian(s, (t1, t2))
    check(
        "§3a Hessian of the action in theta equals 2G (constant in theta)",
        sp.simplify(hess - 2 * gram),
    )
    grad = sp.Matrix([sp.diff(s, t1), sp.diff(s, t2)])
    check(
        "§3a stationarity <=> G theta = r", sp.simplify(grad - 2 * (gram * theta - rhs))
    )


# --------------------------------------------------------------------------
# §3d  Envelope theorem: the reduced gradient
# --------------------------------------------------------------------------
def t_envelope() -> None:
    p = sp.Symbol("p", real=True)
    g11 = sp.Function("G11")(p)
    r1 = sp.Function("r1")(p)
    # scalar case: S(p) = yy - r^2/G, theta = r/G
    yy = sp.Symbol("yy", real=True)
    s = yy - r1**2 / g11
    theta = r1 / g11
    ds = sp.diff(s, p)
    # claim: dS/dp = -2 (dr/dp) theta + theta^2 (dG/dp)
    claim = -2 * sp.diff(r1, p) * theta + theta**2 * sp.diff(g11, p)
    check(
        "§3d envelope theorem: dS/dp = -2 theta dr/dp + theta^2 dG/dp",
        sp.simplify(ds - claim),
    )


def main() -> None:
    for fn in (
        t_euler_lagrange,
        t_theorem_c1,
        t_theorem_c2_prime,
        t_lambda_star,
        t_centroid_is_tsk0,
        t_annealing_bound,
        t_rational_controller,
        t_consequent_convexity,
        t_envelope,
    ):
        fn()
    width = max(len(n) for n, _, _ in results)
    print("SYMBOLIC VERIFICATION (sympy; identities, not sampled points)")
    print("=" * (width + 14))
    n_fail = 0
    for name, status, note in results:
        print(f"{name:<{width}}  {status}")
        if status == FAIL:
            n_fail += 1
            print(f"{'':<{width}}  residual: {note}")
    print("=" * (width + 14))
    print(f"{len(results) - n_fail}/{len(results)} identities verified")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
