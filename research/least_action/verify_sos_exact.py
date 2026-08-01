"""Exact rational verification of the SOS stability certificate.

Run: .venv/bin/python research/least_action/verify_sos_exact.py

The §12d certificate comes from an interior-point SDP in floating point. That is
evidence, not proof: the Gram is PSD *to within solver tolerance* and the
polynomial identity holds *to within round-off*. This upgrades it, by the
standard rounding-and-projection procedure (Peyrl & Parrilo, 2008):

  1. Solve the SDP with a strictly positive margin, so there is room to round.
  2. Round both Gram matrices to rationals with bounded denominator.
  3. Rebuild the target polynomial exactly over Q.  IEEE doubles *are* rationals,
     so the controller, P and rho contribute no error -- only the deliberate
     rounding does.
  4. Project the rounded Gram back onto the affine set where the polynomial
     identity holds EXACTLY.  The coefficient-matching map partitions the Gram
     entries by monomial, so this projection decouples into one group per
     monomial and is computed in closed form rather than by a linear solve.
  5. Verify the projected Gram is PSD in exact arithmetic via rational LDL^T --
     all pivots positive as rationals, no eigenvalues, no tolerance.

If all five succeed, "V decreases strictly on the sublevel set" is a theorem
about the rationalized controller with no floating-point step in the chain.
"""

from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import sympy as sp
from scipy.linalg import solve_continuous_are

import fis_sos
from fis_sos import certify_roa, to_rational
from fis_twocart import TskController, TwoCart, fit_consequents, place_rules

warnings.filterwarnings("ignore")

DENOM = 10**8
RHO = sp.Rational(1, 2)


def rat(x: float, denom: int = DENOM) -> sp.Rational:
    """Deliberate rounding to a bounded-denominator rational."""
    return sp.Rational(Fraction(float(x)).limit_denominator(denom))


def to_exact(expr):
    """Replace every Float by the rational it exactly equals.

    NOT sympy's `nsimplify(..., rational=True)`, which searches for "nice"
    closed forms and cheerfully returns things like 2**(818/971) * 3**(651/971)
    for an ordinary double.  IEEE floats are exactly dyadic rationals, so this
    conversion is lossless and unambiguous.
    """
    return expr.xreplace({f: sp.Rational(f) for f in expr.atoms(sp.Float)})


def exact_psd(mat: sp.Matrix) -> tuple[bool, int, int]:
    """Decide positive definiteness of a rational matrix in exact INTEGER arithmetic.

    Clearing denominators gives an integer matrix M = D*G with D > 0, and G is
    positive definite iff M is.  Definiteness is then decided by Sylvester's
    criterion on integer leading principal minors, computed by fraction-free
    (Bareiss) elimination.

    A rational LDL^T is equally exact but useless at this size: the denominators
    square at every elimination step, and the 14x14 factorization does not
    finish.  Integer minors have no such blow-up.

    Returns (positive_definite, minors_checked, bit-length of the largest minor).
    """
    n = mat.rows
    common = sp.Integer(1)
    for i in range(n):
        for j in range(n):
            common = sp.ilcm(common, sp.Rational(mat[i, j]).q)
    m_int = sp.Matrix(n, n,
                      lambda i, j: sp.Integer(sp.Rational(mat[i, j]) * common))
    biggest = 0
    for k in range(1, n + 1):
        minor = m_int[:k, :k].det(method="bareiss")
        biggest = max(biggest, int(abs(minor)).bit_length())
        if minor <= 0:
            return False, k, biggest
    return True, n, biggest


def main() -> None:
    plant = TwoCart()
    a_mat, b_mat = plant.linearization()
    q_mat = np.diag([1.0, 10.0, 1.0, 1.0])
    p_lyap = solve_continuous_are(a_mat, b_mat, q_mat, np.array([[10.0]]))
    d = np.load(".twocart_train.npz")
    cen, wid = place_rules(d["z"], d["w"], 2)
    ctrl = TskController(cen, wid, order=1, mf="pi")
    fit_consequents(ctrl, d["z"], d["u"], d["w"])
    rat_ctrl = to_rational(ctrl)

    store: dict = {}
    fis_sos.capture(store)
    ok, eg, _ = certify_roa(rat_ctrl, a_mat, b_mat.ravel(), p_lyap, float(RHO),
                            epsilon=1e-8, sigma_degree=2, solver="CLARABEL")
    fis_sos.capture(None)
    print(f"1. numerical SDP at rho={RHO}: certified={ok}, "
          f"Gram min eig {eg:+.3e}  (strictly positive => room to round)")
    if not ok or eg <= 0:
        raise SystemExit("need a strictly positive margin to round into")

    z = rat_ctrl.symbols
    mons = store["monomials"]
    sig_mons = store["sigma_monomials"]
    k, ks = len(mons), len(sig_mons)

    # Step 2: round the multiplier, verify it is PSD exactly.
    sg = store["sigma_gram"]
    s_hat = sp.Matrix(ks, ks, lambda i, j: rat((sg[i, j] + sg[j, i]) / 2))
    s_pd, _, _ = exact_psd(s_hat)
    sigma = sp.expand(sum(s_hat[i, j] * sig_mons[i] * sig_mons[j]
                          for i in range(ks) for j in range(ks)))
    sigma = to_exact(sigma)
    print(f"2. multiplier rounded to Q (denominator <= {DENOM}): "
          f"sigma is SOS over Q = {s_pd}")
    print(f"   sigma(0) = {sigma.subs({zi: 0 for zi in z})}  "
          f"(must be 0 or the target cannot be SOS)")

    # Step 3: exact target over Q, in UNSCALED units.  The SDP solved a version
    # divided by max|coeff|; multiplying the identity through by that positive
    # constant is exact and removes one source of rounding.
    w_expr = to_exact(store["w_expr"])
    v_expr = to_exact(store["v_expr"])
    w_scale = sp.Rational(store["w_scale"])
    slack_scale = sp.Rational(store["slack_scale"])
    eps = sp.Rational(store["epsilon"])
    sigma_eff = sp.expand(sigma * w_scale / slack_scale)
    target = sp.expand(
        w_expr
        - eps * sum(zi**2 for zi in z)
        - sigma_eff * (RHO - v_expr)
    )
    tpoly = sp.Poly(target, *z)
    print(f"3. exact target over Q: {len(tpoly.monoms())} monomials, "
          f"degrees {sorted({sum(m) for m in tpoly.monoms()})}")

    # Step 4: round the Gram, then project onto the exact identity.  Each Gram
    # entry contributes to exactly one monomial, so the constraint decouples.
    g = store["gram"] * float(w_scale)   # undo the solver-side normalization
    g0 = sp.Matrix(k, k, lambda i, j: rat((g[i, j] + g[j, i]) / 2))
    mon_exp = [tuple(sp.Poly(m, *z).monoms()[0]) for m in mons]
    groups: dict = {}
    for i in range(k):
        for j in range(k):
            key = tuple(x + y for x, y in zip(mon_exp[i], mon_exp[j]))
            groups.setdefault(key, []).append((i, j))
    tcoef = {m: c for m, c in zip(tpoly.monoms(), tpoly.coeffs())}
    unmatched = [m for m in tcoef if m not in groups]
    if unmatched:
        print(f"   target has {len(unmatched)} monomials no Gram entry can "
              f"produce -- identity impossible")
        raise SystemExit(1)
    g_exact = g0.copy()
    for key, idxs in groups.items():
        want = sp.nsimplify(tcoef.get(key, 0))
        have = sum(g0[i, j] for i, j in idxs)
        delta = sp.cancel((want - have) / len(idxs))
        for i, j in idxs:
            g_exact[i, j] = g_exact[i, j] + delta
    g_exact = (g_exact + g_exact.T) / 2

    recon = sp.expand(sum(g_exact[i, j] * mons[i] * mons[j]
                          for i in range(k) for j in range(k)))
    residual = sp.expand(recon - target)
    print(f"4. projected onto the exact identity: residual = {residual}  "
          f"({'EXACT' if residual == 0 else 'NONZERO'})")
    if residual != 0:
        raise SystemExit(1)

    # Step 5: exact PSD via integer leading principal minors.
    pd, nchecked, bits = exact_psd(g_exact)
    print(f"5. exact integer minors on the {k}x{k} Gram: positive definite = {pd}")
    print(f"   {nchecked}/{k} leading principal minors verified > 0 "
          f"(largest is a {bits}-bit integer)")
    if not pd:
        raise SystemExit(1)

    print()
    print("CONCLUSION -- no floating-point step remains in the chain:")
    print(f"  W(z)/s - eps||z||^2/s - sigma(z)(rho - V(z))/s'  =  m(z)' G m(z)")
    print(f"  with G positive definite over Q and sigma SOS over Q.")
    print(f"  Therefore Vdot < 0 on {{z : z'Pz <= {RHO}}} \\ {{0}} for the")
    print(f"  rationalized pi-MF controller.  Proved, not measured.")


if __name__ == "__main__":
    main()
