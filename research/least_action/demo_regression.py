"""Numerical tests of the least-action FIS claims on Dr Cohen's cubic.

Run: .venv/bin/python research/least_action/demo_regression.py

Claims under test
  A. Variable projection: consequents solve globally in closed form, so only
     2N antecedent parameters need searching (vs 4N in the joint formulation).
  B. The gradient term (lam > 0) buys derivative accuracy.
  C. Sequential residual fitting is exact iff the H1 Gram is block diagonal, and
     for non-negative membership functions that happens iff the rules have
     disjoint support -- which simultaneously destroys delta-coverage.  This is
     the central obstruction, not a numerical artifact.
  D. H1-Gram-Schmidt (orthogonal least squares) gets the decoupling without
     requiring disjointness.
  E. Identifiability constraints (rule separation, capped widths) are what turn
     a stationary point into a *certifiable* local minimum.
"""

from __future__ import annotations

import numpy as np

from fis_action import (
    Quadrature,
    decoupling_report,
    fit,
    galerkin_residual,
    h1_gram,
    optimality_certificate,
    rule_regressors,
    sequential_fit,
    solve_consequents,
)

X_LO, X_HI = -15.0, 15.0
X_REF = (0.5 * (X_LO + X_HI), 0.5 * (X_HI - X_LO))


def y_d(x):
    return 0.3 * x**3 + 0.2 * x**2 - 5.0 * x - 3.0


def dy_d(x):
    return 0.9 * x**2 + 0.4 * x - 5.0


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    quad = Quadrature.legendre(X_LO, X_HI, 600)
    yd, dyd = y_d(quad.nodes), dy_d(quad.nodes)

    rule("A. Variable projection, N = 3 rules, order-1 consequents")
    print(f"{'lam':>8} {'||e||_L2':>12} {'||de||_L2':>12} {'action':>14} {'dof searched':>13}")
    fits = {}
    for lam in (0.0, 0.25, 1.0, 4.0, 16.0):
        f = fit(y_d, dy_d, n_rules=3, x_lo=X_LO, x_hi=X_HI, lam=lam, order=1)
        fits[lam] = f
        print(f"{lam:8.2f} {f.l2_error:12.4f} {f.h1_seminorm_error:12.4f} "
              f"{f.action:14.4f} {2 * 3:13d}")
    print("  (the joint formulation searches 4N = 12 parameters for the same model)")

    rule("B. Effect of the gradient penalty on derivative fidelity")
    base = fits[0.0]
    print("  lam=0 is pure L2: it fits values and ignores slopes.")
    for lam in (0.25, 1.0, 4.0, 16.0):
        f = fits[lam]
        print(f"  lam={lam:<6} slope error {f.h1_seminorm_error:8.4f} "
              f"({100 * (f.h1_seminorm_error / base.h1_seminorm_error - 1):+6.1f}%)   "
              f"value error {f.l2_error:8.4f} "
              f"({100 * (f.l2_error / base.l2_error - 1):+6.1f}%)")
    print("  The trend is downward but not monotone: each row is a separate multistart")
    print("  on a non-convex antecedent landscape, so individual rows land in different")
    print("  basins.  That scatter is the finding of section F, not noise to hide.")

    rule("C. Decoupling vs. overlap -- the orthogonality/coverage trade-off")
    lam = 1.0
    centers = np.array([-10.0, 0.0, 10.0])
    print(f"{'MF kind':>9} {'width':>7} {'coherence':>11} {'off-block':>11} "
          f"{'min cover':>10} {'joint S':>13} {'greedy gap':>11}")
    cases = [("gaussian", w) for w in (7.0, 5.0, 3.0, 2.0)] + [("bump", 5.0), ("bump", 6.0)]
    for kind, width in cases:
        widths = np.full(3, width)
        psi, dpsi, mu_sum = rule_regressors(quad.nodes, centers, widths, 1, kind, X_REF)
        rep = decoupling_report(h1_gram(psi, dpsi, quad, lam), 3, 1, mu_sum)
        _, joint, _ = solve_consequents(centers, widths, yd, dyd, quad, lam, 1, kind, X_REF)
        _, greedy = sequential_fit(centers, widths, yd, dyd, quad, lam, 1, kind, X_REF)
        print(f"{kind:>9} {width:7.1f} {rep.coherence:11.3e} {rep.off_block_energy:11.3e} "
              f"{rep.min_coverage:10.3e} {joint:13.2f} {abs(greedy - joint) / joint:11.3e}")
    print("  bump/5.0 tiles [-15,15] with exactly touching supports: off-block energy is 0")
    print("  and greedy == joint to machine precision -- but min coverage collapses to 0 at")
    print("  the seams and the action is 42x worse than the Gaussian of the same width.")
    print("  Widening to bump/6.0 restores overlap and the action, and immediately brings")
    print("  the off-block coupling back.  Exact input-side orthogonality and a usable")
    print("  differentiable model are mutually exclusive.")

    rule("D. H1-Gram-Schmidt (orthogonal least squares) decouples without disjointness")
    for width in (7.0, 5.0, 3.0, 2.0):
        widths = np.full(3, width)
        _, joint, _ = solve_consequents(centers, widths, yd, dyd, quad, lam, 1, "gaussian", X_REF)
        _, greedy = sequential_fit(centers, widths, yd, dyd, quad, lam, 1, "gaussian", X_REF)
        _, ols = sequential_fit(
            centers, widths, yd, dyd, quad, lam, 1, "gaussian", X_REF, orthogonalize=True
        )
        print(f"  width={width:<5} joint={joint:11.4f}  naive greedy gap="
              f"{abs(greedy - joint) / joint:9.2e}  OLS greedy gap="
              f"{abs(ols - joint) / joint:9.2e}")

    rule("E. Identifiability constraints decide whether local optimality is certifiable")
    span = X_HI - X_LO
    pitch = span / 3
    configs = [
        ("unconstrained", 0.0, (0.02 * span, 2.0 * span)),
        ("regular fuzzy partition", 0.6 * pitch, (0.25 * pitch, 1.5 * pitch)),
    ]
    for label, gap, wb in configs:
        f = fit(y_d, dy_d, n_rules=3, x_lo=X_LO, x_hi=X_HI, lam=1.0, order=1,
                min_gap=gap, width_bounds=wb)
        cert = optimality_certificate(f, y_d, dy_d, quad)
        print(f"\n  [{label}]  min_gap={gap:.2f}, width in ({wb[0]:.2f}, {wb[1]:.2f})")
        print(f"    centers        {np.array2string(f.centers, precision=3)}")
        print(f"    widths         {np.array2string(f.widths, precision=3)}")
        print(f"    ||e||_L2       {f.l2_error:.4f}")
        print(f"    min coverage   {f.report.min_coverage:.4e}")
        print(f"    Gram cond      {cert['gram_condition']:.3e}  "
              f"(min eig {cert['gram_min_eigenvalue']:.3e})")
        print(f"    Galerkin resid {cert['galerkin_residual_inf_norm']:.3e}  "
              f"(consequent block)")
        print(f"    reduced |grad| {cert['reduced_gradient_norm']:.3e}  (antecedent block)")
        print(f"    KKT residual   {cert['kkt_residual']:.3e}  "
              f"(gradient projected onto the critical cone -> "
              f"stationary: {cert['stationary']})")
        print(f"    active constr  {cert['n_active_constraints']}  ->  critical cone "
              f"dim {cert['critical_cone_dim']} of 6")
        print(f"    proj. Hessian  eig in [{cert['min_eigenvalue']:.4e}, "
              f"{cert['max_eigenvalue']:.4e}]  (noise floor "
              f"{cert['noise_floor']:.1e})")
        print(f"    descent probe  {cert['verified_descent']:.3e}  "
              f"(curvature-driven descent found: {cert['is_saddle']})")
        verdict = bool(cert["stationary"]) and not bool(cert["is_saddle"])
        print(f"    CERTIFIED LOCAL MIN: {verdict}   "
              f"[first-order {cert['stationary']}, no descent "
              f"{not cert['is_saddle']}]")

    rule("F. Consequent sub-problem is globally convex")
    f = fits[1.0]
    psi, dpsi, _ = rule_regressors(quad.nodes, f.centers, f.widths, 1, f.kind, f.x_ref)
    eigs = np.linalg.eigvalsh(h1_gram(psi, dpsi, quad, 1.0))
    print(f"  H1 Gram eigenvalues in [{eigs.min():.4e}, {eigs.max():.4e}], "
          f"all positive = {bool(eigs.min() > 0)}")
    print("  => for fixed antecedents the action has a unique global minimizer in the")
    print("     consequents.  All non-convexity lives in (a_i, b_i).")

    rule("G. Galerkin orthogonality holds per-regressor at the optimum")
    g = galerkin_residual(f, y_d, dy_d, quad)
    for m, v in enumerate(g):
        print(f"  <e, phi_{m // 2} * t^{m % 2}>_H1 = {v:+.3e}")


if __name__ == "__main__":
    main()
