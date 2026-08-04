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
  H. Because the H1 slope term is negative for adjacent rules, there is a third
     route to decoupling: tune lambda so the value and slope terms cancel.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from fis_action import (
    Quadrature,
    decoupling_report,
    fit,
    galerkin_residual,
    h1_gram,
    h1_project,
    normalized_weights,
    optimality_certificate,
    reduced_action_and_gradient,
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

    rule("G'. Analytic reduced gradient vs. finite differences")
    print("  dS/dp = -2 <e, dy_c/dp>_H1 by the envelope theorem (see fis_action.py).")
    print("  The FD reference is itself only good to ~cond(G) * eps / h, so the honest")
    print("  comparison sweeps h and reports the best agreement; a single fixed h")
    print("  measures the reference's round-off, not the gradient's error.")
    print()
    steps = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)

    def fd_gap(c, wdt, p_order, kind):
        _, _, g = reduced_action_and_gradient(
            c, wdt, yd, dyd, quad, lam, p_order, kind, X_REF
        )
        p0 = np.concatenate([c, wdt])

        def red(p):
            return solve_consequents(p[:3], p[3:], yd, dyd, quad, lam, p_order, kind,
                                     X_REF)[1]

        best, best_h = np.inf, 0.0
        for h in steps:
            fd = np.array([
                (red(p0 + np.eye(6)[i] * h) - red(p0 - np.eye(6)[i] * h)) / (2 * h)
                for i in range(6)
            ])
            rel = np.linalg.norm(g - fd) / max(np.linalg.norm(fd), 1e-30)
            if rel < best:
                best, best_h = rel, h
        psi_, dpsi_, mus = rule_regressors(quad.nodes, c, wdt, p_order, kind, X_REF)
        gram_ = h1_gram(psi_, dpsi_, quad, lam)
        return best, best_h, float(np.linalg.cond(gram_)), float(mus.min())

    print(f"{'MF kind':>9} {'order':>6} {'coverage':>11} {'cond(G)':>11} "
          f"{'best rel err':>13} {'at h':>8}")
    configs = {
        "gaussian": (np.array([-9.0, 0.5, 9.0]), np.array([10.0, 11.0, 10.0])),
        "cauchy": (np.array([-9.0, 0.5, 9.0]), np.array([7.0, 8.0, 7.0])),
        # Bumps must be wide enough to overlap, or coverage -- and with it
        # differentiability -- is lost; see the contrast rows below.
        "bump": (np.array([-10.0, 0.0, 10.0]), np.array([13.0, 13.0, 13.0])),
    }
    for kind, (c, wdt) in configs.items():
        for p_order in (0, 1, 2):
            err, h_at, cond, cov = fd_gap(c, wdt, p_order, kind)
            print(f"{kind:>9} {p_order:>6} {cov:11.3e} {cond:11.2e} {err:13.3e} {h_at:8.0e}")
    print("  Agreement degrades with consequent order purely because cond(G) grows and")
    print("  the FD reference loses digits -- the h-sweep shows the error rising as h")
    print("  shrinks, which is the signature of round-off in the reference, not")
    print("  truncation in the gradient.")
    print()
    print("  Same bump membership functions with delta-coverage deliberately broken:")
    for width in (5.0, 7.0, 13.0):
        err, h_at, cond, cov = fd_gap(np.array([-10.0, 0.0, 10.0]),
                                      np.full(3, width), 1, "bump")
        print(f"{'bump':>9} {1:>6} {cov:11.3e} {cond:11.2e} {err:13.3e} {h_at:8.0e}")
    print("  => exact to ~1e-8 exactly where coverage (C2) holds, and meaningless where")
    print("     it fails -- because there the model is not differentiable at all.  C2 is")
    print("     the differentiability condition, not bookkeeping.")

    rule("G. Galerkin orthogonality holds per-regressor at the optimum")
    g = galerkin_residual(f, y_d, dy_d, quad)
    for m, v in enumerate(g):
        print(f"  <e, phi_{m // 2} * t^{m % 2}>_H1 = {v:+.3e}")

    rule("H. A third route: lambda-tuned orthogonality without disjointness")
    print("  <phi_i,phi_j>_H1 = INT phi_i phi_j + lam INT phi_i' phi_j'.  For adjacent")
    print("  rules the slope term is NEGATIVE (one rises where the other falls), so the")
    print("  two terms can cancel at a positive lambda* -- giving exact H1 orthogonality")
    print("  with overlapping, C-infinity, positively-covering membership functions.")
    print()
    fine = Quadrature.legendre(X_LO, X_HI, 2000)
    print("  N=2, order 0.  lambda* = -INT(phi_0 phi_1) / INT(phi_0' phi_1'), closed form:")
    print(f"{'width':>8} {'INT phi.phi':>13} {'INT dphi.dphi':>15} {'lambda*':>11} "
          f"{'off-block at lam*':>19}")
    for width in (2.0, 4.0, 6.0, 10.0):
        cen = np.array([-5.0, 5.0])
        wid = np.full(2, width)
        ph, dph, _ = normalized_weights(fine.nodes, cen, wid, "gaussian")
        num = fine.integrate(ph[0] * ph[1])
        den = fine.integrate(dph[0] * dph[1])
        lam_star = -num / den
        ps, dps, _ = rule_regressors(fine.nodes, cen, wid, 0, "gaussian", X_REF)
        obe = decoupling_report(h1_gram(ps, dps, fine, lam_star), 2, 0).off_block_energy
        print(f"{width:8.1f} {num:13.4e} {den:15.4e} {lam_star:11.4f} {obe:19.2e}")
    print()
    print("  Does it actually make greedy fitting exact?  N=2, centers +-5, width 4:")
    cen = np.array([-5.0, 5.0])
    wid = np.full(2, 4.0)
    ph, dph, _ = normalized_weights(fine.nodes, cen, wid, "gaussian")
    lam_star = -fine.integrate(ph[0] * ph[1]) / fine.integrate(dph[0] * dph[1])
    yd_f, dyd_f = y_d(fine.nodes), dy_d(fine.nodes)
    print(f"{'order':>6} {'lambda':>10} {'off-block':>12} {'greedy vs joint':>17}")
    for p_order in (0, 1):
        for lam_v in (1.0, lam_star, 10.0):
            _, j_act, _ = solve_consequents(cen, wid, yd_f, dyd_f, fine, lam_v,
                                            p_order, "gaussian", X_REF)
            _, g_act = sequential_fit(cen, wid, yd_f, dyd_f, fine, lam_v,
                                      p_order, "gaussian", X_REF)
            ps, dps, _ = rule_regressors(fine.nodes, cen, wid, p_order, "gaussian", X_REF)
            obe = decoupling_report(h1_gram(ps, dps, fine, lam_v), 2, p_order).off_block_energy
            tag = "  <-- lambda*" if abs(lam_v - lam_star) < 1e-9 else ""
            print(f"{p_order:>6} {lam_v:10.4f} {obe:12.3e} "
                  f"{abs(g_act - j_act) / j_act:17.3e}{tag}")
    print("  order 0 at lambda*: off-block at machine zero, greedy == joint.")
    print("  order 1 at lambda*: no good -- (p+1)^2 = 4 conditions, only one lambda.")
    print()
    print("  Scope for N > 2 (order 0): N(N-1)/2 pairwise conditions, still one lambda,")
    print("  so exact cancellation is generically impossible -- but it still helps:")
    print(f"{'N':>4} {'width':>7} {'off-block at lam=1':>20} {'best lam':>10} "
          f"{'off-block there':>17}")
    for n_r in (2, 3, 4, 5):
        cen = np.linspace(-10.0, 10.0, n_r)
        wid = np.full(n_r, 20.0 / max(n_r - 1, 1) * 0.6)

        def off(lam_v, c=cen, w_=wid, n_=n_r):
            ps, dps, _ = rule_regressors(fine.nodes, c, w_, 0, "gaussian", X_REF)
            return decoupling_report(h1_gram(ps, dps, fine, lam_v), n_, 0).off_block_energy

        best = minimize_scalar(off, bounds=(1e-3, 1e5), method="bounded",
                               options={"xatol": 1e-8})
        print(f"{n_r:>4} {wid[0]:7.2f} {off(1.0):20.3e} {best.x:10.3f} {best.fun:17.3e}")
    print("  N=2 reaches machine zero; N>=3 improves the coupling ~20x but cannot")
    print("  eliminate it.  Section I explains why, and evaluates what lambda* costs.")

    rule("I. Evaluating lambda*: closed form, meaning, and price")
    wide = np.linspace(-300.0, 300.0, 700001)

    def lam_star(c, b, kind="gaussian"):
        ph, dph, _ = normalized_weights(wide, np.array([-c, c]), np.full(2, b), kind)
        return -np.trapezoid(ph[0] * ph[1], wide) / np.trapezoid(dph[0] * dph[1], wide)

    print("  For two equal-width Gaussians at +-c the normalized weights collapse to a")
    print("  logistic: phi_1 = sigmoid(k x) with k = 4c/b^2, because the log-ratio of two")
    print("  equal-width Gaussians is linear in x.  Both integrals are then elementary,")
    print("    INT phi_0 phi_1 dx = 1/k        INT phi_0' phi_1' dx = -k/6")
    print("  (the second by substituting s = sigmoid(u), giving INT_0^1 s(1-s) ds = 1/6),")
    print("  so lambda* = 6/k^2 = 3 b^4 / (8 c^2).")
    print()
    print(f"{'c':>5} {'b':>5} {'k=4c/b^2':>10} {'INT pp':>10} {'1/k':>10} "
          f"{'INT dd':>11} {'-k/6':>10} {'lam* num':>10} {'6/k^2':>10}")
    for c, b in ((5.0, 2.0), (5.0, 4.0), (3.0, 2.0), (8.0, 3.0), (6.0, 2.5)):
        ph, dph, _ = normalized_weights(wide, np.array([-c, c]), np.full(2, b), "gaussian")
        pp = np.trapezoid(ph[0] * ph[1], wide)
        dd = np.trapezoid(dph[0] * dph[1], wide)
        k = 4 * c / b**2
        print(f"{c:5.1f} {b:5.1f} {k:10.5f} {pp:10.6f} {1 / k:10.6f} {dd:11.6f} "
              f"{-k / 6:10.6f} {-pp / dd:10.5f} {6 / k**2:10.5f}")
    print()
    print("  MEANING.  w = 1/k = b^2/(4c) is the crossover width of the rule handover.")
    print("  Then ell* = sqrt(lambda*) = sqrt(6) * w exactly:")
    for c, b in ((5.0, 2.0), (5.0, 4.0), (3.0, 2.0), (8.0, 3.0)):
        w_c = b**2 / (4 * c)
        print(f"    c={c:4.1f} b={b:4.1f}  w={w_c:8.5f}  ell*={np.sqrt(lam_star(c, b)):8.5f}"
              f"  ratio={np.sqrt(lam_star(c, b)) / w_c:8.5f}  (sqrt 6 = {np.sqrt(6):.5f})")
    print()
    print("  So lambda* is fixed entirely by the PARTITION GEOMETRY.  y_d appears nowhere")
    print("  in it.  It is therefore not a distinguished correlation length of the target")
    print("  -- it is the correlation length of the rule crossover region.")
    print()
    print("  WHY N>=3 CANNOT WORK.  lambda* ~ 1/separation^2, so on a uniform partition")
    print("  of pitch d, adjacent pairs and next-nearest pairs demand lambda* in ratio")
    print("  (2d/d)^2 = 4.  One lambda cannot serve both:")
    for d, b in ((6.0, 3.0), (8.0, 4.0), (10.0, 4.0)):
        la, ln = lam_star(d / 2, b), lam_star(d, b)
        print(f"    pitch={d:5.1f} b={b:4.1f}  lam*_adjacent={la:9.5f}  "
              f"lam*_next={ln:9.5f}  ratio={la / ln:7.4f}")
    print()
    print("  PRICE.  L2 error of the H1-optimal consequents at lambda*, against the")
    print("  L2-optimal (lambda=0) fit of the same partition; target tanh(x/3):")
    q_t = Quadrature.legendre(X_LO, X_HI, 800)
    yt = np.tanh(q_t.nodes / 3.0)
    dyt = (1.0 - np.tanh(q_t.nodes / 3.0) ** 2) / 3.0
    print(f"{'c':>5} {'b':>5} {'w':>8} {'lambda*':>10} {'ell*':>8} {'L2 @ 0':>9} "
          f"{'L2 @ lam*':>10} {'cost':>8}")
    for c, b in ((5.0, 1.0), (5.0, 2.0), (5.0, 4.0), (5.0, 6.0), (5.0, 8.0), (3.0, 4.0)):
        ls = lam_star(c, b)
        ps, dps, _ = rule_regressors(q_t.nodes, np.array([-c, c]), np.full(2, b),
                                     0, "gaussian", X_REF)
        errs = []
        for lam_v in (0.0, ls):
            gm = h1_gram(ps, dps, q_t, lam_v)
            th = np.linalg.solve(gm + 1e-12 * np.trace(gm) / 2 * np.eye(2),
                                 h1_project(ps, dps, yt, dyt, q_t, lam_v))
            errs.append(np.sqrt(q_t.integrate((yt - ps.T @ th) ** 2)))
        print(f"{c:5.1f} {b:5.1f} {b**2 / (4 * c):8.4f} {ls:10.4f} {np.sqrt(ls):8.4f} "
              f"{errs[0]:9.4f} {errs[1]:10.4f} {100 * (errs[1] / errs[0] - 1):7.2f}%")
    print("  The price stays under ~8% across the sweep, and is under 2% for sharp")
    print("  crossovers.  Reason: ell* = sqrt(6) b^2/(4c) is SUB-RULE-SCALE for any")
    print("  sensible partition, and weighting slopes at a short correlation length")
    print("  barely perturbs the L2 solution.  lambda* is cheaper than expected.")


if __name__ == "__main__":
    main()
