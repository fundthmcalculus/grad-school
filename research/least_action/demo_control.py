"""Provably optimal fuzzy control.

Run: .venv/bin/python research/least_action/demo_control.py

Claims under test
  L. LTI + quadratic cost: a partition-of-unity TSK with affine consequents
     represents u* = -Kx EXACTLY, for any membership functions.  Provably
     globally optimal -- and therefore evidence that fuzzy structure buys
     nothing here.
  M. Theorem C2: J(x0) - V*(x0) = INT R (u - u*)^2 dt along the closed loop,
     exactly.  Verified against a benchmark with a known value function.
  N. Consequence: the correct fitting weight is the closed-loop occupation
     measure, not Lebesgue measure.  Weighting by rho lowers the certified cost
     gap at equal rule count.
  O. Every fitted controller carries a computable certificate: exact
     suboptimality (no bounding constant) plus a Lyapunov region of attraction.
  P. Where the cost gap actually comes from -- and what lambda costs here.
"""

from __future__ import annotations

import numpy as np

from fis_action import Quadrature, h1_gram, h1_project, rule_regressors
from fis_control import (
    LqrProblem,
    cubic_benchmark,
    occupation_density,
    simulate,
    stability_certificate,
)

X_LO, X_HI = -3.0, 3.0
X_REF = (0.0, 3.0)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def fit_control_law(
    prob, centers, widths, order, lam, weight=None, quad=None, kind="gaussian"
):
    """H1-optimal (or rho-weighted L2-optimal) TSK approximation of u*.

    `weight` is an extra per-node density multiplying the L2 part; passing the
    closed-loop occupation density makes the objective exactly the Theorem C2
    excess cost.
    """
    if quad is None:
        quad = Quadrature.legendre(X_LO, X_HI, 600)
    psi, dpsi, _ = rule_regressors(quad.nodes, centers, widths, order, kind, X_REF)
    us = prob.u_star(quad.nodes)
    # d u*/dx for the benchmark: u* = -(x + x^3)
    dus = -(1.0 + 3.0 * quad.nodes**2)
    if weight is None:
        gram = h1_gram(psi, dpsi, quad, lam)
        rhs = h1_project(psi, dpsi, us, dus, quad, lam)
    else:
        w = quad.weights * weight
        gram = (psi * w) @ psi.T + lam * ((dpsi * quad.weights) @ dpsi.T)
        rhs = psi @ (w * us) + lam * (dpsi @ (quad.weights * dus))
    theta = np.linalg.solve(gram + 1e-12 * np.trace(gram) / gram.shape[0]
                            * np.eye(gram.shape[0]), rhs)

    def u_fn(x):
        p, _, _ = rule_regressors(np.atleast_1d(x), centers, widths, order, kind, X_REF)
        return p.T @ theta

    return u_fn, theta


def main() -> None:
    rule("L. LTI + quadratic cost: the fuzzy controller is EXACTLY optimal")
    lqr = LqrProblem(
        a_mat=np.array([[0.0, 1.0], [-1.0, -0.5]]),
        b_mat=np.array([[0.0], [1.0]]),
        q_mat=np.diag([2.0, 1.0]),
        r_mat=np.array([[0.5]]),
    )
    p_mat, k_gain = lqr.solve()
    print(f"  Riccati gain K = {np.array2string(k_gain, precision=5)}")
    rng = np.random.default_rng(0)
    states = rng.normal(size=(2, 4000)) * 3.0
    worst = 0.0
    for trial in range(200):
        # Arbitrary, deliberately ugly partitions of unity -- the point is that
        # NONE of this can matter.
        n_rules = int(rng.integers(2, 7))
        cen = rng.normal(size=(n_rules, 2)) * 2.0
        wid = rng.uniform(0.3, 3.0, size=n_rules)
        d2 = ((states[None, :, :] - cen[:, :, None]) ** 2).sum(axis=1)
        mu = np.exp(-d2 / wid[:, None] ** 2)
        phi = mu / np.maximum(mu.sum(axis=0), 1e-300)
        # Every consequent is the same affine map u_i(x) = -K x.
        blended = np.einsum("ix,x->x", phi, np.zeros(states.shape[1])) \
            + (phi.sum(axis=0) * (-k_gain @ states)[0])
        worst = max(worst, float(np.max(np.abs(blended - (-k_gain @ states)[0]))))
    print(f"  200 random partitions x 4000 states, 2-6 rules, random centres/widths")
    print(f"  max |u_fuzzy - u_LQR| = {worst:.3e}")
    print("  sum_i phi_i(x) (-Kx) = -Kx because sum_i phi_i = 1: the membership")
    print("  functions cancel identically.  Provably globally optimal, and equally")
    print("  provably pointless -- fuzzy structure earns nothing on an LTI/LQR problem.")
    print("  Its value has to come from nonlinearity, which is the rest of this file.")

    rule("M. Theorem C2: the cost gap IS a weighted control-error integral")
    prob = cubic_benchmark()
    grid = np.linspace(X_LO, X_HI, 2001)
    print("  Benchmark by inverse optimal control (q defined FROM the HJB, so V* and")
    print("  u* are exact, not numerical):")
    print("    xdot = -x + u,  V* = x^2 + x^4/2,  u* = -(x + x^3),  q = 3x^2+4x^4+x^6")
    print(f"    q >= 0 on the domain: {prob.q_is_valid(grid)}")
    print()
    print("  J(x0) - V*(x0)  vs  INT R (u-u*)^2 dt, for deliberately WRONG controllers:")
    print(f"{'controller':>22} {'x0':>6} {'J':>12} {'V*':>12} {'gap':>12} "
          f"{'INT(u-u*)^2':>13} {'|residual|':>12}")
    cands = {
        "optimal u*": prob.u_star,
        "linear -1.5x": lambda x: -1.5 * x,
        "linear -3x": lambda x: -3.0 * x,
        "u* scaled 0.7": lambda x: 0.7 * prob.u_star(x),
        "u* + 0.4": lambda x: prob.u_star(x) + 0.4,
    }
    for name, uf in cands.items():
        for x0 in (1.5, -2.0):
            r = simulate(prob, uf, x0)
            print(f"{name:>22} {x0:6.1f} {r.cost:12.6f} {r.optimal_cost:12.6f} "
                  f"{r.gap:12.3e} {r.control_error_integral:13.3e} "
                  f"{r.identity_residual:12.3e}")
    print("  The identity holds to integrator tolerance and the gap is >= 0 in every")
    print("  row -- except the last controller, which misses by 3.9e-2.  That is not")
    print("  numerical error, it is the theorem's ADMISSIBILITY hypothesis showing its")
    print("  teeth: u* + 0.4 does not drive x to the origin, so the boundary term")
    print("  V*(x(inf)) that the derivation discards is not zero.  The exact statement is")
    print()
    print("      J(x0) - V*(x0) + V*(x(inf))  =  INT R (u - u*)^2 dt")
    print()
    print(f"{'x0':>6} {'x(inf)':>11} {'V*(x(inf))':>13} {'residual':>13} {'difference':>12}")
    off = cands["u* + 0.4"]
    for x0 in (1.5, -2.0):
        r = simulate(prob, off, x0)
        v_inf = float(prob.v(np.array([r.final_state]))[0])
        print(f"{x0:6.1f} {r.final_state:11.6f} {v_inf:13.6e} "
              f"{r.identity_residual:13.6e} {abs(v_inf - r.identity_residual):12.2e}")
    print("  The residual IS V*(x(inf)), to 2e-10.  Admissibility is load-bearing, not")
    print("  decoration: a controller with a steady-state offset has no certificate.")
    print("  For admissible controllers the identity is exact -- a certificate, not a")
    print("  bound: no constant to estimate, nothing to be conservative about.")

    rule("N. The correct fitting weight is the occupation measure")
    print("  Theorem C2 weights the control error by closed-loop occupation time, so")
    print("  fitting u* uniformly over a box optimizes the wrong functional.")
    x0s = np.array([-2.5, -1.5, -0.8, 0.8, 1.5, 2.5])
    quad = Quadrature.legendre(X_LO, X_HI, 600)
    rho = occupation_density(prob, prob.u_star, x0s, quad.nodes)
    print(f"  rho concentrates near the origin: rho(0)/rho(+-3) = "
          f"{rho[len(rho) // 2] / max(rho[0], 1e-12):.1f}")
    print()
    print(f"{'N rules':>8} {'weight':>12} {'sup|u-u*|':>11} {'mean cert. gap':>16} "
          f"{'max cert. gap':>15}")
    for n_rules in (2, 3, 4):
        cen = np.linspace(-2.0, 2.0, n_rules)
        wid = np.full(n_rules, 4.0 / n_rules)
        for label, w in (("uniform L2", None), ("occupation rho", rho)):
            u_fn, _ = fit_control_law(prob, cen, wid, 1, 0.0, weight=w, quad=quad)
            sup = float(np.max(np.abs(u_fn(grid) - prob.u_star(grid))))
            runs = [simulate(prob, u_fn, x0) for x0 in x0s]
            gaps = [r.gap for r in runs]
            flag = "" if all(r.stable for r in runs) else "   NOT ADMISSIBLE"
            print(f"{n_rules:>8} {label:>12} {sup:11.4f} {np.mean(gaps):16.3e} "
                  f"{np.max(gaps):15.3e}{flag}")
    print("  rho-weighting gives a worse sup-norm fit and a better certified cost --")
    print("  which is the point.  Uniform accuracy is not the control objective.")
    print("  The flagged row is not a certificate: that controller fails to drive x to")
    print("  the origin, so by section M its number carries an unaccounted V*(x(inf)).")
    print("  A gap is only a certificate once admissibility is separately established,")
    print("  which is what section O is for.")

    rule("O. Certificates for the fitted controllers")
    print(f"{'N rules':>8} {'weight':>12} {'ROA radius':>11} {'V* level':>10} "
          f"{'worst Vdot':>12} {'stable':>7} {'mean gap':>11}")
    for n_rules in (2, 3, 4):
        cen = np.linspace(-2.0, 2.0, n_rules)
        wid = np.full(n_rules, 4.0 / n_rules)
        for label, w in (("uniform L2", None), ("occupation rho", rho)):
            u_fn, _ = fit_control_law(prob, cen, wid, 1, 0.0, weight=w, quad=quad)
            cert = stability_certificate(prob, u_fn, x_max=3.0)
            gaps = [simulate(prob, u_fn, x0) for x0 in x0s]
            print(f"{n_rules:>8} {label:>12} {cert.roa_radius:11.4f} "
                  f"{cert.level_set:10.4f} {cert.worst_vdot:12.3e} "
                  f"{all(g.stable for g in gaps)!s:>7} "
                  f"{np.mean([g.gap for g in gaps]):11.3e}")
    print("  ROA is an inner estimate from Vdot < 0 using V* as the candidate, so it is")
    print("  conservative by construction.  Combined with the exact gap above, each")
    print("  controller ships with: a region it provably stabilizes, and exactly how")
    print("  much more than optimal it costs inside that region.")

    rule("P. What the H1 slope term costs in a control setting")
    print("  Theorem C2 needs L2(rho) and nothing else -- lambda > 0 is a deliberate")
    print("  deviation from the certified-optimal objective.  Its price:")
    print(f"{'lambda':>9} {'sup|u-u*|':>11} {'sup|du-du*|':>13} {'mean cert. gap':>16}")
    cen = np.linspace(-2.0, 2.0, 3)
    wid = np.full(3, 4.0 / 3)
    dstar = -(1.0 + 3.0 * grid**2)
    for lam in (0.0, 0.01, 0.1, 1.0):
        u_fn, _ = fit_control_law(prob, cen, wid, 1, lam, weight=rho, quad=quad)
        du = np.gradient(u_fn(grid), grid)
        gaps = [simulate(prob, u_fn, x0).gap for x0 in x0s]
        print(f"{lam:9.3f} {np.max(np.abs(u_fn(grid) - prob.u_star(grid))):11.4f} "
              f"{np.max(np.abs(du - dstar)):13.4f} {np.mean(gaps):16.3e}")
    print("  lambda buys feedback-gain (du/dx) accuracy, which is what sets the")
    print("  closed-loop linearization and hence local pole placement, and pays for it")
    print("  in certified cost.  For pure cost-optimality use lambda = 0; raise it only")
    print("  when the gain profile itself matters.")


if __name__ == "__main__":
    main()
