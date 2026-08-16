"""Certify the directly-optimized controller (README §8h, §12e).

Run: .venv/bin/python research/least_action/demo_certified_policy.py

§8h produced the best controller in the study and the only one without a
stability certificate. §12d built the machinery. Joining them needs one thing
that is not obvious: the policy search must be restricted to the subspace
u(0) = 0, or it moves the equilibrium off the origin and there is nothing left
to certify.

Two Lyapunov candidates are tried for each controller -- the Riccati matrix, and
one built from the controller's own closed-loop linearization -- because neither
dominates and the certified region depends strongly on the shape of V.
"""

from __future__ import annotations

import time
import warnings
from math import comb

import numpy as np
from scipy.linalg import solve_continuous_are

from fis_sos import (
    certify_roa,
    lyapunov_from_linearization,
    max_certified_rho,
    to_rational,
    unsaturated_radius,
)
from fis_twocart import (
    Metrics,
    TskController,
    TwoCart,
    fit_consequents,
    place_rules,
    policy_optimize,
    simulate,
)

warnings.filterwarnings("ignore")

Z0S = [
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.5, -0.5, 0.0, 0.0]),
    np.array([1.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0]),
    np.array([-0.8, 0.3, 0.2, 0.0]),
]


def metrics_over(plant, fn):
    ms = [simulate(plant, fn, z)[0] for z in Z0S]
    ok = [m for m in ms if m.settled]
    if len(ok) < len(ms):
        return None
    return Metrics(
        float(np.mean([m.settling_time for m in ok])),
        float(np.max([m.peak_force for m in ok])),
        float(np.mean([m.energy for m in ok])),
        True,
    )


def best_certificate(ctrl, plant, a_mat, b_mat, candidates):
    """Largest certified ball over the supplied Lyapunov candidates."""
    rat = to_rational(ctrl)
    best = (0.0, None, None, None)
    for name, p_lyap in candidates:
        if p_lyap is None:
            continue
        rho = max_certified_rho(
            rat, a_mat, b_mat.ravel(), p_lyap, lo=1e-3, hi=1e3, iters=14
        )
        if rho <= 0:
            continue
        rho *= 0.95
        ok, _, rel = certify_roa(rat, a_mat, b_mat.ravel(), p_lyap, rho)
        if not ok:
            continue
        radius = float(np.sqrt(rho / float(np.linalg.eigvalsh(p_lyap).min())))
        if radius > best[0]:
            best = (radius, name, rho, rel)
    return best


def main() -> None:
    plant = TwoCart()
    a_mat, b_mat = plant.linearization()
    q_mat = np.diag([1.0, 10.0, 1.0, 1.0])
    k = plant.lqr(q_mat, 10.0)
    ref = metrics_over(plant, lambda z: float(-k @ z))
    p_riccati = solve_continuous_are(a_mat, b_mat, q_mat, np.array([[10.0]]))
    d = np.load(".twocart_train.npz")

    print("A. SOS problem size against rule count")
    print("   deg Q = 2(N-1), so the certificate cost grows fast in N:")
    print(f"   {'N':>3} {'deg Q':>6} {'deg P':>6} {'SOS deg':>8} {'Gram':>9}")
    for n in (2, 3, 4, 6, 8):
        dq = 2 * (n - 1)
        sd = max(dq + 2, dq + 4)
        half = (sd + 1) // 2
        kk = sum(comb(4 + j - 1, j) for j in range(half + 1))
        print(f"   {n:>3} {dq:>6} {dq + 1:>6} {sd:>8} {f'{kk}x{kk}':>9}")
    print("   N=2 solves in seconds; N=8 is a 715x715 SDP and is out of reach")
    print("   with these solvers.  Certification is tractable at low rule count,")
    print("   which is where §8b said the method converges anyway.")

    print("\nB. Certified controllers (pi MFs, u(0)=0, multiplier degree 4, CLARABEL)")
    print(
        f"   {'controller':>18} {'score':>8} {'u(0)':>11} {'V from':>12} "
        f"{'ball':>7} {'unsat':>7} {'rel':>9}"
    )

    radius_opt = 0.0
    rows = []
    for n_rules in (2, 3):
        cen, wid = place_rules(d["z"], d["w"], n_rules)
        imit = TskController(cen, wid, order=1, mf="pi")
        fit_consequents(imit, d["z"], d["u"], d["w"])
        rows.append((f"imitation N={n_rules}", imit))

    cen, wid = place_rules(d["z"], d["w"], 2)
    warm = TskController(cen, wid, order=1, mf="pi")
    fit_consequents(warm, d["z"], d["u"], d["w"])
    t = time.time()
    opt, _, nev = policy_optimize(
        plant, warm, Z0S, ref, maxfev=600, preserve_origin=True
    )
    opt_time = time.time() - t
    rows.append(("direct opt N=2", opt))

    for tag, ctrl in rows:
        m = metrics_over(plant, ctrl)
        score = "inf" if m is None else f"{m.score(ref):.4f}"
        cands = [
            ("Riccati", p_riccati),
            ("lin", lyapunov_from_linearization(ctrl, a_mat, b_mat.ravel())),
        ]
        radius, name, rho, rel = best_certificate(ctrl, plant, a_mat, b_mat, cands)
        if tag.startswith("direct opt"):
            radius_opt = radius
        r_sat = unsaturated_radius(ctrl, plant.u_max)
        if radius <= 0:
            print(
                f"   {tag:>18} {score:>8} {ctrl(np.zeros(4)):+11.1e} "
                f"{'--':>12} {'none':>7} {r_sat:7.3f} {'--':>9}"
            )
        else:
            print(
                f"   {tag:>18} {score:>8} {ctrl(np.zeros(4)):+11.1e} "
                f"{name:>12} {radius:7.3f} {r_sat:7.3f} {rel:9.1e}"
            )
    print(f"   (direct optimization: {nev} evals, {opt_time:.0f}s)")

    print("\nC. What the subspace restriction actually buys")
    free, _, _ = policy_optimize(
        plant, warm, Z0S, ref, maxfev=200, preserve_origin=False
    )
    cands_f = [
        ("Riccati", p_riccati),
        ("lin", lyapunov_from_linearization(free, a_mat, b_mat.ravel())),
    ]
    r_free, n_free, _, _ = best_certificate(free, plant, a_mat, b_mat, cands_f)
    print(
        f"   unconstrained search: u(0) = {free(np.zeros(4)):+.3e}, "
        f"certified ball {r_free:.3f} ({n_free})"
    )
    print(
        f"   constrained search:   u(0) = {opt(np.zeros(4)):+.3e}, "
        f"certified ball {radius_opt:.3f}"
    )
    print("   The unconstrained optimum still certifies here, and to a LARGER")
    print("   ball -- because starting from a constrained warm start it drifts")
    print("   only to u(0) ~ 1e-9, which is below what the SDP can resolve.")
    print("   So the constraint does not rescue a certificate that would")
    print("   otherwise fail; it makes the certificate EXACT rather than valid")
    print("   only up to a 1e-9 perturbation of the plant.  The u(0) = 2.9e-4 of")
    print("   an unconstrained *imitation* fit (§12b) is a different matter -- at")
    print("   that magnitude the certificate genuinely fails.")


if __name__ == "__main__":
    main()
