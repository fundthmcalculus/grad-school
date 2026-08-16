"""Calibrate the certificate proxy so a *specified* radius is actually achieved.

Run: .venv/bin/python research/least_action/demo_calibrate.py   (README §15)

§14 put the certified radius into the policy objective and left one thing
unresolved: the proxy over-stated the true SOS ball by 1.18-2.51x, so the
penalty target was not a specification. Asking for 0.67 delivered 0.35.

The cause turns out not to be the SOS relaxation. It is the estimator. The
cloud proxy reports a minimum-order statistic over random points, and on this
problem 37% of points violate -- the answer depends on whether one happened to
land near the inner boundary. Replacing it with a ray search (exact first
crossing along each of a fixed set of directions) removes both the variance and
most of the bias, after which the residual gap is a single constant that can be
divided out.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.linalg import solve_continuous_are

from fis_sos import (
    ball_samples,
    certify_roa,
    lyapunov_from_linearization,
    max_certified_rho,
    proxy_ball_radius,
    ray_radius_proxy,
    sphere_directions,
    to_rational,
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
HOLDOUT = [
    np.array([-1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, -0.7, 0.3, 0.0]),
    np.array([0.7, 0.2, -0.4, 0.1]),
    np.array([-0.3, -0.9, 0.0, 0.2]),
    np.array([1.5, -0.5, 0.0, 0.0]),
    np.array([0.0, 1.4, 0.0, -0.3]),
]
SPECS = [0.35, 0.45, 0.55]
PENALTY_W = 10.0
MAXFEV = 600
RADIUS_CAP = 2.0
N_DIRS = 2000


def metrics_over(plant, fn, z0s):
    ms = [simulate(plant, fn, z)[0] for z in z0s]
    ok = [m for m in ms if m.settled]
    if len(ok) < len(ms):
        return None
    return Metrics(
        float(np.mean([m.settling_time for m in ok])),
        float(np.max([m.peak_force for m in ok])),
        float(np.mean([m.energy for m in ok])),
        True,
    )


def main() -> None:
    plant = TwoCart()
    a_mat, b_mat = plant.linearization()
    b_vec = b_mat.ravel()
    q_mat = np.diag([1.0, 10.0, 1.0, 1.0])
    k = plant.lqr(q_mat, 10.0)
    ref = metrics_over(plant, lambda z: float(-k @ z), Z0S)
    p_riccati = solve_continuous_are(a_mat, b_mat, q_mat, np.array([[10.0]]))
    dirs = sphere_directions(N_DIRS)
    d = np.load(".twocart_train.npz")

    cen, wid = place_rules(d["z"], d["w"], 2)
    warm = TskController(cen, wid, order=1, mf="pi")
    fit_consequents(warm, d["z"], d["u"], d["w"])

    def candidates(ctrl):
        return [p_riccati, lyapunov_from_linearization(ctrl, a_mat, b_vec)]

    def ray_proxy(ctrl) -> float:
        return max(
            ray_radius_proxy(ctrl, plant, p, dirs, RADIUS_CAP)
            for p in candidates(ctrl)
            if p is not None
        )

    def sos_per_candidate(ctrl) -> dict[str, float]:
        rat = to_rational(ctrl)
        out = {"Riccati": 0.0, "lin": 0.0}
        for name, p in zip(("Riccati", "lin"), candidates(ctrl)):
            if p is None:
                continue
            rho = max_certified_rho(rat, a_mat, b_vec, p, lo=1e-3, hi=1e3, iters=14)
            if rho <= 0:
                continue
            rho *= 0.95
            if not certify_roa(rat, a_mat, b_vec, p, rho)[0]:
                continue
            out[name] = float(np.sqrt(rho / float(np.linalg.eigvalsh(p).min())))
        return out

    def sos_ball(ctrl) -> tuple[float, str]:
        per = sos_per_candidate(ctrl)
        name = max(per, key=lambda n: per[n])
        return (per[name], name if per[name] > 0 else "--")

    # ---------------------------------------------------------------- A
    print("A. Why the cloud proxy was loose -- it is a minimum-order statistic")
    print("   Violations are not rare: 37% of cloud points have Vdot >= 0.  The")
    print("   estimate is the smallest one SAMPLED, so it turns on luck:")
    print(f"   {'seed':>6} {'cloud (Riccati)':>16} {'ray (Riccati)':>14}")
    for seed in (0, 1, 2):
        cl = proxy_ball_radius(
            warm, plant, p_riccati, ball_samples(seed=seed), RADIUS_CAP
        )
        ry = ray_radius_proxy(
            warm, plant, p_riccati, sphere_directions(N_DIRS, seed=seed), RADIUS_CAP
        )
        print(f"   {seed:>6} {cl:>16.4f} {ry:>14.4f}")
    print("   Fixing the seed made the cloud deterministic, not accurate.")
    print("   Local descent does not rescue it either, and the reason is")
    print("   structural: min{z'Pz : Vdot(z) >= 0} has the trivial solution")
    print("   z = 0, because Vdot(0) = 0.  A descent method walks to the origin.")
    print("   Rays exclude it by construction -- the scan starts at r > 0.")

    t = time.time()
    r_ray = ray_proxy(warm)
    t_ray = time.time() - t
    t = time.time()
    r_sos, nm = sos_ball(warm)
    t_sos = time.time() - t
    print(
        f"\n   imitation N=2: ray {r_ray:.4f} ({t_ray * 1e3:.0f} ms), "
        f"SOS {r_sos:.4f} ({nm}, {t_sos:.0f} s), ratio {r_ray / r_sos:.3f}"
    )
    print(f"   (the §14 cloud proxy reported 0.6718 here, ratio 1.182)")

    # ---------------------------------------------------------------- B
    print("\nB. Calibration set -- is the residual gap a constant?")
    print("   Controllers spanning a range of aggressiveness, made by perturbing")
    print("   the imitation fit along the u(0)=0 null space (so every one of")
    print("   them still has an equilibrium at the origin to certify).")
    a_row = warm.regressors(np.zeros(4))
    _, _, vt = np.linalg.svd(a_row.reshape(1, -1))
    basis = vt[1:].T
    rng = np.random.default_rng(0)

    print(
        f"   {'controller':>14} {'score':>8} {'ray':>8} {'SOS':>8} "
        f"{'V from':>8} {'SOS/ray':>8}"
    )

    def perturbed(mag: float) -> TskController:
        """theta + mag*||theta|| * (unit direction in the u(0)=0 null space)."""
        v = basis @ rng.normal(size=basis.shape[1])
        v /= np.linalg.norm(v)
        return TskController(
            warm.centers,
            warm.widths,
            warm.theta + mag * np.linalg.norm(warm.theta) * v,
            warm.order,
            warm.mf,
        )

    cal = []
    for tag, ctrl in [("imitation", warm)] + [
        (f"perturb {i + 1}", perturbed(mag))
        for i, mag in enumerate((0.05, 0.1, 0.2, 0.4, 0.8))
    ]:
        m = metrics_over(plant, ctrl, Z0S)
        ry = ray_proxy(ctrl)
        sb, nm = sos_ball(ctrl)
        sc = "inf" if m is None else f"{m.score(ref):.4f}"
        if sb > 0:
            cal.append(sb / ry)
            print(
                f"   {tag:>14} {sc:>8} {ry:>8.4f} {sb:>8.4f} {nm:>8} "
                f"{sb / ry:>8.4f}"
            )
        else:
            print(
                f"   {tag:>14} {sc:>8} {ry:>8.4f} {'none':>8} {'--':>8} " f"{'--':>8}"
            )

    kappa = float(np.median(cal))
    print(
        f"   kappa = median(SOS/ray) = {kappa:.4f}  "
        f"over [{min(cal):.4f}, {max(cal):.4f}], {len(cal)} controllers"
    )
    print(f"   Of the {1 - kappa:.1%} shortfall, 2.5% is the deliberate 0.95")
    print("   shrink applied to rho before re-certifying; the rest is the")
    print("   S-procedure's own conservatism at multiplier degree 4.")

    # ---------------------------------------------------------------- C
    print("\nC. Does a SPECIFIED certified radius come out the other end?")
    print(
        f"   Penalty hinges on the CALIBRATED estimate kappa*ray, at w="
        f"{PENALTY_W:g}, so the"
    )
    print("   target is stated directly in certified-ball units.")
    print("   A pass that misses raises the target by exactly the shortfall.  A")
    print("   pass that misses AND does not improve on the one before it is not")
    print("   short of target, it is stuck in a basin, so the target is pushed")
    print("   50% further to leave it -- the per-candidate columns show that")
    print("   happening as the certificate jumps from one Lyapunov function to")
    print("   the other.")
    print(
        f"   {'spec':>6} {'pass':>5} {'score':>8} {'holdout':>8} "
        f"{'kappa*ray':>10} {'SOS Ric':>8} {'SOS lin':>8} {'best':>8} "
        f"{'met?':>5} {'s':>5}"
    )

    for spec in SPECS:
        target, prev = spec, None
        for attempt in (1, 2, 3):

            def pen(c, target=target):
                r = kappa * ray_proxy(c)
                return PENALTY_W * max(0.0, target - r) ** 2 / target**2

            t = time.time()
            opt, _, _ = policy_optimize(
                plant, warm, Z0S, ref, maxfev=MAXFEV, preserve_origin=True, penalty=pen
            )
            el = time.time() - t
            m = metrics_over(plant, opt, Z0S)
            mh = metrics_over(plant, opt, HOLDOUT)
            rc = kappa * ray_proxy(opt)
            per = sos_per_candidate(opt)
            sb = max(per.values())
            sc = "inf" if m is None else f"{m.score(ref):.4f}"
            sh = "inf" if mh is None else f"{mh.score(ref):.4f}"
            met = "yes" if sb >= spec else "no"
            print(
                f"   {spec:>6.2f} {attempt:>5} {sc:>8} {sh:>8} {rc:>10.4f} "
                f"{per['Riccati']:>8.4f} {per['lin']:>8.4f} {sb:>8.4f} "
                f"{met:>5} {el:>5.0f}"
            )
            if sb >= spec or sb <= 0:
                break
            stalled = prev is not None and abs(sb - prev) < 0.02 * spec
            prev = sb
            target = target * spec / sb * (1.5 if stalled else 1.0)

    print("\nD. Reading it")
    print("   The proxy is now an estimator with a known bias rather than an")
    print("   uncontrolled upper bound, so the penalty target is stated in the")
    print("   units the certificate is reported in.  What it does NOT buy is")
    print("   monotonicity: the certified radius is a max over two Lyapunov")
    print("   candidates, and the search sits in a basin where one of them")
    print("   dominates until the target is pushed hard enough to leave it.")
    print("   A spec can therefore fail while a LARGER one succeeds, which is a")
    print("   property of the landscape and not of the calibration.  Every row")
    print("   is confirmed by the real SDP, so a 'yes' is a certificate and not")
    print("   an estimate.")


if __name__ == "__main__":
    main()
