"""Certificate-aware policy optimization: trade performance against the proof.

Run: .venv/bin/python research/least_action/demo_pareto.py   (README §14)

§12e left an uncontrolled trade.  Direct policy optimization improved the
three-objective score from 1.044 to 0.773 while the SOS-certified ball shrank
from 0.568 to 0.274.  Nothing in the objective knew the certificate existed, so
the shrinkage was an accident rather than a choice.

This puts the certified radius into the objective.  The obstacle is cost: a real
SOS solve is ~1.5 s and the search takes 600 evaluations, so certifying inside
the loop costs hours.  Instead the loop uses a sampled proxy -- the smallest
sublevel value at which any sampled point has Vdot >= 0 -- which is a NECESSARY
condition for the SOS certificate and 20x cheaper than one trajectory rollout.
Optimizing against a necessary condition is sound; *reporting* it would not be,
so every controller on the front is then put through the real certificate and
the proxy's optimism is measured rather than assumed.
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
    to_rational,
)
from fis_twocart import (
    Metrics,
    TskController,
    TwoCart,
    fit_consequents,
    place_rules,
    policy_optimize,
    shaped_cost,
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
WEIGHTS = [0.0, 0.1, 0.3, 1.0, 3.0]
MAXFEV = 600
RADIUS_CAP = 2.0


def metrics_over(plant, fn, z0s):
    ms = [simulate(plant, fn, z)[0] for z in z0s]
    ok = [m for m in ms if m.settled]
    if len(ok) < len(ms):
        return None
    return Metrics(
        float(np.mean([m.settling_time for m in ok])),
        float(np.max([m.peak_force for m in ok])),
        float(np.mean([m.energy for m in ok])), True,
    )


def best_certificate(ctrl, a_mat, b_mat, p_riccati):
    """Largest SOS-certified ball over both Lyapunov candidates."""
    rat = to_rational(ctrl)
    cands = [("Riccati", p_riccati),
             ("lin", lyapunov_from_linearization(ctrl, a_mat, b_mat.ravel()))]
    best = (0.0, "--")
    for name, p_lyap in cands:
        if p_lyap is None:
            continue
        rho = max_certified_rho(rat, a_mat, b_mat.ravel(), p_lyap,
                                lo=1e-3, hi=1e3, iters=14)
        if rho <= 0:
            continue
        rho *= 0.95
        if not certify_roa(rat, a_mat, b_mat.ravel(), p_lyap, rho)[0]:
            continue
        radius = float(np.sqrt(rho / float(np.linalg.eigvalsh(p_lyap).min())))
        if radius > best[0]:
            best = (radius, name)
    return best


def main() -> None:
    plant = TwoCart()
    a_mat, b_mat = plant.linearization()
    q_mat = np.diag([1.0, 10.0, 1.0, 1.0])
    k = plant.lqr(q_mat, 10.0)
    ref = metrics_over(plant, lambda z: float(-k @ z), Z0S)
    p_riccati = solve_continuous_are(a_mat, b_mat, q_mat, np.array([[10.0]]))
    cloud = ball_samples(radius=RADIUS_CAP)
    d = np.load(".twocart_train.npz")

    cen, wid = place_rules(d["z"], d["w"], 2)
    warm = TskController(cen, wid, order=1, mf="pi")
    fit_consequents(warm, d["z"], d["u"], d["w"])

    def proxy(ctrl) -> float:
        """Certified ball radius, estimated by sampling, over both candidates.

        Taking the max mirrors how the real certificate is reported: whichever
        Lyapunov candidate certifies more wins.  Using only one would let the
        search degrade the other candidate for free.
        """
        best = proxy_ball_radius(ctrl, plant, p_riccati, cloud, RADIUS_CAP)
        p_lin = lyapunov_from_linearization(ctrl, a_mat, b_mat.ravel())
        if p_lin is not None:
            best = max(best, proxy_ball_radius(ctrl, plant, p_lin, cloud,
                                               RADIUS_CAP))
        return best

    print("A. Proxy validation on the warm start")
    t = time.time()
    r_warm = proxy(warm)
    t_proxy = time.time() - t
    t = time.time()
    r_true_warm, name_warm = best_certificate(warm, a_mat, b_mat, p_riccati)
    t_true = time.time() - t
    print(f"   imitation N=2: proxy ball {r_warm:.4f} in {t_proxy * 1e3:.0f} ms, "
          f"SOS ball {r_true_warm:.4f} ({name_warm}) in {t_true:.0f} s")
    print(f"   proxy is {r_warm / r_true_warm:.3f}x optimistic and "
          f"{t_true / t_proxy:.0f}x faster.  It over-states, which is the safe")
    print("   direction for a search and the wrong one for a claim: a sampled")
    print("   Vdot >= 0 proves SOS must fail at that rho, so the proxy upper-")
    print("   bounds what any certificate can deliver.  Optimize against it,")
    print("   never report it -- hence the SOS column below.")

    print("\nB. Pareto sweep -- performance vs certified region")
    print(f"   target radius = warm-start proxy = {r_warm:.4f}; "
          f"penalty = w * max(0, r_t - r)^2 / r_t^2")
    print(f"   {'w':>6} {'score':>8} {'holdout':>8} {'shaped':>8} "
          f"{'proxy r':>8} {'SOS ball':>9} {'V from':>8} {'u(0)':>10} {'evals':>6} {'s':>5}")

    rows = []
    m = metrics_over(plant, warm, Z0S)
    mh = metrics_over(plant, warm, HOLDOUT)
    rows.append(("none", warm, m, mh, shaped_cost(plant, warm, Z0S, ref),
                 r_warm, r_true_warm, name_warm, 0, 0.0))

    for w in WEIGHTS:
        pen = None
        if w > 0.0:
            def pen(c, w=w):
                return w * max(0.0, r_warm - proxy(c)) ** 2 / r_warm**2
        t = time.time()
        opt, _, nev = policy_optimize(plant, warm, Z0S, ref, maxfev=MAXFEV,
                                      preserve_origin=True, penalty=pen)
        el = time.time() - t
        m = metrics_over(plant, opt, Z0S)
        mh = metrics_over(plant, opt, HOLDOUT)
        r_p = proxy(opt)
        r_t, nm = best_certificate(opt, a_mat, b_mat, p_riccati)
        rows.append((f"{w:g}", opt, m, mh, shaped_cost(plant, opt, Z0S, ref),
                     r_p, r_t, nm, nev, el))

    for tag, ctrl, m, mh, sc, r_p, r_t, nm, nev, el in rows:
        s = "inf" if m is None else f"{m.score(ref):.4f}"
        sh = "inf" if mh is None else f"{mh.score(ref):.4f}"
        ball = "none" if r_t <= 0 else f"{r_t:.4f}"
        print(f"   {tag:>6} {s:>8} {sh:>8} {sc:>8.4f} {r_p:>8.4f} {ball:>9} "
              f"{nm:>8} {ctrl(np.zeros(4)):+10.1e} {nev:>6} {el:>5.0f}")

    print("\nC. Reading the front")
    real = [r for r in rows if r[2] is not None and r[6] > 0]
    swept = [r for r in real if r[0] != "none"]
    if len(real) >= 2:
        best_score = min(real, key=lambda r: r[2].score(ref))
        best_ball = max(real, key=lambda r: r[6])
        print(f"   best score {best_score[2].score(ref):.4f} at w={best_score[0]} "
              f"with ball {best_score[6]:.4f}")
        print(f"   best ball  {best_ball[6]:.4f} at w={best_ball[0]} "
              f"with score {best_ball[2].score(ref):.4f}")
        ratios = [r[5] / r[6] for r in real]
        print(f"   proxy/SOS ratio across the front: "
              f"{min(ratios):.3f} to {max(ratios):.3f}")

    balls = [r[6] for r in swept]
    breaks = [swept[i][0] for i in range(1, len(balls)) if balls[i] < balls[i - 1]]
    print("   certified ball vs w: "
          + " ".join(f"{b:.4f}" for b in balls)
          + "  (w = " + ", ".join(r[0] for r in swept) + ")")
    if breaks:
        print(f"   monotone in w except at w={', '.join(breaks)} -- Powell is a "
              f"local search on a")
        print("   non-convex landscape, so each row is a local optimum at its own")
        print("   weight, not the Pareto-optimal point for that trade-off.")
    else:
        print("   monotone in w across every tested weight.")

    base = swept[0]
    dom = [r for r in swept[1:]
           if r[2].score(ref) <= base[2].score(ref)
           and r[3].score(ref) <= base[3].score(ref) and r[6] >= base[6]]
    if dom:
        b = min(dom, key=lambda r: r[2].score(ref))
        print(f"   w={b[0]} DOMINATES w=0 on all three axes: score "
              f"{b[2].score(ref):.4f} < {base[2].score(ref):.4f}, held out "
              f"{b[3].score(ref):.4f} < {base[3].score(ref):.4f},")
        print(f"   certified ball {b[6]:.4f} > {base[6]:.4f}.  The first increment")
        print("   of certificate-awareness costs nothing: penalizing aggression")
        print("   also regularizes a search that was overfitting six initial")
        print("   conditions.  Past that point the trade is real and w prices it.")
    else:
        print("   no weight dominates w=0; the trade is strict throughout.")


if __name__ == "__main__":
    main()
