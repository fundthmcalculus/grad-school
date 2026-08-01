"""Direct policy optimization on the FIS parameters.

Run: .venv/bin/python research/least_action/demo_policyopt.py

§9e ruled out data as the cause of the 0.8629 plateau; §9f ruled out the model
class and showed open-loop fit accuracy and closed-loop score are *inversely*
related. By elimination the binding constraint was the imitation objective
itself. This tests the implied fix: stop fitting sampled targets and optimize
the closed-loop objective directly over the FIS parameters.

The price is explicit. The shaped cost is not quadratic in the consequents, so
variable projection no longer applies -- there is no globally optimal linear
solve here, only a derivative-free search over a non-convex landscape, and every
guarantee from §3a and §8b is surrendered.

  AA. Warm-started from the imitation fit, by rule count.
  AB. Does the warm start matter, or would a cold start do?
  AC. Tuning the antecedents too.
  AD. Where the gain comes from, per objective.
"""

from __future__ import annotations

import os
import time

import numpy as np

from fis_twocart import (
    Metrics,
    TskController,
    TwoCart,
    fit_consequents,
    optimal_trajectory,
    place_rules,
    policy_optimize,
    shaped_cost,
    simulate,
)

CACHE = os.path.join(os.path.dirname(__file__), ".twocart_train.npz")

Z0S = [
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.5, -0.5, 0.0, 0.0]),
    np.array([1.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0]),
    np.array([-0.8, 0.3, 0.2, 0.0]),
]
BUDGET = 600


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def metrics_of(plant, u_fn):
    ms = [simulate(plant, u_fn, z0)[0] for z0 in Z0S]
    ok = [m for m in ms if m.settled]
    if len(ok) < len(ms):
        return None
    return Metrics(
        float(np.mean([m.settling_time for m in ok])),
        float(np.max([m.peak_force for m in ok])),
        float(np.mean([m.energy for m in ok])),
        True,
    )


def training_set(plant):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["z"], d["u"], d["w"]
    zs, us, ws = [], [], []
    for z0 in Z0S:
        ts, z, u, _ = optimal_trajectory(plant, z0, n_knots=30, t_end=25.0)
        zs.append(z)
        us.append(u)
        ws.append(np.full(len(ts), ts[1] - ts[0]))
    z, u, w = np.vstack(zs), np.concatenate(us), np.concatenate(ws)
    np.savez(CACHE, z=z, u=u, w=w)
    return z, u, w


def main() -> None:
    plant = TwoCart(m1=1.0, m2=1.0, k_lin=1.0, k_nl=0.0, u_max=1.0)
    q_mat = np.diag([1.0, 10.0, 1.0, 1.0])
    k_best = plant.lqr(q_mat, 10.0)
    ref = metrics_of(plant, lambda z: float(-k_best @ z))
    assert ref is not None
    samples, targets, weights = training_set(plant)

    def score(ct):
        m = metrics_of(plant, ct)
        return float("inf") if m is None else m.score(ref)

    print(f"reference (best LQR): score 1.0000, settle {ref.settling_time:.3f}, "
          f"peak {ref.peak_force:.4f}, energy {ref.energy:.4f}")
    print(f"best imitation result (§9f): 0.8487   |   open-loop reference: 0.8183")
    print(f"budget: {BUDGET} closed-loop evaluations per optimization")

    rule("AA. Direct policy optimization, warm-started from the imitation fit")
    print(f"{'rules':>6} {'params':>7} {'imitation':>10} {'direct':>9} "
          f"{'improve':>9} {'evals':>7} {'sec':>6}")
    best_overall = None
    for n_rules in (2, 3, 4):
        cen, wid = place_rules(samples, weights, n_rules)
        ctrl = TskController(cen, wid, order=1)
        fit_consequents(ctrl, samples, targets, weights)
        s0 = score(ctrl)
        t0 = time.time()
        opt, _, n_ev = policy_optimize(plant, ctrl, Z0S, ref, maxfev=BUDGET)
        s1 = score(opt)
        print(f"{n_rules:>6} {opt.n_params():>7} {s0:10.4f} {s1:9.4f} "
              f"{100 * (1 - s1 / s0):8.1f}% {n_ev:>7} {time.time() - t0:6.0f}")
        if best_overall is None or s1 < best_overall[0]:
            best_overall = (s1, n_rules, opt)

    rule("AB. Does the warm start matter?")
    print("  Same 3-rule structure, consequents started from zero instead of the")
    print("  imitation fit.  If a cold start matches, the imitation stage is")
    print("  redundant; if it does not, imitation is a useful initializer even")
    print("  though it caps out on its own.")
    cen, wid = place_rules(samples, weights, 3)
    cold = TskController(cen, wid, np.zeros(15), order=1)
    t0 = time.time()
    cold_opt, _, n_ev = policy_optimize(plant, cold, Z0S, ref, maxfev=BUDGET)
    print(f"  cold start: {score(cold):.4f} -> {score(cold_opt):.4f} "
          f"({n_ev} evals, {time.time() - t0:.0f} s)")

    rule("AC. Tuning the antecedents as well")
    print("  Rule centres and widths join the search: 15 + 24 = 39 parameters.")
    cen, wid = place_rules(samples, weights, 3)
    ctrl = TskController(cen, wid, order=1)
    fit_consequents(ctrl, samples, targets, weights)
    t0 = time.time()
    full, _, n_ev = policy_optimize(plant, ctrl, Z0S, ref, maxfev=BUDGET,
                                    tune_antecedents=True)
    print(f"  consequents only: {best_overall[0]:.4f}")
    print(f"  + antecedents:    {score(full):.4f}  ({n_ev} evals, "
          f"{time.time() - t0:.0f} s)")
    print("  (equal evaluation budget over 2.6x the parameters, so a worse result")
    print("   here is a budget statement, not a capacity statement)")

    rule("AD. Where the gain comes from")
    best_score, best_n, best_ctrl = best_overall
    cen, wid = place_rules(samples, weights, best_n)
    imit = TskController(cen, wid, order=1)
    fit_consequents(imit, samples, targets, weights)
    mi, mo = metrics_of(plant, imit), metrics_of(plant, best_ctrl)
    print(f"{'':>14} {'settle':>9} {'peak |u|':>9} {'energy':>9} {'score':>8}")
    print(f"{'best LQR':>14} {ref.settling_time:9.3f} {ref.peak_force:9.4f} "
          f"{ref.energy:9.4f} {1.0:8.4f}")
    print(f"{'imitation':>14} {mi.settling_time:9.3f} {mi.peak_force:9.4f} "
          f"{mi.energy:9.4f} {mi.score(ref):8.4f}")
    print(f"{'direct opt':>14} {mo.settling_time:9.3f} {mo.peak_force:9.4f} "
          f"{mo.energy:9.4f} {mo.score(ref):8.4f}")
    print()
    print(f"  vs best LQR:        {100 * (1 - mo.score(ref)):.1f}% better")
    print(f"  vs best imitation:  {100 * (1 - mo.score(ref) / 0.8487):.1f}% better")
    print(f"  vs open-loop ref:   {100 * (1 - mo.score(ref) / 0.8183):.1f}% better")
    print("  The open-loop reference had full knowledge of z0; a feedback law")
    print("  beating it is not a contradiction, it is what feedback buys.")


if __name__ == "__main__":
    main()
