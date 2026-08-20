"""Held-out generalization check for the §9h direct-optimized controller.

Run: .venv/bin/python research/least_action/demo_holdout.py

Every score in §9 is measured on the same six initial conditions the controller
was fitted and optimized on.  For the imitation fits that is a mild concern --
they have few parameters and never see the objective.  For direct policy
optimization it is a real one: 10 parameters tuned against 6 trajectories could
simply be memorizing them.

This measures the gap rather than arguing about it: train on Z0_TRAIN, score on
Z0_TEST, and compare against LQR, which has no opportunity to overfit at all
because it never saw either set.
"""

from __future__ import annotations

import os

import numpy as np

from fis_twocart import (
    Metrics,
    TskController,
    TwoCart,
    fit_consequents,
    optimal_trajectory,
    place_rules,
    policy_optimize,
    simulate,
)

CACHE = os.path.join(os.path.dirname(__file__), ".twocart_train.npz")

Z0_TRAIN = [
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.5, -0.5, 0.0, 0.0]),
    np.array([1.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0]),
    np.array([-0.8, 0.3, 0.2, 0.0]),
]

# Held out: different directions, different magnitudes, and two that are larger
# than anything in training so extrapolation is genuinely tested.
Z0_TEST = [
    np.array([-1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, -0.7, 0.3, 0.0]),
    np.array([0.7, 0.2, -0.4, 0.1]),
    np.array([-0.3, -0.9, 0.0, 0.2]),
    np.array([1.5, -0.5, 0.0, 0.0]),
    np.array([0.0, 1.4, 0.0, -0.3]),
]


def metrics_over(plant, u_fn, z0s):
    ms = [simulate(plant, u_fn, z0)[0] for z0 in z0s]
    ok = [m for m in ms if m.settled]
    if len(ok) < len(ms):
        return None, sum(1 for m in ms if m.settled), len(ms)
    return (
        Metrics(
            float(np.mean([m.settling_time for m in ok])),
            float(np.max([m.peak_force for m in ok])),
            float(np.mean([m.energy for m in ok])),
            True,
        ),
        len(ok),
        len(ms),
    )


def main() -> None:
    plant = TwoCart(m1=1.0, m2=1.0, k_lin=1.0, k_nl=0.0, u_max=1.0)
    q_mat = np.diag([1.0, 10.0, 1.0, 1.0])
    k_best = plant.lqr(q_mat, 10.0)

    def lqr_fn(z):
        return float(-k_best @ z)

    # Normalize each split against LQR on that same split, so train and test
    # scores are directly comparable rather than sharing a reference.
    ref_tr, _, _ = metrics_over(plant, lqr_fn, Z0_TRAIN)
    ref_te, _, _ = metrics_over(plant, lqr_fn, Z0_TEST)
    assert ref_tr is not None and ref_te is not None

    if os.path.exists(CACHE):
        d = np.load(CACHE)
        samples, targets, weights = d["z"], d["u"], d["w"]
    else:
        zs, us, ws = [], [], []
        for z0 in Z0_TRAIN:
            ts, z, u, _ = optimal_trajectory(plant, z0, n_knots=30, t_end=25.0)
            zs.append(z)
            us.append(u)
            ws.append(np.full(len(ts), ts[1] - ts[0]))
        samples = np.vstack(zs)
        targets = np.concatenate(us)
        weights = np.concatenate(ws)

    print("Held-out initial conditions, including two larger than any in training.")
    print(
        f"{'controller':>22} {'train':>9} {'test':>9} {'test settled':>13} "
        f"{'degradation':>12}"
    )

    def report(name, fn):
        mt, _, _ = metrics_over(plant, fn, Z0_TRAIN)
        me, n_ok, n_all = metrics_over(plant, fn, Z0_TEST)
        s_tr = "inf" if mt is None else f"{mt.score(ref_tr):.4f}"
        s_te = "inf" if me is None else f"{me.score(ref_te):.4f}"
        deg = (
            "--"
            if mt is None or me is None
            else f"{100 * (me.score(ref_te) / mt.score(ref_tr) - 1):+.1f}%"
        )
        print(f"{name:>22} {s_tr:>9} {s_te:>9} {f'{n_ok}/{n_all}':>13} {deg:>12}")

    report("best LQR", lqr_fn)

    cen, wid = place_rules(samples, weights, 2)
    imit = TskController(cen, wid, order=1)
    fit_consequents(imit, samples, targets, weights)
    report("imitation N=2", imit)

    cen3, wid3 = place_rules(samples, weights, 3)
    imit3 = TskController(cen3, wid3, order=1)
    fit_consequents(imit3, samples, targets, weights)
    report("imitation N=3", imit3)

    print("  optimizing (600 evals, train split only)...", flush=True)
    opt, _, _ = policy_optimize(plant, imit, Z0_TRAIN, ref_tr, maxfev=600)
    report("direct opt N=2", opt)

    print()
    print("  LQR is the control: it never saw either split, so its train/test")
    print("  difference is pure split-to-split variation and calibrates how much")
    print("  degradation is meaningful.")


if __name__ == "__main__":
    main()
