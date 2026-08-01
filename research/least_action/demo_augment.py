"""Does off-trajectory data recover the high-rule-count failures of §9c?

Run: .venv/bin/python research/least_action/demo_augment.py

§9c diagnosed the breakdown past N=3 as distribution shift: the controller is
fitted only on optimal trajectories, so it has no data where its own closed loop
actually goes, and more rules extrapolate harder.  This tests the fix.

  V. Tube augmentation -- perturb the optimal trajectories and re-label.
     Controller-independent, computed once, cheap.
  W. DAgger -- fit, roll out, label the states the controller ACTUALLY visits,
     refit.  This is not a generic trick here: README §8d requires fitting under
     the occupation measure of the deployed controller, and training on the
     optimal controller's trajectories uses the wrong measure.  DAgger is the
     fixed-point iteration that removes the mismatch.
  X. Verdict: how much of the gap is data, and how much is the model class.
"""

from __future__ import annotations

import time

import numpy as np

from fis_twocart import (
    Metrics,
    TskController,
    TwoCart,
    augment_tube,
    dagger_states,
    distribution_shift,
    fit_consequents,
    label_state,
    optimal_trajectory,
    place_rules,
    simulate,
)

Z0S = [
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0, 0.0]),
    np.array([0.5, -0.5, 0.0, 0.0]),
    np.array([1.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 0.0]),
    np.array([-0.8, 0.3, 0.2, 0.0]),
]


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def aggregate(plant, u_fn):
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


def build_base(plant):
    samples, targets, weights = [], [], []
    for z0 in Z0S:
        ts, zs, us, _ = optimal_trajectory(plant, z0, n_knots=30, t_end=25.0)
        samples.append(zs)
        targets.append(us)
        weights.append(np.full(len(ts), ts[1] - ts[0]))
    return (np.vstack(samples), np.concatenate(targets), np.concatenate(weights))


def fit_and_eval(plant, n_rules, samples, targets, weights, seed=0):
    cen, wid = place_rules(samples, weights, n_rules, seed=seed)
    ctrl = TskController(cen, wid)
    fit_consequents(ctrl, samples, targets, weights)
    agg = aggregate(plant, ctrl)
    shift = max(distribution_shift(simulate(plant, ctrl, z0)[2], samples)
                for z0 in Z0S)
    return ctrl, agg, shift


def main() -> None:
    plant = TwoCart(m1=1.0, m2=1.0, k_lin=1.0, k_nl=0.0, u_max=1.0)
    q_mat = np.diag([1.0, 10.0, 1.0, 1.0])
    k_best = plant.lqr(q_mat, 10.0)
    lqr_ref = aggregate(plant, lambda z: float(-k_best @ z))
    assert lqr_ref is not None

    print(f"reference (best LQR, R=10): settle={lqr_ref.settling_time:.3f} "
          f"peak={lqr_ref.peak_force:.4f} energy={lqr_ref.energy:.4f}")
    base_z, base_u, base_w = build_base(plant)
    print(f"baseline training set: {len(base_z)} on-trajectory samples")

    rule("V. Tube augmentation: perturb the optimal trajectories and re-label")
    t0 = time.time()
    # Subsample before labelling: the expert solve is ~1.7 s, so labelling every
    # trajectory point would cost more than the entire rest of the study for
    # information that is largely redundant along a smooth trajectory.
    sub = np.arange(0, len(base_z), 3)
    tube_z, tube_u, tube_w = augment_tube(
        plant, base_z[sub], base_w[sub], sigma=0.35, n_per=1, seed=0
    )
    aug_z = np.vstack([base_z, tube_z])
    aug_u = np.concatenate([base_u, tube_u])
    aug_w = np.concatenate([base_w, tube_w])
    print(f"  +{len(tube_z)} off-trajectory labels in {time.time() - t0:.1f} s "
          f"(sigma = 0.35 of per-axis spread)")
    print()
    print(f"{'rules':>6} {'on-traj score':>14} {'on-traj shift':>14} "
          f"{'+tube score':>12} {'+tube shift':>12}")
    for n_rules in (3, 4, 8, 12, 16):
        _, a0, s0 = fit_and_eval(plant, n_rules, base_z, base_u, base_w)
        _, a1, s1 = fit_and_eval(plant, n_rules, aug_z, aug_u, aug_w)
        f0 = "inf" if a0 is None else f"{a0.score(lqr_ref):.4f}"
        f1 = "inf" if a1 is None else f"{a1.score(lqr_ref):.4f}"
        print(f"{n_rules:>6} {f0:>14} {s0:14.2f} {f1:>12} {s1:12.2f}")

    rule("W. DAgger: label the states the controller actually visits")
    print("  Each round: fit -> roll out -> label visited states -> add -> refit.")
    print("  This makes the training measure equal the deployment measure, which")
    print("  is what README 8d requires and what the on-trajectory fit violates.")
    print()
    print(f"{'rules':>6} {'round':>6} {'samples':>8} {'score':>9} {'shift':>7} "
          f"{'settle':>8} {'peak':>7} {'energy':>8}")
    for n_rules in (3, 8, 16):
        z, u, w = base_z.copy(), base_u.copy(), base_w.copy()
        for rnd in range(3):
            ctrl, agg, shift = fit_and_eval(plant, n_rules, z, u, w)
            sc = "inf" if agg is None else f"{agg.score(lqr_ref):.4f}"
            if agg is None:
                print(f"{n_rules:>6} {rnd:>6} {len(z):>8} {sc:>9} {shift:7.2f} "
                      f"{'--':>8} {'--':>7} {'--':>8}")
            else:
                print(f"{n_rules:>6} {rnd:>6} {len(z):>8} {sc:>9} {shift:7.2f} "
                      f"{agg.settling_time:8.3f} {agg.peak_force:7.4f} "
                      f"{agg.energy:8.4f}")
            if rnd == 2:
                break
            new_z, new_w = dagger_states(plant, ctrl, Z0S, n_per_traj=10)
            if len(new_z) == 0:
                print(f"{n_rules:>6} {rnd:>6} -- controller diverges, no states "
                      f"to label")
                break
            new_u = np.array([label_state(plant, zz) for zz in new_z])
            z = np.vstack([z, new_z])
            u = np.concatenate([u, new_u])
            w = np.concatenate([w, new_w])

    rule("X. Verdict")
    print("  See README 9f.")


if __name__ == "__main__":
    main()
