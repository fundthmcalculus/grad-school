"""Two-cart benchmark: how fast does the fuzzy least-action model converge?

Run: .venv/bin/python research/least_action/demo_twocart.py

Objectives: settling time, peak force, total energy INT u^2 dt.

Sections
  Q. Plant, and confirmation that the LINEAR case is degenerate (Theorem C1).
  R. Baselines: LQR family sweep, and the open-loop trajectory-optimization
     reference used as the training target (NOT a certified optimum -- see R).
  S. Convergence of the fuzzy least-action controller against rule count, with
     NO refinement: cluster the optimal trajectories, variable-project the
     consequents, stop.
  T. Same on the nonlinear-spring plant, where Theorem C1 no longer applies.
  U. Verdict.
"""

from __future__ import annotations

import numpy as np

from fis_twocart import (
    Metrics,
    TskController,
    TwoCart,
    distribution_shift,
    fit_consequents,
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


def aggregate(plant, u_fn, z0s, ref=None):
    ms = [simulate(plant, u_fn, z0)[0] for z0 in z0s]
    ok = [m for m in ms if m.settled]
    if not ok:
        return None, ms
    agg = Metrics(
        settling_time=float(np.mean([m.settling_time for m in ok])),
        peak_force=float(np.max([m.peak_force for m in ok])),
        energy=float(np.mean([m.energy for m in ok])),
        settled=len(ok) == len(ms),
    )
    return agg, ms


def build_training_set(plant, z0s, t_end=25.0, n_knots=30):
    """Optimal trajectories -> occupation-weighted (state, control) samples."""
    samples, targets, weights = [], [], []
    per_z0 = {}
    for z0 in z0s:
        ts, zs, us, _ = optimal_trajectory(plant, z0, n_knots=n_knots, t_end=t_end)
        dt = ts[1] - ts[0]
        samples.append(zs)
        targets.append(us)
        # dt weighting IS the occupation measure -- no kernel estimate needed,
        # because the samples already lie along the closed-loop trajectory.
        weights.append(np.full(len(ts), dt))
        per_z0[tuple(z0)] = (ts, zs, us)
    return (
        np.vstack(samples),
        np.concatenate(targets),
        np.concatenate(weights),
        per_z0,
    )


def main() -> None:
    plant = TwoCart(m1=1.0, m2=1.0, k_lin=1.0, k_nl=0.0, u_max=1.0)
    q_mat = np.diag([1.0, 10.0, 1.0, 1.0])

    rule("Q. Plant, and the degeneracy the objectives have to break")
    a_mat, b_mat = plant.linearization()
    eig = np.linalg.eigvals(a_mat)
    print(f"  m1=m2=1, k=1, |u| <= {plant.u_max}.  Force on cart 1, cart 2 is the")
    print(f"  one that must be stilled -- non-collocated, flexible mode.")
    print(f"  open-loop poles: {np.array2string(np.sort_complex(eig), precision=4)}")
    print(f"  (undamped oscillatory pair on the imaginary axis: nothing decays")
    print(f"   on its own, so every metric below is entirely the controller's doing)")
    print()
    k_lqr = plant.lqr(q_mat, 1.0)
    print(f"  LQR gain K = {np.array2string(k_lqr, precision=4)}")
    print("  Theorem C1 (README 8a): on this LINEAR plant with a quadratic cost, any")
    print("  partition-of-unity TSK with affine consequents reproduces -Kx exactly and")
    print("  the membership functions cancel.  A fuzzy controller cannot beat LQR at")
    print("  LQR's own objective.  What follows is only a fair contest because")
    print("  settling time and peak force are NOT quadratic, and |u| <= u_max makes")
    print("  the true optimum non-smooth.")

    rule("R. Baselines: LQR sweep, and the open-loop training reference")
    print(
        f"{'controller':>22} {'settle':>9} {'peak |u|':>9} {'energy':>9} {'all settled':>12}"
    )
    lqr_results = {}
    for r_val in (0.1, 0.3, 1.0, 3.0, 10.0):
        k = plant.lqr(q_mat, r_val)
        agg, ms = aggregate(plant, lambda z, k=k: float(-k @ z), Z0S)
        lqr_results[r_val] = agg
        if agg is None:
            print(
                f"{'LQR R=' + str(r_val):>22} {'--':>9} {'--':>9} {'--':>9} {'no':>12}"
            )
        else:
            print(
                f"{'LQR R=' + str(r_val):>22} {agg.settling_time:9.3f} "
                f"{agg.peak_force:9.4f} {agg.energy:9.4f} {str(agg.settled):>12}"
            )
    print()
    print("  Open-loop trajectory optimization (full knowledge of z0):")
    samples, targets, weights, per_z0 = build_training_set(plant, Z0S)
    ceil_settle, ceil_peak, ceil_energy = [], [], []
    for z0 in Z0S:
        ts, zs, us = per_z0[tuple(z0)]
        norms = np.linalg.norm(zs, axis=1)
        thr = 0.02 * np.linalg.norm(z0)
        out = np.where(norms > thr)[0]
        ceil_settle.append(ts[out[-1]] if out.size else 0.0)
        ceil_peak.append(np.max(np.abs(us)))
        ceil_energy.append(np.trapezoid(us**2, ts))
    ceiling = Metrics(
        float(np.mean(ceil_settle)),
        float(np.max(ceil_peak)),
        float(np.mean(ceil_energy)),
        True,
    )
    print(
        f"{'open-loop reference':>22} {ceiling.settling_time:9.3f} "
        f"{ceiling.peak_force:9.4f} {ceiling.energy:9.4f} {'yes':>12}"
    )
    print(f"  training set: {len(samples)} occupation-weighted (state, u*) samples")

    best_lqr = min(
        (v for v in lqr_results.values() if v is not None),
        key=lambda m: m.score(ceiling),
    )
    print()
    print("  IMPORTANT: this open-loop reference is a local optimum of a SMOOTH")
    print("  SURROGATE of the objective, not a certified optimum of the objective")
    print("  itself.  Section S beats it, which is proof that it is not a lower")
    print("  bound -- so it is used only as the training target, and all scores")
    print("  below are normalized against the best LQR instead, which is a")
    print("  reference that can actually be computed exactly.")

    rule("S. Fuzzy least-action convergence vs rule count -- NO refinement")
    print("  Recipe, applied once with no tuning: weighted k-means on the optimal")
    print("  trajectories -> variable-project the consequents (one linear solve,")
    print("  globally optimal for those rule positions) -> simulate.  Nothing else.")
    print()
    print(
        f"{'rules':>6} {'params':>7} {'settle':>9} {'peak |u|':>9} {'energy':>9} "
        f"{'score':>8} {'shift':>7} {'settled':>8}"
    )
    print(
        f"{'LQR':>6} {'4':>7} {best_lqr.settling_time:9.3f} {best_lqr.peak_force:9.4f} "
        f"{best_lqr.energy:9.4f} {1.0:8.4f} {'--':>7} {str(best_lqr.settled):>8}"
    )
    print(
        f"{'ref':>6} {'open':>7} {ceiling.settling_time:9.3f} {ceiling.peak_force:9.4f} "
        f"{ceiling.energy:9.4f} {ceiling.score(best_lqr):8.4f} {'--':>7} {'yes':>8}"
    )
    for n_rules in (1, 2, 3, 4, 6, 8, 12, 16):
        cen, wid = place_rules(samples, weights, n_rules)
        ctrl = TskController(cen, wid)
        fit_consequents(ctrl, samples, targets, weights)
        agg, ms = aggregate(plant, ctrl, Z0S)
        shift = max(
            distribution_shift(simulate(plant, ctrl, z0)[2], samples) for z0 in Z0S
        )
        if agg is None:
            print(
                f"{n_rules:>6} {5 * n_rules:>7} {'--':>9} {'--':>9} {'--':>9} "
                f"{'inf':>8} {shift:7.2f} {'no':>8}"
            )
            continue
        print(
            f"{n_rules:>6} {5 * n_rules:>7} {agg.settling_time:9.3f} "
            f"{agg.peak_force:9.4f} {agg.energy:9.4f} {agg.score(best_lqr):8.4f} "
            f"{shift:7.2f} {str(agg.settled):>8}"
        )
    print("  score < 1 beats the best LQR on the equal-weight three-objective score.")
    print("  'shift' is the max distance from a closed-loop state to the nearest")
    print("  training sample, in units of the training set's own per-axis spread.")

    rule("T. Nonlinear spring, where Theorem C1 no longer applies")
    nl = TwoCart(m1=1.0, m2=1.0, k_lin=1.0, k_nl=2.0, u_max=1.0)
    print("  Same plant plus a hardening cubic spring (k_nl = 2).  The LQR gain is now")
    print("  only valid to first order, and the exact-representation argument is void.")
    nl_samples, nl_targets, nl_weights, nl_per = build_training_set(nl, Z0S)
    ceil_s, ceil_p, ceil_e = [], [], []
    for z0 in Z0S:
        ts, zs, us = nl_per[tuple(z0)]
        norms = np.linalg.norm(zs, axis=1)
        out = np.where(norms > 0.02 * np.linalg.norm(z0))[0]
        ceil_s.append(ts[out[-1]] if out.size else 0.0)
        ceil_p.append(np.max(np.abs(us)))
        ceil_e.append(np.trapezoid(us**2, ts))
    nl_ceiling = Metrics(
        float(np.mean(ceil_s)), float(np.max(ceil_p)), float(np.mean(ceil_e)), True
    )
    nl_best = None
    for r_val in (0.3, 1.0, 3.0):
        k = nl.lqr(q_mat, r_val)
        agg, _ = aggregate(nl, lambda z, k=k: float(-k @ z), Z0S)
        if agg is not None and (
            nl_best is None or agg.score(nl_ceiling) < nl_best.score(nl_ceiling)
        ):
            nl_best = agg
    print(
        f"{'controller':>22} {'settle':>9} {'peak |u|':>9} {'energy':>9} {'score':>8}"
    )
    if nl_best is None:
        print("  no linear LQR in the sweep stabilizes the nonlinear plant.")
        return
    print(
        f"{'best linear LQR':>22} {nl_best.settling_time:9.3f} "
        f"{nl_best.peak_force:9.4f} {nl_best.energy:9.4f} {1.0:8.4f}"
    )
    print(
        f"{'open-loop reference':>22} {nl_ceiling.settling_time:9.3f} "
        f"{nl_ceiling.peak_force:9.4f} {nl_ceiling.energy:9.4f} "
        f"{nl_ceiling.score(nl_best):8.4f}"
    )
    for n_rules in (1, 2, 4, 8, 16):
        cen, wid = place_rules(nl_samples, nl_weights, n_rules)
        ctrl = TskController(cen, wid)
        fit_consequents(ctrl, nl_samples, nl_targets, nl_weights)
        agg, _ = aggregate(nl, ctrl, Z0S)
        label = f"fuzzy N={n_rules}"
        if agg is None:
            print(f"{label:>22} {'--':>9} {'--':>9} {'--':>9} {'inf':>8}")
        else:
            print(
                f"{label:>22} {agg.settling_time:9.3f} {agg.peak_force:9.4f} "
                f"{agg.energy:9.4f} {agg.score(nl_best):8.4f}"
            )


if __name__ == "__main__":
    main()
