"""Does raising the consequent order move the §9b plateau?

Run: .venv/bin/python research/least_action/demo_order.py

§9b found the least-action fit converging at 2-3 affine-consequent rules to a
score of 0.8629, and §9e showed that plateau survives every form of
off-trajectory augmentation.  Two candidates were left: the model class, and the
training targets.  This isolates the first.

Raising the consequent polynomial order enlarges the model class while keeping
the consequents LINEAR in the parameters, so variable projection still returns
the global optimum for each rule placement (README §3a).  Nothing else in the
recipe changes -- same rule placement, same occupation weighting, same single
linear solve.

  Y. Score against rule count, for consequent orders 0-3.
  Z. The fair comparison: score against total PARAMETER count, since order-2
     with 1 rule and order-1 with 3 rules cost the same 15 parameters.
"""

from __future__ import annotations

import os

import numpy as np

from fis_twocart import (
    Metrics,
    TskController,
    TwoCart,
    basis_size,
    distribution_shift,
    fit_consequents,
    optimal_trajectory,
    place_rules,
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


def training_set(plant):
    """Cached, because rebuilding it costs six trajectory optimizations."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["z"], d["u"], d["w"]
    zs_all, us_all, ws_all = [], [], []
    for z0 in Z0S:
        ts, zs, us, _ = optimal_trajectory(plant, z0, n_knots=30, t_end=25.0)
        zs_all.append(zs)
        us_all.append(us)
        ws_all.append(np.full(len(ts), ts[1] - ts[0]))
    z, u, w = np.vstack(zs_all), np.concatenate(us_all), np.concatenate(ws_all)
    np.savez(CACHE, z=z, u=u, w=w)
    return z, u, w


def main() -> None:
    plant = TwoCart(m1=1.0, m2=1.0, k_lin=1.0, k_nl=0.0, u_max=1.0)
    q_mat = np.diag([1.0, 10.0, 1.0, 1.0])
    k_best = plant.lqr(q_mat, 10.0)
    lqr_ref = aggregate(plant, lambda z: float(-k_best @ z))
    assert lqr_ref is not None
    samples, targets, weights = training_set(plant)
    print(f"training set: {len(samples)} on-trajectory samples")
    print(f"reference (best LQR): settle={lqr_ref.settling_time:.3f} "
          f"peak={lqr_ref.peak_force:.4f} energy={lqr_ref.energy:.4f}")
    print(f"affine-consequent plateau from 9b: 0.8629 at N=3 (15 parameters)")

    results = []

    rule("Y. Score vs rule count, by consequent order")
    print(f"{'order':>6} {'basis':>6}   " + "".join(f"{f'N={n}':>9}" for n in
                                                    (1, 2, 3, 4, 6, 8, 12)))
    for order in (0, 1, 2, 3):
        bsz = basis_size(4, order)
        row = []
        for n_rules in (1, 2, 3, 4, 6, 8, 12):
            cen, wid = place_rules(samples, weights, n_rules)
            ctrl = TskController(cen, wid, order=order)
            fit_consequents(ctrl, samples, targets, weights)
            agg = aggregate(plant, ctrl)
            n_par = ctrl.n_params()
            if agg is None:
                row.append("inf")
                results.append((order, n_rules, n_par, None, None))
            else:
                sc = agg.score(lqr_ref)
                row.append(f"{sc:.4f}")
                shift = max(distribution_shift(simulate(plant, ctrl, z0)[2], samples)
                            for z0 in Z0S)
                results.append((order, n_rules, n_par, sc, agg))
        print(f"{order:>6} {bsz:>6}   " + "".join(f"{v:>9}" for v in row))
    print("  order 0 = Sugeno constants, 1 = affine (classic TSK), 2 = quadratic,")
    print("  3 = cubic in the four states.  Consequents stay linear in the")
    print("  parameters at every order, so each cell is still one globally")
    print("  optimal linear solve -- no optimizer, no local minima.")

    rule("Z. The fair comparison: score vs total parameter count")
    print("  order-2 with 1 rule and order-1 with 3 rules both cost 15 parameters,")
    print("  so rule count alone is not a like-for-like axis.")
    print()
    ok = [r for r in results if r[3] is not None]
    ok.sort(key=lambda r: r[2])
    print(f"{'params':>7} {'order':>6} {'rules':>6} {'score':>8} {'settle':>8} "
          f"{'peak':>7} {'energy':>8}")
    for order, n_rules, n_par, sc, agg in ok:
        if n_par > 200:
            continue
        print(f"{n_par:>7} {order:>6} {n_rules:>6} {sc:8.4f} "
              f"{agg.settling_time:8.3f} {agg.peak_force:7.4f} {agg.energy:8.4f}")
    print()
    best = min(ok, key=lambda r: r[3])
    print(f"  BEST OVERALL: order {best[0]}, {best[1]} rules, {best[2]} params, "
          f"score {best[3]:.4f}")
    base = 0.8629
    if best[3] < base - 1e-4:
        print(f"  The plateau MOVED: {base:.4f} -> {best[3]:.4f} "
              f"({100 * (1 - best[3] / base):.1f}% better).")
    else:
        print(f"  The plateau did NOT move: best {best[3]:.4f} vs {base:.4f} "
              f"from affine consequents at 15 parameters.")

    # Is the limit the fit itself, or the closed loop?
    rule("Z'. Open-loop fit quality vs closed-loop score")
    print("  If higher order fits u* better but does not score better, the model")
    print("  class was never the binding constraint.")
    print(f"{'order':>6} {'rules':>6} {'params':>7} {'weighted RMS fit':>18} {'score':>8}")
    for order in (0, 1, 2, 3):
        for n_rules in (3, 8):
            cen, wid = place_rules(samples, weights, n_rules)
            ctrl = TskController(cen, wid, order=order)
            fit_consequents(ctrl, samples, targets, weights)
            pred = np.array([ctrl(z) for z in samples])
            wn = weights / weights.sum()
            rms = float(np.sqrt(np.sum(wn * (pred - targets) ** 2)))
            agg = aggregate(plant, ctrl)
            sc = "inf" if agg is None else f"{agg.score(lqr_ref):.4f}"
            print(f"{order:>6} {n_rules:>6} {ctrl.n_params():>7} {rms:18.6f} {sc:>8}")


if __name__ == "__main__":
    main()
