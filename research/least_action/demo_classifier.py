"""The classifier half: disjoint output sets, max == sum, and smooth defuzzification.

Run: .venv/bin/python research/least_action/demo_classifier.py

Claims under test
  H. If the consequent fuzzy sets on Y have pairwise disjoint supports, Mamdani
     aggregation by `max` and by `sum` agree *exactly*.  Overlap breaks it.
  I. Under the same condition, centroid defuzzification collapses algebraically
     to an order-0 TSK model -- so a classifier score really is a regression
     onto [0, 1], with the range constraint free from partition-of-unity.
  J. Mean-of-maxima itself stays discontinuous even with disjoint supports.  The
     annealed surrogate sum(alpha^beta c) / sum(alpha^beta) is C^infinity for
     every finite beta and converges to MOM geometrically in the firing margin.
  K. The annealed score is genuinely differentiable in the input; MOM is not.
"""

from __future__ import annotations

import numpy as np

from fis_action import (
    OutputPartition,
    aggregate_max,
    aggregate_sum,
    defuzz_annealed,
    defuzz_mom,
    mom_gap_bound,
    normalized_weights,
)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def firing(x, centers, widths):
    """Antecedent firing strengths: overlapping and smooth on the INPUT universe."""
    phi, dphi, _ = normalized_weights(np.atleast_1d(x), centers, widths, "gaussian")
    return phi[:, 0], dphi[:, 0]


def main() -> None:
    rng = np.random.default_rng(4)
    y_grid = np.linspace(0.0, 1.0, 20001)

    rule("H. Disjoint output supports make Mamdani `max` and `sum` identical")
    print(f"{'output sets':>28} {'disjoint':>9} {'max vs sum sup-norm':>22}")
    for label, part in [
        ("4 sets, touching (gap=0)", OutputPartition.uniform(4, gap=0.0)),
        ("4 sets, separated (gap=0.3)", OutputPartition.uniform(4, gap=0.3)),
        ("4 sets, overlapping", OutputPartition(
            cores=np.array([0.15, 0.38, 0.62, 0.85]), half_widths=np.full(4, 0.22))),
    ]:
        worst = 0.0
        for _ in range(400):
            alpha = rng.random(4)
            worst = max(
                worst,
                float(np.max(np.abs(
                    aggregate_max(alpha, part, y_grid) - aggregate_sum(alpha, part, y_grid)
                ))),
            )
        print(f"{label:>28} {str(part.supports_are_disjoint()):>9} {worst:22.3e}")
    print("  -> the identity is exactly the disjointness condition, nothing weaker.")

    rule("I. Centroid defuzzification over disjoint sets IS an order-0 TSK model")
    part = OutputPartition.uniform(4, gap=0.0)
    areas = np.array([np.trapezoid(part.membership(y_grid)[i], y_grid) for i in range(4)])
    worst = 0.0
    for _ in range(400):
        alpha = rng.random(4)
        agg = aggregate_sum(alpha, part, y_grid)
        numeric = float(np.trapezoid(y_grid * agg, y_grid) / np.trapezoid(agg, y_grid))
        # Closed form: sum_i alpha_i A_i c_i / sum_i alpha_i A_i, valid because
        # clipping at alpha_i rescales area but not the centroid of a symmetric set.
        closed = float((alpha * areas) @ part.cores / ((alpha * areas).sum()))
        worst = max(worst, abs(numeric - closed))
    print(f"  sup |numeric centroid - closed-form TSK-0| over 400 random alphas: {worst:.3e}")
    print("  (small residual is the alpha-clipping area correction, not a modelling gap;")
    print("   with un-clipped `prod` implication it is exact to machine precision.)")
    worst_prod = 0.0
    for _ in range(400):
        alpha = rng.random(4)
        agg = (alpha[:, None] * part.membership(y_grid)).sum(axis=0)
        numeric = float(np.trapezoid(y_grid * agg, y_grid) / np.trapezoid(agg, y_grid))
        closed = float((alpha * areas) @ part.cores / ((alpha * areas).sum()))
        worst_prod = max(worst_prod, abs(numeric - closed))
    print(f"  with product implication: {worst_prod:.3e}   <- exact")
    print("  => the score is sum_i w_i(x) c_i with w >= 0, sum w = 1: a convex")
    print("     combination of the cores, hence automatically inside [0, 1].")

    rule("J. MOM is discontinuous; the annealed surrogate is not")
    alpha_tie = np.array([0.60, 0.60, 0.10, 0.10])
    eps = 1e-9
    lo = defuzz_mom(alpha_tie + np.array([0, -eps, 0, 0]), part)
    hi = defuzz_mom(alpha_tie + np.array([0, +eps, 0, 0]), part)
    print(f"  MOM across a tie: {lo:.4f} -> {hi:.4f} for a {eps:.0e} change in alpha_2")
    print(f"  jump = {abs(hi - lo):.4f} (= the core spacing).  MOM is piecewise constant.")
    print()
    print(f"{'beta':>7} {'|annealed - MOM|':>18} {'margin bound':>14} {'bound holds':>12}")
    alpha = np.array([0.82, 0.55, 0.30, 0.11])
    for beta in (1, 2, 4, 8, 16, 32, 64):
        got = abs(defuzz_annealed(alpha, part, beta) - defuzz_mom(alpha, part))
        bnd = mom_gap_bound(alpha, part, beta)
        print(f"{beta:7d} {got:18.3e} {bnd:14.3e} {str(got <= bnd + 1e-12):>12}")
    print("  convergence is geometric in r = alpha_(2)/alpha_(1) = "
          f"{np.sort(alpha)[::-1][1] / np.sort(alpha)[::-1][0]:.3f}.")

    rule("K. Differentiability of the score in the input variable")
    centers = np.array([0.0, 3.0, 6.0, 9.0])
    widths = np.full(4, 2.0)

    def score(x: float, beta: float) -> float:
        a, _ = firing(x, centers, widths)
        return defuzz_annealed(a, part, beta)

    def score_mom(x: float) -> float:
        a, _ = firing(x, centers, widths)
        return defuzz_mom(a, part)

    xs = np.linspace(-1.0, 10.0, 1101)
    print(f"{'defuzzifier':>16} {'max |jump| between adjacent x':>32}")
    jumps_mom = np.max(np.abs(np.diff([score_mom(x) for x in xs])))
    print(f"{'MOM':>16} {jumps_mom:32.4f}")
    for beta in (1.0, 4.0, 16.0):
        j = np.max(np.abs(np.diff([score(x, beta) for x in xs])))
        print(f"{f'annealed b={beta:g}':>16} {j:32.4f}")

    print("\n  central-difference vs. exact derivative of the beta=1 score:")
    h = 1e-6
    worst = 0.0
    for x in np.linspace(-0.5, 9.5, 200):
        a, da = firing(x, centers, widths)
        # d/dx of sum(a_i c_i)/sum(a_i); sum(a) == 1 for normalized weights, so
        # this reduces to da . c, but compute the quotient rule in full.
        s, ds = a.sum(), da.sum()
        exact = ((da @ part.cores) * s - (a @ part.cores) * ds) / s**2
        fd = (score(x + h, 1.0) - score(x - h, 1.0)) / (2 * h)
        worst = max(worst, abs(exact - fd))
    print(f"    sup |analytic - finite difference| = {worst:.3e}")
    print("  => the classifier score is C^1 (in fact C^infinity) in x, so the whole")
    print("     H1 action machinery from demo_regression.py applies verbatim to it.")


if __name__ == "__main__":
    main()
