"""Exactness tests for the FIS -> ReLU-network conversion.

These are the load-bearing claims. If the membership expansion or the 1-D
conversion is not exact to machine precision, the "conversion" in this
experiment is really a fit, and every downstream comparison would be measuring
something other than what it says it measures.

Run: ``python -m pytest experiments/fis-to-neural-net/test_fis2nn.py``
(or ``python experiments/fis-to-neural-net/test_fis2nn.py`` for a bare report).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tribblefis.gauss_data import (  # noqa: E402
    GaussianMembership,
    TrapezoidMembership,
    TriangularMembership,
)
from tribblefis.ruspini import (  # noqa: E402
    build_triangular_partition,
    verify_partition_of_unity,
)
from tribblefis.triangle_fit import fit_triangle_to_gaussian  # noqa: E402

import fis2nn  # noqa: E402

GRID = np.linspace(-12.0, 12.0, 24_001)
TOL = 1e-12


def _max_err(mf, expansion) -> float:
    return float(np.max(np.abs(mf.evaluate(GRID) - expansion.evaluate(GRID))))


def test_interior_triangle_is_three_relus():
    mf = TriangularMembership(a=-2.0, b=0.5, c=3.25)
    exp = fis2nn.triangle_to_relu(mf)
    assert exp.knots.size == 3
    assert _max_err(mf, exp) < TOL


def test_asymmetric_triangle():
    mf = TriangularMembership(a=-7.5, b=-7.0, c=6.0)
    assert _max_err(mf, fis2nn.triangle_to_relu(mf)) < TOL


def test_left_shoulder():
    mf = TriangularMembership(a=-np.inf, b=-3.0, c=1.5)
    exp = fis2nn.triangle_to_relu(mf)
    assert exp.bias == 1.0
    assert _max_err(mf, exp) < TOL


def test_right_shoulder():
    mf = TriangularMembership(a=-1.0, b=4.0, c=np.inf)
    exp = fis2nn.triangle_to_relu(mf)
    assert exp.bias == 0.0
    assert _max_err(mf, exp) < TOL


def test_trapezoid_is_four_relus():
    mf = TrapezoidMembership(a=-5.0, b=-1.0, c=2.0, d=6.0)
    exp = fis2nn.trapezoid_to_relu(mf)
    assert exp.knots.size == 4
    assert _max_err(mf, exp) < TOL


def test_gaussian_routes_through_the_package_triangle_fit():
    """The Gaussian branch must be exactly the package's own triangle fit.

    It is the only lossy step in the pipeline, so it has to be *that* step and
    not a second, differently-derived approximation living in this experiment.
    """
    g = GaussianMembership(mu=1.25, sigma=0.8)
    exp = fis2nn.membership_to_relu(g)
    tri = fit_triangle_to_gaussian(g)
    assert _max_err(tri, exp) < TOL


def test_ruspini_partition_converts_exactly():
    """The theorem: 1-D partition-of-unity TSK == one-hidden-layer ReLU net.

    Verified against the package's own partition builder, so the object being
    converted is the same one `tribblefis.ruspini` produces rather than a
    hand-rolled stand-in.
    """
    rng = np.random.default_rng(0)
    for trial in range(25):
        k = int(rng.integers(2, 9))
        apexes = np.sort(rng.uniform(-8.0, 8.0, size=k))
        if np.min(np.diff(apexes)) < 1e-3:
            continue
        terms = build_triangular_partition(apexes)
        assert verify_partition_of_unity(terms, GRID)

        singletons = rng.uniform(-40.0, 90.0, size=k)
        fis = sum(m * t.evaluate(GRID) for m, t in zip(singletons, terms))

        net = fis2nn.fis_to_relu_net_1d(terms, singletons)
        err = float(np.max(np.abs(fis - net.predict(GRID[:, None]))))
        assert err < 1e-10, f"trial {trial}: max abs error {err:.3e}"
        assert net.n_hidden == k, "one hidden unit per apex knot, no more"


def test_conversion_needs_no_data():
    """The 1-D conversion is analytic -- same weights whatever data exists."""
    terms = build_triangular_partition([-2.0, 0.0, 3.0, 5.0])
    a = fis2nn.fis_to_relu_net_1d(terms, [1.0, 2.0, 3.0, 4.0])
    b = fis2nn.fis_to_relu_net_1d(terms, [1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(a.W1, b.W1) and np.array_equal(a.b1, b.b1)
    assert np.array_equal(a.w2, b.w2) and a.c == b.c


def test_readout_solve_recovers_a_linear_target():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, size=(400, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5
    net = fis2nn._axis_aligned_net(3, [(f, np.array([0.25, 0.75])) for f in range(3)])
    fitted = fis2nn.solve_readout(net, X, y, l2=1e-12)
    assert fis2nn.rmse(y, fitted.predict(X)) < 1e-8


def test_knot_merging_is_idempotent_and_sorted():
    knots = fis2nn.merge_knots([3.0, 1.0, 1.0 + 1e-15, np.inf, -np.inf, 2.0])
    assert np.all(np.diff(knots) > 0)
    assert np.array_equal(knots, fis2nn.merge_knots(knots))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
