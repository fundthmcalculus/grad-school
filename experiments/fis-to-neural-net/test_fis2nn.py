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


def test_pwl_decomposition_is_exact_at_and_between_knots():
    """Slope-change weights reproduce the interpolant, not merely the samples."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        t = np.sort(rng.uniform(-5, 5, size=int(rng.integers(2, 12))))
        if t.size > 1 and np.min(np.diff(t)) < 1e-3:
            continue
        v = rng.uniform(-10, 10, size=t.size)
        base, intercept, coeffs = fis2nn.pwl_to_relu_weights(t, v)

        def g(x):
            act = np.maximum(x[:, None] - t[None, :], 0.0)
            return intercept + base * x + act @ coeffs

        assert np.max(np.abs(g(t) - v)) < 1e-10, "knot values not reproduced"
        if t.size > 1:
            # A midpoint of each segment must sit on the straight line joining
            # its endpoints -- this is what distinguishes an exact PWL
            # decomposition from one that merely interpolates the samples.
            mid = 0.5 * (t[:-1] + t[1:])
            want = 0.5 * (v[:-1] + v[1:])
            assert np.max(np.abs(g(mid) - want)) < 1e-10
        # The end knots carry no slope change: the function extends linearly.
        assert coeffs[0] == 0.0 and coeffs[-1] == 0.0


def test_analytic_seed_reproduces_a_1d_system_at_its_knots():
    """The equivalence, backed out into weights: 1-D seed == the system itself.

    With one input there is nothing to average over, so the partial-dependence
    profile *is* the system's own function and the seed must reproduce it
    wherever the conversion places a knot -- with no labels involved anywhere.
    """
    import pandas as pd

    terms = build_triangular_partition([-3.0, -1.0, 0.5, 2.0, 4.5])
    singletons = [7.0, -2.0, 11.0, 3.0, -5.0]

    def fis(frame):
        x = np.asarray(frame["x"], dtype=float)
        return sum(m * t.evaluate(x) for m, t in zip(singletons, terms))

    knots = {"x": fis2nn.merge_knots([-3.0, -1.0, 0.5, 2.0, 4.5])}
    X = pd.DataFrame({"x": np.linspace(-3.0, 4.5, 200)})
    net = fis2nn.analytic_seed_from_fis(fis, X, ["x"], knots, background_size=None)

    at_knots = knots["x"][:, None]
    assert (
        np.max(np.abs(net.predict(at_knots) - fis(pd.DataFrame({"x": knots["x"]}))))
        < 1e-9
    )
    # And, because this system is piecewise linear with breakpoints exactly at
    # those knots, everywhere in between too.
    dense = np.linspace(-3.0, 4.5, 5001)
    err = np.max(np.abs(net.predict(dense[:, None]) - fis(pd.DataFrame({"x": dense}))))
    assert err < 1e-9, f"max abs error {err:.3e}"


def test_analytic_seed_uses_no_labels():
    """The seed takes no target: it converts the FIS, it does not refit it."""
    import inspect

    assert "y" not in inspect.signature(fis2nn.analytic_seed_from_fis).parameters


def test_knot_merging_is_idempotent_and_sorted():
    knots = fis2nn.merge_knots([3.0, 1.0, 1.0 + 1e-15, np.inf, -np.inf, 2.0])
    assert np.all(np.diff(knots) > 0)
    assert np.array_equal(knots, fis2nn.merge_knots(knots))


def _raises(fn, *args, exc=ValueError) -> bool:
    try:
        fn(*args)
    except exc:
        return True
    return False


def test_degenerate_triangle_raises_instead_of_lying():
    """A vertical side is discontinuous, so no finite ReLU sum represents it.

    This is a regression test with teeth: the old code returned an array rather
    than raising, and for ``T(1, 1, 2)`` that array was the *negation* of the
    correct falling ramp -- ``max|err| = 1.0``, a full unit of membership, with
    nothing to tell the caller. Both orientations are pinned, and so is the
    inverted-foot case the old `elif b > a` guard also swallowed.
    """
    D = fis2nn.DegenerateMembership
    # Zero width -> DegenerateMembership, which callers walking a fitted FIS
    # are expected to catch and skip.
    assert _raises(
        fis2nn.triangle_to_relu, TriangularMembership(a=1.0, b=1.0, c=2.0), exc=D
    )
    assert _raises(
        fis2nn.triangle_to_relu, TriangularMembership(a=0.0, b=1.0, c=1.0), exc=D
    )
    # Inverted feet are a different failure: the caller passed nonsense.
    assert _raises(fis2nn.triangle_to_relu, TriangularMembership(a=2.0, b=1.0, c=3.0))
    assert _raises(fis2nn.triangle_to_relu, TriangularMembership(a=0.0, b=3.0, c=1.0))
    # ...and the shoulder forms, which look degenerate but are not, still work.
    assert not _raises(
        fis2nn.triangle_to_relu, TriangularMembership(a=-np.inf, b=1.0, c=2.0)
    )
    assert not _raises(
        fis2nn.triangle_to_relu, TriangularMembership(a=0.0, b=1.0, c=np.inf)
    )


def test_degenerate_trapezoid_raises_instead_of_lying():
    D = fis2nn.DegenerateMembership
    assert _raises(
        fis2nn.trapezoid_to_relu, TrapezoidMembership(a=1.0, b=1.0, c=2.0, d=3.0), exc=D
    )
    assert _raises(
        fis2nn.trapezoid_to_relu, TrapezoidMembership(a=0.0, b=1.0, c=2.0, d=2.0), exc=D
    )


def test_fis_knots_skips_degenerate_terms_and_says_so():
    """A fitted FIS really does contain zero-width terms; walking it must not die.

    A feature with (near-)zero variance gives a Gaussian whose sigma underflows,
    and `fit_triangle_to_gaussian` turns that into a zero-width triangle. Real
    N-CMAPSS DS02 fits produce them. `fis_knots` skips those terms -- but warns,
    because a FIS that is mostly degenerate converts to a seed that means
    nothing, and silence would hide that.
    """
    import warnings

    from tribblefis.gauss_data import GaussianMixtureModel

    good = GaussianMembership(mu=0.0, sigma=1.0)
    dead = GaussianMembership(mu=5.0, sigma=0.0)

    class _Label:
        def __init__(self, mfs):
            self.memberships = mfs

    class _Feature:
        def __init__(self, mfs):
            self.label_models = {"L": _Label(mfs)}

    class _Model:
        feature_models = {"x": _Feature([good, dead])}

    assert isinstance(_Model(), object) and GaussianMixtureModel is not None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        knots = fis2nn.fis_knots(_Model(), ["x"])
    assert knots["x"].size == 3, "the healthy term's three knots survive"
    assert any("zero width" in str(w.message) for w in caught), "must warn"


def test_infinite_trapezoid_foot_is_rejected_not_guessed():
    """A shouldered trapezoid has no ground truth to convert *to*.

    `gauss_data.TrapezoidMembership.evaluate` computes the rising branch as
    ``(x - a) / (b - a)``, so an ``a = -inf`` foot gives ``inf / inf`` -- NaN
    across the entire left side. The old expansion silently returned an
    all-zero rising side for it; guessing a shoulder instead would have been
    the same mistake in the other direction. Rejecting is the only answer that
    does not invent semantics the package does not have.
    """
    assert np.isnan(
        TrapezoidMembership(a=-np.inf, b=-2.0, c=2.0, d=6.0).evaluate(np.array([-5.0]))
    ).all()
    assert _raises(
        fis2nn.trapezoid_to_relu,
        TrapezoidMembership(a=-np.inf, b=-2.0, c=2.0, d=6.0),
    )
    assert _raises(
        fis2nn.trapezoid_to_relu,
        TrapezoidMembership(a=-5.0, b=-1.0, c=2.0, d=np.inf),
    )


def test_recorded_seconds_exclude_evaluation_time():
    """``hist.seconds`` must measure gradient descent, not the ruler.

    Evaluation cost scales with hidden width, and the comparison this module
    exists for puts arms of very different widths side by side, so charging
    eval time to training biases exactly the number under test. A wide net
    evaluated on a large test set every epoch is the case that used to inflate.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 4))
    y = X[:, 0] * 2.0 - X[:, 1]
    X_big = rng.normal(size=(20_000, 4))
    y_big = X_big[:, 0] * 2.0 - X_big[:, 1]

    net = fis2nn.he_start(rng, 4, 256)
    _, hist = fis2nn.train_adam(
        net,
        X,
        y,
        X_test=X_big,
        y_test=y_big,
        epochs=3,
        batch_size=64,
        track_train=False,
    )
    # Three epochs of 4 minibatches on 256x4 is microseconds of real work; if
    # eval time leaked in, 4 passes over 20k rows through 256 hidden units would
    # dominate and push this into the tens of milliseconds.
    assert hist.seconds[-1] < 0.05, hist.seconds
    assert all(b >= a for a, b in zip(hist.seconds, hist.seconds[1:]))


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
