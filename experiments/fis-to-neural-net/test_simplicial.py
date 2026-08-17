"""Exactness tests for the tetrahedral (simplicial) construction.

The n-dimensional equivalence rests on three claims, and all three are checked
here at zero or machine-precision error rather than at a tolerance chosen to
pass: the hat's closed form *is* the Freudenthal hat, the hats partition unity
in any dimension, and only n+1 of them are ever nonzero.

Run: ``python experiments/fis-to-neural-net/test_simplicial.py``
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simplicial  # noqa: E402


def _kuhn_reference(X, F_at):
    """Kuhn interpolation, written out longhand as the definition to check against.

    Deliberately a slow, literal, per-row walk of the staircase path: it is the
    thing `simplicial.barycentric`'s vectorized argsort is *claimed* to be, so it
    must not share any code with it.
    """
    X = np.atleast_2d(X)
    out = np.zeros(len(X))
    base = np.floor(X).astype(int)
    frac = X - base
    for r in range(len(X)):
        order = np.argsort(-frac[r])
        v = base[r].copy()
        acc = F_at(v)
        for i in order:
            v2 = v.copy()
            v2[i] += 1
            acc += frac[r, i] * (F_at(v2) - F_at(v))
            v = v2
        out[r] = acc
    return out


def test_hat_matches_kuhn_interpolation():
    """phi_v is the Freudenthal hat, exactly, in every dimension."""
    rng = np.random.default_rng(1)
    for n in (1, 2, 3, 5, 8):
        P = rng.uniform(-2.0, 2.0, size=(3000, n))
        reference = _kuhn_reference(P, lambda v: 1.0 if not np.any(v) else 0.0)
        closed = simplicial.hat(
            P, np.zeros((1, n), dtype=np.int64), np.zeros(n), np.ones(n)
        ).ravel()
        err = float(np.max(np.abs(reference - closed)))
        assert err == 0.0, f"n={n}: closed form differs from Kuhn by {err:.3e}"


def test_barycentric_is_a_partition_of_unity_in_any_dimension():
    rng = np.random.default_rng(2)
    for n in (1, 2, 4, 8, 16, 32):
        X = rng.uniform(-5.0, 5.0, size=(500, n))
        vertices, weights = simplicial.barycentric(X, np.zeros(n), np.ones(n))
        assert vertices.shape == (500, n + 1, n)
        assert weights.shape == (500, n + 1)
        assert np.all(weights >= -1e-15), f"n={n}: negative barycentric weight"
        err = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
        assert err < 1e-12, f"n={n}: weights sum to 1 +/- {err:.3e}"


def test_only_n_plus_one_hats_are_active():
    """The sparsity claim: dimension does not increase the active rule count."""
    rng = np.random.default_rng(3)
    for n in (2, 4, 8):
        X = rng.uniform(0.05, 2.95, size=(200, n))
        origin, h = np.zeros(n), np.ones(n)
        # Every vertex of the enclosing grid region, not just the active ones.
        span = range(-1, 5)
        allv = (
            np.array(list(itertools.product(span, repeat=n)), dtype=np.int64)
            if n <= 4
            else None
        )
        if allv is None:  # n=8 would be 6**8; use the occupied set instead
            allv, _ = simplicial.occupied_vertices(X, origin, h)
        phi = simplicial.hat(X, allv, origin, h)
        active = (phi > 1e-12).sum(axis=1)
        assert (
            active.max() <= n + 1
        ), f"n={n}: {active.max()} active hats, expected <= {n+1}"
        assert np.allclose(
            phi.sum(axis=1), 1.0, atol=1e-12
        ), f"n={n}: hats do not sum to 1"


def test_interpolant_reproduces_affine_functions_exactly():
    """A simplicial interpolant is exact on anything already piecewise linear."""
    rng = np.random.default_rng(4)
    for n in (2, 5):
        w = rng.normal(size=n)
        b = float(rng.normal())
        X = rng.uniform(0.0, 3.0, size=(400, n))
        origin, h = np.zeros(n), np.ones(n)
        vertices, _ = simplicial.occupied_vertices(X, origin, h)
        centres = origin + vertices * h
        net = simplicial.SimplicialNet(
            vertices=vertices,
            origin=origin,
            h=h,
            c=centres @ w + b,
            skip=np.zeros(n),
            bias=0.0,
        )
        err = float(np.max(np.abs(net.predict(X) - (X @ w + b))))
        assert err < 1e-10, f"n={n}: affine target reproduced to only {err:.3e}"


def test_sparse_and_dense_paths_agree():
    rng = np.random.default_rng(5)
    n = 3
    X = rng.uniform(0.0, 2.0, size=(300, n))
    origin, h = np.zeros(n), np.ones(n)
    vertices, _ = simplicial.occupied_vertices(X, origin, h)
    net = simplicial.SimplicialNet(
        vertices=vertices,
        origin=origin,
        h=h,
        c=rng.normal(size=len(vertices)),
        skip=rng.normal(size=n),
        bias=0.5,
    )
    assert np.max(np.abs(net.predict(X) - net.predict_sparse(X))) < 1e-12


def test_max_fold_expands_to_pure_relu():
    """max(a, b) = a + relu(b - a), so the hat's maxes are ReLU units.

    This is what licenses calling the tetrahedral membership function a ReLU
    circuit rather than something with a max operator bolted on.

    Checked to a tolerance rather than bit-exactly, and the distinction is
    arithmetic rather than algebraic: the identity is exact, but evaluating it
    computes ``a + (b - a)``, which is not the same float as ``b``. That is a
    rounding difference of one ulp, not a gap in the construction.
    """
    rng = np.random.default_rng(6)
    for n in (2, 3, 5, 8):
        Z = rng.normal(size=(2000, n))
        folded = Z[:, 0].copy()
        for i in range(1, n):
            folded = folded + np.maximum(Z[:, i] - folded, 0.0)
        err = float(np.max(np.abs(folded - Z.max(axis=1))))
        assert err < 1e-15, f"n={n}: fold differs from max by {err:.3e}"


def test_relu_spec_is_linear_in_dimension_and_vertices():
    """The scalability claim, as an assertion rather than a paragraph."""
    n, v = 12, 500
    net = simplicial.SimplicialNet(
        vertices=np.zeros((v, n), dtype=np.int64),
        origin=np.zeros(n),
        h=np.ones(n),
        c=np.zeros(v),
        skip=np.zeros(n),
        bias=0.0,
    )
    spec = net.to_relu_spec()
    assert spec["relu_units"] == v * (2 * (n - 1) + 3)
    assert spec["depth"] == int(np.ceil(np.log2(n))) + 2
    # The dense grid this replaces, for scale: 5**12 = 244 million vertices.
    assert spec["relu_units"] < 5**n


def test_occupied_vertices_are_bounded_by_the_data():
    rng = np.random.default_rng(7)
    n, N = 8, 400
    X = rng.uniform(0.0, 1.0, size=(N, n))
    origin, h = simplicial.grid_from_data(X, resolution=4)
    vertices, support = simplicial.occupied_vertices(X, origin, h)
    assert len(vertices) <= N * (n + 1)
    assert len(vertices) < (4 + 1) ** n  # vastly fewer than the dense grid
    assert np.all(np.diff(support) <= 1e-12), "support must come back sorted"
    assert abs(support.sum() - N) < 1e-9, "total support is one unit per row"


def test_warp_puts_the_lattice_on_the_knots():
    """Each feature's knots land exactly on lattice integers, and order survives."""
    rng = np.random.default_rng(8)
    for _ in range(20):
        k = np.sort(rng.uniform(-3, 3, size=int(rng.integers(2, 9))))
        if np.min(np.diff(k)) < 1e-3:
            continue
        w = simplicial.AxisWarp(knots=[k])
        u = w.forward(k[:, None]).ravel()
        assert np.max(np.abs(u - np.arange(k.size))) < 1e-9, "knots must map to 0..m-1"
        # Strictly increasing everywhere, including outside the knot range, so
        # rows beyond the training extent are not clamped on top of each other.
        x = np.sort(rng.uniform(k[0] - 2, k[-1] + 2, size=500))
        assert np.all(np.diff(w.forward(x[:, None]).ravel()) > 0)


def test_from_knots_merges_near_duplicates():
    """Near-duplicate knots must not become full lattice cells.

    The regression this guards: on WEC the FIS's knot gaps span 4.6e4-to-1, and
    building the warp straight off them put a cell boundary between two points
    4e-6 apart, driving conversion fidelity to 16.46 against an additive seed's
    1.47.
    """
    knots = {"a": [0.0, 1e-6, 2e-6, 0.5, 1.0], "b": [0.0, 1.0], "c": [0.3]}
    w = simplicial.AxisWarp.from_knots(knots, ["a", "b", "c"])
    assert w.knots[0].size == 3, "the three colliding knots must collapse to one"
    assert np.min(np.diff(w.knots[0])) >= simplicial.AxisWarp.MIN_GAP
    assert w.knots[1].size == 2
    assert w.knots[2].size == 0, "a single knot cannot define a warp"


def test_warped_lattice_still_partitions_unity():
    """Warping the axes cannot break the property the equivalence rests on."""
    rng = np.random.default_rng(9)
    n = 3
    knots = [np.sort(rng.uniform(0, 1, size=6)) for _ in range(n)]
    warp = simplicial.AxisWarp(knots=knots)
    X = rng.uniform(0.05, 0.95, size=(300, n))
    U = warp.forward(X)
    origin, h = simplicial.grid_from_data(U, resolution=3)
    vertices, _ = simplicial.occupied_vertices(U, origin, h)
    phi = simplicial.hat(U, vertices, origin, h)
    assert np.allclose(phi.sum(axis=1), 1.0, atol=1e-12)
    assert (phi > 1e-12).sum(axis=1).max() <= n + 1


def test_warp_is_a_relu_circuit():
    """The warp costs one ReLU per interior knot per axis -- linear, not free."""
    knots = [np.linspace(0, 1, 7), np.linspace(0, 1, 4), np.asarray([])]
    w = simplicial.AxisWarp(knots=knots)
    assert w.relu_units() == (7 - 2) + (4 - 2)
    # An axis with no knots passes through untouched.
    X = np.zeros((5, 3))
    X[:, 2] = np.arange(5.0)
    assert np.array_equal(w.forward(X)[:, 2], np.arange(5.0))


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
