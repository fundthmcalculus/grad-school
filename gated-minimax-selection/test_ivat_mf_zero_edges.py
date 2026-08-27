"""Regression tests for grad-school #176: exact-zero dissimilarities.

`minimax_transform_fast` routes through `csr_matrix`, where a stored zero is an
ABSENT edge -- so duplicate points (distance exactly 0) lost their edges and
came back with inflated D*. Real data has such points: unconstrained DTW
collapses consecutive repeats, so distinct binary on/off traces sit at distance
0 (ElectricDevices: 130 zero-cliques over 1,292 points).

Every test here compares against `minimax_transform`, the O(n^3) reference,
which never had the bug.

Run: ``python -m pytest gated-minimax-selection/test_ivat_mf_zero_edges.py``
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ivat_mf as im  # noqa: E402


def test_hand_case_two_coincident_points():
    """The minimal reproduction from #176: points 0 and 1 are coincident, so
    their minimax distance is 0, not the 5.0 the csr path used to return."""
    D = np.array(
        [
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 5.0, 6.0],
            [5.0, 5.0, 0.0, 1.0],
            [6.0, 6.0, 1.0, 0.0],
        ]
    )
    ref = im.minimax_transform(D)
    fast = im.minimax_transform_fast(D)
    assert fast[0, 1] == 0.0
    assert np.allclose(ref, fast)


def test_duplicate_clique_matches_reference():
    """A block of mutually coincident points (the ElectricDevices shape: a
    zero-clique, not a chain) must be exactly 0 inside and correct outside."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(6, 3))
    X = np.vstack([base, np.repeat(base[:1], 5, axis=0)])  # 5 copies of row 0
    D = im.dissimilarity(X)
    ref = im.minimax_transform(D)
    fast = im.minimax_transform_fast(D)
    assert np.allclose(ref, fast)
    dup = [0] + list(range(6, 11))
    assert np.all(fast[np.ix_(dup, dup)] == 0.0)


@pytest.mark.parametrize("n_dup", [1, 3, 10])
def test_random_matrices_with_duplicates(n_dup):
    rng = np.random.default_rng(n_dup)
    X = rng.normal(size=(20, 4))
    X = np.vstack([X, np.repeat(X[:1], n_dup, axis=0)])
    D = im.dissimilarity(X)
    assert np.allclose(im.minimax_transform(D), im.minimax_transform_fast(D))


def test_all_points_coincident():
    """Degenerate extreme: every distance is 0, so D* is entirely 0."""
    D = np.zeros((5, 5))
    fast = im.minimax_transform_fast(D)
    assert np.array_equal(fast, np.zeros((5, 5)))
    assert np.allclose(im.minimax_transform(D), fast)


def test_no_zeros_path_unchanged():
    """Matrices without duplicates must be untouched by the sentinel logic."""
    rng = np.random.default_rng(7)
    D = im.dissimilarity(rng.normal(size=(30, 3)))
    assert np.allclose(im.minimax_transform(D), im.minimax_transform_fast(D))


def test_input_matrix_not_mutated():
    """The sentinel shift must not leak into the caller's array."""
    D = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )
    before = D.copy()
    im.minimax_transform_fast(D)
    assert np.array_equal(D, before)


def test_tiny_but_nonzero_distances_survive():
    """Near-duplicates are NOT duplicates: a genuinely tiny edge must stay
    distinguishable from an exact zero (the sentinel sits strictly below every
    real edge, so it can never swallow one)."""
    D = np.array(
        [
            [0.0, 1e-9, 4.0],
            [1e-9, 0.0, 4.0],
            [4.0, 4.0, 0.0],
        ]
    )
    fast = im.minimax_transform_fast(D)
    assert fast[0, 1] == pytest.approx(1e-9)
    assert np.allclose(im.minimax_transform(D), fast)
