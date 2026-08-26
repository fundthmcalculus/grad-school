"""Tests for nonmetric_data.py: kernels, diagnostics, violation injection, and
the structural claim run_nonmetric.py rests on -- the minimax transform of ANY
symmetric dissimilarity is an ultrametric, hence Euclidean-embeddable.

Run: ``python -m pytest gated-minimax-selection/test_nonmetric.py``
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ivat_mf as im  # noqa: E402
import nonmetric_data as ND  # noqa: E402
from nerfcm import nerfcm  # noqa: E402

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def test_levenshtein_known_values():
    assert ND.levenshtein("kitten", "sitting") == 3
    assert ND.levenshtein("", "abc") == 3
    assert ND.levenshtein("abc", "abc") == 0
    assert ND.levenshtein("abc", "axc") == 1
    # symmetry
    assert ND.levenshtein("abcd", "xyz") == ND.levenshtein("xyz", "abcd")


def test_dtw_known_values():
    assert ND.dtw_distance([0, 1, 2], [0, 1, 2]) == 0.0
    # constant offset, equal length: every aligned pair costs 1, and warping
    # cannot help; the diagonal path (2 cells) is minimal.
    assert ND.dtw_distance([0, 0], [1, 1]) == pytest.approx(2.0)
    # warping helps: [0, 5, 0] vs [0, 5, 5, 0] aligns the plateau at no cost.
    assert ND.dtw_distance([0, 5, 0], [0, 5, 5, 0]) == pytest.approx(0.0)
    a, b = np.sin(np.linspace(0, 3, 25)), np.cos(np.linspace(0, 3, 25))
    assert ND.dtw_distance(a, b) == pytest.approx(ND.dtw_distance(b, a))


def test_pairwise_symmetry_zero_diagonal():
    items = [np.array([0.0]), np.array([1.0]), np.array([3.0])]
    D = ND.pairwise(items, lambda a, b: abs(float(a[0] - b[0])))
    assert np.allclose(D, D.T)
    assert np.all(np.diag(D) == 0)
    assert D[0, 2] == 3.0


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def test_euclidean_matrix_is_admissible_and_metric():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(X))
    assert ND.triangle_violation_stats(D)["pair_violation_fraction"] == 0.0
    assert ND.euclidean_embeddability(D)["neg_ratio"] < 1e-9


def test_triangle_violation_detected():
    # d(0,2) = 10 but d(0,1) + d(1,2) = 2: a maximal violation.
    D = np.array([[0.0, 1.0, 10.0], [1.0, 0.0, 1.0], [10.0, 1.0, 0.0]])
    stats = ND.triangle_violation_stats(D)
    assert stats["pair_violation_fraction"] > 0
    assert stats["max_violation_depth"] == pytest.approx(4.0)  # 10/2 - 1


def test_is_ultrametric():
    # A genuine ultrametric: two tight pairs joined at a higher level.
    U = np.array(
        [
            [0.0, 1.0, 3.0, 3.0],
            [1.0, 0.0, 3.0, 3.0],
            [3.0, 3.0, 0.0, 2.0],
            [3.0, 3.0, 2.0, 0.0],
        ]
    )
    assert ND.is_ultrametric(U)
    V = U.copy()
    V[0, 1] = V[1, 0] = 5.0  # now d(0,1) > max(d(0,2), d(2,1)) = 3
    assert not ND.is_ultrametric(V)


# ---------------------------------------------------------------------------
# violation injection
# ---------------------------------------------------------------------------


def _base_D():
    D, _, _ = ND.euclidean_blobs(n_per=8, seed=1)
    return D


def test_violate_pairs_identity_at_zero():
    D = _base_D()
    assert np.array_equal(ND.violate_pairs(D, 0.0, 0.8, "stretch"), D)
    assert np.array_equal(ND.violate_pairs(D, 0.2, 0.0, "shortcut"), D)


def test_violate_pairs_direction_and_symmetry():
    D = _base_D()
    Dst = ND.violate_pairs(D, 0.3, 0.8, "stretch", seed=2)
    Dsc = ND.violate_pairs(D, 0.3, 0.8, "shortcut", seed=2)
    assert np.allclose(Dst, Dst.T)
    assert np.allclose(Dsc, Dsc.T)
    assert np.all(Dst >= D - 1e-12)  # stretch never shrinks
    assert np.all(Dsc <= D + 1e-12)  # shortcut never grows
    n_pairs = D.shape[0] * (D.shape[0] - 1) // 2
    iu = np.triu_indices(D.shape[0], k=1)
    changed = np.sum(~np.isclose(Dst[iu], D[iu]))
    assert changed == pytest.approx(round(0.3 * n_pairs), abs=1)


def test_violate_pairs_deterministic():
    D = _base_D()
    a = ND.violate_pairs(D, 0.2, 0.5, "stretch", seed=7)
    b = ND.violate_pairs(D, 0.2, 0.5, "stretch", seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, ND.violate_pairs(D, 0.2, 0.5, "stretch", seed=8))


def test_violate_pairs_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ND.violate_pairs(_base_D(), 0.2, 0.5, "wormhole")


# ---------------------------------------------------------------------------
# the structural claim: minimax(D) is ultrametric, hence admissible -- for
# EVERY input family, including non-metric ones and corrupted ones
# ---------------------------------------------------------------------------

SMALL_GENERATORS = [
    lambda: ND.dtw_traces(n_per=6, length=25, seed=11)[0],
    lambda: ND.edit_strings(n_per=6, seed=12)[0],
    lambda: ND.hamming_categorical(n_per=6, seed=13)[0],
    lambda: ND.graph_communities(n_per=6, seed=14)[0],
    lambda: ND.cosine_topics(n_per=6, seed=15)[0],
    lambda: ND.spiked_random(n=15, seed=16)[0],
    lambda: ND.violate_pairs(_base_D(), 0.4, 1.0, "stretch", seed=17),
    lambda: ND.violate_pairs(_base_D(), 0.4, 1.0, "shortcut", seed=18),
]


@pytest.mark.parametrize("gen", SMALL_GENERATORS)
def test_minimax_is_ultrametric_and_admissible(gen):
    D = gen()
    Dstar = im.minimax_transform_fast(D)
    assert ND.is_ultrametric(Dstar)
    assert ND.euclidean_embeddability(Dstar)["neg_ratio"] < 1e-9
    # idempotence: an ultrametric is a fixed point of the minimax transform
    assert np.allclose(im.minimax_transform_fast(Dstar), Dstar)


def test_beta_spread_positive_control():
    """The spiked matrix must fire NERFCM's beta-spread (and its D* must not),
    so 'beta = 0' rows elsewhere in the diagnostics are measurements."""
    D, _ = ND.spiked_random(n=20, seed=208)
    betas = [nerfcm(D, 2, seed=s)[1] for s in range(5)]
    assert max(betas) > 1.0
    Dstar = im.minimax_transform_fast(D)
    betas_star = [nerfcm(Dstar, 2, seed=s)[1] for s in range(5)]
    assert max(betas_star) == 0.0


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


def test_generators_deterministic_and_labeled():
    for fn, k in ND.BATTERY.values():
        D1, y1 = fn()
        D2, y2 = fn()
        assert np.array_equal(D1, D2)
        assert np.array_equal(y1, y2)
        assert D1.shape[0] == len(y1)
        assert len(np.unique(y1)) == k
        assert np.allclose(D1, D1.T)
        assert np.all(np.diag(D1) == 0)
        assert np.all(D1 >= 0)


def test_graph_communities_all_finite():
    D, _ = ND.graph_communities(n_per=10, p_in=0.15, p_out=0.005, seed=3)
    assert np.all(np.isfinite(D))


def test_relational_nested_hierarchy_structure():
    D, y_fine, y_coarse = ND.relational_nested_hierarchy()
    assert len(y_fine) == len(y_coarse) == D.shape[0] == 48
    assert len(np.unique(y_fine)) == 6
    assert len(np.unique(y_coarse)) == 3
    assert np.array_equal(y_coarse, y_fine // 2)
    # the three scales must be ordered: intra-sub < inter-sub < inter-super
    same_fine = (y_fine[:, None] == y_fine[None, :]) & ~np.eye(48, dtype=bool)
    same_coarse_diff_fine = (y_coarse[:, None] == y_coarse[None, :]) & (
        y_fine[:, None] != y_fine[None, :]
    )
    diff_coarse = y_coarse[:, None] != y_coarse[None, :]
    assert D[same_fine].max() < D[same_coarse_diff_fine].min()
    assert D[same_coarse_diff_fine].max() < D[diff_coarse].min()


def test_dtw_multivariate_reduces_to_univariate():
    a = np.array([0.0, 1.0, 3.0, 2.0])
    b = np.array([0.0, 2.0, 2.5, 2.0])
    assert ND.dtw_distance_multivariate(a, b) == pytest.approx(ND.dtw_distance(a, b))
    # and as explicit column vectors
    assert ND.dtw_distance_multivariate(
        a.reshape(-1, 1), b.reshape(-1, 1)
    ) == pytest.approx(ND.dtw_distance(a, b))


def test_dtw_multivariate_basic_invariants():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(12, 3))
    b = rng.normal(size=(15, 3))
    assert ND.dtw_distance_multivariate(a, a) == 0.0
    assert ND.dtw_distance_multivariate(a, b) == pytest.approx(
        ND.dtw_distance_multivariate(b, a)
    )
    assert ND.dtw_distance_multivariate(a, b) > 0


def test_bootstrap_jackknife_variant():
    """The jackknife (replace=False) path must run, discover the same k as the
    with-replacement original on a clean structured case, and be deterministic
    under a fixed seed."""
    from run_nonmetric import select_bottleneck_bootstrap_relational

    D, y = ND.edit_strings(n_per=8, seed=12)
    k_w, sel_w, meta_w = select_bottleneck_bootstrap_relational(D, replace=True)
    k_j, sel_j, meta_j = select_bottleneck_bootstrap_relational(D, replace=False)
    assert k_w == k_j == 3
    k_j2, _, meta_j2 = select_bottleneck_bootstrap_relational(D, replace=False)
    assert k_j2 == k_j
    assert meta_j2["gap_frequency"] == meta_j["gap_frequency"]


def test_knn_graph_hubs_invariants():
    D, y = ND.knn_graph_hubs(n_per=8, n_hubs=2, seed=3)
    assert D.shape[0] == len(y) == 26
    assert (y == -1).sum() == 2  # hubs are noise-labeled
    assert set(np.unique(y)) == {-1, 0, 1, 2}
    assert np.all(np.isfinite(D))
    assert np.allclose(D, D.T)
    D2, y2 = ND.knn_graph_hubs(n_per=8, n_hubs=2, seed=3)
    assert np.array_equal(D, D2) and np.array_equal(y, y2)
    # zero hubs must reduce to a clean 3-community graph
    D0, y0 = ND.knn_graph_hubs(n_per=8, n_hubs=0, seed=3)
    assert (y0 == -1).sum() == 0


def test_heavy_tailed_blobs_invariants():
    D, y = ND.heavy_tailed_blobs(n_per=8, df=1.5, seed=4)
    assert D.shape[0] == len(y) == 24
    assert set(np.unique(y)) == {0, 1, 2}  # every point keeps its true label
    assert np.allclose(D, D.T)
    assert np.all(D >= 0)
    D2, y2 = ND.heavy_tailed_blobs(n_per=8, df=1.5, seed=4)
    assert np.array_equal(D, D2)


def test_constrained_minimax_and_hub_drop_helpers():
    """H3 helpers: constraint injection produces a valid symmetric transform
    with must-link pairs at minimax distance ~0; hub-drop keeps shapes and
    ordering consistent."""
    from run_hard_cases import constrained_minimax, drop_low_mean_rows

    D, y = ND.knn_graph_hubs(n_per=8, n_hubs=2, seed=3)
    ml = [(0, 1), (1, 2)]
    cl = [(0, 8)]
    Dstar = constrained_minimax(D, ml, cl)
    assert np.allclose(Dstar, Dstar.T)
    tiny_ceiling = 1e-6 * D.max()
    assert Dstar[0, 1] < tiny_ceiling and Dstar[0, 2] < tiny_ceiling  # closure
    Dk, keep = drop_low_mean_rows(D, 2)
    assert Dk.shape == (D.shape[0] - 2, D.shape[0] - 2)
    assert len(keep) == D.shape[0] - 2
    assert np.array_equal(Dk, D[np.ix_(keep, keep)])
