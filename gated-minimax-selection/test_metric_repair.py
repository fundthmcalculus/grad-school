"""Tests for metric_repair.reverse_ti_repair: the two structural properties
(identity on metric inputs at any quantile; one-sidedness), the repair it
exists to perform (a planted shortcut lifted back to cluster scale), and the
scope boundary (deeply non-metric inputs get lifted heavily -- by design the
repair is for metric-plus-sparse-corruption data, not intrinsic non-metricity).

Run: ``python -m pytest gated-minimax-selection/test_metric_repair.py``
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ivat_mf as im  # noqa: E402
import nonmetric_data as ND  # noqa: E402
from metric_repair import reverse_ti_repair, witness_lower_bounds  # noqa: E402


def _euclidean_D(n=30, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(X))


# ---------------------------------------------------------------------------
# structural properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.25, 0.5, 0.75, 0.9, 1.0])
def test_identity_on_euclidean_input_at_any_quantile(q):
    D = _euclidean_D()
    assert np.allclose(reverse_ti_repair(D, q), D)


@pytest.mark.parametrize(
    "gen",
    [
        lambda: ND.edit_strings(n_per=6, seed=12)[0],
        lambda: ND.hamming_categorical(n_per=6, seed=13)[0],
        lambda: ND.graph_communities(n_per=6, seed=14)[0],
    ],
)
def test_identity_on_metric_families(gen):
    """Every witness bound is <= D_ij in a metric, so no quantile can lift --
    including graph shortest paths, whose thin inter-community edges are REAL
    structure the repair must not touch."""
    D = gen()
    assert np.allclose(reverse_ti_repair(D, 0.5), D)
    assert np.allclose(reverse_ti_repair(D, 0.9), D)


def test_one_sided_never_decreases():
    D = ND.violate_pairs(_euclidean_D(), 0.3, 1.0, "shortcut", seed=5)
    R = reverse_ti_repair(D, 0.75)
    assert np.all(R >= D - 1e-12)
    assert np.allclose(R, R.T)
    assert np.all(np.diag(R) == 0)


def test_witness_bounds_are_lower_bounds_in_metric():
    D = _euclidean_D()
    LB = witness_lower_bounds(D, 1.0)  # even the MAX witness bound
    assert np.all(LB <= D + 1e-9)


# ---------------------------------------------------------------------------
# the repair it exists to perform
# ---------------------------------------------------------------------------


def test_planted_shortcut_lifted_back_to_scale():
    """Deflate one cross-cluster entry to near zero; the repair must lift it
    back to (at least) cluster-separation scale, and the minimax transform of
    the repaired matrix must re-separate the clusters."""
    D0, y, _ = ND.euclidean_blobs(n_per=10, seed=1)
    i = int(np.where(y == 0)[0][0])
    j = int(np.where(y == 1)[0][0])
    true_val = D0[i, j]
    Dv = D0.copy()
    Dv[i, j] = Dv[j, i] = 0.01  # a hard bridge
    R = reverse_ti_repair(Dv, 0.5)
    assert R[i, j] > 0.5 * true_val  # lifted back to cluster scale
    # and the bridge no longer fuses the two clusters in minimax space. A
    # fused pair of clusters merges at INTRA-cluster height (the bottleneck
    # path routes through the bridge, so the max edge is an intra edge), so
    # the fusion signature is cross ~ intra, and repair must restore
    # cross >> intra.
    Dstar_raw = im.minimax_transform_fast(Dv)
    Dstar_rep = im.minimax_transform_fast(R)
    intra_raw = np.median(Dstar_raw[np.ix_(y == 0, y == 0)])
    cross_raw = np.median(Dstar_raw[np.ix_(y == 0, y == 1)])
    intra_rep = np.median(Dstar_rep[np.ix_(y == 0, y == 0)])
    cross_rep = np.median(Dstar_rep[np.ix_(y == 0, y == 1)])
    assert cross_raw < 2 * intra_raw  # fused before repair
    assert cross_rep > 5 * intra_rep  # re-separated after


def test_genuine_short_pairs_untouched_under_sparse_corruption():
    """Sparse shortcut corruption must not cause the repair to lift
    UNCORRUPTED entries (at the median quantile)."""
    D0, _, _ = ND.euclidean_blobs(n_per=10, seed=2)
    Dv = ND.violate_pairs(D0, 0.1, 1.0, "shortcut", seed=3)
    corrupted = ~np.isclose(Dv, D0)
    R = reverse_ti_repair(Dv, 0.5)
    lifted = R > Dv + 1e-12
    assert not np.any(lifted & ~corrupted)


# ---------------------------------------------------------------------------
# the scope boundary
# ---------------------------------------------------------------------------


def test_deeply_nonmetric_input_is_out_of_scope():
    """Deep upper-bound TI violations imply reverse-TI violations at other
    pairs of the same triple (D_ij > D_ik + D_jk  <=>  D_ik < D_ij - D_jk), so
    an intrinsically, DENSELY deeply non-metric matrix gets lifted heavily --
    the repair is for metric-plus-sparse-corruption data. (Sparse deep
    violations do NOT trigger the median witness -- that robustness is tested
    above -- so this uses dense violations, the regime real flight-profile DTW
    turned out to occupy: 70% of pairs violated, 50% of entries lifted.)"""
    rng = np.random.default_rng(0)
    n = 25
    D = rng.uniform(1.0, 2.0, size=(n, n))
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    # densely deeply non-metric: half of all entries far above any two-hop path
    iu = np.triu_indices(n, k=1)
    hit = rng.choice(len(iu[0]), size=len(iu[0]) // 2, replace=False)
    D[iu[0][hit], iu[1][hit]] *= 4.0
    D[iu[1][hit], iu[0][hit]] = D[iu[0][hit], iu[1][hit]]
    R = reverse_ti_repair(D, 0.5)
    assert np.mean(R > D + 1e-12) > 0.05  # a nontrivial fraction lifted
