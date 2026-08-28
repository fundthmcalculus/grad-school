"""Tests for relationdata.py's generators, pinning the ground-truth integrity
that issue #160 found violated: every declared label must agree with the
sub-cluster the distances actually encode.

Run: ``python -m pytest gated-minimax-selection/test_relationdata.py``
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import relationdata as RD  # noqa: E402


def _structural_components(D, threshold):
    from scipy.sparse.csgraph import connected_components

    n_comp, comp = connected_components(D < threshold, directed=False)
    return n_comp, comp


def test_multi_scale_hierarchy_labels_match_structure():
    """The #160 regression: zero declared labels may disagree with the
    connected component their point belongs to at the intra-sub-cluster scale
    (the construction's scales are ~0.8 / ~4.6 / ~12.6, so 2.0 separates them
    unambiguously)."""
    D, y = RD.multi_scale_hierarchy()
    n_comp, comp = _structural_components(D, 2.0)
    assert n_comp == 6  # 3 large clusters x 2 sub-clusters
    for c in range(n_comp):
        labs = y[comp == c]
        assert (
            labs == labs[0]
        ).all(), f"component {c} carries mixed labels {set(labs)}"
    # and the six components carry six DISTINCT labels
    assert len({y[comp == c][0] for c in range(n_comp)}) == 6


def test_multi_scale_hierarchy_n_and_determinism():
    D, y = RD.multi_scale_hierarchy(n=45)
    assert len(y) == 45  # the old top-up loop counted internal nodes: gave 39
    assert D.shape == (45, 45)
    D2, y2 = RD.multi_scale_hierarchy(n=45)
    assert np.array_equal(D, D2)
    assert np.array_equal(y, y2)


def test_all_generators_basic_invariants():
    for fn, k in [
        (RD.three_clusters_tree, 3),
        (RD.chain_then_ring, 2),
        (RD.multi_scale_hierarchy, 6),
    ]:
        D, y = fn()
        assert D.shape[0] == len(y)
        assert np.allclose(D, D.T)
        assert np.all(np.diag(D) == 0)
        assert np.all(D >= 0)
        assert len(np.unique(y)) == k
