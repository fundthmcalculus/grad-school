"""The selection rule returns a disjoint ANTICHAIN, never overlapping blocks.

`select_coverage_cover` is written as a greedy set-cover and its docstring used
to claim it tolerated overlap. It cannot: dendrogram nodes are a laminar family
(any two nested or disjoint), so greedy-by-uncovered-gain always takes a
maximal eligible node and then finds every descendant at gain 0. What it
actually computes is a local cut through the hierarchy, in the sense of
Campello et al.'s FOSC / HDBSCAN* framework.

Pinned so the corrected docstring cannot rot back into the false claim.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import battery as B  # noqa: E402
import ivat_mf as im  # noqa: E402
import nonmetric_data as ND  # noqa: E402
import relationdata as RD  # noqa: E402
import selection as S  # noqa: E402


def _pairs_overlapping(sel):
    return sum(
        1 for a, b in itertools.combinations(sel, 2) if a["members"] & b["members"]
    )


COORD = [
    B.two_gaussians,
    B.bridged_gaussians,
    B.concentric_rings,
    B.varying_density,
    B.uniform_noise,
]
REL = [RD.three_clusters_tree, RD.chain_then_ring, RD.multi_scale_hierarchy]


@pytest.mark.parametrize("gen", COORD)
def test_no_overlap_coordinate_battery(gen):
    X, _ = gen()
    sel = S.select_coverage_cover(im.minimax_transform_fast(im.dissimilarity(X)))
    assert _pairs_overlapping(sel) == 0


@pytest.mark.parametrize("gen", REL)
def test_no_overlap_relational_battery(gen):
    D, _ = gen()
    sel = S.select_coverage_cover(im.minimax_transform_fast(D))
    assert _pairs_overlapping(sel) == 0


@pytest.mark.parametrize("name", sorted(ND.BATTERY))
def test_no_overlap_nonmetric_battery(name):
    D, _ = ND.BATTERY[name][0]()
    sel = S.select_coverage_cover(im.minimax_transform_fast(D))
    assert _pairs_overlapping(sel) == 0


def test_selection_is_an_antichain_not_merely_disjoint():
    """No selected block may be an ancestor/descendant of another -- the
    stronger structural statement, which disjointness implies for a laminar
    family but is worth asserting directly."""
    X, _ = B.varying_density()
    sel = S.select_coverage_cover(im.minimax_transform_fast(im.dissimilarity(X)))
    for a, b in itertools.combinations(sel, 2):
        assert not (a["members"] <= b["members"] or b["members"] <= a["members"])
