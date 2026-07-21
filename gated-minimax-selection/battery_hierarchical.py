"""
Hierarchical / multi-scale synthetic datasets with GROUND TRUTH AT EACH LEVEL.

The ordinary battery (`battery.py`) has a single ground-truth partition per
dataset, so it cannot score whether a method recovers *nested* structure. These
generators return one label vector per scale level (fine, [medium,] coarse), so
a multi-scale selector can be scored against every level simultaneously.

Design principle (learned from probing the flat coverage_cover baseline): within
each super-cluster the sub-clusters are well separated *relative to their own
spread*, and super-clusters are separated by a much larger gap. This creates a
genuine gap in the (log) birth-height axis of the single-linkage dendrogram --
one contiguous band of births per generation -- which is exactly the structure
`multiscale_persistence.discover_band_edges` keys on. If the levels were NOT
scale-separated (overlapping birth bands) the problem would be ill-posed for any
band-based method; that regime is called out as future work in the writeup.

Each generator returns (X, *label_vectors_fine_to_coarse).
"""

import numpy as np


def nested_gaussians(n_per=20, seed=7, super_sep=40.0, sub_sep=6.0, sub_sigma=0.4):
    """Two super-clusters, each with three tight sub-clusters (6 fine, 2 coarse).

    Returns (X, y_fine[0..5], y_coarse[0..1]).
    """
    rng = np.random.default_rng(seed)
    supers = [np.array([0.0, 0.0]), np.array([super_sep, 0.0])]
    layout = [np.array([0.0, 0.0]),
              np.array([sub_sep, 0.0]),
              np.array([sub_sep / 2, sub_sep * 0.83])]
    X, y_fine, y_coarse = [], [], []
    fine = 0
    for si, sc in enumerate(supers):
        for off in layout:
            X.append(rng.normal(sc + off, sub_sigma, (n_per, 2)))
            y_fine += [fine] * n_per
            y_coarse += [si] * n_per
            fine += 1
    return np.vstack(X), np.array(y_fine), np.array(y_coarse)


def three_level_hierarchy(n_per=12, seed=11,
                          l1_sep=200.0, l2_sep=28.0, l3_sep=5.0, sigma=0.35):
    """A 2 x 2 x 2 balanced tree: 8 fine, 4 medium, 2 coarse clusters.

    Three genuinely distinct scales. Returns (X, y_fine, y_medium, y_coarse).
    """
    rng = np.random.default_rng(seed)
    X = []
    y_fine, y_med, y_coarse = [], [], []
    fine = med = 0
    for c in range(2):                      # coarse
        cx = np.array([c * l1_sep, 0.0])
        for m in range(2):                  # medium
            mx = cx + np.array([0.0, m * l2_sep])
            for f in range(2):              # fine
                fx = mx + np.array([f * l3_sep, 0.0])
                X.append(rng.normal(fx, sigma, (n_per, 2)))
                y_fine += [fine] * n_per
                y_med += [med] * n_per
                y_coarse += [c] * n_per
                fine += 1
            med += 1
    return np.vstack(X), np.array(y_fine), np.array(y_med), np.array(y_coarse)


def density_hierarchy(n_per=25, seed=19, super_sep=30.0, sub_sep=5.0):
    """Two super-clusters whose sub-clusters have DIFFERENT spreads within each
    super (tight + diffuse side by side). Stresses that a single per-band spread
    assumption is not required. Returns (X, y_fine[0..3], y_coarse[0..1]).
    """
    rng = np.random.default_rng(seed)
    supers = [np.array([0.0, 0.0]), np.array([super_sep, 0.0])]
    # within each super: one tight sub, one looser sub
    sub_layout = [(np.array([0.0, 0.0]), 0.30),
                  (np.array([sub_sep, 0.0]), 0.75)]
    X, y_fine, y_coarse = [], [], []
    fine = 0
    for si, sc in enumerate(supers):
        for off, sg in sub_layout:
            X.append(rng.normal(sc + off, sg, (n_per, 2)))
            y_fine += [fine] * n_per
            y_coarse += [si] * n_per
            fine += 1
    return np.vstack(X), np.array(y_fine), np.array(y_coarse)


# Registry: name -> (generator, [level names fine->coarse])
HIERARCHICAL = {
    'nested_gaussians': (nested_gaussians, ['fine(6)', 'coarse(2)']),
    'three_level_hierarchy': (three_level_hierarchy, ['fine(8)', 'medium(4)', 'coarse(2)']),
    'density_hierarchy': (density_hierarchy, ['fine(4)', 'coarse(2)']),
}
