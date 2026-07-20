"""
Relational data benchmark: NERFCM on distance-matrix-only data.

Demonstrates that D* (minimax distance) can improve clustering on data where
only pairwise distances are available (no vector coordinates). Since these are
non-Euclidean-embeddable structures, vector-space methods don't apply; NERFCM
on D* is the natural comparison point.

Datasets are synthetic trees/hierarchies where D* should help uncover structure
that raw D might obscure.
"""

import numpy as np
import sys
import os

# Import from the actual project root
# Worktree is at: /home/scott/PycharmProjects/grad-school/.claude/worktrees/relationdata-dataset
# Project root is at: /home/scott/PycharmProjects/grad-school/gated-minimax-selection
project_root = "/home/scott/PycharmProjects/grad-school/gated-minimax-selection"
sys.path.insert(0, project_root)

from sklearn.metrics import adjusted_rand_score
import ivat_mf as im
from nerfcm import nerfcm
import relationdata as RD


DATASETS = [
    ("three_clusters_tree", RD.three_clusters_tree, 3),
    ("chain_then_ring", RD.chain_then_ring, 2),
    ("multi_scale_hierarchy", RD.multi_scale_hierarchy, 3),
]

SEEDS = [0, 1, 2]


def nerfcm_ari(M, y, c, seeds=SEEDS):
    """Run NERFCM on matrix M and score ARI over multiple seeds."""
    mask = y >= 0
    aris = []
    for s in seeds:
        U, beta, it = nerfcm(M, c, seed=s)
        lab = np.argmax(U, axis=0)
        if mask.sum() > 0:
            aris.append(adjusted_rand_score(y[mask], lab[mask]))
    return (np.mean(aris), np.std(aris)) if aris else (np.nan, np.nan)


def main():
    print("=" * 75)
    print("Relational Data Benchmark: NERFCM on Distance Matrices Only")
    print("=" * 75)
    print()
    print("These datasets have no vector coordinates—only pairwise distances.")
    print("Matrix methods (NERFCM) are the natural/only choice.")
    print("Columns show ARI improvement: does D* (minimax distance) help?")
    print()

    print(f"{'dataset':<25}{'NERFCM(D)':>20}{'NERFCM(D*)':>20}{'Δ ARI':>12}")
    print("-" * 75)

    gaps = []
    for name, fn, c_true in DATASETS:
        D, y = fn()
        Dstar = im.minimax_transform(D)

        # Run on raw D
        m_d, s_d = nerfcm_ari(D, y, c_true)
        # Run on D*
        m_ds, s_ds = nerfcm_ari(Dstar, y, c_true)

        def fmt(mean, std):
            if np.isnan(mean):
                return "n/a"
            return f"{mean:.2f}±{std:.2f}"

        delta = m_ds - m_d if not np.isnan(m_d) and not np.isnan(m_ds) else np.nan
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "n/a"

        print(f"{name:<25}{fmt(m_d, s_d):>20}{fmt(m_ds, s_ds):>20}{delta_str:>12}")
        if not np.isnan(delta):
            gaps.append(delta)

    print("-" * 75)
    if gaps:
        print(f"Mean Δ ARI (D* vs D): {np.mean(gaps):+.3f}")
    print()

    print("Interpretation:")
    print("  - These datasets are relational only: no feature vectors.")
    print("  - NERFCM is the baseline method (designed for distance matrices).")
    print("  - Positive Δ ARI: D* helps NERFCM find structure better than raw D.")
    print("  - D* is the minimax bottleneck-distance transform.")
    print()
    print("Dataset descriptions:")
    print("  - three_clusters_tree: 3 clusters on a tree (tight intra, far inter).")
    print("  - chain_then_ring: elongated vs circular clusters; tests non-convexity.")
    print("  - multi_scale_hierarchy: nested clusters at different scales.")
    print()


if __name__ == "__main__":
    main()
