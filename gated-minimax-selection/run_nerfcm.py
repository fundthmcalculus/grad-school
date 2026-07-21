"""
NERFCM partition-quality baseline vs the iVAT coverage-cover selector.

Fair-comparison design:
  - NERFCM is given the true c (it has no cluster-count discovery).
  - Run NERFCM on BOTH raw Euclidean D and minimax D*, to separate what the
    minimax transform contributes from what NERFCM contributes.
  - Score ARI (on non-ambiguous points) + coverage (trivially 1.0 for NERFCM
    since it partitions everything - reported for parity).
  - c-sensitivity: ARI at c-1, c, c+1.
  - iVAT coverage-cover column reproduced from selection.py for side-by-side.

Averaged over several seeds because NERFCM depends on random init.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
import sys
import os

project_root = "/home/scott/PycharmProjects/grad-school/gated-minimax-selection"
sys.path.insert(0, project_root)

import ivat_mf as im
import battery as B
import selection as S
from nerfcm import nerfcm
import relationdata as RD


DATASETS = [
    ("two_gaussians",     B.two_gaussians,     2),
    ("bridged_gaussians", B.bridged_gaussians, 2),
    ("concentric_rings",  B.concentric_rings,  2),
    ("varying_density",   B.varying_density,   3),
    ("uniform_noise",     B.uniform_noise,     2),
]

# Relational datasets (distance-matrix-only, no vector coordinates)
RELATIONAL_DATASETS = [
    ("three_clusters_tree", RD.three_clusters_tree, 3),
    ("chain_then_ring",     RD.chain_then_ring,     2),
    ("multi_scale_hierarchy", RD.multi_scale_hierarchy, 3),
]

SEEDS = [0, 1, 2, 3, 4]


def nerfcm_ari(M, y, c, seeds=SEEDS):
    mask = y >= 0
    aris = []
    for s in seeds:
        U, beta, it = nerfcm(M, c, seed=s)
        lab = np.argmax(U, axis=0)
        if mask.sum() > 0:
            aris.append(adjusted_rand_score(y[mask], lab[mask]))
    return (np.mean(aris), np.std(aris)) if aris else (np.nan, np.nan)


def ivat_cover_ari(Dstar, y):
    """Score the coverage-cover selection by assigning each point to its
    highest-membership selected block (proximity in D*)."""
    sel = S.select_coverage_cover(Dstar)
    if not sel:
        return np.nan, 0
    n = Dstar.shape[0]
    # membership by minimax proximity to each selected block
    Dblock = np.zeros((len(sel), n))
    for k, b in enumerate(sel):
        mem = np.array(sorted(b['members']), dtype=int)
        Dblock[k] = Dstar[:, mem].min(axis=1)
    lab = np.argmin(Dblock, axis=0)
    mask = y >= 0
    if mask.sum() == 0:
        return np.nan, len(sel)
    return adjusted_rand_score(y[mask], lab[mask]), len(sel)


def main():
    print("Partition-quality baseline. ARI on non-ambiguous points, mean over "
          f"{len(SEEDS)} seeds for NERFCM.\n")
    print(f"{'dataset':<19}{'NERFCM(D)':>18}{'NERFCM(D*)':>18}"
          f"{'iVAT-cover':>14}")
    print("-" * 69)
    for name, fn, c in DATASETS:
        X, y = fn()
        D = im.dissimilarity(X)
        Dstar = im.minimax_transform(D)

        m_d, s_d = nerfcm_ari(D, y, c)
        m_ds, s_ds = nerfcm_ari(Dstar, y, c)
        iv, nblk = ivat_cover_ari(Dstar, y)

        def fmt(mean, std):
            if np.isnan(mean):
                return "n/a"
            return f"{mean:.2f}+-{std:.2f}"

        ivs = "n/a" if np.isnan(iv) else f"{iv:.2f} ({nblk}blk)"
        print(f"{name:<19}{fmt(m_d,s_d):>18}{fmt(m_ds,s_ds):>18}{ivs:>14}")

    print("\nc-sensitivity (NERFCM on D*, mean ARI at c-1, c, c+1):")
    for name, fn, c in DATASETS:
        X, y = fn()
        D = im.dissimilarity(X)
        Dstar = im.minimax_transform(D)
        row = []
        for cc in [max(2, c - 1), c, c + 1]:
            m, _ = nerfcm_ari(Dstar, y, cc)
            row.append("n/a" if np.isnan(m) else f"{m:.2f}")
        print(f"  {name:<19} [{', '.join(row)}]")

    print("\n" + "=" * 69)
    print("RELATIONAL DATA BENCHMARK")
    print("=" * 69)
    print("\nDistance-matrix-only datasets (no vector coordinates).")
    print("These showcase scenarios where matrix methods are the ONLY option.")
    print(f"Format: dataset, NERFCM(D), NERFCM(D*), where D* = minimax distance.\n")
    print(f"{'dataset':<25}{'NERFCM(D)':>20}{'NERFCM(D*)':>20}")
    print("-" * 69)
    for name, fn, c in RELATIONAL_DATASETS:
        D, y = fn()
        Dstar = im.minimax_transform(D)

        m_d, s_d = nerfcm_ari(D, y, c, seeds=SEEDS[:3])  # use fewer seeds for relational
        m_ds, s_ds = nerfcm_ari(Dstar, y, c, seeds=SEEDS[:3])

        def fmt(mean, std):
            if np.isnan(mean):
                return "n/a"
            return f"{mean:.2f}±{std:.2f}"

        print(f"{name:<25}{fmt(m_d, s_d):>20}{fmt(m_ds, s_ds):>20}")

    print("\nInterpretation:")
    print("  NERFCM(D) vs NERFCM(D*): gap isolates what the minimax transform")
    print("    contributes independent of any selection machinery.")
    print("  rings: expect NERFCM(D) to FAIL (non-convex) and NERFCM(D*) to")
    print("    succeed - evidence the transform is the load-bearing piece.")
    print("  uniform_noise: NERFCM forced to c=2 fabricates structure (ARI is")
    print("    meaningless but it still returns a confident partition); iVAT-")
    print("    cover can decline. That capability gap is the qualitative story.")
    print("\n  RELATIONAL DATASETS: These have no vector coordinates, only distances.")
    print("    Matrix methods like NERFCM are the natural choice. The presence of")
    print("    a gap between NERFCM(D) and NERFCM(D*) shows where minimax transform")
    print("    helps on purely relational data.")


if __name__ == "__main__":
    main()
