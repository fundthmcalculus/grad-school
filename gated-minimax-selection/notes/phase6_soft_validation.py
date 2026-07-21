"""Phase 6 soft-calibration check (see MF_PROGRESS_LOG.md Phase 6).

Are the minimax fuzzy memberships calibrated soft posteriors? Score the Brier
distance of the generated membership to the ANALYTIC posterior of a 2-Gaussian
mixture, versus crisp 0/1 labels. Finding: the fuzzy MF is WORSE than crisp
labels -- ultrametric distances make it a constant step per cluster, with no
boundary resolution.

Run:  python notes/phase6_soft_validation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import ivat_mf as im
import multiscale_persistence as MS


def gt_block(Ds, y, lab):
    """Ground-truth block with birth/death from dendrogram geometry (isolates MF
    quality from the selection question)."""
    mem = set(np.where(y == lab)[0].tolist())
    m = np.fromiter(mem, int)
    birth = Ds[np.ix_(m, m)].max()
    others = np.array([i for i in range(Ds.shape[0]) if i not in mem])
    death = max(Ds[np.ix_(m, others)].min(), birth * 1.01)
    return {'members': mem, 'birth': birth, 'death': death, 'size': len(mem)}


def gauss(x, m, s):
    return np.exp(-((x - m) ** 2).sum(1) / (2 * s * s))


def main():
    print(f"{'sep':>5}{'Brier fuzzy':>13}{'Brier hard':>12}{'graded_frac':>13}{'boundary':>10}")
    for sep in [2.0, 3.0, 4.0, 6.0]:
        rng = np.random.default_rng(1)
        m0, m1, s = np.array([0, 0]), np.array([sep, 0]), 1.0
        X = np.vstack([rng.normal(m0, s, (120, 2)), rng.normal(m1, s, (120, 2))])
        y = np.array([0] * 120 + [1] * 120)
        Ds = im.minimax_transform_fast(im.dissimilarity(X))
        blocks = [gt_block(Ds, y, 0), gt_block(Ds, y, 1)]
        U = np.vstack([MS.block_membership(b, Ds, kernel='gaussian') for b in blocks])
        graded = np.mean((U > 1e-6) & (U < 1 - 1e-6))
        U = MS.normalize_partition(U)
        mu0 = U[0]
        N0, N1 = gauss(X, m0, s), gauss(X, m1, s)
        post0 = N0 / (N0 + N1)
        bf = np.mean((mu0 - post0) ** 2)
        bh = np.mean(((y == 0).astype(float) - post0) ** 2)
        boundary = np.mean((post0 > 0.2) & (post0 < 0.8))
        print(f"{sep:>5.1f}{bf:>13.4f}{bh:>12.4f}{graded:>13.3f}{boundary:>10.2f}")
    print("\nlower Brier vs TRUE posterior = better-calibrated soft memberships.")
    print("Finding: fuzzy MF worse than crisp 0/1 -> ultrametric step, no boundary "
          "resolution (see MF_PROGRESS_LOG.md Phase 6).")


if __name__ == '__main__':
    main()
