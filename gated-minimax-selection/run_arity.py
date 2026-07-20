"""
Compare disjunct-arity detection modes against ground truth.

For each dataset we know the "right" answer:
  two_gaussians     : 2 blocks, each arity 1  (simple convex sets)
  bridged_gaussians : ideally 2 blocks arity 1; chaining may merge -> watch
  concentric_rings  : 2 blocks. In D* each ring is ONE component (arity 1).
                      In feature space each ring is spatially connected too
                      (it's a closed loop), so arity should also be 1 -- the
                      projection multimodality is per-AXIS, not a spatial split.
                      This is the subtle case.
  varying_density   : 3 blocks, each arity 1
  uniform_noise     : no real structure
"""

import numpy as np
import ivat_mf as im
import battery as B
import disjunct as dj


DATASETS = [
    ("two_gaussians",     B.two_gaussians,     2),
    ("bridged_gaussians", B.bridged_gaussians, 2),
    ("concentric_rings",  B.concentric_rings,  2),
    ("varying_density",   B.varying_density,   3),
    ("uniform_noise",     B.uniform_noise,     2),
]

MODES = ['dstar', 'geometric']


def gt_block(Dstar, y, label):
    """Build a block dict from ground-truth membership, with birth/death heights
    estimated from the within-cluster and to-nearest-other-cluster minimax
    distances. This isolates the ARITY question from the SELECTION question."""
    members = set(np.where(y == label)[0].tolist())
    mem = np.array(sorted(members), dtype=int)
    within = Dstar[np.ix_(mem, mem)]
    birth = within.max()  # height at which the whole cluster is connected
    others = np.array([i for i in range(Dstar.shape[0]) if i not in members])
    if len(others) > 0:
        death = Dstar[np.ix_(mem, others)].min()  # nearest escape to another cluster
        death = max(death, birth * 1.01)
    else:
        death = birth * 1.5
    return {'members': members, 'birth': birth, 'death': death, 'size': len(members)}


def main():
    print(f"{'dataset':<20}{'mode':<9}{'ground-truth block arities':<40}")
    print("-" * 70)
    for name, fn, c in DATASETS:
        X, y = fn()
        D = im.dissimilarity(X)
        Dstar = im.minimax_transform(D)
        labels = sorted(set(y[y >= 0].tolist()))
        if len(labels) == 0:
            print(f"{name:<20}(no ground-truth structure)\n")
            continue
        blocks = [gt_block(Dstar, y, lab) for lab in labels]
        for mode in MODES:
            arities = []
            for blk in blocks:
                nc, _ = dj.block_arity(Dstar, X, blk, mode=mode)
                arities.append(nc)
            sizes = [blk['size'] for blk in blocks]
            arity_str = ", ".join(f"{a}(n={s})" for a, s in zip(arities, sizes))
            print(f"{name:<20}{mode:<9}{arity_str:<40}")
        print()


if __name__ == "__main__":
    main()
