"""
Run the full iVAT->MF battery and print a verdict table.
"""

import numpy as np
import ivat_mf as im
import battery as B


def medoids_from_labels(Dstar, y, c):
    """Pick a medoid per ground-truth cluster to seed Mapping 1 fairly."""
    meds = []
    for lab in sorted(set(y[y >= 0]))[:c]:
        idx = np.where(y == lab)[0]
        sub = Dstar[np.ix_(idx, idx)]
        meds.append(idx[np.argmin(sub.sum(axis=1))])
    # pad if needed
    while len(meds) < c:
        meds.append(int(np.argmax(Dstar.sum(axis=1))))
    return meds


DATASETS = [
    ("two_gaussians",     B.two_gaussians,     2),
    ("bridged_gaussians", B.bridged_gaussians, 2),
    ("concentric_rings",  B.concentric_rings,  2),
    ("varying_density",   B.varying_density,   3),
    ("uniform_noise",     B.uniform_noise,     2),
]


def main():
    header = (f"{'dataset':<20}{'M1 ARI':>8}{'M2 ARI':>8}{'kmeans':>8}"
              f"{'cover':>7}{'convex':>8}   c-sensitivity(ARI @ c-1,c,c+1)")
    print(header)
    print("-" * len(header))

    for name, fn, c in DATASETS:
        X, y = fn()
        D = im.dissimilarity(X)
        Dstar = im.minimax_transform(D)

        # Mapping 1
        meds = medoids_from_labels(Dstar, y, c)
        U1 = im.mapping1_medoid(Dstar, meds)
        ari1 = B.score_ari(U1, y)

        # Mapping 2
        U2, info, Dblock = im.mapping2_persistence(Dstar, c)
        ari2 = B.score_ari(U2, y, Dblock)

        cov = B.coverage(U2)
        conv = B.univariate_convexity(U2, X, axis=0)
        km = B.kmeans_ari(X, y, c)
        csens = B.c_sensitivity(Dstar, y, c)
        csens_str = ", ".join(f"{v:.2f}" for v in csens.values())

        ari1s = "  n/a" if np.isnan(ari1) else f"{ari1:6.2f}"
        ari2s = "  n/a" if np.isnan(ari2) else f"{ari2:6.2f}"
        kms   = "  n/a" if np.isnan(km)   else f"{km:6.2f}"

        print(f"{name:<20}{ari1s:>8}{ari2s:>8}{kms:>8}"
              f"{cov:7.2f}{conv:8.2f}   [{csens_str}]")

    print()
    print("Reading the table:")
    print("  M2 ARI ~1.0 on concentric_rings while kmeans ~0.0  => the WIN case")
    print("    (minimax/SL structure captures non-convex clusters; centroids can't).")
    print("  M2 ARI on bridged_gaussians is the KILL test: low ARI + bridge points")
    print("    absorbed => single-linkage chaining has propagated into the MFs.")
    print("  c-sensitivity spread small  => headline 'robust to cluster count' holds.")
    print("  convex<1.0 => some generated MFs are multi-modal on that axis")
    print("    (linguistic-labeling problem flagged in the design review).")
    print("  uniform_noise: any confident high-coverage membership is a FALSE structure")
    print("    warning - ideally persistence is low and we'd decline to commit.")


if __name__ == "__main__":
    main()
