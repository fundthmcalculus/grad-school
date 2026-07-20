"""
ConiVAT across the full battery, compared to plain iVAT single-linkage and to
the coverage-cover selector. Scored by ARI (non-ambiguous points) at true k.

ConiVAT needs constraints; we generate them from labels (as the paper does),
averaged over several constraint draws since the sample is random.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import ivat_mf as im
import battery as B
import selection as S
from conivat import conivat, sl_labels_from_mtd

DATASETS = [
    ("two_gaussians",     B.two_gaussians,     2),
    ("bridged_gaussians", B.bridged_gaussians, 2),
    ("concentric_rings",  B.concentric_rings,  2),
    ("varying_density",   B.varying_density,   3),
]
SEEDS = [0, 1, 2, 3, 4]


def ivat_sl_ari(X, y, k):
    D = im.dissimilarity(X); Ds = im.minimax_transform(D)
    Z = linkage(squareform(Ds, checks=False), method='single')
    lab = fcluster(Z, t=k, criterion='maxclust') - 1
    m = y >= 0
    return adjusted_rand_score(y[m], lab[m])


def conivat_ari(X, y, k, seeds=SEEDS, n_con=40):
    m = y >= 0
    aris = []
    for s in seeds:
        Dp = conivat(X, y, n_constraints=n_con, seed=s)
        lab = sl_labels_from_mtd(Dp, k)
        aris.append(adjusted_rand_score(y[m], lab[m]))
    return np.mean(aris), np.std(aris)


def cover_ari(X, y):
    D = im.dissimilarity(X); Ds = im.minimax_transform(D)
    sel = S.select_coverage_cover(Ds)
    if not sel:
        return np.nan
    n = Ds.shape[0]
    Db = np.zeros((len(sel), n))
    for k, b in enumerate(sel):
        mem = np.array(sorted(b['members']), int)
        Db[k] = Ds[:, mem].min(1)
    lab = np.argmin(Db, 0)
    m = y >= 0
    return adjusted_rand_score(y[m], lab[m])


def main():
    print(f"{'dataset':<19}{'iVAT-SL':>9}{'ConiVAT':>16}{'cover':>9}")
    print("-" * 54)
    for name, fn, k in DATASETS:
        X, y = fn()
        iv = ivat_sl_ari(X, y, k)
        cm, cs = conivat_ari(X, y, k)
        cv = cover_ari(X, y)
        cvs = "n/a" if np.isnan(cv) else f"{cv:.2f}"
        print(f"{name:<19}{iv:>9.2f}{f'{cm:.2f}+-{cs:.2f}':>16}{cvs:>9}")

    print()
    print("iVAT-SL : plain single-linkage on minimax D* at true k (no constraints)")
    print("ConiVAT : + metric learning from 40 label-derived constraints, mean/5 seeds")
    print("cover   : coverage-cover selector, k discovered (no k, no constraints)")
    print()
    print("Expected story:")
    print("  bridged_gaussians: iVAT-SL fails (chaining), ConiVAT fixes it with")
    print("    constraints. Quantifies the value of building on ConiVAT.")
    print("  Cost: ConiVAT needs labeled constraints + metric learning; cover")
    print("    needs neither but trails on varying_density. Different trade-offs.")


if __name__ == "__main__":
    main()
