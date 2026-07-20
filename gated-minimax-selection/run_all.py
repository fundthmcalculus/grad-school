"""
MASTER REPRODUCIBLE ANALYSIS for the iVAT -> Membership Function investigation.

Runs the entire chain deterministically and writes:
  - results.json      : every numeric result, one source of truth
  - fig1_datasets.png : the five synthetic datasets (ground truth)
  - fig2_transform.png: raw D vs minimax D* heatmaps (why the transform matters)
  - fig3_methods_ari.png : ARI comparison across methods (the headline chart)
  - fig4_persistence.png : sorted-persistence curves + knee (selection story)
  - fig5_membership.png  : example generated membership functions (1-D profiles)
  - fig6_conivat_bridge.png : the 0.00 -> 1.00 chaining repair, visualized

All randomness is seeded. Re-running reproduces identical numbers and figures.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import ivat_mf as im
import battery as B
import selection as S
from nerfcm import nerfcm
from conivat import coniv at, sl_labels_from_mtd

OUT = "/mnt/user-data/outputs"
SEEDS = [0, 1, 2, 3, 4]

DATASETS = [
    ("two_gaussians",     B.two_gaussians,     2),
    ("bridged_gaussians", B.bridged_gaussians, 2),
    ("concentric_rings",  B.concentric_rings,  2),
    ("varying_density",   B.varying_density,   3),
    ("uniform_noise",     B.uniform_noise,     2),
]

results = {}


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------

def ivat_sl_ari(X, y, k):
    D = im.dissimilarity(X); Ds = im.minimax_transform(D)
    Z = linkage(squareform(Ds, checks=False), method='single')
    lab = fcluster(Z, t=k, criterion='maxclust') - 1
    m = y >= 0
    return adjusted_rand_score(y[m], lab[m]) if m.sum() else np.nan


def nerfcm_ari(M, y, c):
    m = y >= 0
    if m.sum() == 0:
        return np.nan, np.nan
    aris = [adjusted_rand_score(y[m], np.argmax(nerfcm(M, c, seed=s)[0], 0)[m])
            for s in SEEDS]
    return float(np.mean(aris)), float(np.std(aris))


def conivat_ari(X, y, k, n_con=40):
    m = y >= 0
    if m.sum() == 0:
        return np.nan, np.nan
    aris = [adjusted_rand_score(y[m], sl_labels_from_mtd(conivat(X, y, n_con, s), k)[m])
            for s in SEEDS]
    return float(np.mean(aris)), float(np.std(aris))


def cover_result(X, y):
    D = im.dissimilarity(X); Ds = im.minimax_transform(D)
    sel = S.select_coverage_cover(Ds)
    n = Ds.shape[0]
    if not sel:
        return np.nan, 0, 0.0
    Db = np.zeros((len(sel), n))
    for k, b in enumerate(sel):
        mem = np.array(sorted(b['members']), int)
        Db[k] = Ds[:, mem].min(1)
    lab = np.argmin(Db, 0)
    m = y >= 0
    cov = S.coverage_of(sel, n)
    ari = adjusted_rand_score(y[m], lab[m]) if m.sum() else np.nan
    return (float(ari) if not np.isnan(ari) else np.nan), len(sel), float(cov)


# ---------------------------------------------------------------------------
# run all numeric analysis
# ---------------------------------------------------------------------------

def run_numeric():
    table = {}
    for name, fn, k in DATASETS:
        X, y = fn()
        D = im.dissimilarity(X); Ds = im.minimax_transform(D)
        nd_m, nd_s = nerfcm_ari(D, y, k)
        nds_m, nds_s = nerfcm_ari(Ds, y, k)
        cv_ari, cv_nblk, cv_cov = cover_result(X, y)
        entry = {
            "k_true": k,
            "n": int(len(y)),
            "iVAT_SL_ari": None if np.isnan(ivat_sl_ari(X, y, k)) else round(ivat_sl_ari(X, y, k), 3),
            "NERFCM_D_ari": None if np.isnan(nd_m) else round(nd_m, 3),
            "NERFCM_Dstar_ari": None if np.isnan(nds_m) else round(nds_m, 3),
            "cover_ari": None if np.isnan(cv_ari) else round(cv_ari, 3),
            "cover_nblocks": cv_nblk,
            "cover_coverage": round(cv_cov, 3),
        }
        if y.max() >= 0 and (y >= 0).sum() > 0 and name != "uniform_noise":
            cm, cs = conivat_ari(X, y, k)
            entry["ConiVAT_ari"] = None if np.isnan(cm) else round(cm, 3)
            entry["ConiVAT_std"] = None if np.isnan(cs) else round(cs, 3)
        else:
            entry["ConiVAT_ari"] = None
            entry["ConiVAT_std"] = None
        # c-sensitivity for NERFCM(D*)
        cs_row = {}
        for cc in [max(2, k - 1), k, k + 1]:
            m_, _ = nerfcm_ari(Ds, y, cc)
            cs_row[cc] = None if np.isnan(m_) else round(m_, 3)
        entry["NERFCM_Dstar_csens"] = cs_row
        table[name] = entry
    results["main_table"] = table
    return table


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_datasets():
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    axes = axes.ravel()
    for ax, (name, fn, k) in zip(axes, DATASETS):
        X, y = fn()
        for lab in sorted(set(y)):
            mask = y == lab
            c = 'lightgray' if lab == -1 else None
            ax.scatter(X[mask, 0], X[mask, 1], s=12, c=c,
                       label=('bridge/noise' if lab == -1 else f'cluster {lab}'))
        ax.set_title(f"{name}\n(n={len(y)}, k={k})", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=7, loc='upper right')
    axes[-1].axis('off')  # 6th panel unused (5 datasets)
    fig.suptitle("Figure 1: Synthetic test battery (ground truth)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_datasets.png", dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_transform():
    """Raw D vs minimax D*, VAT-reordered, for two datasets: rings + varying."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for row, (name, fn, k) in enumerate([("concentric_rings", B.concentric_rings, 2),
                                          ("varying_density", B.varying_density, 3)]):
        X, y = fn()
        D = im.dissimilarity(X)
        Ds = im.minimax_transform(D)
        order = im.vat_order(D)
        Dre = D[np.ix_(order, order)]
        Dsre = Ds[np.ix_(order, order)]
        for ax, M, ttl in [(axes[row, 0], Dre, "raw D (VAT-ordered)"),
                           (axes[row, 1], Dsre, "minimax D* (iVAT image)")]:
            im_ = ax.imshow(M, cmap='viridis', aspect='equal')
            ax.set_title(f"{name}\n{ttl}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im_, ax=ax, fraction=0.046)
        axc = axes[row, 2]
        for lab in sorted(set(y)):
            if lab < 0:
                continue
            mask = y == lab
            axc.scatter(X[mask, 0], X[mask, 1], s=8)
        axc.set_title(f"{name}\nscatter", fontsize=10)
        axc.set_xticks([]); axc.set_yticks([])
        axc.set_aspect('equal', adjustable='datalim')
    fig.suptitle("Figure 2: The minimax transform sharpens block structure\n"
                 "(dark diagonal blocks = clusters)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_transform.png", dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_methods_ari(table):
    names = [n for n, _, _ in DATASETS if n != "uniform_noise"]
    methods = [
        ("NERFCM(D)", "NERFCM_D_ari", "#c44"),
        ("NERFCM(D*)", "NERFCM_Dstar_ari", "#4a7"),
        ("ConiVAT", "ConiVAT_ari", "#47c"),
        ("iVAT-cover (no k)", "cover_ari", "#a5a"),
    ]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (label, key, color) in enumerate(methods):
        vals = [table[n][key] if table[n][key] is not None else 0 for n in names]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Figure 3: Partition quality by method (ARI vs ground truth, at true k "
                 "except iVAT-cover which discovers k)\n"
                 "NERFCM given true k; NERFCM(D) fails non-convex rings; "
                 "the minimax transform (D*) rescues it", fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_methods_ari.png", dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_persistence():
    """Sorted-persistence curves with knee detection - the selection story."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    persistence_data = {}
    panels = [d for d in DATASETS if d[0] != "uniform_noise"] + \
             [("uniform_noise", B.uniform_noise, 0)]
    for ax, (name, fn, k) in zip(axes, panels):
        X, y = fn()
        D = im.dissimilarity(X); Ds = im.minimax_transform(D)
        blocks, n = S._all_blocks(Ds)
        p = sorted([b['persistence'] for b in blocks
                    if 3 <= b['size'] <= 0.6 * n], reverse=True)[:8]
        p = np.array(p)
        ax.plot(range(1, len(p) + 1), p, 'o-', color='#333')
        if len(p) > 1:
            ratios = p[:-1] / (p[1:] + 1e-9)
            knee = int(np.argmax(ratios)) + 1
            ax.axvline(knee + 0.5, color='crimson', ls='--',
                       label=f'knee @ {knee}')
            ax.legend(fontsize=9)
        ax.set_title(f"{name}  (true k={k})", fontsize=10)
        ax.set_xlabel("block rank"); ax.set_ylabel("persistence")
        ax.grid(alpha=0.3)
        persistence_data[name] = [round(float(v), 4) for v in p]
    results["persistence_curves"] = persistence_data
    fig.suptitle("Figure 4: Sorted-persistence curves. Clean knee at true k "
                 "(two_gaussians, rings);\nknee MISFIRES on varying_density "
                 "(k=2, true 3) and bridged (k=4) - the multi-scale hard cases",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_persistence.png", dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_membership():
    """Example generated membership functions projected on x-axis, for
    two_gaussians and concentric_rings, from the coverage-cover blocks."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    for ax, (name, fn, k) in zip(axes, [("two_gaussians", B.two_gaussians, 2),
                                        ("concentric_rings", B.concentric_rings, 2)]):
        X, y = fn()
        D = im.dissimilarity(X); Ds = im.minimax_transform(D)
        sel = S.select_coverage_cover(Ds)
        xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
        for bi, b in enumerate(sel[:4]):
            mem = np.array(sorted(b['members']), int)
            # membership of a probe point = ramp over birth-death of minimax dist to block
            # approximate on the x-axis by nearest data point's block distance
            d_to_block = Ds[:, mem].min(1)
            h_b, h_d = b['birth'], b['death']
            mu_pts = np.clip((h_d - d_to_block) / (h_d - h_b + 1e-9), 0, 1)
            # bin by x to get a 1-D profile
            order = np.argsort(X[:, 0])
            ax.plot(X[order, 0], mu_pts[order], '.', ms=3, alpha=0.5,
                    label=f'set {bi} (n={b["size"]})')
        ax.set_title(f"{name}: generated membership vs x-feature", fontsize=11)
        ax.set_xlabel("x feature"); ax.set_ylabel("membership")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Figure 5: Example generated membership functions "
                 "(minimax-derived, projected on one feature)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_membership.png", dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_conivat_bridge():
    """The 0.00 -> 1.00 chaining repair, visualized on the bridge dataset."""
    X, y = B.bridged_gaussians()
    D = im.dissimilarity(X); Ds = im.minimax_transform(D)
    Zi = linkage(squareform(Ds, checks=False), method='single')
    lab_ivat = fcluster(Zi, t=2, criterion='maxclust') - 1
    Dp = conivat(X, y, n_constraints=40, seed=0)
    lab_coni = sl_labels_from_mtd(Dp, 2)
    m = y >= 0
    ari_ivat = adjusted_rand_score(y[m], lab_ivat[m])
    ari_coni = adjusted_rand_score(y[m], lab_coni[m])

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    # ground truth
    for lab in sorted(set(y)):
        mask = y == lab
        c = 'lightgray' if lab == -1 else None
        axes[0].scatter(X[mask, 0], X[mask, 1], s=12, c=c,
                        label=('bridge' if lab == -1 else f'true {lab}'))
    axes[0].set_title("Ground truth\n(two blobs + noise bridge)", fontsize=11)
    axes[0].legend(fontsize=8)
    # iVAT SL
    for lab in sorted(set(lab_ivat)):
        mask = lab_ivat == lab
        axes[1].scatter(X[mask, 0], X[mask, 1], s=12)
    axes[1].set_title(f"plain iVAT single-linkage\nARI = {ari_ivat:.2f}  (chaining fails)",
                      fontsize=11, color='crimson')
    # ConiVAT
    for lab in sorted(set(lab_coni)):
        mask = lab_coni == lab
        axes[2].scatter(X[mask, 0], X[mask, 1], s=12)
    axes[2].set_title(f"ConiVAT (40 constraints)\nARI = {ari_coni:.2f}  (repaired)",
                      fontsize=11, color='green')
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='datalim')
    fig.suptitle("Figure 6: ConiVAT metric learning repairs single-linkage "
                 "chaining across the noise bridge", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_conivat_bridge.png", dpi=130, bbox_inches='tight')
    plt.close(fig)
    results["bridge_repair"] = {"iVAT_SL_ari": round(float(ari_ivat), 3),
                                "ConiVAT_ari": round(float(ari_coni), 3)}


# ---------------------------------------------------------------------------
def main():
    print("Running numeric analysis (deterministic)...")
    table = run_numeric()
    print("Generating figures...")
    fig_datasets()
    fig_transform()
    fig_methods_ari(table)
    fig_persistence()
    fig_membership()
    fig_conivat_bridge()
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results and 6 figures written to", OUT)
    # echo the main table
    print("\nMAIN TABLE:")
    for name, e in table.items():
        print(f"  {name}: iVAT-SL={e['iVAT_SL_ari']} NERFCM(D)={e['NERFCM_D_ari']} "
              f"NERFCM(D*)={e['NERFCM_Dstar_ari']} ConiVAT={e['ConiVAT_ari']} "
              f"cover={e['cover_ari']} (k={e['cover_nblocks']}, cov={e['cover_coverage']})")


if __name__ == "__main__":
    main()
