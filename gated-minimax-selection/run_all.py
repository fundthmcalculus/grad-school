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
  - fig7_relationdata_distances_*.png : relational data distance matrices (D vs D*)
  - fig7_relationdata_memberships_*.png : relational data membership functions
  - fig7_relationdata_ari.png : relational data ARI comparison
  - fig8_multiscale_hierarchy.png : Option D multi-scale persistence selection
                                    (flat cover collapses a hierarchy; band-wise
                                    selection recovers every level)
  - fig9_selection_comparison.png : persistence-gap vs beta-plateau vs
                                    bottleneck-bootstrap (k, coverage, ARI)
  - fig10_persistence_thresholds.png : persistence curves with each method's knee

This is the SINGLE entry point for the whole investigation: it consolidates what
were previously separate run_*.py scripts. results.json holds every number:
  main_table (two mappings, NERFCM D/D*, ConiVAT, coverage-cover, k-means,
  convexity, c-sensitivity), selector_comparison (topc/relpersist/coverage),
  persistence_methods (gap/beta-plateau/bottleneck-bootstrap), arity_detection
  (D*/geometric), ruspini (Option B), feature_space (Options A auto-tuned + C),
  relational_table, and the multiscale_* blocks (Option D).

All randomness is seeded. Re-running reproduces identical numbers and figures.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import ivat_mf as im
import battery as B
import selection as S
from nerfcm import nerfcm
from conivat import conivat, sl_labels_from_mtd
import relationdata as RD
import multiscale_persistence as MS
import battery_hierarchical as BH
import selection_comparison as SC
import disjunct as dj
import ruspini_mf as RM
import auto_select_mf_v2 as ASM2
import feature_space_mf as FSM

OUT = "./outputs"
SEEDS = [0, 1, 2, 3, 4]

# Figure output configuration
OUTPUT_CONFIG = {
    "dpi": 96,           # Default: low-res for easy sharing (set to 130+ for high-res)
    "fmt": "png",        # Format: "png" or "svg"
}


def save_figure(fig, filename, dpi=None, fmt=None):
    """Save figure with configurable resolution and format."""
    dpi = dpi or OUTPUT_CONFIG["dpi"]
    fmt = fmt or OUTPUT_CONFIG["fmt"]
    full_path = f"{OUT}/{filename}".replace(".png", f".{fmt}").replace(".jpg", f".{fmt}")
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight', format=fmt)
    plt.close(fig)
    return full_path

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
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
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
    save_figure(fig, "fig1_datasets.png")


def fig_transform():
    """Raw D vs minimax D*, VAT-reordered, for two datasets: rings + varying."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
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
    save_figure(fig, "fig2_transform.png")


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
    fig, ax = plt.subplots(figsize=(9, 5.5))
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
    save_figure(fig, "fig3_methods_ari.png")


def fig_persistence():
    """Sorted-persistence curves with knee detection - the selection story."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
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
    save_figure(fig, "fig4_persistence.png")


def fig_membership():
    """Example generated membership functions projected on x-axis, for
    two_gaussians and concentric_rings, from the coverage-cover blocks."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for ax, (name, fn, k) in zip(axes, [("two_gaussians", B.two_gaussians, 2),
                                        ("concentric_rings", B.concentric_rings, 2)]):
        ax = axes[row, 0]
        X, y = fn()
        D = im.dissimilarity(X); Ds = im.minimax_transform(D)
        sel = S.select_coverage_cover(Ds)
        for bi, b in enumerate(sel[:4]):
            mem = np.array(sorted(b['members']), int)
            # membership of a probe point = ramp over birth-death of minimax dist to block
            # approximate on the x-axis by nearest data point's block distance
            d_to_block = Ds[:, mem].min(1)
            h_b, h_d = b['birth'], b['death']
            mu_pts = np.clip((h_d - d_to_block) / (h_d - h_b + 1e-9), 0, 1)
            # bin by x to get a 1-D profile
            order = np.argsort(X[:, 0])
            x_sorted = X[order, 0]
            mu_sorted = mu_pts[order]
            ax.fill_between(x_sorted, 0, mu_sorted, alpha=0.4, color=colors[bi],
                            label=f'set {bi} (n={b["size"]})')
            ax.plot(x_sorted, mu_sorted, color=colors[bi], linewidth=1.5, alpha=0.8)
        ax.set_title(f"{name}: generated membership vs x-feature", fontsize=12, fontweight='bold')
        ax.set_xlabel("x feature"); ax.set_ylabel("membership")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)

    # Top-right: empty
    axes[0, 1].axis('off')

    # Bottom-right: 2D heatmap for concentric_rings (first block only)
    ax_hm = axes[1, 1]
    X_cr, y_cr = B.concentric_rings()
    D_cr = im.dissimilarity(X_cr); Ds_cr = im.minimax_transform(D_cr)
    sel_cr = S.select_coverage_cover(Ds_cr)

    from scipy.spatial.distance import cdist
    b = sel_cr[0]
    mem_idx = np.array(sorted(b['members']), int)

    # Create a 2D grid for the heatmap
    x_min, x_max = X_cr[:, 0].min() - 0.5, X_cr[:, 0].max() + 0.5
    y_min, y_max = X_cr[:, 1].min() - 0.5, X_cr[:, 1].max() + 0.5
    grid_res = 100
    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([XX.ravel(), YY.ravel()])

    # Compute membership for the first block on the grid
    dist_to_members = cdist(grid_points, X_cr[mem_idx], metric='euclidean').min(1)
    h_b, h_d = b['birth'], b['death']
    mu_grid = np.clip((h_d - dist_to_members) / (h_d - h_b + 1e-9), 0, 1)
    mu_grid = mu_grid.reshape(XX.shape)

    # Plot the heatmap
    im_hm = ax_hm.contourf(XX, YY, mu_grid, levels=20, cmap='RdYlGn')
    ax_hm.scatter(X_cr[mem_idx, 0], X_cr[mem_idx, 1], s=20, color='black',
                  alpha=0.5, edgecolors='white', linewidth=0.5)
    ax_hm.set_title(f"concentric_rings: membership heatmap (set 0)", fontsize=11)
    ax_hm.set_xlabel("x feature"); ax_hm.set_ylabel("y feature")
    cbar = fig.colorbar(im_hm, ax=ax_hm)
    cbar.set_label("membership", fontsize=9)

    fig.suptitle("Figure 5: Example generated membership functions "
                 "(minimax-derived, projected on one feature and 2D)", fontsize=12, y=1.00)
    fig.tight_layout()
    save_figure(fig, "fig5_membership.png")


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

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
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
    save_figure(fig, "fig6_conivat_bridge.png")
    results["bridge_repair"] = {"iVAT_SL_ari": round(float(ari_ivat), 3),
                                "ConiVAT_ari": round(float(ari_coni), 3)}


# ---------------------------------------------------------------------------
# relational data analysis and figures
# ---------------------------------------------------------------------------

def run_relational_numeric():
    """Analyze relational datasets with NERFCM(D) vs NERFCM(D*)."""
    table = {}
    for name, fn, c_true in RELATIONAL_DATASETS:
        D, y = fn()
        Dstar = im.minimax_transform(D)

        # Run NERFCM on both D and D*
        m_d, s_d = nerfcm_ari(D, y, c_true)
        m_ds, s_ds = nerfcm_ari(Dstar, y, c_true)

        entry = {
            "k_true": c_true,
            "n": int(D.shape[0]),
            "NERFCM_D_ari": None if np.isnan(m_d) else round(m_d, 3),
            "NERFCM_D_std": None if np.isnan(s_d) else round(s_d, 3),
            "NERFCM_Dstar_ari": None if np.isnan(m_ds) else round(m_ds, 3),
            "NERFCM_Dstar_std": None if np.isnan(s_ds) else round(s_ds, 3),
        }
        table[name] = entry
    results["relational_table"] = table
    return table


def fig_relational_distances():
    """Plot D vs D* distance matrices for each relational dataset."""
    for name, fn, c_true in RELATIONAL_DATASETS:
        D, y = fn()
        Dstar = im.minimax_transform(D)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

        # Plot D
        im1 = ax1.imshow(D, cmap='viridis', aspect='auto')
        ax1.set_title(f'{name}\nRaw Distance D', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Point j')
        ax1.set_ylabel('Point i')
        plt.colorbar(im1, ax=ax1, label='Distance')

        # Plot D*
        im2 = ax2.imshow(Dstar, cmap='viridis', aspect='auto')
        ax2.set_title(f'{name}\nMinimax Distance D*', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Point j')
        ax2.set_ylabel('Point i')
        plt.colorbar(im2, ax=ax2, label='Distance')

        fig.suptitle(f"Relational Data: {name}\nD vs D* (minimax bottleneck)",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, f"fig7_relationdata_distances_{name}.png")


def fig_relational_memberships():
    """Plot NERFCM membership functions for D and D*."""
    NERFCM_SEEDS = [0, 1, 2]
    for name, fn, c_true in RELATIONAL_DATASETS:
        D, y = fn()
        Dstar = im.minimax_transform(D)

        # Get average membership over seeds
        U_d_list = []
        U_ds_list = []
        for s in NERFCM_SEEDS:
            U_d, _, _ = nerfcm(D, c_true, seed=s)
            U_ds, _, _ = nerfcm(Dstar, c_true, seed=s)
            U_d_list.append(U_d)
            U_ds_list.append(U_ds)
        U_d = np.mean(U_d_list, axis=0)
        U_ds = np.mean(U_ds_list, axis=0)

        fig, axes = plt.subplots(c_true, 2, figsize=(9.5, 2.4*c_true))
        if c_true == 1:
            axes = axes.reshape(1, -1)

        for k in range(c_true):
            # Left: NERFCM(D)
            ax = axes[k, 0]
            ax.bar(range(U_d.shape[1]), U_d[k, :], alpha=0.7, color='steelblue')
            ax.set_title(f'Cluster {k}: NERFCM(D)', fontsize=10)
            ax.set_ylabel('Membership')
            ax.set_ylim([0, 1])
            ax.set_xticks([])

            # Right: NERFCM(D*)
            ax = axes[k, 1]
            ax.bar(range(U_ds.shape[1]), U_ds[k, :], alpha=0.7, color='coral')
            ax.set_title(f'Cluster {k}: NERFCM(D*)', fontsize=10)
            ax.set_ylabel('Membership')
            ax.set_ylim([0, 1])
            ax.set_xticks([])

        fig.suptitle(f'{name}: NERFCM Membership Functions (c={c_true})',
                     fontsize=12, fontweight='bold', y=1.00)
        plt.tight_layout()
        save_figure(fig, f"fig7_relationdata_memberships_{name}.png")


def fig_relational_ari(relational_table):
    """Plot ARI comparison across relational datasets."""
    fig, ax = plt.subplots(figsize=(8.5, 4))

    names = list(relational_table.keys())
    x = np.arange(len(names))
    width = 0.35

    nerfcm_d = [relational_table[name]['NERFCM_D_ari'] for name in names]
    nerfcm_ds = [relational_table[name]['NERFCM_Dstar_ari'] for name in names]

    bars1 = ax.bar(x - width/2, nerfcm_d, width, label='NERFCM(D)', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, nerfcm_ds, width, label='NERFCM(D*)', alpha=0.8, color='coral')

    ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=11)
    ax.set_title('Relational Data: ARI Comparison\nD vs D* (Minimax Distance)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_figure(fig, "fig7_relationdata_ari.png")


# ---------------------------------------------------------------------------
# Option D: multi-scale persistence selection (density-stratified block cover)
# ---------------------------------------------------------------------------

def run_multiscale_numeric():
    """Headline: on nested data, flat coverage_cover recovers ONE level while
    multi-scale recovers EVERY level. Populates results["multiscale_hierarchy"]."""
    table = {}
    for name, (gen, level_names) in BH.HIERARCHICAL.items():
        out = gen()
        X, levels = out[0], list(out[1:])
        Ds = im.minimax_transform(im.dissimilarity(X))

        # flat baseline: one hard assignment scored against every level
        flat = S.select_coverage_cover(Ds)
        flat_a = MS.assign(flat, Ds)
        flat_ari = [round(float(adjusted_rand_score(y, flat_a)), 3) for y in levels]

        # multi-scale: one band per scale, each scored against every level
        msel = MS.select_multiscale(Ds)
        band_a = [MS.assign_band(b, Ds) for b in msel.bands]
        best = [round(float(max([adjusted_rand_score(y, a) for a in band_a] or [np.nan])), 3)
                for y in levels]
        band_rows = []
        for b, a in zip(msel.bands, band_a):
            band_rows.append({
                "band_id": b.band_id, "k": b.k,
                "birth_lo": round(float(b.birth_lo), 3),
                "birth_hi": (None if b.birth_hi == float("inf") else round(float(b.birth_hi), 3)),
                "coverage": round(float(b.coverage_fraction(msel.n)), 3),
                "ari_per_level": [round(float(adjusted_rand_score(y, a)), 3) for y in levels],
            })
        table[name] = {
            "level_names": level_names,
            "n": int(len(levels[0])),
            "flat_k": len(flat),
            "flat_ari_per_level": flat_ari,
            "flat_mean_ari": round(float(np.mean(flat_ari)), 3),
            "ms_n_scales": msel.n_scales,
            "ms_granularities": msel.granularities(),
            "ms_best_ari_per_level": best,
            "ms_mean_ari": round(float(np.mean(best)), 3),
            "bands": band_rows,
        }
    results["multiscale_hierarchy"] = table
    return table


def run_multiscale_no_regression():
    """Strict-generalization check: on the single-scale battery, multi-scale
    discovers ONE band (or zero on noise) reproducing the flat baseline.
    Populates results["multiscale_no_regression"]."""
    table = {}
    for name, fn, k in DATASETS:
        X, y = fn()
        Ds = im.minimax_transform(im.dissimilarity(X))
        m = y >= 0
        flat = S.select_coverage_cover(Ds)
        flat_a = MS.assign(flat, Ds)
        flat_ari = round(float(adjusted_rand_score(y[m], flat_a[m])), 3) if m.sum() else None
        msel = MS.select_multiscale(Ds)
        if msel.n_scales:
            fb = msel.finest()
            fb_a = MS.assign_band(fb, Ds)
            fb_ari = round(float(adjusted_rand_score(y[m], fb_a[m])), 3) if m.sum() else None
            fk = fb.k
        else:
            fb_ari, fk = None, 0
        table[name] = {
            "flat_k": len(flat), "flat_ari": flat_ari,
            "ms_n_bands": msel.n_scales, "ms_finest_k": fk, "ms_finest_ari": fb_ari,
        }
    results["multiscale_no_regression"] = table
    return table


def run_multiscale_scale_invariance():
    """Falsification experiment: flat coverage_cover is ALREADY scale-invariant on
    single-level data (so we make no flat-ARI claim there). Separation is scaled
    with spread so clusters stay separable; only the scale gap varies.
    Populates results["multiscale_scale_invariance"]."""
    def make(contrast, n=180, seed=104, sep=6.0):
        rng = np.random.default_rng(seed)
        base = 0.25
        sig = [base, base * contrast, base * contrast ** 2]
        xs = [0.0]
        for j in range(1, 3):
            xs.append(xs[-1] + sep * (sig[j - 1] + sig[j]) / 2)
        parts, ys = [], []
        for j, (x, s) in enumerate(zip(xs, sig)):
            parts.append(rng.normal([x, 0], s, (n // 3, 2)))
            ys += [j] * (n // 3)
        return np.vstack(parts), np.array(ys)

    rows = {}
    for contrast in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        X, y = make(contrast)
        Ds = im.minimax_transform(im.dissimilarity(X))
        flat = S.select_coverage_cover(Ds)
        a = MS.assign(flat, Ds)
        rows[f"contrast_{contrast}"] = {
            "spread_ratio": f"1:{contrast:g}:{contrast**2:g}",
            "flat_k": len(flat),
            "flat_ari": round(float(adjusted_rand_score(y, a)), 3),
        }
    results["multiscale_scale_invariance"] = rows
    return rows


def fig_multiscale_hierarchy():
    """Visualize Option D on nested_gaussians: data, persistence diagram colored
    by discovered scale band, and the band x level ARI matrix."""
    X, y_fine, y_coarse = BH.nested_gaussians()
    Ds = im.minimax_transform(im.dissimilarity(X))
    blocks, n = S._all_blocks(Ds)
    sig = MS.significant_blocks(blocks, n)
    msel = MS.select_multiscale(Ds)
    edges = msel.band_edges_log

    def band_of(birth):
        lb = np.log(birth + 1e-12)
        return sum(1 for e in edges if lb >= e)

    fig, ax = plt.subplots(3, 1, figsize=(6, 12.5))

    # (a) data colored by fine truth
    ax[0].scatter(X[:, 0], X[:, 1], c=y_fine, cmap='tab10', s=18)
    ax[0].set_title('nested_gaussians\n(6 fine / 2 coarse clusters)')
    ax[0].set_aspect('equal', 'datalim')
    ax[0].set_xticks([]); ax[0].set_yticks([])

    # (b) persistence diagram colored by discovered band
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(1, len(edges) + 1)))
    for b in blocks:
        if 3 <= b['size'] <= 0.6 * n:
            ax[1].scatter(b['birth'], b['persistence'], s=12, color='0.8', zorder=1)
    for b in sig:
        bi = band_of(b['birth'])
        ax[1].scatter(b['birth'], b['persistence'], s=55,
                      color=colors[min(bi, len(colors) - 1)],
                      edgecolor='k', linewidth=0.4, zorder=3)
    for e in edges:
        ax[1].axvline(np.exp(e), ls='--', color='crimson', lw=1)
    ax[1].set_xscale('log')
    ax[1].set_xlabel('birth height (log)')
    ax[1].set_ylabel('persistence (death - birth)')
    ax[1].set_title('persistence diagram\n(significant blocks by scale band;\n'
                    'dashed = discovered band edges)')

    # (c) band x level ARI matrix
    levels = [y_fine, y_coarse]
    level_names = ['fine (6)', 'coarse (2)']
    M = np.array([[adjusted_rand_score(y, MS.assign_band(b, Ds)) for y in levels]
                  for b in msel.bands])
    im_ = ax[2].imshow(M, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
    ax[2].set_xticks(range(len(level_names))); ax[2].set_xticklabels(level_names)
    ax[2].set_yticks(range(len(msel.bands)))
    ax[2].set_yticklabels([f'band {b.band_id}\n(k={b.k})' for b in msel.bands])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax[2].text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                       color='k' if M[i, j] < 0.6 else 'w')
    ax[2].set_title('ARI: discovered band vs ground-truth level\n'
                    '(diagonal ~1 = each level recovered by its own band)')
    fig.colorbar(im_, ax=ax[2], fraction=0.046)

    fig.suptitle('Figure 8: Option D - multi-scale persistence selection recovers '
                 'a nested hierarchy\nthat flat set-cover collapses to a single level',
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, "fig8_multiscale_hierarchy.png")


# ---------------------------------------------------------------------------
# consolidated analyses (ported from the former standalone run_*.py scripts)
# ---------------------------------------------------------------------------

def _medoids_from_labels(Dstar, y, c):
    """Pick a medoid per ground-truth cluster to seed Mapping 1 fairly.
    (from the former run_battery.py)"""
    meds = []
    for lab in sorted(set(y[y >= 0]))[:c]:
        idx = np.where(y == lab)[0]
        sub = Dstar[np.ix_(idx, idx)]
        meds.append(idx[np.argmin(sub.sum(axis=1))])
    while len(meds) < c:
        meds.append(int(np.argmax(Dstar.sum(axis=1))))
    return meds


def run_mappings_numeric():
    """Two candidate MF mappings (Mapping 1 medoid, Mapping 2 persistence) vs the
    k-means vector-space anchor, with coverage, 1-D convexity, and Mapping-2
    c-sensitivity. Folds these into results["main_table"] (was run_battery.py)."""
    table = results.get("main_table", {})
    for name, fn, c in DATASETS:
        X, y = fn()
        Ds = im.minimax_transform(im.dissimilarity(X))
        meds = _medoids_from_labels(Ds, y, c)
        ari1 = B.score_ari(im.mapping1_medoid(Ds, meds), y)
        U2, _info, Dblock = im.mapping2_persistence(Ds, c)
        ari2 = B.score_ari(U2, y, Dblock)
        cov = B.coverage(U2)
        conv = B.univariate_convexity(U2, X, axis=0)
        km = B.kmeans_ari(X, y, c)
        csens = B.c_sensitivity(Ds, y, c)
        entry = table.setdefault(name, {})
        entry["mapping1_ari"] = None if np.isnan(ari1) else round(float(ari1), 3)
        entry["mapping2_ari"] = None if np.isnan(ari2) else round(float(ari2), 3)
        entry["mapping2_coverage"] = round(float(cov), 3)
        entry["mapping2_convexity"] = round(float(conv), 3)
        entry["kmeans_ari"] = None if np.isnan(km) else round(float(km), 3)
        entry["mapping2_csens"] = {int(k): (None if np.isnan(v) else round(float(v), 3))
                                   for k, v in csens.items()}
    results["main_table"] = table
    return table


def run_selector_comparison_numeric():
    """Block-selector bake-off under the t-conorm coverage model: topc_disjoint,
    relpersist, coverage_cover by #blocks, coverage, purity (was run_selection.py)."""
    table = {}
    for name, fn, c in DATASETS:
        X, y = fn()
        Ds = im.minimax_transform(im.dissimilarity(X))
        n = Ds.shape[0]
        selectors = {
            "topc_disjoint": S.select_topc_disjoint(Ds, c),
            "relpersist": S.select_relpersist(Ds, c),
            "coverage_cover": S.select_coverage_cover(Ds),
        }
        table[name] = {}
        for sname, sel in selectors.items():
            pur = S.purity_vs_truth(sel, y)
            table[name][sname] = {
                "n_blocks": len(sel),
                "coverage": round(float(S.coverage_of(sel, n)), 3),
                "purity": None if np.isnan(pur) else round(float(pur), 3),
                "sizes": sorted((b["size"] for b in sel), reverse=True),
            }
    results["selector_comparison"] = table
    return table


def run_persistence_methods_numeric():
    """Persistence-gap vs beta-plateau vs bottleneck-bootstrap on the battery
    (was run_selection_comparison.py). Returns the raw dict used by figs 9-10."""
    METHODS = ["persistence_gap", "beta_plateau", "bottleneck_bootstrap"]
    table = {}
    for name, fn, k_true in DATASETS:
        X, y = fn()
        n = len(X)
        res = SC.compare_all_methods(X, y_true=y, verbose=False)
        entry = {"k_true": k_true, "n": n, "methods": {}}
        for method in METHODS:
            blocks = res[method]["blocks"]
            if blocks:
                lab = np.zeros(n, dtype=int)
                for bi, b in enumerate(blocks):
                    for idx in b["members"]:
                        lab[idx] = bi
                m = y >= 0
                ari = adjusted_rand_score(y[m], lab[m]) if m.sum() else np.nan
            else:
                ari = np.nan
            entry["methods"][method] = {
                "k_discovered": res[method]["n_clusters"],
                "coverage": round(float(res[method]["coverage"]), 3),
                "ari": None if np.isnan(ari) else round(float(ari), 3),
            }
        table[name] = entry
    results["persistence_methods"] = table
    return table


def _gt_block(Dstar, y, label):
    """Ground-truth block with birth/death from within- and cross-cluster minimax
    distances -- isolates the ARITY question from selection (from run_arity.py)."""
    members = set(np.where(y == label)[0].tolist())
    mem = np.array(sorted(members), dtype=int)
    within = Dstar[np.ix_(mem, mem)]
    birth = within.max()
    others = np.array([i for i in range(Dstar.shape[0]) if i not in members])
    if len(others) > 0:
        death = max(Dstar[np.ix_(mem, others)].min(), birth * 1.01)
    else:
        death = birth * 1.5
    return {"members": members, "birth": birth, "death": death, "size": len(members)}


def run_arity_numeric():
    """Disjunct-arity of ground-truth blocks in D* vs geometric modes
    (was run_arity.py). arity should be 1 for every real cluster."""
    table = {}
    for name, fn, c in DATASETS:
        X, y = fn()
        Ds = im.minimax_transform(im.dissimilarity(X))
        labels = sorted(set(y[y >= 0].tolist()))
        if not labels:
            table[name] = None
            continue
        blocks = [_gt_block(Ds, y, lab) for lab in labels]
        table[name] = {}
        for mode in ("dstar", "geometric"):
            table[name][mode] = [
                {"size": blk["size"], "arity": int(dj.block_arity(Ds, X, blk, mode=mode)[0])}
                for blk in blocks
            ]
    results["arity_detection"] = table
    return table


def run_ruspini_numeric():
    """Option B: Ruspini partition-of-unity extraction on coverage-cover blocks
    (was run_ruspini.py / run_ruspini_integrated.py)."""
    table = {}
    for name, fn, k in DATASETS:
        X, y = fn()
        Ds = im.minimax_transform(im.dissimilarity(X))
        blocks = S.select_coverage_cover(Ds)
        if not blocks:
            table[name] = {"k_found": 0, "ari": None, "coverage": None,
                           "partition_error_max": None, "partition_error_mean": None,
                           "status": "noise_rejection"}
            continue
        ext = RM.RuspiniPartitionExtractor(verbose=False)
        block_sets = [set(b["members"]) for b in blocks]
        medoids = [b.get("medoid_idx", list(b["members"])[0]) for b in blocks]
        birth_death = {i: (b.get("birth", 0.0), b.get("death", np.inf))
                       for i, b in enumerate(blocks)}
        _mf, mu = ext.extract_partition(Ds, block_sets, medoids, birth_death, normalize=True)
        max_err, mean_err, _std = ext.partition_of_unity_error(mu)
        assign = ext.defuzzify_hardmax(mu)
        m = y >= 0
        ari = adjusted_rand_score(y[m], assign[m]) if m.sum() else np.nan
        table[name] = {
            "k_found": len(blocks),
            "ari": None if np.isnan(ari) else round(float(ari), 3),
            "coverage": round(float(ext.coverage(mu, threshold=0.5)), 3),
            "partition_error_max": round(float(max_err), 6),
            "partition_error_mean": round(float(mean_err), 6),
        }
    results["ruspini"] = table
    return table


def run_feature_space_numeric():
    """Options A + C: auto-tuned Ruspini (dissimilarity-space) vs feature-space
    surrogate rules -- surrogate L2 fidelity, ARI gap, #linguistic rules
    (was run_feature_space.py; also represents run_auto_select.py's Option A,
    whose standalone script imported a since-removed module)."""
    table = {}
    for name, fn, k in DATASETS:
        X, y = fn()
        Ds = im.minimax_transform(im.dissimilarity(X))
        blocks = S.select_coverage_cover(Ds)
        if not blocks:
            table[name] = {"num_clusters": 0, "status": "noise_rejection"}
            continue
        ext_d = ASM2.AutoTunedRuspiniExtractor(verbose=False)
        dissim_mfs, mu_d = ext_d.extract_partition(Ds, blocks)
        ari_d = adjusted_rand_score(y, ext_d.defuzzify(dissim_mfs, mu_d))
        ext_f = FSM.FeatureSpaceExtractor(verbose=False)
        feat_mfs = ext_f.extract_feature_space_mfs(X, Ds, blocks)
        comp = ext_f.compare_surrogates(X, Ds, blocks, dissim_mfs, feat_mfs)
        ari_f = adjusted_rand_score(y, np.argmax(comp["mu_feature"], axis=1))
        rules = ext_f.generate_linguistic_rules(feat_mfs)
        table[name] = {
            "num_clusters": len(blocks),
            "surrogate_l2_mean": round(float(comp["l2_error_mean"]), 4),
            "surrogate_l2_max": round(float(comp["l2_error_max"]), 4),
            "ari_dissimilarity": round(float(ari_d), 4),
            "ari_feature_space": round(float(ari_f), 4),
            "ari_gap": round(float(abs(ari_d - ari_f)), 4),
            "n_rules": len(rules),
        }
    results["feature_space"] = table
    return table


def fig_selection_comparison(pm_table):
    """Persistence-gap vs beta-plateau vs bottleneck-bootstrap: discovered k,
    coverage, ARI (was run_selection_comparison.fig_method_comparison)."""
    methods = ["persistence_gap", "beta_plateau", "bottleneck_bootstrap"]
    datasets = [n for n, _, _ in DATASETS if n != "uniform_noise"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    x = np.arange(len(datasets)); w = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    # k discovered
    for i, method in enumerate(methods):
        vals = [pm_table[ds]["methods"][method]["k_discovered"] for ds in datasets]
        axes[0].bar(x + (i - 1) * w, vals, w, label=method, color=colors[i], alpha=0.85)
    axes[0].plot(x, [pm_table[ds]["k_true"] for ds in datasets], "ko--", lw=2, ms=7, label="k_true")
    axes[0].set_title("Discovered k vs truth"); axes[0].set_ylabel("k")
    # coverage
    for i, method in enumerate(methods):
        vals = [pm_table[ds]["methods"][method]["coverage"] for ds in datasets]
        axes[1].bar(x + (i - 1) * w, vals, w, color=colors[i], alpha=0.85)
    axes[1].set_title("Coverage"); axes[1].set_ylim(0, 1.1)
    # ari
    for i, method in enumerate(methods):
        vals = [pm_table[ds]["methods"][method]["ari"] or 0 for ds in datasets]
        axes[2].bar(x + (i - 1) * w, vals, w, color=colors[i], alpha=0.85)
    axes[2].set_title("ARI vs truth"); axes[2].set_ylim(0, 1.1)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=15, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure 9: Selection-method comparison "
                 "(persistence-gap vs beta-plateau vs bottleneck-bootstrap)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "fig9_selection_comparison.png")


def fig_persistence_thresholds(pm_table):
    """Persistence curves with each method's selected knee marked
    (was run_selection_comparison.fig_persistence_curves_all)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    cmap = {"persistence_gap": "#e74c3c", "beta_plateau": "#3498db",
            "bottleneck_bootstrap": "#2ecc71"}
    for ax, (name, fn, k_true) in zip(axes, DATASETS):
        X, y = fn()
        Ds = im.minimax_transform(im.dissimilarity(X))
        blocks, n = S._all_blocks(Ds)
        persist = np.array([b["persistence"] for b in blocks if 3 <= b["size"] <= 0.6 * n])
        if len(persist) < 2:
            ax.text(0.5, 0.5, f"{name}\n(too few blocks)", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([]); continue
        ps = np.sort(persist)[::-1][:10]
        ax.plot(range(1, len(ps) + 1), ps, "o-", color="#333", lw=2, ms=5)
        for method, color in cmap.items():
            k_sel = pm_table[name]["methods"][method]["k_discovered"]
            if 0 < k_sel <= len(ps):
                ax.axvline(k_sel + 0.5, color=color, ls="--", alpha=0.6, lw=1.5, label=method)
        ax.set_title(f"{name} (k_true={k_true})", fontsize=9)
        ax.set_xlabel("block rank"); ax.set_ylabel("persistence")
        ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Figure 10: Persistence curves with each method's selected knee", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, "fig10_persistence_thresholds.png")


# ---------------------------------------------------------------------------
def main(high_res=False, svg=False):
    """Generate analysis, numeric results, and figures.

    Args:
        high_res (bool): If True, generate figures at 300 dpi for reports (default: 96 dpi for sharing)
        svg (bool): If True, save as SVG format instead of PNG (for archival)
    """
    import os
    os.makedirs(OUT, exist_ok=True)

    if high_res:
        OUTPUT_CONFIG["dpi"] = 300
        print("Generating high-resolution figures (300 dpi)...")
    if svg:
        OUTPUT_CONFIG["fmt"] = "svg"
        print("Generating figures in SVG format...")

    print("Running numeric analysis (deterministic)...")
    table = run_numeric()
    run_mappings_numeric()
    run_selector_comparison_numeric()
    print("Running persistence selection-method comparison...")
    persistence_methods = run_persistence_methods_numeric()
    run_arity_numeric()
    print("Running membership-extraction variants (Ruspini / feature-space)...")
    run_ruspini_numeric()
    run_feature_space_numeric()
    relational_table = run_relational_numeric()
    print("Running multi-scale (Option D) analysis...")
    multiscale_table = run_multiscale_numeric()
    run_multiscale_no_regression()
    run_multiscale_scale_invariance()
    print("Generating figures...")
    fig_datasets()
    fig_transform()
    fig_methods_ari(table)
    fig_persistence()
    fig_membership()
    fig_conivat_bridge()
    print("Generating relational data figures...")
    fig_relational_distances()
    fig_relational_memberships()
    fig_relational_ari(relational_table)
    print("Generating multi-scale (Option D) figure...")
    fig_multiscale_hierarchy()
    print("Generating selection-method comparison figures...")
    fig_selection_comparison(persistence_methods)
    fig_persistence_thresholds(persistence_methods)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results and figures written to", OUT)
    # echo the main table
    print("\nMAIN TABLE (Euclidean/Synthetic):")
    for name, e in table.items():
        print(f"  {name}: iVAT-SL={e['iVAT_SL_ari']} NERFCM(D)={e['NERFCM_D_ari']} "
              f"NERFCM(D*)={e['NERFCM_Dstar_ari']} ConiVAT={e['ConiVAT_ari']} "
              f"cover={e['cover_ari']} (k={e['cover_nblocks']}, cov={e['cover_coverage']})")
    print("\nRELATIONAL DATA TABLE (Distance-Matrix-Only):")
    for name, e in relational_table.items():
        d_ari = e['NERFCM_D_ari'] if e['NERFCM_D_ari'] is not None else "n/a"
        ds_ari = e['NERFCM_Dstar_ari'] if e['NERFCM_Dstar_ari'] is not None else "n/a"
        delta = (e['NERFCM_Dstar_ari'] - e['NERFCM_D_ari']) if (e['NERFCM_D_ari'] is not None and e['NERFCM_Dstar_ari'] is not None) else None
        delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
        print(f"  {name}: NERFCM(D)={d_ari} NERFCM(D*)={ds_ari} ΔAI={delta_str} (k={e['k_true']}, n={e['n']})")

    print("\nMULTI-SCALE HIERARCHY TABLE (Option D): mean ARI over ALL "
          "ground-truth levels")
    for name, e in multiscale_table.items():
        print(f"  {name}: flat(k={e['flat_k']})={e['flat_mean_ari']} "
              f"multi-scale(scales={e['ms_n_scales']}, k={e['ms_granularities']})="
              f"{e['ms_mean_ari']}  | flat/level={e['flat_ari_per_level']} "
              f"ms/level={e['ms_best_ari_per_level']}")

    print("\nMAPPINGS + BASELINES (folded into main_table):")
    for name, e in table.items():
        print(f"  {name}: M1={e.get('mapping1_ari')} M2={e.get('mapping2_ari')} "
              f"kmeans={e.get('kmeans_ari')} M2cov={e.get('mapping2_coverage')} "
              f"M2convex={e.get('mapping2_convexity')}")

    print("\nPERSISTENCE SELECTION METHODS (k / coverage / ARI):")
    for name, e in persistence_methods.items():
        parts = []
        for meth, m in e["methods"].items():
            parts.append(f"{meth}: k={m['k_discovered']} cov={m['coverage']} ARI={m['ari']}")
        print(f"  {name} (k_true={e['k_true']}): " + " | ".join(parts))

    print("\nMEMBERSHIP VARIANTS: Ruspini (Option B) partition-of-unity + "
          "feature-space (Options A/C):")
    for name in results["ruspini"]:
        r = results["ruspini"][name]
        f = results["feature_space"].get(name, {})
        if r.get("status") == "noise_rejection":
            print(f"  {name}: (noise rejected)")
            continue
        print(f"  {name}: RuspiniARI={r['ari']} POU_err(max)={r['partition_error_max']} "
              f"| autoTunedARI={f.get('ari_dissimilarity')} featARI={f.get('ari_feature_space')} "
              f"L2={f.get('surrogate_l2_mean')} rules={f.get('n_rules')}")


if __name__ == "__main__":
    import sys
    use_high_res = '--high-res' in sys.argv or '-hr' in sys.argv
    use_svg = '--svg' in sys.argv or '-s' in sys.argv
    main(high_res=use_high_res, svg=use_svg)
