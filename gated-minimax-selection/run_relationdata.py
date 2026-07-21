"""
Relational data benchmark: NERFCM on distance-matrix-only data with visualizations.

Demonstrates that D* (minimax distance) can improve clustering on data where
only pairwise distances are available (no vector coordinates). Since these are
non-Euclidean-embeddable structures, vector-space methods don't apply; NERFCM
on D* is the natural comparison point.

Generates plots showing distance matrices, NERFCM membership functions, and ARI comparisons.
"""

import numpy as np
import sys
import os

# Import from the actual project root
project_root = "/home/scott/PycharmProjects/grad-school/gated-minimax-selection"
sys.path.insert(0, project_root)

from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import ivat_mf as im
from nerfcm import nerfcm
import relationdata as RD

OUT = "./outputs"
os.makedirs(OUT, exist_ok=True)


DATASETS = [
    ("three_clusters_tree", RD.three_clusters_tree, 3),
    ("chain_then_ring", RD.chain_then_ring, 2),
    ("multi_scale_hierarchy", RD.multi_scale_hierarchy, 3),
]

SEEDS = [0, 1, 2]


def save_figure(fig, filename, dpi=96, fmt="png"):
    """Save figure to outputs directory."""
    full_path = f"{OUT}/{filename}".replace(".png", f".{fmt}").replace(".jpg", f".{fmt}")
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight', format=fmt)
    plt.close(fig)
    print(f"  → {full_path}")
    return full_path


def nerfcm_ari(M, y, c, seeds=SEEDS):
    """Run NERFCM on matrix M and score ARI over multiple seeds."""
    mask = y >= 0
    aris = []
    U_list = []
    for s in seeds:
        U, beta, it = nerfcm(M, c, seed=s)
        lab = np.argmax(U, axis=0)
        if mask.sum() > 0:
            aris.append(adjusted_rand_score(y[mask], lab[mask]))
        U_list.append(U)
    mean_U = np.mean(U_list, axis=0)  # Average membership over seeds
    return (np.mean(aris), np.std(aris)) if aris else (np.nan, np.nan), mean_U


def plot_distance_matrices(D, Dstar, name):
    """Plot raw distance matrix D and minimax distance D* side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

    plt.tight_layout()
    return fig


def plot_memberships(U_d, U_dstar, y, name, c):
    """Plot NERFCM membership functions for D and D*."""
    fig, axes = plt.subplots(c, 2, figsize=(12, 3*c))
    if c == 1:
        axes = axes.reshape(1, -1)

    for k in range(c):
        # Membership for cluster k
        sort_idx = np.argsort(np.arange(U_d.shape[1]))

        # Left: NERFCM(D)
        ax = axes[k, 0]
        ax.bar(range(U_d.shape[1]), U_d[k, sort_idx], alpha=0.7, color='steelblue')
        ax.set_title(f'Cluster {k}: NERFCM(D)', fontsize=10)
        ax.set_ylabel('Membership')
        ax.set_ylim([0, 1])
        ax.set_xticks([])

        # Right: NERFCM(D*)
        ax = axes[k, 1]
        ax.bar(range(U_dstar.shape[1]), U_dstar[k, sort_idx], alpha=0.7, color='coral')
        ax.set_title(f'Cluster {k}: NERFCM(D*)', fontsize=10)
        ax.set_ylabel('Membership')
        ax.set_ylim([0, 1])
        ax.set_xticks([])

    fig.suptitle(f'{name}: NERFCM Membership Functions (c={c})',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_ari_comparison(results):
    """Plot ARI comparison across datasets and methods."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.35

    nerfcm_d = [results[name]['nerfcm_d'][0] for name in names]
    nerfcm_ds = [results[name]['nerfcm_dstar'][0] for name in names]

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
    return fig


def main():
    print("=" * 75)
    print("Relational Data Benchmark: NERFCM on Distance Matrices Only")
    print("=" * 75)
    print()
    print("These datasets have no vector coordinates—only pairwise distances.")
    print("Matrix methods (NERFCM) are the natural/only choice.")
    print()
    print("Generating plots...")
    print()

    results = {}

    for name, fn, c_true in DATASETS:
        print(f"Processing: {name}")
        D, y = fn()
        Dstar = im.minimax_transform(D)

        # Run on raw D
        (m_d, s_d), U_d = nerfcm_ari(D, y, c_true)
        # Run on D*
        (m_ds, s_ds), U_dstar = nerfcm_ari(Dstar, y, c_true)

        results[name] = {
            'nerfcm_d': (m_d, s_d),
            'nerfcm_dstar': (m_ds, s_ds),
            'c': c_true,
            'n': D.shape[0],
        }

        # Plot distance matrices
        print(f"  Plotting distance matrices...")
        fig = plot_distance_matrices(D, Dstar, name)
        save_figure(fig, f"relationdata_distances_{name}.png")

        # Plot membership functions
        print(f"  Plotting membership functions...")
        fig = plot_memberships(U_d, U_dstar, y, name, c_true)
        save_figure(fig, f"relationdata_memberships_{name}.png")

    print()

    # Plot overall ARI comparison
    print(f"Plotting ARI comparison...")
    fig = plot_ari_comparison(results)
    save_figure(fig, "relationdata_ari_comparison.png")

    print()
    print("=" * 75)
    print("Relational Data Results")
    print("=" * 75)
    print()
    print(f"{'dataset':<25}{'n':>5}{'c':>3}{'NERFCM(D)':>20}{'NERFCM(D*)':>20}{'Δ ARI':>12}")
    print("-" * 75)

    gaps = []
    for name, fn, c_true in DATASETS:
        r = results[name]
        m_d, s_d = r['nerfcm_d']
        m_ds, s_ds = r['nerfcm_dstar']
        n = r['n']

        def fmt(mean, std):
            if np.isnan(mean):
                return "n/a"
            return f"{mean:.2f}±{std:.2f}"

        delta = m_ds - m_d if not np.isnan(m_d) and not np.isnan(m_ds) else np.nan
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "n/a"

        print(f"{name:<25}{n:>5}{c_true:>3}{fmt(m_d, s_d):>20}{fmt(m_ds, s_ds):>20}{delta_str:>12}")
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
    print(f"All plots saved to: {OUT}/")
    print()


if __name__ == "__main__":
    main()
