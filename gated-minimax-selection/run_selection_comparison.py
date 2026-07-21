"""
Runner for comprehensive selection method comparison.

Compares:
  1. Persistence-gap / knee selection (current)
  2. Beta-plateau (Bonis & Oudot)
  3. Bottleneck-bootstrap (AuToMATo)

On the full synthetic battery + produces comparison table and figures.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

import ivat_mf as im
import battery as B
import selection_comparison as SC

OUT = "./outputs"

DATASETS = [
    ("two_gaussians",     B.two_gaussians,     2),
    ("bridged_gaussians", B.bridged_gaussians, 2),
    ("concentric_rings",  B.concentric_rings,  2),
    ("varying_density",   B.varying_density,   3),
    ("uniform_noise",     B.uniform_noise,     2),
]

OUTPUT_CONFIG = {
    "dpi": 96,
    "fmt": "png",
}


def save_figure(fig, filename):
    """Save figure with configurable resolution and format."""
    full_path = f"{OUT}/{filename}".replace(".png", f".{OUTPUT_CONFIG['fmt']}")
    fig.savefig(full_path, dpi=OUTPUT_CONFIG["dpi"], bbox_inches='tight', format=OUTPUT_CONFIG['fmt'])
    plt.close(fig)
    return full_path


def compute_ari_for_assignment(blocks, y, n):
    """Compute ARI given blocks and ground truth."""
    if not blocks:
        return np.nan
    lab = np.zeros(n, dtype=int)
    for bi, b in enumerate(blocks):
        for idx in b['members']:
            lab[idx] = bi
    m = y >= 0
    if m.sum() == 0:
        return np.nan
    return adjusted_rand_score(y[m], lab[m])


def run_comparison():
    """Run all three methods on all datasets."""
    results = {}

    for name, fn, k_true in DATASETS:
        X, y = fn()
        n = len(X)

        print(f"\n{name} (n={n}, k_true={k_true}):")

        # Run all three methods
        res = SC.compare_all_methods(X, y_true=y, verbose=True)

        # Compute ARI for each method
        aris = {}
        for method in ['persistence_gap', 'beta_plateau', 'bottleneck_bootstrap']:
            ari = compute_ari_for_assignment(res[method]['blocks'], y, n)
            aris[method] = float(ari) if not np.isnan(ari) else None

        results[name] = {
            'k_true': k_true,
            'n': n,
            'methods': {}
        }

        for method in ['persistence_gap', 'beta_plateau', 'bottleneck_bootstrap']:
            results[name]['methods'][method] = {
                'k_discovered': res[method]['n_clusters'],
                'coverage': round(res[method]['coverage'], 3),
                'ari': aris[method],
                'metadata': res[method]['metadata'],
            }

    return results


def make_comparison_table(results):
    """Print and return comparison table."""
    print("\n" + "="*120)
    print("SELECTION METHOD COMPARISON TABLE")
    print("="*120)

    header_fmt = "{:<20} {:<8} {:<10} {:<30} {:<30} {:<30}"
    print(header_fmt.format("Dataset", "k_true", "n", "Persistence-Gap", "Beta-Plateau", "Bottleneck-Bootstrap"))
    print("-"*120)

    table_data = []
    for name in [n for n, _, _ in DATASETS]:
        entry = results[name]
        row_data = {
            'name': name,
            'k_true': entry['k_true'],
            'n': entry['n'],
        }

        for method in ['persistence_gap', 'beta_plateau', 'bottleneck_bootstrap']:
            m_res = entry['methods'][method]
            k = m_res['k_discovered']
            cov = m_res['coverage']
            ari = m_res['ari']
            ari_str = f"{ari:.3f}" if ari is not None else "n/a"
            row_data[method] = {
                'k': k,
                'coverage': cov,
                'ari': ari_str,
            }

        table_data.append(row_data)

        fmt_str = f"{name:<20} {entry['k_true']:<8} {entry['n']:<10} "
        for method in ['persistence_gap', 'beta_plateau', 'bottleneck_bootstrap']:
            m = row_data[method]
            fmt_str += f"k={m['k']} cov={m['coverage']:.2f} ARI={m['ari']:<6} "
        print(fmt_str)

    print("="*120)
    return table_data


def fig_method_comparison(results):
    """Compare methods on the same datasets."""
    methods = ['persistence_gap', 'beta_plateau', 'bottleneck_bootstrap']
    datasets = [n for n, _, _ in DATASETS if n != 'uniform_noise']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(datasets))
    w = 0.25
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    # Plot 1: k_discovered vs k_true
    ax = axes[0]
    for i, method in enumerate(methods):
        vals = [results[ds]['methods'][method]['k_discovered'] for ds in datasets]
        ax.bar(x + (i - 1) * w, vals, w, label=method, color=colors[i], alpha=0.8)

    k_true = [results[ds]['k_true'] for ds in datasets]
    ax.plot(x, k_true, 'ko--', linewidth=2, markersize=8, label='k_true')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylabel('Cluster Count (k)')
    ax.set_title('Discovered k vs Ground Truth')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Coverage
    ax = axes[1]
    for i, method in enumerate(methods):
        vals = [results[ds]['methods'][method]['coverage'] for ds in datasets]
        ax.bar(x + (i - 1) * w, vals, w, label=method, color=colors[i], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylabel('Coverage')
    ax.set_ylim(0, 1.1)
    ax.set_title('Point Coverage by Method')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: ARI
    ax = axes[2]
    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            ari = results[ds]['methods'][method]['ari']
            vals.append(ari if ari is not None else 0)
        ax.bar(x + (i - 1) * w, vals, w, label=method, color=colors[i], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.set_ylabel('Adjusted Rand Index (ARI)')
    ax.set_ylim(0, 1.1)
    ax.set_title('Partition Quality (vs Ground Truth)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Selection Method Comparison on Synthetic Battery\n'
                 '(Persistence-Gap vs Beta-Plateau vs Bottleneck-Bootstrap)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, 'fig_selection_comparison.png')


def fig_persistence_curves_all(results):
    """Show persistence curves for all datasets with method-specific thresholds."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    for ax, (name, fn, k_true) in zip(axes, DATASETS):
        X, y = fn()
        D = im.dissimilarity(X)
        Ds = im.minimax_transform(D)
        blocks, n = SC._all_blocks(Ds)

        ceiling = 0.6 * n
        persist = np.array([b['persistence'] for b in blocks
                            if 3 <= b['size'] <= ceiling])
        if len(persist) < 2:
            ax.text(0.5, 0.5, f"{name}\n(too few blocks)", ha='center', va='center')
            continue

        persist_sorted = np.sort(persist)[::-1][:10]
        x_range = np.arange(1, len(persist_sorted) + 1)

        # Plot persistence curve
        ax.plot(x_range, persist_sorted, 'o-', color='#333', linewidth=2, markersize=6)

        # Get thresholds from each method
        res = SC.compare_all_methods(X, y_true=y, verbose=False)

        # Try to mark the selected thresholds
        colors_methods = {'persistence_gap': '#e74c3c', 'beta_plateau': '#3498db', 'bottleneck_bootstrap': '#2ecc71'}
        for method, color in colors_methods.items():
            k_sel = res[method]['n_clusters']
            if k_sel > 0 and k_sel <= len(persist_sorted):
                ax.axvline(k_sel + 0.5, color=color, linestyle='--', alpha=0.6, linewidth=1.5, label=method)

        ax.set_title(f"{name}\n(k_true={k_true}, n={len(y)})", fontsize=10)
        ax.set_xlabel("Block Rank")
        ax.set_ylabel("Persistence")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].axis('off')  # 6th panel unused
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=9)

    fig.suptitle('Persistence Curves with Method-Specific Selections\n'
                 '(vertical lines show where each method selects the knee)',
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_figure(fig, 'fig_persistence_thresholds.png')


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    print("Running comprehensive selection method comparison...")
    print("=" * 120)

    results = run_comparison()

    table = make_comparison_table(results)

    print("\nGenerating comparison figures...")
    fig_method_comparison(results)
    fig_persistence_curves_all(results)

    # Save results
    with open(f"{OUT}/selection_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUT}/")
    print(f"  - selection_comparison_results.json")
    print(f"  - fig_selection_comparison.png")
    print(f"  - fig_persistence_thresholds.png")


if __name__ == "__main__":
    main()
