"""
Test Ruspini partitioning extraction on the synthetic battery.

Compares against baseline (persistence-based) membership functions.
Metrics: ARI, coverage, partition-of-unity error, and defuzzification quality.
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from sklearn.metrics import adjusted_rand_score

# Import existing modules
import sys
sys.path.insert(0, '/home/scott/PycharmProjects/grad-school/gated-minimax-selection')

import battery as B
import ivat_mf as im
import selection as S
import ruspini_mf as RM


def run_ruspini_battery() -> Dict:
    """
    Run Ruspini extraction on all synthetic datasets.

    Returns:
        dict with results for all datasets
    """
    datasets = [
        ('two_gaussians', B.two_gaussians),
        ('bridged_gaussians', B.bridged_gaussians),
        ('concentric_rings', B.concentric_rings),
        ('varying_density', B.varying_density),
        ('uniform_noise', B.uniform_noise),
    ]

    results = {}

    for name, dataset_fn in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"{'='*70}")

        X, y_true = dataset_fn()
        n = len(X)

        # Compute dissimilarity and minimax transform
        D = im.dissimilarity(X)
        Dstar = im.minimax_transform(D)

        # Select blocks using coverage-cover (same as baseline)
        blocks = S.select_coverage_cover(Dstar)

        if len(blocks) == 0:
            print(f"  ⚠ No clusters selected (noise detection)")
            results[name] = {
                'num_clusters': 0,
                'ari': np.nan,
                'coverage': np.nan,
                'partition_error_max': np.nan,
                'partition_error_mean': np.nan,
                'status': 'noise_rejection',
            }
            continue

        # Extract Ruspini partition
        extractor = RM.RuspiniPartitionExtractor(verbose=False)

        # Build birth/death dict from blocks
        birth_death = {
            i: (b.get('birth', 0.0), b.get('death', np.inf))
            for i, b in enumerate(blocks)
        }

        block_sets = [set(b['members']) for b in blocks]
        medoids = [b.get('medoid_idx', list(b['members'])[0]) for b in blocks]

        # Extract partition (with normalization)
        mf_list, mu = extractor.extract_partition(Dstar, block_sets, medoids, birth_death, normalize=True)

        # Metrics
        max_err, mean_err, std_err = extractor.partition_of_unity_error(mu)
        cov = extractor.coverage(mu, threshold=0.5)
        assignments = extractor.defuzzify_hardmax(mu)

        # Compute ARI (against ground truth)
        ari = adjusted_rand_score(y_true, assignments)

        # Print results
        print(f"  Clusters selected: {len(blocks)}")
        print(f"  Partition-of-unity error:")
        print(f"    max:  {max_err:.6f}")
        print(f"    mean: {mean_err:.6f}")
        print(f"    std:  {std_err:.6f}")
        print(f"  Coverage (μ > 0.5): {cov:.2%}")
        print(f"  ARI (vs ground truth): {ari:.4f}")

        results[name] = {
            'num_clusters': len(blocks),
            'ari': float(ari),
            'coverage': float(cov),
            'partition_error_max': float(max_err),
            'partition_error_mean': float(mean_err),
            'partition_error_std': float(std_err),
            'mf_list': [
                {
                    'cluster_id': mf.cluster_id,
                    'medoid_idx': mf.medoid_idx,
                    'birth': float(mf.birth_height),
                    'death': float(mf.death_height),
                    'center_dissim': float(mf.center_dissim),
                    'support_width': float(mf.support_width),
                }
                for mf in mf_list
            ],
        }

    return results


def compare_to_baseline(results_ruspini: Dict) -> None:
    """
    Print a comparison table: Ruspini vs. baseline (coverage_cover).

    The baseline numbers are from FINDINGS.md (corrected numbers).
    """
    baseline = {
        'two_gaussians': {'ari': 1.00, 'coverage': 1.00},
        'bridged_gaussians': {'ari': 0.98, 'coverage': 0.53},
        'concentric_rings': {'ari': 1.00, 'coverage': 1.00},
        'varying_density': {'ari': 0.98, 'coverage': 1.00},
        'uniform_noise': {'ari': np.nan, 'coverage': 0.125},
    }

    print(f"\n{'='*90}")
    print("Comparison: Ruspini vs. Baseline (coverage_cover)")
    print(f"{'='*90}")
    print(f"{'Dataset':<20} {'Ruspini ARI':<15} {'Baseline ARI':<15} {'Δ ARI':<10}")
    print(f"{'-'*90}")

    for name, result in results_ruspini.items():
        ruspini_ari = result['ari']
        baseline_ari = baseline[name]['ari']

        if np.isnan(ruspini_ari):
            ari_str = "  (noise)"
        else:
            delta = ruspini_ari - baseline_ari
            ari_str = f"{ruspini_ari:.4f}    {baseline_ari:.4f}    {delta:+.4f}"

        print(f"{name:<20} {ari_str}")

    print(f"\nPartition-of-Unity Error Summary:")
    print(f"{'Dataset':<20} {'Max Error':<15} {'Mean Error':<15} {'Std Error':<15}")
    print(f"{'-'*65}")

    for name, result in results_ruspini.items():
        if not np.isnan(result.get('partition_error_max', np.nan)):
            print(f"{name:<20} {result['partition_error_max']:.6f}    "
                  f"{result['partition_error_mean']:.6f}    "
                  f"{result['partition_error_std']:.6f}")

    print(f"\n{'='*90}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("RUSPINI PARTITIONING BATTERY TEST")
    print("="*70)

    results = run_ruspini_battery()

    # Compare to baseline
    compare_to_baseline(results)

    # Save results
    output_file = 'ruspini_results.json'
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if not np.isnan(obj) and not np.isinf(obj) else None
            return obj

        json.dump(results, f, default=convert, indent=2)

    print(f"\nResults saved to {output_file}")
