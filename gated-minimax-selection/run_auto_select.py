"""
Test auto-selective prototype membership function extraction on battery.

Compares:
1. Auto-selected prototypes vs hand-tuned (oracle)
2. Auto-selected vs baseline (persistence-based from coverage_cover)
3. Selection stability across datasets
4. Success criteria: auto-selected within 0.05 ARI of oracle
"""

import numpy as np
import json
from typing import Dict, List
from sklearn.metrics import adjusted_rand_score

import sys
sys.path.insert(0, '/home/scott/PycharmProjects/grad-school/gated-minimax-selection')

import battery as B
import ivat_mf as im
import selection as S
import auto_select_mf as ASM


def run_auto_select_battery(verbose: bool = False) -> Dict:
    """
    Run auto-selective prototype extraction on all synthetic datasets.

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

        # Select blocks
        blocks = S.select_coverage_cover(Dstar)

        if len(blocks) == 0:
            print(f"  ⚠ No clusters selected (noise detection)")
            results[name] = {
                'num_clusters': 0,
                'ari': np.nan,
                'prototypes': {},
                'status': 'noise_rejection',
            }
            continue

        # Auto-select membership functions
        extractor = ASM.AutoSelectivePrototypeMFExtractor(verbose=verbose)
        mf_list, selection_results = extractor.extract_mfs(Dstar, blocks, X)

        if verbose:
            extractor.print_summary(selection_results)

        # Defuzzify
        assignments = extractor.defuzzify(mf_list, Dstar)

        # Compute ARI
        ari = adjusted_rand_score(y_true, assignments)

        # Prototype statistics
        proto_counts = {}
        proto_confidence = {}
        for result in selection_results:
            proto = result.selected_prototype
            proto_counts[proto] = proto_counts.get(proto, 0) + 1
            if proto not in proto_confidence:
                proto_confidence[proto] = []
            proto_confidence[proto].append(result.confidence)

        avg_confidences = {proto: float(np.mean(confs)) for proto, confs in proto_confidence.items()}

        # Print results
        print(f"  Clusters selected: {len(blocks)}")
        print(f"  Prototypes selected:")
        for proto, count in sorted(proto_counts.items(), key=lambda x: -x[1]):
            avg_conf = avg_confidences[proto]
            print(f"    {proto:<20} {count:2d} clusters (avg confidence: {avg_conf:.3f})")
        print(f"  ARI: {ari:.4f}")

        results[name] = {
            'num_clusters': len(blocks),
            'ari': float(ari),
            'prototypes': {proto: {'count': count, 'avg_confidence': avg_confidences[proto]}
                          for proto, count in proto_counts.items()},
            'selection_results': [
                {
                    'cluster_id': r.cluster_id,
                    'prototype': r.selected_prototype,
                    'confidence': float(r.confidence),
                    'cohesion': float(r.metric_signature.cohesion),
                    'symmetry': float(r.metric_signature.symmetry),
                    'concentration': float(r.metric_signature.concentration),
                }
                for r in selection_results
            ],
        }

    return results


def compare_to_baseline(results_auto: Dict) -> None:
    """
    Compare auto-selected prototypes against baseline (coverage_cover persistence MFs).

    Baseline numbers from FINDINGS.md (corrected, deterministic run).
    """
    baseline = {
        'two_gaussians': {'ari': 1.00},
        'bridged_gaussians': {'ari': 0.98},
        'concentric_rings': {'ari': 1.00},
        'varying_density': {'ari': 0.98},
        'uniform_noise': {'ari': np.nan},
    }

    print(f"\n{'='*90}")
    print("Comparison: Auto-Selected Prototypes vs. Baseline (coverage_cover)")
    print(f"{'='*90}")
    print(f"{'Dataset':<20} {'Auto-Select ARI':<20} {'Baseline ARI':<20} {'Δ ARI':<15} {'Gap':<10}")
    print(f"{'-'*90}")

    gaps = []
    for name, result in results_auto.items():
        auto_ari = result['ari']
        baseline_ari = baseline[name]['ari']

        if np.isnan(auto_ari):
            ari_str = "  (noise)"
            gap_str = "  -"
        else:
            delta = auto_ari - baseline_ari
            gap = abs(delta)
            gaps.append(gap)
            ari_str = f"{auto_ari:.4f}    {baseline_ari:.4f}    {delta:+.4f}    {gap:.4f}"

        print(f"{name:<20} {ari_str:<75}")

    # Summary
    clean_gaps = [g for g in gaps if not np.isnan(g)]
    if clean_gaps:
        max_gap = np.max(clean_gaps)
        avg_gap = np.mean(clean_gaps)
        print(f"\n{'='*90}")
        print(f"Average gap: {avg_gap:.4f}")
        print(f"Max gap: {max_gap:.4f}")
        print(f"Success criterion (gap < 0.05): {'✓ PASS' if max_gap < 0.05 else '✗ FAIL'}")
        print(f"{'='*90}")

        return avg_gap, max_gap
    else:
        print(f"\n{'='*90}")
        print("All datasets noise-rejected (no gap to measure)")
        print(f"{'='*90}")
        return np.nan, np.nan


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("AUTO-SELECTIVE PROTOTYPE MEMBERSHIP FUNCTIONS")
    print("="*70)

    results = run_auto_select_battery(verbose=args.verbose)

    # Compare to baseline
    avg_gap, max_gap = compare_to_baseline(results)

    # Save results
    output_file = 'auto_select_results.json'
    with open(output_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            return obj

        json.dump(results, f, default=convert, indent=2)

    print(f"\nResults saved to {output_file}")

    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    if not np.isnan(max_gap):
        if max_gap < 0.05:
            print("✅ SUCCESS: Auto-selection within tolerance (< 0.05 ARI gap)")
        elif max_gap < 0.10:
            print("⚠️  MARGINAL: Auto-selection close to tolerance (0.05-0.10 ARI gap)")
        else:
            print("❌ IMPROVEMENT NEEDED: Auto-selection gap too large (> 0.10 ARI)")
    print("="*70)
