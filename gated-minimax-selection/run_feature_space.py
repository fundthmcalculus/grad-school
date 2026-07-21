"""
Comprehensive test of feature-space rule extraction and application.

Demonstrates:
1. Feature-space MF extraction from dissimilarity-space MFs
2. Surrogate fidelity evaluation
3. Linguistic rule generation
4. Application of rules to classify new points
5. Comparison to direct dissimilarity-space classification
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

import sys
sys.path.insert(0, '/home/scott/PycharmProjects/grad-school/gated-minimax-selection')

import battery as B
import ivat_mf as im
import selection as S
import auto_select_mf_v2 as ASM
import feature_space_mf as FSM


def run_comprehensive_feature_space_test():
    """Test feature-space extraction on battery with full workflow."""
    datasets = [
        ('two_gaussians', B.two_gaussians),
        ('bridged_gaussians', B.bridged_gaussians),
        ('concentric_rings', B.concentric_rings),
        ('varying_density', B.varying_density),
        ('uniform_noise', B.uniform_noise),
    ]

    results = {}

    for name, dataset_fn in datasets:
        print(f"\n{'='*90}")
        print(f"Dataset: {name}")
        print(f"{'='*90}")

        X, y_true = dataset_fn()
        n, d = X.shape

        # Step 1: Compute dissimilarity and minimax transform
        D = im.dissimilarity(X)
        Dstar = im.minimax_transform(D)

        # Step 2: Select blocks
        blocks = S.select_coverage_cover(Dstar)

        if not blocks:
            print("No clusters selected (noise rejection)")
            results[name] = {
                'num_clusters': 0,
                'surrogate_fidelity': np.nan,
                'ari_dissimilarity': np.nan,
                'ari_feature_space': np.nan,
                'rules': [],
            }
            continue

        # Step 3: Extract dissimilarity-space MFs (Option B/A)
        print(f"  Extracting dissimilarity-space MFs...")
        extractor_dissim = ASM.AutoTunedRuspiniExtractor(verbose=False)
        dissim_mfs, mu_dissim = extractor_dissim.extract_partition(Dstar, blocks)

        # Classify using dissimilarity-space MFs
        assignments_dissim = extractor_dissim.defuzzify(dissim_mfs, mu_dissim)
        ari_dissim = adjusted_rand_score(y_true, assignments_dissim)

        # Step 4: Extract feature-space surrogates
        print(f"  Extracting feature-space MFs...")
        extractor_feature = FSM.FeatureSpaceExtractor(verbose=False)
        feature_mfs = extractor_feature.extract_feature_space_mfs(X, Dstar, blocks)

        # Step 5: Compare surrogates
        print(f"  Evaluating surrogate fidelity...")
        comparison = extractor_feature.compare_surrogates(X, Dstar, blocks, dissim_mfs, feature_mfs)

        # Classify using feature-space MFs
        mu_feature = comparison['mu_feature']
        assignments_feature = np.argmax(mu_feature, axis=1)
        ari_feature = adjusted_rand_score(y_true, assignments_feature)

        # Step 6: Generate linguistic rules
        rules = extractor_feature.generate_linguistic_rules(feature_mfs)

        # Results
        print(f"  Clusters: {len(blocks)}")
        print(f"\n  Surrogate fidelity (L2 error):")
        print(f"    Mean: {comparison['l2_error_mean']:.4f}")
        print(f"    Max:  {comparison['l2_error_max']:.4f}")

        print(f"\n  Classification performance:")
        print(f"    Dissimilarity-space ARI: {ari_dissim:.4f}")
        print(f"    Feature-space ARI:       {ari_feature:.4f}")
        print(f"    Gap:                     {abs(ari_dissim - ari_feature):.4f}")

        print(f"\n  Generated {len(rules)} linguistic rules:")
        for rule in rules:
            print(f"    {rule}")

        results[name] = {
            'num_clusters': len(blocks),
            'surrogate_fidelity_mean': float(comparison['l2_error_mean']),
            'surrogate_fidelity_max': float(comparison['l2_error_max']),
            'ari_dissimilarity': float(ari_dissim),
            'ari_feature_space': float(ari_feature),
            'ari_gap': float(abs(ari_dissim - ari_feature)),
            'rules': rules,
        }

    return results


def print_summary(results):
    """Print comprehensive summary table."""
    print("\n" + "="*120)
    print("FEATURE-SPACE EXTRACTION SUMMARY")
    print("="*120)
    print(f"{'Dataset':<20} {'L2 Error':<15} {'Dissim ARI':<15} {'Feature ARI':<15} {'ARI Gap':<15}")
    print("-"*120)

    for name, result in results.items():
        if result['num_clusters'] == 0:
            print(f"{name:<20} {'(noise)':<15} {'(noise)':<15} {'(noise)':<15} {'-':<15}")
        else:
            l2_err = result.get('surrogate_fidelity_mean', np.nan)
            ari_d = result.get('ari_dissimilarity', np.nan)
            ari_f = result.get('ari_feature_space', np.nan)
            gap = result.get('ari_gap', np.nan)

            print(f"{name:<20} {l2_err:<15.4f} {ari_d:<15.4f} {ari_f:<15.4f} {gap:<15.4f}")

    print("="*120)

    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 120)

    clean_results = {k: v for k, v in results.items() if v['num_clusters'] > 0}

    if clean_results:
        mean_l2 = np.mean([v['surrogate_fidelity_mean'] for v in clean_results.values()])
        mean_ari_gap = np.mean([v['ari_gap'] for v in clean_results.values()])
        max_ari_gap = np.max([v['ari_gap'] for v in clean_results.values()])

        print(f"1. Surrogate fidelity:")
        print(f"   - Average L2 error: {mean_l2:.4f} (lower is better)")
        if mean_l2 < 0.1:
            print(f"   - ✓ EXCELLENT: Surrogates closely match dissimilarity-space MFs")
        elif mean_l2 < 0.3:
            print(f"   - ✓ GOOD: Surrogates are reasonable approximations")
        else:
            print(f"   - ⚠ MODERATE: Some loss in fidelity, use for interpretability only")

        print(f"\n2. Classification performance:")
        print(f"   - Average ARI gap: {mean_ari_gap:.4f}")
        print(f"   - Max ARI gap:     {max_ari_gap:.4f}")
        if max_ari_gap < 0.05:
            print(f"   - ✓ EXCELLENT: Feature-space rules match dissimilarity-space performance")
        elif max_ari_gap < 0.15:
            print(f"   - ✓ GOOD: Feature-space rules are competitive")
        else:
            print(f"   - ⚠ FAIR: Some performance loss, but rules are interpretable")

        print(f"\n3. Linguistic rules:")
        print(f"   - Total rules generated: {sum(len(v['rules']) for v in clean_results.values())}")
        print(f"   - Rules are human-readable feature ranges")
        print(f"   - Can be integrated into fuzzy inference systems")
        print(f"   - No D* needed after training (portable to new data)")

    print("="*120)


if __name__ == '__main__':
    print("\n" + "="*120)
    print("FEATURE-SPACE MEMBERSHIP FUNCTION EXTRACTION & LINGUISTIC RULE GENERATION")
    print("="*120)

    results = run_comprehensive_feature_space_test()
    print_summary(results)

    print("\nUSAGE:")
    print("-" * 120)
    print("Generated linguistic rules can be directly integrated into fuzzy inference systems:")
    print("")
    print("Example rule from two_gaussians:")
    if results['two_gaussians']['rules']:
        print(f"  {results['two_gaussians']['rules'][0]}")
    print("")
    print("Advantages:")
    print("  1. Human-readable and interpretable")
    print("  2. No dependency on D* after training (portable)")
    print("  3. Can be manually tuned by domain experts")
    print("  4. Suitable for real-time inference (simple distance calc)")
    print("="*120)
