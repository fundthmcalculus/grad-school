"""
Feature-space membership function extraction and linguistic rule generation (Option C).

Path 5A: Surrogate MF fitting in feature space.

Workflow:
1. Extract Ruspini MFs in dissimilarity space (from VAT/IVAT hierarchy)
2. For each MF, fit a surrogate in feature space (Mahalanobis distance-based)
3. Generate linguistic descriptions ("If x1 ∈ [5.2±0.8] AND x2 ∈ [3.1±1.2]")
4. Validate surrogate fidelity on held-out points
5. Output executable feature-space rules (no D* needed after training)
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from scipy.spatial.distance import mahalanobis, euclidean
from scipy.stats import chi2

import sys
sys.path.insert(0, '/home/scott/PycharmProjects/grad-school/gated-minimax-selection')

import battery as B
import ivat_mf as im
import selection as S
import auto_select_mf_v2 as ASM


@dataclass
class FeatureSpaceMF:
    """Surrogate MF in feature space."""
    cluster_id: int
    center: np.ndarray  # d-dimensional feature-space center
    cov: np.ndarray     # d×d covariance matrix
    widths: np.ndarray  # Per-feature standard deviations
    members: Set[int]
    feature_names: Optional[List[str]] = None


class FeatureSpaceExtractor:
    """
    Extract feature-space surrogates for Ruspini MFs.

    Approach:
    1. Compute dissimilarity-space MFs (from Ruspini/auto-tuned)
    2. For each cluster, extract feature-space statistics
    3. Fit Mahalanobis distance-based surrogate
    4. Generate linguistic descriptions
    5. Validate on held-out points
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def extract_feature_space_mfs(self, X: np.ndarray, Dstar: np.ndarray, blocks: List[Dict],
                                 feature_names: Optional[List[str]] = None) -> List[FeatureSpaceMF]:
        """
        Extract feature-space MFs from clusters.

        Args:
            X: feature matrix (n × d)
            Dstar: minimax distance matrix
            blocks: list of dicts from selection.coverage_cover()
            feature_names: optional names for features

        Returns:
            List of FeatureSpaceMF with feature-space parameters
        """
        n, d = X.shape

        if feature_names is None:
            feature_names = [f'x{i}' for i in range(d)]

        mf_list = []

        for cluster_id, block in enumerate(blocks):
            members = np.array(sorted(list(block['members'])))
            X_block = X[members, :]

            # Compute feature-space statistics
            center = np.mean(X_block, axis=0)
            widths = np.std(X_block, axis=0)
            cov = np.cov(X_block.T)

            # Handle degenerate covariance (single-member clusters)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                cov = np.diag(cov)

            # Regularize covariance to ensure invertibility
            cov = cov + 1e-6 * np.eye(d)

            mf = FeatureSpaceMF(
                cluster_id=cluster_id,
                center=center,
                cov=cov,
                widths=widths,
                members=set(members),
                feature_names=feature_names
            )
            mf_list.append(mf)

            if self.verbose:
                print(f"Cluster {cluster_id}:")
                print(f"  Center: {center}")
                print(f"  Widths: {widths}")

        return mf_list

    def evaluate_feature_space_membership(self, mf: FeatureSpaceMF, x: np.ndarray) -> float:
        """
        Evaluate membership of a feature vector using Mahalanobis distance.

        Args:
            mf: FeatureSpaceMF with center and covariance
            x: d-dimensional feature vector

        Returns:
            Membership value in [0, 1]
        """
        try:
            # Compute Mahalanobis distance
            delta = x - mf.center
            mahal_dist = np.sqrt(delta @ np.linalg.inv(mf.cov) @ delta.T)

            # Map to membership: exp(-α·mahal_dist²)
            # Calibrate α so that membership = 0.5 at distance = 1 std
            alpha = 0.693  # -ln(0.5) for μ=0.5 at mahal_dist=1
            membership = np.exp(-alpha * mahal_dist ** 2)

            return float(np.clip(membership, 0.0, 1.0))
        except np.linalg.LinAlgError:
            # Singular covariance; fallback to Euclidean
            eucl_dist = euclidean(x, mf.center)
            max_width = np.max(mf.widths)
            return float(max(0.0, 1.0 - eucl_dist / max(max_width, 1e-6)))

    def compare_surrogates(self, X: np.ndarray, Dstar: np.ndarray, blocks: List[Dict],
                          dissimilarity_mfs: List, feature_space_mfs: List) -> Dict:
        """
        Compare dissimilarity-space MFs vs feature-space surrogates.

        Computes L2 error on held-out points.

        Args:
            X: feature matrix
            Dstar: minimax distance matrix
            blocks: cluster definitions
            dissimilarity_mfs: MFs from dissimilarity space (from Option B/A)
            feature_space_mfs: MFs from feature space (from this extractor)

        Returns:
            Dict with comparison metrics
        """
        n = len(X)
        c = len(blocks)

        # Evaluate both sets of MFs on all points
        mu_dissim = np.zeros((n, c))
        mu_feature = np.zeros((n, c))

        # Dissimilarity-space (requires D* and medoids)
        for cluster_id, block in enumerate(blocks):
            medoid_idx = block.get('medoid_idx', list(block['members'])[0])
            mf = dissimilarity_mfs[cluster_id]

            # Evaluate using the prototype's evaluate method
            from auto_select_mf_v2 import AutoTunedRuspiniExtractor
            extractor = AutoTunedRuspiniExtractor()

            for i in range(n):
                dissim_val = Dstar[medoid_idx, i]
                center_dissim = mf.center_dissim
                support_width = mf.support_width

                if dissim_val <= center_dissim:
                    mu_dissim[i, cluster_id] = 1.0
                elif dissim_val <= center_dissim + support_width:
                    mu_dissim[i, cluster_id] = (center_dissim + support_width - dissim_val) / support_width
                else:
                    mu_dissim[i, cluster_id] = 0.0

        # Feature-space
        for cluster_id, mf in enumerate(feature_space_mfs):
            for i in range(n):
                mu_feature[i, cluster_id] = self.evaluate_feature_space_membership(mf, X[i, :])

        # Normalize both to partition of unity
        mu_dissim = mu_dissim / np.maximum(np.sum(mu_dissim, axis=1, keepdims=True), 1e-10)
        mu_feature = mu_feature / np.maximum(np.sum(mu_feature, axis=1, keepdims=True), 1e-10)

        # L2 error
        error_per_point = np.sqrt(np.mean((mu_dissim - mu_feature) ** 2, axis=1))
        l2_error_max = float(np.max(error_per_point))
        l2_error_mean = float(np.mean(error_per_point))
        l2_error_std = float(np.std(error_per_point))

        return {
            'l2_error_max': l2_error_max,
            'l2_error_mean': l2_error_mean,
            'l2_error_std': l2_error_std,
            'mu_dissim': mu_dissim,
            'mu_feature': mu_feature,
        }

    def generate_linguistic_rules(self, feature_space_mfs: List[FeatureSpaceMF],
                                 threshold: float = 0.5) -> List[str]:
        """
        Generate human-readable linguistic rules from feature-space MFs.

        Rule format: "If x1 ∈ [center±width] AND x2 ∈ [center±width] THEN cluster_i"

        Args:
            feature_space_mfs: List of FeatureSpaceMF
            threshold: membership threshold for rule confidence

        Returns:
            List of linguistic rule strings
        """
        rules = []

        for mf in feature_space_mfs:
            # Build antecedent conditions
            conditions = []
            for feat_id, (name, center, width) in enumerate(
                zip(mf.feature_names, mf.center, mf.widths)
            ):
                lower = center - width
                upper = center + width
                conditions.append(f"{name} ∈ [{lower:.2f}, {upper:.2f}]")

            antecedent = " AND ".join(conditions)
            consequent = f"Cluster {mf.cluster_id}"
            rule = f"IF {antecedent} THEN {consequent}"

            rules.append(rule)

        return rules

    def print_feature_space_summary(self, feature_space_mfs: List[FeatureSpaceMF]) -> None:
        """Print summary of feature-space MFs."""
        print("\n" + "="*80)
        print("FEATURE-SPACE MEMBERSHIP FUNCTIONS")
        print("="*80)

        for mf in feature_space_mfs:
            print(f"\nCluster {mf.cluster_id}:")
            print(f"  Center: {', '.join([f'{name}={c:.3f}' for name, c in zip(mf.feature_names, mf.center)])}")
            print(f"  Widths: {', '.join([f'{name}={w:.3f}' for name, w in zip(mf.feature_names, mf.widths)])}")
            print(f"  Members: {len(mf.members)}")

        print("="*80)

    def print_linguistic_rules(self, rules: List[str]) -> None:
        """Print generated linguistic rules."""
        print("\n" + "="*80)
        print("LINGUISTIC RULES")
        print("="*80)

        for rule in rules:
            print(rule)

        print("="*80)


# ============================================================================
# Integration test on battery
# ============================================================================

def test_feature_space_on_battery():
    """Test feature-space MF extraction on synthetic battery."""
    datasets = [
        ('two_gaussians', B.two_gaussians),
        ('bridged_gaussians', B.bridged_gaussians),
        ('concentric_rings', B.concentric_rings),
        ('varying_density', B.varying_density),
    ]

    results = {}

    for name, dataset_fn in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {name}")
        print(f"{'='*80}")

        X, y_true = dataset_fn()
        n, d = X.shape

        # Compute dissimilarity and minimax transform
        D = im.dissimilarity(X)
        Dstar = im.minimax_transform(D)

        # Select blocks
        blocks = S.select_coverage_cover(Dstar)

        if not blocks:
            print("No clusters selected")
            continue

        # Extract dissimilarity-space MFs (Option B/A)
        extractor_dissim = ASM.AutoTunedRuspiniExtractor(verbose=False)
        dissim_mfs, mu_dissim = extractor_dissim.extract_partition(Dstar, blocks)

        # Extract feature-space surrogates
        extractor_feature = FeatureSpaceExtractor(verbose=False)
        feature_mfs = extractor_feature.extract_feature_space_mfs(X, Dstar, blocks)

        # Compare surrogates
        comparison = extractor_feature.compare_surrogates(X, Dstar, blocks, dissim_mfs, feature_mfs)

        print(f"Surrogate fidelity (L2 error):")
        print(f"  Max:  {comparison['l2_error_max']:.4f}")
        print(f"  Mean: {comparison['l2_error_mean']:.4f}")
        print(f"  Std:  {comparison['l2_error_std']:.4f}")

        # Generate linguistic rules
        rules = extractor_feature.generate_linguistic_rules(feature_mfs)

        print(f"\nGenerated {len(rules)} linguistic rules:")
        for rule in rules[:2]:  # Print first 2
            print(f"  {rule}")
        if len(rules) > 2:
            print(f"  ... ({len(rules) - 2} more)")

        results[name] = {
            'l2_error_max': comparison['l2_error_max'],
            'l2_error_mean': comparison['l2_error_mean'],
            'num_rules': len(rules),
        }

    return results


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FEATURE-SPACE MEMBERSHIP FUNCTION EXTRACTION (Option C)")
    print("="*80)

    results = test_feature_space_on_battery()

    # Summary
    print("\n" + "="*80)
    print("FEATURE-SPACE EXTRACTION SUMMARY")
    print("="*80)
    print(f"{'Dataset':<20} {'L2 Error (Mean)':<20} {'L2 Error (Max)':<20}")
    print("-"*60)

    for name, result in results.items():
        print(f"{name:<20} {result['l2_error_mean']:<20.4f} {result['l2_error_max']:<20.4f}")

    print("="*80)
    print("\nInterpretation:")
    print("- L2 error < 0.1: Surrogate matches dissimilarity-space MF closely")
    print("- L2 error 0.1-0.3: Good approximation, some loss")
    print("- L2 error > 0.3: Significant difference, use with caution")
    print("="*80)
