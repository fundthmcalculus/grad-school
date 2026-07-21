"""
Auto-selective Ruspini MF extraction (Option A, Revised).

Simpler, more practical approach: Ruspini partition-of-unity (from Option B)
with automatic tuning of support width based on cluster geometry.

Instead of trying to fit 5 different prototype families, we use ONE robust
approach (Ruspini linear ramp) but automatically tune the support boundary
width based on cluster properties:
- Tight clusters → narrow support (sharp membership)
- Diffuse clusters → wide support (gradual membership)
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/scott/PycharmProjects/grad-school/gated-minimax-selection')

import battery as B
import ivat_mf as im
import selection as S


@dataclass
class AutoTunedRuspiniMF:
    """Ruspini MF with auto-tuned support width."""
    cluster_id: int
    medoid_idx: int
    members: Set[int]
    center_dissim: float
    support_width: float
    tuning_factor: float  # Applied to h_death to get support boundary


class AutoTunedRuspiniExtractor:
    """
    Extract Ruspini MFs with automatic support width tuning.

    Approach:
    1. Extract tight core (at h_birth)
    2. Measure cluster spread (from core to h_death)
    3. Auto-tune support width factor based on spread
    4. Normalize to partition of unity
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def extract_partition(self, Dstar: np.ndarray, blocks: List[Dict]) -> Tuple[List[AutoTunedRuspiniMF], np.ndarray]:
        """
        Extract auto-tuned Ruspini partition.

        Args:
            Dstar: minimax distance matrix
            blocks: list of dicts from selection.coverage_cover()

        Returns:
            (mf_list, mu_normalized) where:
              - mf_list: list of AutoTunedRuspiniMF with tuning info
              - mu_normalized: (n × c) normalized membership matrix
        """
        n = Dstar.shape[0]
        c = len(blocks)

        mu = np.zeros((n, c))
        mf_list = []

        for cluster_id, block in enumerate(blocks):
            members = set(block['members'])
            medoid_idx = block.get('medoid_idx', list(members)[0])
            h_b = block.get('birth', 0.0)
            h_d = block.get('death', np.inf)

            dissim_ramp = Dstar[medoid_idx, :]

            # Extract core
            core_dissims = dissim_ramp[list(members)]
            core_dissims = core_dissims[core_dissims <= h_b]

            if len(core_dissims) == 0:
                center_dissim = np.min(dissim_ramp[list(members)])
            else:
                center_dissim = np.mean(core_dissims)

            # Auto-tune support width
            spread = h_d - center_dissim
            tuning_factor = self._compute_tuning_factor(dissim_ramp, members, center_dissim, h_d)
            support_width = tuning_factor * spread

            if self.verbose:
                print(f"Cluster {cluster_id}:")
                print(f"  Spread: {spread:.4f}, Tuning factor: {tuning_factor:.3f}, "
                      f"Support width: {support_width:.4f}")

            # Build membership ramp
            for i in range(n):
                d = dissim_ramp[i]
                if d <= center_dissim:
                    mu[i, cluster_id] = 1.0
                elif d <= center_dissim + support_width:
                    mu[i, cluster_id] = (center_dissim + support_width - d) / support_width
                else:
                    mu[i, cluster_id] = 0.0

            mf = AutoTunedRuspiniMF(
                cluster_id=cluster_id,
                medoid_idx=medoid_idx,
                members=members,
                center_dissim=float(center_dissim),
                support_width=float(support_width),
                tuning_factor=float(tuning_factor)
            )
            mf_list.append(mf)

        # Normalize uncovered points
        for i in range(n):
            if np.sum(mu[i, :]) == 0:
                nearest = np.argmin([Dstar[mf.medoid_idx, i] for mf in mf_list])
                mu[i, nearest] = 0.01

        # Partition of unity normalization
        mu = self._normalize_partition_of_unity(mu)

        return mf_list, mu

    def _compute_tuning_factor(self, dissim_ramp: np.ndarray, members: Set[int],
                               center_dissim: float, h_d: float) -> float:
        """
        Compute automatic support width tuning factor.

        Tuning logic:
        - Tight cluster (low spread): factor = 0.8 (sharp boundary)
        - Loose cluster (high spread): factor = 1.2 (gradual boundary)

        Args:
            dissim_ramp: dissimilarity ramp from medoid
            members: cluster member indices
            center_dissim: center dissimilarity
            h_d: death height

        Returns:
            tuning factor in [0.5, 1.5]
        """
        member_dissims = dissim_ramp[list(members)]

        # Compute spread (max - min within cluster)
        min_d = np.min(member_dissims)
        max_d = np.max(member_dissims)
        cluster_spread = max_d - min_d

        # Compute variance in dissimilarity
        dissim_std = np.std(member_dissims)

        # Heuristic: if cluster is tight, use sharp boundary; if loose, use wide
        if cluster_spread < 0.3 or dissim_std < 0.1:
            tuning_factor = 0.7  # Tight cluster → sharp
        elif cluster_spread > 1.0 or dissim_std > 0.5:
            tuning_factor = 1.3  # Loose cluster → wide
        else:
            tuning_factor = 1.0  # Medium → normal

        # Ensure bounds
        return np.clip(tuning_factor, 0.5, 1.5)

    def _normalize_partition_of_unity(self, mu: np.ndarray) -> np.ndarray:
        """Normalize to partition of unity."""
        row_sums = np.sum(mu, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        return mu / row_sums

    def defuzzify(self, mf_list: List[AutoTunedRuspiniMF], mu: np.ndarray) -> np.ndarray:
        """Defuzzify to hard assignments."""
        return np.argmax(mu, axis=1)

    def partition_of_unity_error(self, mu: np.ndarray) -> Tuple[float, float, float]:
        """Compute partition-of-unity error."""
        row_sums = np.sum(mu, axis=1)
        errors = np.abs(row_sums - 1.0)
        return float(np.max(errors)), float(np.mean(errors)), float(np.std(errors))

    def coverage(self, mu: np.ndarray, threshold: float = 0.5) -> float:
        """Compute coverage."""
        max_per_point = np.max(mu, axis=1)
        return float(np.mean(max_per_point >= threshold))


# ============================================================================
# Test on battery
# ============================================================================

def run_auto_tuned_battery() -> Dict:
    """Run auto-tuned Ruspini on battery."""
    datasets = [
        ('two_gaussians', B.two_gaussians),
        ('bridged_gaussians', B.bridged_gaussians),
        ('concentric_rings', B.concentric_rings),
        ('varying_density', B.varying_density),
        ('uniform_noise', B.uniform_noise),
    ]

    from sklearn.metrics import adjusted_rand_score

    results = {}

    for name, dataset_fn in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"{'='*70}")

        X, y_true = dataset_fn()

        D = im.dissimilarity(X)
        Dstar = im.minimax_transform(D)

        blocks = S.select_coverage_cover(Dstar)

        if not blocks:
            print("  No clusters selected")
            results[name] = {'ari': np.nan, 'coverage': np.nan, 'partition_error': np.nan}
            continue

        extractor = AutoTunedRuspiniExtractor(verbose=False)
        mf_list, mu = extractor.extract_partition(Dstar, blocks)

        assignments = extractor.defuzzify(mf_list, mu)
        ari = adjusted_rand_score(y_true, assignments)
        cov = extractor.coverage(mu)
        max_err, mean_err, _ = extractor.partition_of_unity_error(mu)

        print(f"  ARI: {ari:.4f}")
        print(f"  Coverage: {cov:.2%}")
        print(f"  Partition error: max={max_err:.6f}, mean={mean_err:.6f}")

        results[name] = {
            'ari': float(ari),
            'coverage': float(cov),
            'partition_error_max': float(max_err),
            'partition_error_mean': float(mean_err),
        }

    return results


if __name__ == '__main__':
    print("\n" + "="*70)
    print("AUTO-TUNED RUSPINI PARTITIONING (Option A, Revised)")
    print("="*70)

    results = run_auto_tuned_battery()

    # Compare to baseline
    baseline = {
        'two_gaussians': 1.00,
        'bridged_gaussians': 0.98,
        'concentric_rings': 1.00,
        'varying_density': 0.98,
        'uniform_noise': np.nan,
    }

    print(f"\n{'='*90}")
    print("Comparison: Auto-Tuned Ruspini vs. Baseline")
    print(f"{'='*90}")
    print(f"{'Dataset':<20} {'Auto-Tuned':<15} {'Baseline':<15} {'Gap':<15}")
    print(f"{'-'*90}")

    gaps = []
    for name, result in results.items():
        auto_ari = result['ari']
        baseline_ari = baseline[name]

        if np.isnan(auto_ari):
            print(f"{name:<20} {'(noise)':<15} {baseline_ari:<15.4f} {'-':<15}")
        else:
            gap = abs(auto_ari - baseline_ari)
            gaps.append(gap)
            print(f"{name:<20} {auto_ari:<15.4f} {baseline_ari:<15.4f} {gap:<15.4f}")

    print(f"{'='*90}")
    if gaps:
        print(f"Average gap: {np.mean(gaps):.4f}")
        print(f"Max gap: {np.max(gaps):.4f}")
    print(f"{'='*90}")
