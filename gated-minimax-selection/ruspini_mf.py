"""
Ruspini partitioning from VAT/IVAT ultrametric structure.

Path 1A: Core extraction → normalization → Ruspini parameters.

A Ruspini partition is a fuzzy partition of unity where:
  1. ∑_c μ_c(x) = 1 everywhere (partition property)
  2. Each μ_c is unimodal with a peak and linear descent (linguistic property)
  3. Parameters (center, width) are interpretable in feature space

This module extracts disjoint cores from the minimax hierarchy, then normalizes
the membership functions to satisfy the partition-of-unity property.
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


@dataclass
class RuspiniMF:
    """A Ruspini membership function with partition properties."""
    cluster_id: int
    medoid_idx: int
    members: Set[int]
    birth_height: float
    death_height: float
    # Parameters for reconstruction
    center_dissim: float  # μ = 1 at this dissimilarity value
    support_width: float  # support extends to center_dissim + support_width


class RuspiniPartitionExtractor:
    """
    Extract Ruspini partitions from VAT/IVAT hierarchy.

    Core idea: From a dendrogram, extract disjoint blocks with their birth/death
    heights, then normalize membership functions across all blocks so that
    ∑_c μ_c(x) = 1 everywhere.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def extract_partition(self, Dstar: np.ndarray, blocks: List[Set[int]],
                         medoids: List[int], birth_death: Dict[int, Tuple[float, float]],
                         normalize: bool = True, extend_support: bool = True) -> Tuple[List[RuspiniMF], np.ndarray]:
        """
        Extract a Ruspini partition from VAT blocks.

        Args:
            Dstar: minimax distance matrix (n × n)
            blocks: list of member sets for each cluster
            medoids: list of medoid indices
            birth_death: dict mapping cluster_id → (h_birth, h_death)
            normalize: if True, normalize to partition of unity
            extend_support: if True, extend support to ensure every point is covered

        Returns:
            (mf_list, mu_matrix) where:
              - mf_list: list of RuspiniMF objects
              - mu_matrix: (n × c) membership matrix before or after normalization
        """
        n = Dstar.shape[0]
        c = len(blocks)

        # Step 1: Build unnormalized membership functions (one per cluster)
        mu = np.zeros((n, c))
        mf_list = []
        supports = []  # Track actual support for each cluster

        for cluster_id, (block_members, medoid_idx) in enumerate(zip(blocks, medoids)):
            h_b, h_d = birth_death.get(cluster_id, (0.0, np.inf))

            # Compute dissimilarity ramp from medoid
            dissim_ramp = Dstar[medoid_idx, :]

            # Extract core and support
            core_dissims = dissim_ramp[list(block_members)]
            core_dissims = core_dissims[core_dissims <= h_b]

            if len(core_dissims) == 0:
                # Degenerate case: no tight core; use the tightest point
                center_dissim = np.min(dissim_ramp[list(block_members)])
            else:
                center_dissim = np.mean(core_dissims)

            support_width = h_d - center_dissim
            support_width = max(support_width, 1e-6)

            # Membership function: linear ramp from 1 at center_dissim to 0 at h_d
            for i in range(n):
                d = dissim_ramp[i]
                if d <= center_dissim:
                    mu[i, cluster_id] = 1.0
                elif d <= h_d:
                    mu[i, cluster_id] = (h_d - d) / support_width
                else:
                    mu[i, cluster_id] = 0.0

            supports.append(h_d)

            mf = RuspiniMF(
                cluster_id=cluster_id,
                medoid_idx=medoid_idx,
                members=block_members,
                birth_height=h_b,
                death_height=h_d,
                center_dissim=float(center_dissim),
                support_width=float(support_width)
            )
            mf_list.append(mf)

            if self.verbose:
                print(f"Cluster {cluster_id}: center_d={center_dissim:.4f}, "
                      f"support_width={support_width:.4f}, coverage={np.sum(mu[:, cluster_id] > 0)}")

        # Step 2 (optional): Extend support to ensure every point has membership
        if extend_support:
            max_support = np.max(supports)
            for i in range(n):
                row_sum = np.sum(mu[i, :])
                if row_sum == 0:
                    # Point is outside all supports; assign to nearest cluster
                    distances_to_medoids = np.array([Dstar[mf.medoid_idx, i] for mf in mf_list])
                    nearest = np.argmin(distances_to_medoids)
                    # Give it a small non-zero membership to the nearest cluster
                    mu[i, nearest] = 0.01

        # Step 3: Normalize to partition of unity
        if normalize:
            mu = self._normalize_partition_of_unity(mu)

        return mf_list, mu

    def _normalize_partition_of_unity(self, mu: np.ndarray) -> np.ndarray:
        """
        Normalize membership matrix so that ∑_c μ_c(x) = 1 for all points x.

        This is a simple sum-normalization: μ_c(x) := μ_c(x) / ∑_c' μ_c'(x).

        Args:
            mu: (n × c) membership matrix

        Returns:
            (n × c) normalized membership matrix with ∑_c μ_c(x) = 1 for all x
        """
        row_sums = np.sum(mu, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        mu_normalized = mu / row_sums
        return mu_normalized

    def evaluate_membership(self, mf_list: List[RuspiniMF], Dstar: np.ndarray,
                           point_idx: int, mu_normalized: np.ndarray = None) -> np.ndarray:
        """
        Evaluate membership of a single point across all clusters.

        Args:
            mf_list: list of RuspiniMF objects
            Dstar: minimax distance matrix
            point_idx: index of the point to evaluate
            mu_normalized: optional pre-computed normalized matrix (faster)

        Returns:
            1-D array of membership values across clusters
        """
        if mu_normalized is not None:
            return mu_normalized[point_idx, :]

        # Compute on-the-fly
        memberships = np.zeros(len(mf_list))
        for cluster_id, mf in enumerate(mf_list):
            d = Dstar[mf.medoid_idx, point_idx]
            if d <= mf.center_dissim:
                memberships[cluster_id] = 1.0
            elif d <= mf.death_height:
                memberships[cluster_id] = (mf.death_height - d) / mf.support_width
            else:
                memberships[cluster_id] = 0.0

        # Normalize
        s = np.sum(memberships)
        if s > 0:
            memberships = memberships / s

        return memberships

    def defuzzify_hardmax(self, mu: np.ndarray) -> np.ndarray:
        """
        Defuzzify membership matrix using hardmax (arg-max per point).

        Args:
            mu: (n × c) membership matrix

        Returns:
            1-D array of cluster assignments (0 to c-1)
        """
        return np.argmax(mu, axis=1)

    def defuzzify_proximity_tiebreak(self, mu: np.ndarray, Dstar: np.ndarray,
                                      mf_list: List[RuspiniMF]) -> np.ndarray:
        """
        Defuzzify using max membership, with distance-based tie-breaking.

        When two clusters have nearly equal membership, pick the one with smaller
        dissimilarity (closer in the ultrametric space).

        Args:
            mu: (n × c) membership matrix
            Dstar: minimax distance matrix
            mf_list: list of RuspiniMF objects

        Returns:
            1-D array of cluster assignments
        """
        n = mu.shape[0]
        assignments = np.zeros(n, dtype=int)

        for i in range(n):
            mu_i = mu[i, :]
            max_mu = np.max(mu_i)

            # Find all clusters within epsilon of the max
            epsilon = 0.01
            candidates = np.where(mu_i >= max_mu - epsilon)[0]

            if len(candidates) == 1:
                assignments[i] = candidates[0]
            else:
                # Tie-break by distance to medoid
                distances = np.array([Dstar[mf_list[c].medoid_idx, i] for c in candidates])
                assignments[i] = candidates[np.argmin(distances)]

        return assignments

    def partition_of_unity_error(self, mu: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute partition-of-unity error metrics.

        Args:
            mu: (n × c) membership matrix

        Returns:
            (max_error, mean_error, std_error) where each is |∑_c μ_c(x) - 1|
        """
        row_sums = np.sum(mu, axis=1)
        errors = np.abs(row_sums - 1.0)
        return float(np.max(errors)), float(np.mean(errors)), float(np.std(errors))

    def coverage(self, mu: np.ndarray, threshold: float = 0.5) -> float:
        """
        Compute coverage: fraction of points with max membership ≥ threshold.

        Args:
            mu: (n × c) membership matrix
            threshold: membership threshold

        Returns:
            fraction of points with confident membership (0 to 1)
        """
        max_per_point = np.max(mu, axis=1)
        return float(np.mean(max_per_point >= threshold))

    def ruspini_parameters(self, mf_list: List[RuspiniMF], X: np.ndarray = None,
                          feature_names: List[str] = None) -> Dict:
        """
        Extract Ruspini parameters for linguistic interpretation.

        Args:
            mf_list: list of RuspiniMF objects
            X: optional feature matrix (n × d) for feature-space extraction
            feature_names: optional names for features

        Returns:
            dict with cluster parameters (center, widths, etc.)
        """
        params = {}

        for mf in mf_list:
            cluster_params = {
                'cluster_id': mf.cluster_id,
                'medoid_idx': mf.medoid_idx,
                'center_dissim': mf.center_dissim,
                'support_width': mf.support_width,
                'birth_height': mf.birth_height,
                'death_height': mf.death_height,
                'num_members': len(mf.members),
            }

            # If feature data provided, compute feature-space statistics
            if X is not None:
                members_array = np.array(sorted(list(mf.members)))
                X_block = X[members_array, :]
                cluster_params['center_features'] = np.mean(X_block, axis=0).tolist()
                cluster_params['std_features'] = np.std(X_block, axis=0).tolist()

                if feature_names is not None:
                    cluster_params['feature_names'] = feature_names

            params[mf.cluster_id] = cluster_params

        return params

    def linguistic_description(self, params: Dict, cluster_id: int) -> str:
        """
        Generate human-readable linguistic description of a cluster.

        Args:
            params: dict from ruspini_parameters()
            cluster_id: ID of the cluster

        Returns:
            human-readable string
        """
        p = params.get(cluster_id, {})
        desc = f"Cluster {cluster_id}:"
        desc += f" members={p.get('num_members', 'unknown')}"
        desc += f" center_dissim={p.get('center_dissim', np.nan):.4f}"
        desc += f" support_width={p.get('support_width', np.nan):.4f}"

        if 'center_features' in p and 'feature_names' in p:
            center_feat = p['center_features']
            feature_names = p['feature_names']
            feat_str = ", ".join([f"{name}={c:.2f}" for name, c in zip(feature_names, center_feat)])
            desc += f" [features: {feat_str}]"

        return desc


# ============================================================================
# Integration with existing selection module
# ============================================================================

def extract_ruspini_from_blocks(Dstar: np.ndarray, blocks: List[Dict],
                                 X: np.ndarray = None) -> Tuple[List[RuspiniMF], np.ndarray]:
    """
    Convenience function: extract Ruspini partition from selection.py blocks.

    Expects blocks in the format from selection.coverage_cover():
    Each block is a dict with keys: 'members', 'birth', 'death', 'medoid_idx'

    Args:
        Dstar: minimax distance matrix
        blocks: list of dicts (from coverage_cover or similar selector)
        X: optional feature matrix for feature-space extraction

    Returns:
        (mf_list, mu_normalized) from RuspiniPartitionExtractor
    """
    # Convert block dicts to the format expected by RuspiniPartitionExtractor
    block_sets = [set(b['members']) for b in blocks]
    medoids = [b.get('medoid_idx', list(b['members'])[0]) for b in blocks]
    birth_death = {
        i: (b.get('birth', 0.0), b.get('death', np.inf))
        for i, b in enumerate(blocks)
    }

    extractor = RuspiniPartitionExtractor(verbose=False)
    mf_list, mu = extractor.extract_partition(Dstar, block_sets, medoids, birth_death, normalize=True)

    return mf_list, mu
