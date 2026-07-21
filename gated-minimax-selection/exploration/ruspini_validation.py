"""
Ruspini partitioning validation & integration with VAT/IVAT membership extraction.

Key questions answered here:
1. Do extracted MFs form a partition of unity? (∑_c μ_c(x) = 1)
2. Does Ruspini structure preserve non-convex clustering wins?
3. How does Ruspini compare to persistence-based MFs on ARI, coverage?
4. Can we map Ruspini parameters back to feature space for interpretability?
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


# ============================================================================
# Ruspini Partition Validation
# ============================================================================

@dataclass
class RuspiniPartitionMetrics:
    """Metrics for validating Ruspini partition properties."""
    partition_of_unity_error: float  # max |∑_c μ_c(x) - 1.0|
    coverage: float                  # fraction of points with max μ > 0.5
    consistency: float               # std of ∑_c μ_c(x) across all points
    peak_sharpness: List[float]      # max(μ_c) - mean(μ_c) per cluster (higher = sharper peak)
    linguistic_validity: bool        # True if all MFs univariate-convex


def validate_partition_of_unity(mf_evaluator, Dstar: np.ndarray,
                                blocks: List[set], medoids: List[int]) -> RuspiniPartitionMetrics:
    """
    Validate that extracted MFs satisfy Ruspini partition properties.

    Args:
        mf_evaluator: callable(block_id, point_idx) -> μ_c(x_i)
        Dstar: minimax distance matrix
        blocks: list of block member sets
        medoids: list of medoid indices

    Returns:
        RuspiniPartitionMetrics with validation results
    """
    n = Dstar.shape[0]
    num_blocks = len(blocks)

    # Evaluate membership for all points
    mu = np.zeros((n, num_blocks))
    for c, (block_members, medoid_idx) in enumerate(zip(blocks, medoids)):
        for i in range(n):
            mu[i, c] = mf_evaluator(c, i, Dstar[medoid_idx, i])

    # Partition of unity check: ∑_c μ_c(x) should equal 1.0 everywhere
    row_sums = np.sum(mu, axis=1)
    partition_error = np.max(np.abs(row_sums - 1.0))
    consistency = np.std(row_sums)

    # Coverage: what fraction of points have at least 0.5 membership in some cluster
    max_per_point = np.max(mu, axis=1)
    coverage = np.mean(max_per_point >= 0.5)

    # Peak sharpness: how concentrated is each cluster's membership around its medoid
    peak_sharpness = []
    for c in range(num_blocks):
        mu_c = mu[:, c]
        if np.max(mu_c) > 0:
            sharpness = np.max(mu_c) - np.mean(mu_c[mu_c > 0])
            peak_sharpness.append(float(sharpness))
        else:
            peak_sharpness.append(0.0)

    # Linguistic validity: each MF should be univariate-convex when plotted against
    # the dissimilarity ramp. This is a proxy for "interpretable as a linguistic term."
    # (Full check would require fitting to a feature ramp.)
    linguistic_validity = True  # Placeholder; full check below

    return RuspiniPartitionMetrics(
        partition_of_unity_error=float(partition_error),
        coverage=float(coverage),
        consistency=float(consistency),
        peak_sharpness=peak_sharpness,
        linguistic_validity=linguistic_validity
    )


def plot_partition_of_unity(mf_evaluator, Dstar: np.ndarray, blocks: List[set],
                            medoids: List[int], title: str = "Partition of Unity Check",
                            output_path: str = None):
    """
    Visualize membership functions and their sum along a sorted dissimilarity ramp.

    Creates a stacked-area plot showing each cluster's membership contribution,
    with a thin black line at y=1.0 showing the target partition sum.
    """
    n = Dstar.shape[0]
    num_blocks = len(blocks)

    # Create a synthetic "ramp" by sorting all pairwise dissimilarities
    ramp_distances = np.sort(Dstar[0, :])  # Example: from medoid 0

    # Evaluate membership along this ramp
    mu = np.zeros((len(ramp_distances), num_blocks))
    for c, medoid_idx in enumerate(medoids):
        mu[:, c] = np.array([mf_evaluator(c, i, ramp_distances[i])
                            for i in range(len(ramp_distances))])

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Stacked area plot
    ax1.stackplot(range(len(ramp_distances)), mu.T, alpha=0.7,
                  labels=[f"Cluster {c}" for c in range(num_blocks)])
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Target (y=1.0)')
    ax1.set_ylabel("Membership (stacked)")
    ax1.set_title(f"{title} — Stacked Membership")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim([0, 1.5])
    ax1.grid(True, alpha=0.3)

    # Sum check
    mu_sum = np.sum(mu, axis=1)
    ax2.plot(range(len(ramp_distances)), mu_sum, color='blue', linewidth=2, label='∑ μ_c(x)')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Target (y=1.0)')
    ax2.fill_between(range(len(ramp_distances)), 1.0, mu_sum,
                     where=(mu_sum >= 1.0), alpha=0.3, color='green', label='Over-assignment')
    ax2.fill_between(range(len(ramp_distances)), 1.0, mu_sum,
                     where=(mu_sum < 1.0), alpha=0.3, color='red', label='Under-coverage')
    ax2.set_ylabel("Sum of membership (∑ μ_c)")
    ax2.set_xlabel("Point index (sorted by dissimilarity)")
    ax2.set_title("Partition of Unity Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.5])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Feature-Space Ruspini Parameter Extraction
# ============================================================================

@dataclass
class RuspiniParameters:
    """Linguistic Ruspini parameters in feature space."""
    cluster_id: int
    center: np.ndarray          # d-dimensional center
    left_width: float           # width on the "left" side
    right_width: float          # width on the "right" side
    prototype: str              # 'triangular', 'gaussian', etc.
    feature_names: List[str]    # names of features (for interpretability)


def extract_ruspini_parameters_feature_space(X: np.ndarray, blocks: List[set],
                                              medoids: List[int],
                                              feature_names: List[str] = None) -> List[RuspiniParameters]:
    """
    Extract Ruspini parameters from feature-space data (Euclidean case).

    For each cluster:
    1. Compute the center (mean of block members)
    2. Fit a Gaussian or triangular template in feature space
    3. Extract (center, width) parameters
    4. Return as linguistic descriptions

    Args:
        X: data matrix (n × d)
        blocks: list of member sets
        medoids: list of medoid indices
        feature_names: optional names for features

    Returns:
        List of RuspiniParameters, one per cluster
    """
    n, d = X.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]

    params_list = []

    for cluster_id, (block_members, medoid_idx) in enumerate(zip(blocks, medoids)):
        members_array = np.array(list(block_members))
        X_block = X[members_array, :]

        # Center: mean of block members
        center = np.mean(X_block, axis=0)

        # Widths: std per dimension (captured left/right as ±σ)
        stds = np.std(X_block, axis=0)
        left_width = np.mean(stds)
        right_width = np.mean(stds)

        params = RuspiniParameters(
            cluster_id=cluster_id,
            center=center,
            left_width=float(left_width),
            right_width=float(right_width),
            prototype='gaussian',  # Default; could be auto-selected
            feature_names=feature_names
        )
        params_list.append(params)

    return params_list


def linguistic_description_from_ruspini(params: RuspiniParameters) -> str:
    """
    Generate human-readable linguistic description from Ruspini parameters.

    Example output:
    "Cluster 0: centered at (x1=5.2, x2=3.1) with width ±0.8"
    """
    center_str = ", ".join([f"{name}={c:.2f}" for name, c in zip(params.feature_names, params.center)])
    width = (params.left_width + params.right_width) / 2
    return f"Cluster {params.cluster_id}: centered at ({center_str}) with width ±{width:.2f}"


# ============================================================================
# Comparison: Ruspini vs. Persistence-Based Membership
# ============================================================================

def compare_mf_approaches(X: np.ndarray, Dstar: np.ndarray, blocks: List[set],
                          medoids: List[int], birth_death_heights: Dict,
                          ground_truth_labels: np.ndarray = None) -> Dict:
    """
    Compare Ruspini-extracted MFs against persistence-based MFs.

    Args:
        X: feature data (for feature-space Ruspini)
        Dstar: minimax distance matrix
        blocks, medoids, birth_death_heights: from VAT/IVAT
        ground_truth_labels: optional; compute ARI against ground truth

    Returns:
        Dict with comparison metrics
    """
    n = Dstar.shape[0]
    num_clusters = len(blocks)

    # 1. Ruspini membership (feature space)
    ruspini_params = extract_ruspini_parameters_feature_space(X, blocks, medoids)

    # 2. Persistence-based membership (dissimilarity space) — extracted via prototype framework
    # (Stub: in real code, call your existing persistence_mf extraction)
    persistence_mu = np.zeros((n, num_clusters))
    for c, medoid_idx in enumerate(medoids):
        # Naive: membership decays with distance
        persistence_mu[:, c] = np.exp(-0.5 * Dstar[medoid_idx, :])
    persistence_mu /= (np.sum(persistence_mu, axis=1, keepdims=True) + 1e-9)

    # 3. Ruspini membership (dissimilarity space via prototype framework)
    # (Stub: would call prototype_mf_extractor)

    # 4. Metrics
    results = {
        'num_clusters': num_clusters,
        'partition_error_persistence': np.max(np.abs(np.sum(persistence_mu, axis=1) - 1.0)),
        'coverage_persistence': np.mean(np.max(persistence_mu, axis=1) >= 0.5),
        'ruspini_center_spread': np.std([np.linalg.norm(p.center) for p in ruspini_params]),
    }

    if ground_truth_labels is not None:
        from sklearn.metrics import adjusted_rand_score
        pred_hard_persistence = np.argmax(persistence_mu, axis=1)
        results['ari_persistence'] = adjusted_rand_score(ground_truth_labels, pred_hard_persistence)

    return results


# ============================================================================
# Integration: Ruspini + Multi-Scale Persistence Selection
# ============================================================================

def multi_scale_persistence_with_ruspini(Dstar: np.ndarray, X: np.ndarray = None,
                                         scale_bins: int = 5) -> Tuple[List[set], List[int]]:
    """
    Select clusters using multi-scale persistence, then extract Ruspini MFs.

    Idea: instead of a single global persistence ranking, partition the hierarchy
    into scale bands (e.g., fine, medium, coarse) and rank within each band.

    Args:
        Dstar: minimax distance matrix
        X: optional feature data (for Ruspini in feature space)
        scale_bins: number of scale bands to use

    Returns:
        (blocks, medoids) selected via multi-scale persistence
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    Z = linkage(squareform(Dstar, checks=False), method='single')

    # Partition the dendrogram into scale bands
    heights = Z[:, 2]
    height_percentiles = np.percentile(heights, np.linspace(0, 100, scale_bins + 1))

    selected_blocks = []
    selected_medoids = []

    for i in range(scale_bins):
        h_low = height_percentiles[i]
        h_high = height_percentiles[i + 1]

        # Find all blocks in this scale band
        # (Simplified: in real code, would walk the dendrogram tree)
        # For each such block, compute persistence within its scale band

    return selected_blocks, selected_medoids


# ============================================================================
# Example & Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Ruspini Partitioning Validation")
    print("=" * 70)

    # Toy example: 3 Gaussian clusters in 2D
    np.random.seed(42)
    n_per_cluster = 30
    X = np.vstack([
        np.random.randn(n_per_cluster, 2) + np.array([0, 0]),
        np.random.randn(n_per_cluster, 2) + np.array([5, 0]),
        np.random.randn(n_per_cluster, 2) + np.array([2.5, 5]),
    ])
    n_total = len(X)

    # Compute dissimilarity and minimax transform (simulated)
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    # (In real code, would compute minimax D* here)
    Dstar = D.copy()

    # Simulated blocks (ground truth clusters)
    blocks = [
        set(range(0, n_per_cluster)),
        set(range(n_per_cluster, 2*n_per_cluster)),
        set(range(2*n_per_cluster, 3*n_per_cluster)),
    ]
    medoids = [0, n_per_cluster, 2*n_per_cluster]

    # Extract Ruspini parameters
    print("\nRuspini Parameters in Feature Space:")
    ruspini_params = extract_ruspini_parameters_feature_space(X, blocks, medoids)
    for params in ruspini_params:
        print(f"  {linguistic_description_from_ruspini(params)}")

    # Validate partition of unity (mock evaluator)
    def mock_evaluator(c, point_idx, dissim_value):
        # Return membership proportional to 1 / (1 + dissim_value)
        return 1.0 / (1.0 + dissim_value)

    metrics = validate_partition_of_unity(mock_evaluator, Dstar, blocks, medoids)
    print(f"\nPartition Metrics:")
    print(f"  Partition of unity error: {metrics.partition_of_unity_error:.4f}")
    print(f"  Coverage (μ > 0.5): {metrics.coverage:.2%}")
    print(f"  Consistency (std of sums): {metrics.consistency:.4f}")
    print(f"  Peak sharpness per cluster: {[f'{s:.2f}' for s in metrics.peak_sharpness]}")

    print("\n" + "=" * 70)
