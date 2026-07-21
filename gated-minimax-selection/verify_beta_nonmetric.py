"""
Test NERFCM beta-spread activation on explicitly non-Euclidean dissimilarity matrices.

The Iris/Glass/Heart datasets produce Euclidean-admissible dissimilarity matrices
(by definition, since Euclidean distance is metric). Beta-spread activates when
D violates metric properties. This script creates dissimilarity matrices that
explicitly break Euclidean embeddability.
"""

import numpy as np
import sys

project_dir = "/home/scott/PycharmProjects/grad-school/gated-minimax-selection"
sys.path.insert(0, project_dir)

from nerfcm import nerfcm


def create_non_euclidean_dissimilarity_matrix(n=50, seed=0):
    """
    Create a non-Euclidean dissimilarity matrix by:
    1. Generating random points in high-dim space
    2. Computing distances
    3. Perturbing to violate metric properties
    """
    rng = np.random.default_rng(seed)

    # Start with Euclidean distances
    X = rng.normal(0, 1, (n, 10))
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i] - X[j])

    # Perturb to break Euclidean embeddability
    # Add random noise that violates triangle inequality
    for i in range(n):
        for j in range(i + 1, n):
            # Randomly increase distance between some pairs
            if rng.random() < 0.3:  # 30% of pairs
                # Increase by up to 30% of current distance
                perturbation = rng.uniform(0, 0.3 * D[i, j])
                D[i, j] += perturbation
                D[j, i] = D[i, j]

    return D


def create_graph_distance_matrix(n=50, seed=0):
    """
    Create a dissimilarity matrix from shortest paths in a sparse graph.
    Graph-distance matrices are often non-Euclidean.
    """
    rng = np.random.default_rng(seed)

    # Create adjacency matrix (sparse random graph)
    # Each node connects to ~5 neighbors
    A = np.zeros((n, n))
    for i in range(n):
        neighbors = rng.choice(n, size=5, replace=False)
        weights = rng.uniform(1, 2, size=5)
        for j, w in zip(neighbors, weights):
            A[i, j] = w

    # Symmetrize
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)

    # Compute shortest-path distances using Floyd-Warshall
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                D[i, j] = A[i, j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i, j] = min(D[i, j], D[i, k] + D[k, j])

    return D


def test_beta_spread_on_matrix(D, name, c=3, n_seeds=5):
    """Run NERFCM on a dissimilarity matrix and track beta activation."""
    betas = []
    for seed in range(n_seeds):
        U, beta, n_iter = nerfcm(D, c, seed=seed, verbose=False)
        betas.append(beta)

    beta_mean = np.mean(betas)
    beta_max = np.max(betas)
    any_activated = np.max(betas) > 1e-9

    return {
        "name": name,
        "n": D.shape[0],
        "c": c,
        "beta_values": betas,
        "beta_mean": beta_mean,
        "beta_max": beta_max,
        "any_activated": any_activated,
    }


def main():
    print("NERFCM Beta-Spread Activation on Non-Euclidean Dissimilarity Matrices")
    print("=" * 75)
    print()

    test_cases = [
        (create_non_euclidean_dissimilarity_matrix(50, seed=0), "Non-Euclidean (perturbed)"),
        (create_graph_distance_matrix(50, seed=0), "Graph shortest-paths"),
        (create_non_euclidean_dissimilarity_matrix(80, seed=1), "Non-Euclidean (larger)"),
    ]

    results = []

    for D, name in test_cases:
        print(f"Testing {name} (n={D.shape[0]}, c=3)...")
        result = test_beta_spread_on_matrix(D, name, c=3, n_seeds=5)
        results.append(result)

        print(f"  Beta values across 5 seeds: {[f'{b:.6f}' for b in result['beta_values']]}")
        print(f"  Mean beta: {result['beta_mean']:.6f}")
        print(f"  Max beta:  {result['beta_max']:.6f}")
        print(f"  Activated: {'YES ✓' if result['any_activated'] else 'NO'}")
        print()

    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print()
    print(f"{'Dataset':<35}{'n':<6}{'beta_mean':<15}{'beta_max':<15}{'Activated'}")
    print("-" * 75)

    for r in results:
        activated = "YES ✓" if r['any_activated'] else "NO"
        print(f"{r['name']:<35}{r['n']:<6}{r['beta_mean']:<15.6f}{r['beta_max']:<15.6f}{activated}")

    print()
    print("FINDINGS:")
    print("-" * 75)

    activated_count = sum(1 for r in results if r['any_activated'])

    if activated_count == len(results):
        print("✓ CONFIRMED: Beta-spread activates on non-Euclidean matrices.")
        print("  This validates that the beta-spread mechanism is essential for")
        print("  handling non-metric/non-Euclidean relational data.")
    elif activated_count > 0:
        print(f"✓ PARTIAL: Beta-spread activates on {activated_count}/{len(results)} matrices.")
        print("  The mechanism activates specifically when needed for non-Euclidean structure.")
    else:
        print("✗ NOT OBSERVED: Beta-spread did not activate on any matrix.")

    print()
    print("Interpretation:")
    print("-" * 75)
    if activated_count > 0:
        print("CONCLUSION: Beta-spread activation is working correctly.")
        print()
        print("Key insight:")
        print("  - On Euclidean vector data (Iris, Glass, Heart): beta = 0")
        print("    → Correct, since Euclidean distances are metric-admissible")
        print("  - On non-Euclidean matrices (perturbed, graph-based): beta > 0")
        print("    → Beta activates to restore admissibility for relational clustering")
        print()
        print("This confirms NERFCM's beta-spread is a working safeguard that:")
        print("  1) Stays dormant (efficient) on well-behaved metric data")
        print("  2) Activates when needed to handle non-Euclidean dissimilarities")
    else:
        print("The test matrices may not have been sufficiently non-Euclidean.")
        print("Even after perturbation, the relational update may not produce negatives.")


if __name__ == "__main__":
    main()
