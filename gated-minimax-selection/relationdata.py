"""
Synthetic relational datasets: distance matrices without vector coordinates.

These datasets are designed for relational clustering methods (e.g., NERFCM) where
only pairwise distances are available, not feature vectors. They showcase scenarios
where D* (minimax distance) should improve clustering over raw dissimilarity D.

Datasets are constructed as tree/hierarchical structures with embedded non-convex
or multi-scale properties that are not Euclidean-embeddable, so vector-space
methods cannot be applied.
"""

import numpy as np


rng = np.random.default_rng(42)


def _rng(seed):
    """Deterministic per-dataset RNG."""
    return np.random.default_rng(seed) if seed is not None else rng


def _tree_distance_matrix(tree_edges, leaf_labels, noise_scale=0.0, seed=None):
    """
    Build a distance matrix from a tree structure.

    Parameters
    ----------
    tree_edges : list of (u, v, edge_weight) tuples
        Edges in the tree; nodes are integers.
    leaf_labels : dict
        Mapping from leaf node ID to ground-truth label.
    noise_scale : float
        Add Gaussian noise to distances (proportional to distance).
    seed : int or None
        RNG seed.

    Returns
    -------
    D : (n, n) symmetric distance matrix (leaves only)
    y : (n,) ground-truth labels for leaves
    """
    rng_local = _rng(seed)
    leaves = sorted(leaf_labels.keys())
    n = len(leaves)
    leaf_idx = {lid: i for i, lid in enumerate(leaves)}

    # Build adjacency list and compute all-pairs shortest paths.
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v, w in tree_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    D = np.zeros((n, n))
    for i, li in enumerate(leaves):
        for j, lj in enumerate(leaves):
            if i == j:
                continue
            # BFS to find shortest path distance
            dist = _shortest_path(li, lj, adj)
            D[i, j] = dist

    # Symmetrize (should already be, but just in case)
    D = (D + D.T) / 2

    # Add noise
    if noise_scale > 0:
        noise = rng_local.normal(0, noise_scale * D.mean(), (n, n))
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        D = np.maximum(D + noise, 0)
        D = (D + D.T) / 2

    y = np.array([leaf_labels[lid] for lid in leaves], dtype=int)
    return D, y


def _shortest_path(start, end, adj):
    """BFS shortest path in an edge-weighted graph."""
    if start == end:
        return 0.0
    from collections import deque
    visited = {start}
    queue = deque([(start, 0.0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor, edge_weight in adj[node]:
            if neighbor == end:
                return dist + edge_weight
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + edge_weight))
    return float('inf')


def three_clusters_tree(n=30, seed=106):
    """
    Three clusters on a tree backbone.

    Structure:
      - Root node connects to three backbone subtrees (A, B, C).
      - Within each subtree, points form a tight cluster (low intra-distance).
      - Inter-cluster distances are high.

    Ground truth: 3 clusters.
    """
    rng_local = _rng(seed)

    # Build a tree with 3 main branches from root 0.
    # Branch A: 0-1-2,3,4 (tight)
    # Branch B: 0-5-6,7,8 (tight)
    # Branch C: 0-9-10,11,12 (tight)

    edges = [
        # Backbone (inter-cluster)
        (0, 1, 3.0),   # A root
        (0, 5, 3.0),   # B root
        (0, 9, 3.0),   # C root
        # Cluster A
        (1, 2, 0.3),
        (1, 3, 0.3),
        (1, 4, 0.3),
        # Cluster B
        (5, 6, 0.3),
        (5, 7, 0.3),
        (5, 8, 0.3),
        # Cluster C
        (9, 10, 0.3),
        (9, 11, 0.3),
        (9, 12, 0.3),
    ]

    # Add within-cluster noise edges to mimic additional structure
    leaves = list(range(2, 5)) + list(range(6, 9)) + list(range(10, 13))

    # Expand if n > 13
    node_id = 13
    leaf_labels = {
        2: 0, 3: 0, 4: 0,      # Cluster A
        6: 1, 7: 1, 8: 1,      # Cluster B
        10: 2, 11: 2, 12: 2,   # Cluster C
    }

    while len(leaves) < n:
        # Add more points by attaching to random cluster roots
        cluster_root = rng_local.choice([1, 5, 9])
        cluster_id = [0, 1, 2][{1: 0, 5: 1, 9: 2}[cluster_root]]
        edges.append((cluster_root, node_id, 0.3))
        leaves.append(node_id)
        leaf_labels[node_id] = cluster_id
        node_id += 1

    D, y = _tree_distance_matrix(edges, leaf_labels, noise_scale=0.05, seed=seed)
    return D, y


def chain_then_ring(n=40, seed=107):
    """
    Two clusters: one elongated (chain-like), one circular (ring-like).

    Structure:
      - Chain cluster: a long linear sequence of closely-spaced nodes.
      - Ring cluster: nodes forming a cycle.
      - The two clusters are far apart in distance.

    Ground truth: 2 clusters.
    This tests whether D* helps separate non-convex structures that raw D
    might conflate if intra-cluster distances vary widely.
    """
    rng_local = _rng(seed)

    edges = []
    leaf_labels = {}

    # Chain cluster: 0-1-2-3-...-n1, all at distance 0.5
    n1 = n // 2
    for i in range(n1 - 1):
        edges.append((i, i + 1, 0.5))
        leaf_labels[i] = 0
    leaf_labels[n1 - 1] = 0

    # Ring cluster: nodes n1 to n1+n2-1, arranged in a cycle
    n2 = n - n1
    ring_nodes = list(range(n1, n1 + n2))
    for i in range(n2):
        u = ring_nodes[i]
        v = ring_nodes[(i + 1) % n2]
        edges.append((u, v, 0.4))
        leaf_labels[u] = 1

    # Connect the two clusters far apart
    edges.append((n1 - 1, n1, 5.0))

    D, y = _tree_distance_matrix(edges, leaf_labels, noise_scale=0.03, seed=seed)
    return D, y


def multi_scale_hierarchy(n=45, seed=108):
    """
    Nested clusters at different scales (hierarchy-like).

    Structure:
      - Top level: 3 large clusters (A, B, C).
      - Each large cluster contains 2-3 sub-clusters.
      - Intra-sub-cluster distances are small.
      - Inter-sub-cluster distances are intermediate.
      - Inter-large-cluster distances are large.

    Ground truth (fine-grained): 6-9 sub-clusters.
    Ground truth (coarse): 3 clusters.

    This tests whether D* can adapt to multi-scale structure.
    """
    rng_local = _rng(seed)

    edges = []
    leaf_labels = {}
    node_id = 0

    # Create 3 large clusters
    cluster_roots = []
    for large_c in range(3):
        root = node_id
        cluster_roots.append(root)
        node_id += 1

        # Each large cluster has 2 sub-clusters
        sub_roots = []
        for sub_c in range(2):
            sub_root = node_id
            node_id += 1
            sub_roots.append(sub_root)

            # Connect sub_root to root with intermediate distance
            edges.append((root, sub_root, 2.0))

            # Add points to sub-cluster
            points_in_sub = rng_local.integers(4, 8)
            for _ in range(points_in_sub):
                leaf_node = node_id
                node_id += 1
                edges.append((sub_root, leaf_node, 0.4))
                # Label: we'll assign a sub-cluster ID for fine-grained truth
                label = large_c * 2 + sub_c
                leaf_labels[leaf_node] = label

    # Connect large clusters far apart
    edges.append((cluster_roots[0], cluster_roots[1], 6.0))
    edges.append((cluster_roots[1], cluster_roots[2], 6.0))
    edges.append((cluster_roots[0], cluster_roots[2], 6.5))

    # If we haven't reached n points, add more to random sub-clusters
    while node_id - len(cluster_roots) < n:
        sub_root = rng_local.choice([r for roots in [sub_roots] for r in roots])
        leaf_node = node_id
        node_id += 1
        edges.append((sub_root, leaf_node, 0.4))
        # Find the cluster ID from the root's children
        label = None
        for lbl, lid in list(leaf_labels.items())[:5]:
            if label is None:
                label = rng_local.integers(0, 4)
        if label is None:
            label = 0
        leaf_labels[leaf_node] = label

    # Trim to exactly n leaves
    leaves = [k for k in leaf_labels.keys()]
    if len(leaves) > n:
        trim = sorted(rng_local.choice(leaves, n, replace=False))
        leaf_labels = {k: leaf_labels[k] for k in trim}

    D, y = _tree_distance_matrix(edges, leaf_labels, noise_scale=0.04, seed=seed)
    return D, y


if __name__ == "__main__":
    # Test the datasets
    for name, fn in [
        ("three_clusters_tree", three_clusters_tree),
        ("chain_then_ring", chain_then_ring),
        ("multi_scale_hierarchy", multi_scale_hierarchy),
    ]:
        D, y = fn()
        print(f"{name}: D.shape={D.shape}, clusters={np.unique(y)}, "
              f"D.mean()={D.mean():.3f}, D.std()={D.std():.3f}")
