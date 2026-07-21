# Relational Data Test Datasets

## Overview

**Relational datasets** are distance matrices without vector coordinates. They represent real-world scenarios where only pairwise distances are available (e.g., phylogenetic data, semantic similarities, graph distances). Matrix-based methods like NERFCM are the natural/only choice for these datasets.

This module (`relationdata.py`) provides three synthetic relational datasets designed to test whether D* (minimax distance) improves clustering on data where:
- Only a dissimilarity matrix D is available (no feature vectors)
- Vector-space methods (k-means, FCM, Euclidean clustering) do not apply
- Matrix methods (NERFCM) are the baseline

## Datasets

### 1. `three_clusters_tree(n=30, seed=106)`

**Structure:**
- Three clusters embedded in a tree backbone
- Root node → three subtrees (A, B, C), each with tight internal structure
- Intra-cluster distances: ~0.3 (tight)
- Inter-cluster distances: ~3.0 (far apart)
- Small noise added

**Use case:** Tests whether D* helps identify well-separated clusters when the underlying structure is hierarchical but expressed as distances only.

**Ground truth:** 3 clusters

```
    Root
    / | \
   A  B  C
  /|  |\ |\
 ... ... ...
```

### 2. `chain_then_ring(n=40, seed=107)`

**Structure:**
- One elongated (chain-like) cluster: nodes strung in a line at distance ~0.5
- One circular (ring-like) cluster: nodes forming a cycle at distance ~0.4
- The two clusters are far apart (distance ~5.0)
- Small noise added

**Use case:** Tests multi-scale structure where intra-cluster distances vary. Chain has elongated intra-cluster distances; ring has more uniform distances. This mimics non-convex real-world structures.

**Ground truth:** 2 clusters

The key idea: raw distances D might conflate elongation with separation, but D* (bottleneck distance) should provide more robust structure recovery.

### 3. `multi_scale_hierarchy(n=45, seed=108)`

**Structure:**
- Three large clusters at the coarse level
- Each large cluster contains 2 sub-clusters at an intermediate scale
- Intra-sub-cluster distances: ~0.4 (tight)
- Inter-sub-cluster distances: ~2.0 (intermediate)
- Inter-large-cluster distances: ~6.0+ (far apart)
- Small noise added

**Use case:** Tests adaptive scale discovery. D* should help if it can separate the natural hierarchical scales without a global threshold.

**Ground truth:**
- Coarse: 3 clusters
- Fine: 6 sub-clusters

## Expected Behavior

### Current Results

Running `python3 run_nerfcm.py` on these datasets shows:

```
dataset                             NERFCM(D)          NERFCM(D*)
three_clusters_tree                 1.00±0.00           1.00±0.00
chain_then_ring                     1.00±0.00           1.00±0.00
multi_scale_hierarchy               0.29±0.00           0.29±0.00
```

**Interpretation:**
- `three_clusters_tree` and `chain_then_ring`: NERFCM already recovers structure well from D alone. No gap = NERFCM is robust to tree distances.
- `multi_scale_hierarchy`: Both NERFCM(D) and NERFCM(D*) struggle with ARI 0.29. This suggests the scale-adaptation problem is *harder* than NERFCM's simple approach can handle. This is actually the *real* problem to solve.

## Why These Datasets Matter

1. **Pure relational setting:** No feature vectors → vector-space methods are ruled out → isolation of what a pure matrix method can achieve.

2. **D* comparison:** Since both D and D* are dissimilarity matrices, both can be fed to NERFCM. Any gap between NERFCM(D) and NERFCM(D*) isolates the value of the minimax transform on relational data, independent of any selection machinery.

3. **Progressive difficulty:**
   - `three_clusters_tree`: easy (well-separated, clear hierarchy)
   - `chain_then_ring`: medium (non-convex, but still recoverable)
   - `multi_scale_hierarchy`: hard (adaptive scales)

## How to Extend

### Adding Noise or Variability

Adjust the `noise_scale` parameter in `_tree_distance_matrix()`:
```python
D, y = _tree_distance_matrix(edges, leaf_labels, noise_scale=0.1, seed=seed)
```

Higher noise makes the problem harder.

### Creating New Datasets

1. Define tree structure via edges:
   ```python
   edges = [
       (root, child1, weight1),
       (root, child2, weight2),
       ...
   ]
   ```

2. Label leaves:
   ```python
   leaf_labels = {
       leaf_id: cluster_id,
       ...
   }
   ```

3. Call `_tree_distance_matrix()`:
   ```python
   D, y = _tree_distance_matrix(edges, leaf_labels, noise_scale=0.05)
   ```

### Synthetic Relational Data Without Trees

If you want datasets that are *not* tree-based (e.g., arbitrary metric spaces, non-metric dissimilarities), you can:

1. **Generate synthetic "relational" structures:**
   - Random metric spaces (Sample from a distribution of distances)
   - Graph distances (shortest paths in a random graph)
   - String edit distances (synthetic strings with known similarity structure)

2. **Use real relational data:**
   - Phylogenetic distances (species or protein distances)
   - Document/text similarities (TF-IDF, word embeddings truncated to distances only)
   - Graph spectral distances (eigenvalues of adjacency matrices)

## Running the Tests

```bash
# Test just relationdata module
python3 relationdata.py

# Run NERFCM comparison (Euclidean + relational)
python3 run_nerfcm.py

# Run standalone relational benchmark
python3 run_relationdata.py
```

## Files

- `relationdata.py`: Dataset generation code
- `run_relationdata.py`: Standalone benchmark (NERFCM on relational data only)
- `run_nerfcm.py`: Extended version with relational datasets added

## Integration

The relational datasets integrate seamlessly into your existing test harness:
- `relationdata` module is importable alongside `battery`
- D and D* matrices flow through the same NERFCM pipeline
- ARI scoring reuses existing metrics
- Extends `run_nerfcm.py` to show gaps (or lack thereof) on relational data

## Next Steps

1. **Identify where D* should help:** Modify datasets to create cases where raw D fails but D* succeeds (e.g., add non-metricity, embed multi-scale structure more aggressively).

2. **Combine with selection machinery:** Run `selection.py` on D and D* to see if block-selection+coverage improves over NERFCM.

3. **Performance characterization:** Once you're confident in small test cases, measure O(n²) cost of minimax_transform and profile bottlenecks.
