"""
Comprehensive comparison of three persistence-based cluster count selection methods:

1. PERSISTENCE-GAP / KNEE-SELECTION (current method)
   - Identifies the largest gap in the sorted persistence distribution
   - Gates blocks as eligible if their persistence is a statistical outlier
   - Uses greedy set-cover for selection
   - Source: Current iVAT-based approach

2. BETA-PLATEAU (Bonis & Oudot, arXiv:1406.7130)
   - Uses a temperature parameter (beta) in random-walk diffusion model
   - For each beta, computes a partition quality metric (e.g., cluster count)
   - "Plateau" = where the metric stabilizes as beta varies
   - Interpretation: Beta controls attraction to density peaks
   - Approximation: Varies a "temperature" analog over persistence curves

3. BOTTLENECK-BOOTSTRAP (AuToMATo, arXiv:2408.06958)
   - Resamples data multiple times via bootstrap
   - Computes persistence diagrams for each resample
   - Uses bottleneck distance to find stability in persistence features
   - Selects the gap that appears most consistently across boots
   - More computationally expensive but statistically principled

Each method returns (n_clusters, selected_blocks, metadata) for evaluation.
"""

import numpy as np
import ivat_mf as im
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _all_blocks(Dstar):
    """Enumerate every internal node of the SL dendrogram as a candidate block.
    (Shared utility from selection.py)"""
    n = Dstar.shape[0]
    Z = linkage(squareform(Dstar, checks=False), method='single')
    members = {i: {i} for i in range(n)}
    heights = {i: 0.0 for i in range(n)}
    parent_h = {}
    nodes = []
    for r, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b)
        nid = n + r
        members[nid] = members[a] | members[b]
        heights[nid] = h
        parent_h[a] = h
        parent_h[b] = h
        nodes.append(nid)
    root = n + len(Z) - 1
    parent_h[root] = Z[-1, 2] * 1.5 + 1e-9
    blocks = []
    for nid in nodes:
        if len(members[nid]) < 2:
            continue
        birth = heights[nid]
        death = parent_h.get(nid, birth)
        blocks.append({
            'node_id': nid, 'members': members[nid],
            'birth': birth, 'death': death,
            'persistence': death - birth,
            'rel_persistence': death / (birth + 1e-12),
            'size': len(members[nid]),
        })
    return blocks, n


# ===========================================================================
# METHOD 1: PERSISTENCE-GAP / KNEE-SELECTION
# ===========================================================================

def select_persistence_gap(Dstar, gap_sigma=2.0, max_size_frac=0.6):
    """
    Persistence-gap / knee-selection method (current approach).

    Algorithm:
    1. Compute persistence of all blocks
    2. Find threshold = median + gap_sigma * MAD-scaled-sigma
    3. Gate blocks as eligible if persistence exceeds threshold
    4. Greedy set-cover: iteratively select highest-persistence eligible block
       that covers most uncovered points until all covered
    5. Return # blocks selected (cluster count is OUTPUT, not input)

    Returns: (n_clusters, selected_blocks, metadata)
    """
    blocks, n = _all_blocks(Dstar)
    ceiling = max_size_frac * n
    persist = np.array([b['persistence'] for b in blocks])
    med = np.median(persist)
    mad = np.median(np.abs(persist - med)) + 1e-12
    sigma = 1.4826 * mad
    gap_threshold = med + gap_sigma * sigma

    elig = [b for b in blocks
            if b['size'] >= 3
            and b['size'] <= ceiling
            and b['persistence'] >= gap_threshold]

    if not elig:
        return 0, [], {
            'method': 'persistence_gap',
            'threshold': float(gap_threshold),
            'max_persistence': float(np.max(persist)),
            'reason': 'no_outliers'
        }

    covered = set()
    sel = []
    all_pts = set(range(n))
    while covered != all_pts:
        best, best_gain = None, 0
        for b in elig:
            if b in sel:
                continue
            gain = len(b['members'] - covered)
            if gain > best_gain or (gain == best_gain and best is not None
                                    and b['persistence'] > best['persistence']):
                best, best_gain = b, gain
        if best is None or best_gain == 0:
            break
        sel.append(best)
        covered |= best['members']

    return len(sel), sel, {
        'method': 'persistence_gap',
        'threshold': float(gap_threshold),
        'coverage': len(covered) / n,
        'gap_sigma': gap_sigma,
    }


# ===========================================================================
# METHOD 2: BETA-PLATEAU (Bonis & Oudot)
# ===========================================================================

def select_beta_plateau(Dstar, n_betas=20, plateau_patience=3, max_size_frac=0.6):
    """
    Beta-plateau method inspired by Bonis & Oudot (arXiv:1406.7130).

    Intuition (adapted to persistence framework):
    - Beta is a temperature-like parameter controlling attraction to peaks
    - For each beta, compute a "quality" score (e.g., cluster count from a
      varying threshold)
    - Identify the "plateau" where quality metric stabilizes
    - This plateau indicates robust cluster structure

    Approximation for hierarchy (not diffusion):
    1. Generate a series of candidate thresholds by exponentially decreasing
       a "temperature" analog
    2. For each threshold, count how many blocks exceed it (simulating what
       different beta values would select)
    3. Detect the "plateau" as where cluster count becomes stable
    4. Select the most conservative point in the plateau (fewest clusters)

    Returns: (n_clusters, selected_blocks, metadata)
    """
    blocks, n = _all_blocks(Dstar)
    ceiling = max_size_frac * n
    persist = np.array([b['persistence'] for b in blocks if 3 <= b['size'] <= ceiling])

    if len(persist) < 2:
        return 0, [], {'method': 'beta_plateau', 'reason': 'too_few_blocks'}

    persist_sorted = np.sort(persist)[::-1]
    med = np.median(persist_sorted)
    sigma = np.std(persist_sorted) + 1e-12

    # Generate beta-like sequence: temperature decreases, threshold increases
    # High temp = low threshold (accept many blocks)
    # Low temp = high threshold (accept few blocks)
    temps = np.linspace(2.0, 0.1, n_betas)
    cluster_counts = []
    thresholds = []

    for temp in temps:
        threshold = med - temp * sigma
        count = np.sum(persist_sorted >= threshold)
        cluster_counts.append(count)
        thresholds.append(threshold)

    # Detect plateau: consecutive cluster counts that are identical
    cluster_counts = np.array(cluster_counts)
    plateaus = []
    i = 0
    while i < len(cluster_counts):
        j = i
        while j < len(cluster_counts) and cluster_counts[j] == cluster_counts[i]:
            j += 1
        plateau_len = j - i
        if plateau_len >= plateau_patience:
            plateaus.append({
                'value': cluster_counts[i],
                'start': i,
                'length': plateau_len,
                'threshold': thresholds[i],
            })
        i = j

    if not plateaus:
        # No stable plateau found; pick the most common cluster count
        unique_counts, counts = np.unique(cluster_counts, return_counts=True)
        best_count = unique_counts[np.argmax(counts)]
        threshold_idx = np.where(cluster_counts == best_count)[0][0]
        selected_threshold = thresholds[threshold_idx]
    else:
        # Select the plateau with highest cluster count (most structure)
        # if there are multiple, use the one with longest duration
        best_plateau = max(plateaus, key=lambda p: (p['value'], p['length']))
        best_count = best_plateau['value']
        selected_threshold = best_plateau['threshold']

    # Apply threshold to select blocks
    blocks_full, _ = _all_blocks(Dstar)
    elig_blocks = [b for b in blocks_full
                   if 3 <= b['size'] <= ceiling
                   and b['persistence'] >= selected_threshold]

    if not elig_blocks:
        return 0, [], {
            'method': 'beta_plateau',
            'selected_threshold': float(selected_threshold),
            'reason': 'no_eligible_blocks'
        }

    # Greedy set-cover among eligible blocks
    covered = set()
    sel = []
    all_pts = set(range(n))
    while covered != all_pts:
        best, best_gain = None, 0
        for b in elig_blocks:
            if b in sel:
                continue
            gain = len(b['members'] - covered)
            if gain > best_gain or (gain == best_gain and best is not None
                                    and b['persistence'] > best['persistence']):
                best, best_gain = b, gain
        if best is None or best_gain == 0:
            break
        sel.append(best)
        covered |= best['members']

    return len(sel), sel, {
        'method': 'beta_plateau',
        'selected_threshold': float(selected_threshold),
        'plateaus_found': len(plateaus),
        'plateau_value': int(best_count) if plateaus else None,
        'cluster_counts_curve': cluster_counts.tolist(),
        'coverage': len(covered) / n,
    }


# ===========================================================================
# METHOD 3: BOTTLENECK-BOOTSTRAP (AuToMATo)
# ===========================================================================

def select_bottleneck_bootstrap(X, n_boots=100, boot_frac=0.8, max_size_frac=0.6):
    """
    Bottleneck-bootstrap method inspired by AuToMATo (arXiv:2408.06958).

    Algorithm:
    1. Bootstrap-resample data (with replacement)
    2. For each bootstrap sample:
       - Compute minimax transform D*
       - Extract all persistence values
    3. Identify gaps that appear stably across bootstraps
    4. Select the most robust/stable gap as the cluster count threshold

    Returns: (n_clusters, selected_blocks, metadata)
    """
    n = X.shape[0]
    n_boot_samples = int(n * boot_frac)

    gap_counts = {}  # gap_idx -> count across boots

    np.random.seed(42)  # for reproducibility
    for boot_idx in range(n_boots):
        # Bootstrap resample
        indices = np.random.choice(n, size=n_boot_samples, replace=True)
        X_boot = X[indices]

        # Compute D* for this boot
        try:
            D_boot = im.dissimilarity(X_boot)
            Ds_boot = im.minimax_transform(D_boot)
        except Exception:
            continue

        # Extract persistence values
        blocks_boot, n_boot = _all_blocks(Ds_boot)
        ceiling = max_size_frac * n_boot
        persist_boot = np.array([b['persistence'] for b in blocks_boot
                                 if 3 <= b['size'] <= ceiling])

        if len(persist_boot) < 2:
            continue

        # Find the largest gap in this boot
        diffs = persist_boot[:-1] / (persist_boot[1:] + 1e-9)
        if len(diffs) > 0:
            largest_gap_idx = int(np.argmax(diffs))
            gap_counts[largest_gap_idx] = gap_counts.get(largest_gap_idx, 0) + 1

    if not gap_counts:
        return 0, [], {'method': 'bottleneck_bootstrap', 'reason': 'no_valid_boots'}

    # Select the most common gap index
    most_stable_gap_idx = max(gap_counts, key=gap_counts.get)
    gap_frequency = gap_counts[most_stable_gap_idx] / n_boots

    # Convert gap index back to block count on full dataset
    D_full = im.dissimilarity(X)
    Ds_full = im.minimax_transform(D_full)
    blocks_full, n_full = _all_blocks(Ds_full)
    ceiling = max_size_frac * n_full

    persist_full = np.array([b['persistence'] for b in blocks_full
                             if 3 <= b['size'] <= ceiling])

    if len(persist_full) < 2:
        return 0, [], {'method': 'bottleneck_bootstrap', 'reason': 'too_few_blocks'}

    # Use the stable gap index to determine threshold
    if most_stable_gap_idx < len(persist_full) - 1:
        threshold = (persist_full[most_stable_gap_idx] + persist_full[most_stable_gap_idx + 1]) / 2
    else:
        threshold = persist_full[-1]

    elig_blocks = [b for b in blocks_full
                   if 3 <= b['size'] <= ceiling
                   and b['persistence'] >= threshold]

    if not elig_blocks:
        return 0, [], {'method': 'bottleneck_bootstrap', 'reason': 'no_eligible'}

    # Greedy set-cover
    covered = set()
    sel = []
    all_pts = set(range(n_full))
    while covered != all_pts:
        best, best_gain = None, 0
        for b in elig_blocks:
            if b in sel:
                continue
            gain = len(b['members'] - covered)
            if gain > best_gain or (gain == best_gain and best is not None
                                    and b['persistence'] > best['persistence']):
                best, best_gain = b, gain
        if best is None or best_gain == 0:
            break
        sel.append(best)
        covered |= best['members']

    return len(sel), sel, {
        'method': 'bottleneck_bootstrap',
        'most_stable_gap_idx': int(most_stable_gap_idx),
        'gap_frequency': float(gap_frequency),
        'n_boots': n_boots,
        'boot_frac': boot_frac,
        'selected_threshold': float(threshold),
        'coverage': len(covered) / n_full,
        'gap_frequency_dist': dict(sorted(gap_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
    }


# ===========================================================================
# UNIFIED INTERFACE FOR COMPARISON
# ===========================================================================

def compare_all_methods(X, y_true=None, verbose=False):
    """
    Run all three selection methods on the same data and return results.

    Args:
        X: data matrix (n, p)
        y_true: ground truth labels (for ARI scoring if available)
        verbose: print detailed results

    Returns:
        results_dict with entries for each method
    """
    D = im.dissimilarity(X)
    Ds = im.minimax_transform(D)

    results = {}

    # Method 1: Persistence-gap
    k1, sel1, meta1 = select_persistence_gap(Ds)
    results['persistence_gap'] = {
        'n_clusters': k1,
        'blocks': sel1,
        'coverage': meta1.get('coverage', 0),
        'metadata': meta1,
    }

    # Method 2: Beta-plateau
    k2, sel2, meta2 = select_beta_plateau(Ds)
    results['beta_plateau'] = {
        'n_clusters': k2,
        'blocks': sel2,
        'coverage': meta2.get('coverage', 0),
        'metadata': meta2,
    }

    # Method 3: Bottleneck-bootstrap
    k3, sel3, meta3 = select_bottleneck_bootstrap(X)
    results['bottleneck_bootstrap'] = {
        'n_clusters': k3,
        'blocks': sel3,
        'coverage': meta3.get('coverage', 0),
        'metadata': meta3,
    }

    if verbose:
        print(f"\nResults on dataset (n={len(X)}):")
        for method, res in results.items():
            print(f"  {method}: k={res['n_clusters']}, coverage={res['coverage']:.3f}")
        if y_true is not None:
            from sklearn.metrics import adjusted_rand_score
            for method, res in results.items():
                if res['blocks']:
                    lab = np.zeros(len(X), dtype=int)
                    for bi, b in enumerate(res['blocks']):
                        for idx in b['members']:
                            lab[idx] = bi
                    m = y_true >= 0
                    if m.sum() > 0:
                        ari = adjusted_rand_score(y_true[m], lab[m])
                        print(f"    ARI = {ari:.3f}")

    return results
