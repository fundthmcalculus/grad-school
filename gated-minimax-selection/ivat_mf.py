"""
iVAT -> Membership Function test battery.

Core module: the iVAT (minimax path) transform and the two candidate
mappings from the reordered dissimilarity structure to fuzzy membership
functions.

Mapping 1 (naive)  : monotone-decreasing function of minimax distance to a medoid.
Mapping 2 (defended): persistence-of-block membership derived from the single-linkage
                      dendrogram (birth/death heights of each cluster).

Everything is dependency-light: numpy + scipy only.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage


# ---------------------------------------------------------------------------
# iVAT core
# ---------------------------------------------------------------------------

def minimax_transform(D):
    """
    Compute the all-pairs minimax (bottleneck) path distance matrix D*.

    D*_ij = min over paths p from i to j of ( max edge weight on p ).

    For a full dissimilarity matrix this equals the largest edge on the unique
    MST path between i and j. We compute it directly with a Prim-style
    recurrence (the same recurrence the efficient iVAT formulation uses),
    which is O(n^2).

    This is exactly the iVAT distance transform; the iVAT *image* is just
    squareform(D*) reordered by the VAT ordering.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    Dstar = np.zeros_like(D)

    # Prim's MST from node 0, recording the bottleneck value at which each
    # node is connected. The efficient iVAT recurrence: when we attach node j
    # via the current frontier, D*[j, prev] = the connecting edge, and
    # D*[j, others] = max(edge, D*[prev, others]).
    connected = [0]
    # distance from tree to each node, and which tree-node achieves it
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True

    for _ in range(1, n):
        # find the minimum edge from the tree to any node not in the tree
        best_edge = np.inf
        best_j = -1
        best_i = -1
        tree_nodes = np.where(in_tree)[0]
        for i in tree_nodes:
            for j in range(n):
                if not in_tree[j] and D[i, j] < best_edge:
                    best_edge = D[i, j]
                    best_j = j
                    best_i = i
        # attach best_j via best_i
        for k in range(n):
            if in_tree[k]:
                Dstar[best_j, k] = max(best_edge, Dstar[best_i, k])
                Dstar[k, best_j] = Dstar[best_j, k]
        in_tree[best_j] = True

    return Dstar


def vat_order(D):
    """
    Return the VAT ordering (Prim MST visitation order) of the objects.
    Used only for producing the reordered image; not needed for the mappings.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    # VAT seeds at the pair with the maximum dissimilarity; standard choice is
    # the row containing the global max.
    i0 = np.unravel_index(np.argmax(D), D.shape)[0]
    order = [i0]
    in_tree = np.zeros(n, dtype=bool)
    in_tree[i0] = True
    for _ in range(1, n):
        best = np.inf
        best_j = -1
        for i in order:
            row = D[i].copy()
            row[in_tree] = np.inf
            j = np.argmin(row)
            if row[j] < best:
                best = row[j]
                best_j = j
        order.append(best_j)
        in_tree[best_j] = True
    return np.array(order)


# ---------------------------------------------------------------------------
# Mapping 1: naive medoid / minimax-distance fuzzification
# ---------------------------------------------------------------------------

def mapping1_medoid(Dstar, medoid_indices):
    """
    mu_k(x) = 1 - D*(x, v_k) / sum_j D*(x, v_j)   (normalized, monotone-decreasing)

    Returns U of shape (n_clusters, n_points), columns not forced to sum to 1
    but each membership in [0,1].
    """
    Dstar = np.asarray(Dstar, dtype=float)
    meds = np.asarray(medoid_indices, dtype=int)
    # distance from every point to each medoid
    dist_to_med = Dstar[meds, :]              # (c, n)
    eps = 1e-12
    denom = dist_to_med.sum(axis=0, keepdims=True) + eps
    U = 1.0 - dist_to_med / denom
    U = np.clip(U, 0.0, 1.0)
    return U


# ---------------------------------------------------------------------------
# Mapping 2: persistence-of-block membership from the SL dendrogram
# ---------------------------------------------------------------------------

def single_linkage_from_Dstar(Dstar):
    """
    Single-linkage dendrogram. Because D* is the minimax transform, SL on the
    ORIGINAL D and SL on D* give the same merge structure; we run it on the
    condensed original-equivalent form via D* to keep heights in minimax units.
    """
    condensed = squareform(Dstar, checks=False)
    Z = linkage(condensed, method='single')
    return Z


def _cluster_members_at_merge(Z, n):
    """
    Walk the linkage matrix and record, for each internal node (merge),
    the set of leaf members and the height at which it formed.
    Returns list of dicts: {members:set, birth:float, node_id:int}.
    Node ids follow scipy convention (leaves 0..n-1, merges n..2n-2).
    """
    members = {i: {i} for i in range(n)}
    heights = {i: 0.0 for i in range(n)}
    nodes = []
    for r, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b)
        node_id = n + r
        mem = members[a] | members[b]
        members[node_id] = mem
        heights[node_id] = h
        nodes.append({'members': mem, 'birth': min(heights[a], 0.0) if False else heights[a],
                      'merge_height': h, 'node_id': node_id,
                      'child_births': (heights[a], heights[b])})
    return members, heights, nodes


def mapping2_persistence(Dstar, n_clusters, min_persistence=0.0):
    """
    Persistence-based membership.

    Idea: select the c most persistent blocks in the single-linkage hierarchy.
    A block is "born" when its two children merge (its formation height) and
    "dies" when it merges into its parent. Persistence = death - birth.

    For a selected block C_k with birth height h_b and death height h_d:

        d_k(x) = min over y in C_k of D*(x, y)      # minimax dist to the block
        mu_k(x) = clip( (h_d - d_k(x)) / (h_d - h_b + eps), 0, 1 )

    Points inside the block (d_k <= h_b) -> membership 1.
    Points that only reach the block above its death height -> membership 0.
    Graded in between, with the grade set by *where in the block's lifetime*
    the point attaches. This is the iVAT-native shape.
    """
    Dstar = np.asarray(Dstar, dtype=float)
    n = Dstar.shape[0]
    Z = single_linkage_from_Dstar(Dstar)
    members, heights, nodes = _cluster_members_at_merge(Z, n)

    # death height of a node = the merge height of its PARENT.
    # Build parent map.
    parent_height = {}
    for r, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b)
        node_id = n + r
        parent_height[a] = h
        parent_height[b] = h
    root = n + len(Z) - 1
    parent_height[root] = Z[-1, 2] * 1.5 + 1e-9  # root dies at +inf; use a cap

    # candidate internal nodes with >= 2 members
    cands = []
    for nd in nodes:
        nid = nd['node_id']
        birth = nd['merge_height']         # height at which this block formed
        death = parent_height.get(nid, birth)
        persistence = death - birth
        cands.append({'node_id': nid, 'members': nd['members'],
                      'birth': birth, 'death': death,
                      'persistence': persistence, 'size': len(nd['members'])})

    # pick the c most persistent blocks whose persistence exceeds threshold,
    # BUT require the selected blocks to be mutually disjoint (an antichain in
    # the tree). Selecting nested ancestor/descendant blocks would just give
    # the whole set multiple times. Greedy: take highest-persistence blocks,
    # skipping any that overlap an already-selected block.
    cands = [c for c in cands if c['persistence'] > min_persistence and c['size'] >= 2]
    # Scale-invariant selection. Two guards against the failure modes found in
    # testing: (1) rank by RELATIVE persistence (death/birth ratio) so clusters
    # at different absolute height scales compete fairly; (2) exclude blocks
    # containing more than a fraction of all points, so the near-root block
    # (essentially the whole dataset) can never be selected as "a cluster".
    size_ceiling = 0.9 * n
    for c in cands:
        c['score'] = c['death'] / (c['birth'] + 1e-12)
    cands = [c for c in cands if c['size'] <= size_ceiling]
    cands.sort(key=lambda c: c['score'], reverse=True)
    selected = []
    for cand in cands:
        if any(cand['members'] & s['members'] for s in selected):
            continue  # overlaps an already chosen block -> skip
        selected.append(cand)
        if len(selected) == n_clusters:
            break

    if len(selected) == 0:
        # degenerate; fall back to whole-set
        selected = sorted(cands, key=lambda c: c['size'], reverse=True)[:n_clusters]

    U = np.zeros((len(selected), n))
    Dblock = np.zeros((len(selected), n))   # minimax distance to each block
    eps = 1e-12
    for k, blk in enumerate(selected):
        mem = np.array(sorted(blk['members']), dtype=int)
        d_to_block = Dstar[:, mem].min(axis=1)   # (n,)
        Dblock[k] = d_to_block
        h_b, h_d = blk['birth'], blk['death']
        mu = (h_d - d_to_block) / (h_d - h_b + eps)
        mu = np.clip(mu, 0.0, 1.0)
        core = d_to_block <= h_b + eps
        mu[core] = 1.0
        U[k] = mu
    info = selected
    return U, info, Dblock


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def dissimilarity(X, metric='euclidean'):
    return squareform(pdist(X, metric=metric))


def hard_labels_from_U(U):
    """Argmax defuzzification -> hard labels for scoring against ground truth."""
    return np.argmax(U, axis=0)


def hard_labels_proximity(U, Dblock):
    """
    Defuzzify by assigning each point to the block it is closest to in minimax
    distance, but only among blocks where its membership is nonzero. This fixes
    the saturation problem: when several MFs read 1.0 at a point (because their
    death heights exceed the global scale), argmax is arbitrary, so we break the
    tie by true minimax proximity to the block core.
    """
    U = np.asarray(U)
    Dblock = np.asarray(Dblock)
    n = U.shape[1]
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        active = np.where(U[:, i] > 1e-6)[0]
        if len(active) == 0:
            labels[i] = int(np.argmin(Dblock[:, i]))
        else:
            labels[i] = int(active[np.argmin(Dblock[active, i])])
    return labels
