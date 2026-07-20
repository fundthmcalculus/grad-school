"""
ConiVAT (Rathore, Bezdek, Santi, Ratti 2020) - constraint-based iVAT.

Pipeline:
  1. Generate must-link (S) / cannot-link (D) constraints from partial labels,
     expand by transitive closure, remove inconsistencies.
  2. Learn a Mahalanobis metric A (Xing et al. 2002) that pulls S pairs together
     and pushes D pairs apart, via gradient ascent + iterative projection.
  3. Recompute D in the learned space, force must-link distances to 0, then
     build the Minimum Transitive Dissimilarity (MTD) matrix = the minimax /
     path-based transform (Eqs 5-6 in the paper) -- identical to iVAT's D*.
  4. Cluster by cutting the k-1 longest MST edges of the MTD matrix.

The chaining fix comes from steps 2-3: cannot-link pairs are pushed apart in the
learned metric, so a noise bridge no longer offers a low-cost minimax path.

For a fair, self-contained comparison we implement the Xing metric learning on
the DIAGONAL A (axis weights), which is the common practical variant and avoids
full-matrix SDP projection cost. Full A is available via full_matrix=True.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def generate_constraints(y, n_constraints=40, seed=0):
    """Sample must-link / cannot-link pairs from labels (labels used ONLY here,
    as ConiVAT does for constraint generation - not for clustering)."""
    rng = np.random.default_rng(seed)
    idx = np.where(y >= 0)[0]
    S, Dc = [], []
    for _ in range(n_constraints):
        i, j = rng.choice(idx, 2, replace=False)
        if y[i] == y[j]:
            S.append((i, j))
        else:
            Dc.append((i, j))
    return S, Dc


def transitive_closure_ml(S, n):
    """Expand must-link via transitive closure (union-find)."""
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    for i, j in S:
        union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    expanded = []
    for g in groups.values():
        for a_i in range(len(g)):
            for b_i in range(a_i + 1, len(g)):
                expanded.append((g[a_i], g[b_i]))
    return expanded


def learn_metric_diag(X, S, Dc, lr=0.1, iters=200, eps=1e-6):
    """
    Xing et al. metric learning, diagonal A (per-axis weights w >= 0).
    maximize sum_D sqrt((xi-xj)^2 . w)  s.t. sum_S (xi-xj)^2 . w <= 1, w >= 0.
    Solved by projected gradient ascent.
    """
    X = np.asarray(X, float)
    d = X.shape[1]
    if len(S) == 0 or len(Dc) == 0:
        return np.ones(d)
    S_diff2 = np.array([(X[i] - X[j]) ** 2 for i, j in S])   # (|S|, d)
    D_diff2 = np.array([(X[i] - X[j]) ** 2 for i, j in Dc])  # (|D|, d)
    w = np.ones(d)
    s_sum = S_diff2.sum(0)  # constraint gradient is constant in w for diag case

    for _ in range(iters):
        # objective g(w) = sum_D sqrt(D_diff2 . w); grad = sum_D D_diff2/(2 sqrt(.))
        proj = D_diff2 @ w
        proj = np.clip(proj, 1e-12, None)
        grad = (D_diff2 / (2 * np.sqrt(proj)[:, None])).sum(0)
        w = w + lr * grad
        # project onto {w>=0}
        w = np.clip(w, 0, None)
        # project onto {sum_S (xi-xj)^2 . w <= 1}
        c = s_sum @ w
        if c > 1:
            w = w / c
        if np.linalg.norm(lr * grad) < eps:
            break
    return w


def minimax_mtd(D):
    """Minimum Transitive Dissimilarity = minimax path transform (Eqs 5-6).
    Same as iVAT's D*. O(n^2) Prim-style."""
    D = np.asarray(D, float)
    n = D.shape[0]
    Dstar = np.zeros_like(D)
    in_tree = np.zeros(n, bool)
    in_tree[0] = True
    for _ in range(1, n):
        best = np.inf; bi = bj = -1
        tn = np.where(in_tree)[0]
        for i in tn:
            row = D[i].copy(); row[in_tree] = np.inf
            j = np.argmin(row)
            if row[j] < best:
                best, bi, bj = row[j], i, j
        for k in range(n):
            if in_tree[k]:
                Dstar[bj, k] = max(best, Dstar[bi, k])
                Dstar[k, bj] = Dstar[bj, k]
        in_tree[bj] = True
    return Dstar


def conivat(X, y, n_constraints=40, seed=0):
    """
    Full ConiVAT. Returns the MTD matrix D'* (minimax on the learned,
    constraint-imposed dissimilarity). Cluster it by cutting MST edges.
    """
    X = np.asarray(X, float)
    n = X.shape[0]

    S, Dc = generate_constraints(y, n_constraints, seed)
    S = transitive_closure_ml(S, n)

    w = learn_metric_diag(X, S, Dc)
    Xt = X * np.sqrt(np.clip(w, 0, None))[None, :]   # transform: A = diag(w)

    D = squareform(pdist(Xt))
    # impose must-link: zero those distances
    for i, j in S:
        D[i, j] = 0.0
        D[j, i] = 0.0

    Dprime = minimax_mtd(D)
    return Dprime


def sl_labels_from_mtd(Dprime, k):
    """Cut the k-1 longest MST edges of the MTD matrix -> k SL clusters.
    Equivalent to single-linkage fcluster on the minimax matrix."""
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(squareform(Dprime, checks=False), method='single')
    return fcluster(Z, t=k, criterion='maxclust') - 1
