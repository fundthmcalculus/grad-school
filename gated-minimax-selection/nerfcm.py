"""
NERFCM - Non-Euclidean Relational Fuzzy C-Means (Hathaway & Bezdek).

Relational fuzzy clustering that operates directly on a dissimilarity matrix D
(no vector coordinates needed). The "NE" part is the beta-spread transform:
when D is not Euclidean-embeddable, the relational-dual update can produce
negative "distances"; NERFCM detects this and applies a self-adaptive spread
D_beta = D + beta*(1 - I) that restores admissibility, updating beta as needed.

Reference: Hathaway & Bezdek, "NERF c-means: Non-Euclidean relational fuzzy
clustering," Pattern Recognition 27(3):429-437, 1994.

Fair-baseline notes for the comparison:
  - c must be supplied (NERFCM has no cluster-count discovery).
  - memberships are a probabilistic partition (columns sum to 1).
  - it will always produce c clusters, including on noise.
"""

import numpy as np


def nerfcm(D, c, m=2.0, max_iter=100, tol=1e-5, seed=0, verbose=False):
    """
    Parameters
    ----------
    D : (n,n) symmetric dissimilarity matrix, zero diagonal.
    c : number of clusters.
    m : fuzzifier (>1).
    Returns
    -------
    U : (c, n) membership matrix, columns sum to 1.
    beta : final beta-spread value (0 if D was already Euclidean-admissible).
    n_iter : iterations run.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    rng = np.random.default_rng(seed)

    # init memberships randomly, normalized per column
    U = rng.random((c, n))
    U /= U.sum(axis=0, keepdims=True)

    beta = 0.0
    Dbeta = D.copy()

    def relational_update(U, Dbeta):
        """One RFCM relational update; returns new distances d (c,n) and new V."""
        Um = U ** m                       # (c,n)
        # v_i = Um_i / sum(Um_i)  (relational "membership vector" per cluster)
        denom = Um.sum(axis=1, keepdims=True)          # (c,1)
        V = Um / denom                                  # (c,n), each row sums to 1

        # d_ik = (D v_i)_k - 0.5 * v_i^T D v_i
        d = np.zeros((c, n))
        for i in range(c):
            vi = V[i]                                   # (n,)
            Dv = Dbeta @ vi                             # (n,)
            quad = 0.5 * (vi @ Dv)
            d[i] = Dv - quad
        return d, V

    for it in range(max_iter):
        d, V = relational_update(U, Dbeta)

        # NE fix: if any relational distance is negative, increase beta enough
        # to make all d >= 0, and re-derive distances under the new spread.
        dmin = d.min()
        if dmin < 0:
            # find the increment to beta that restores non-negativity.
            # Under D_beta = D + beta*(J - I), each d_ik increases by
            # beta * (1 - 0.5 * sum_j v_ij^2)  (see Hathaway-Bezdek).
            # Compute the max required delta over all (i,k) with d<0.
            deltas = []
            for i in range(c):
                vi = V[i]
                factor = 1.0 - 0.5 * np.sum(vi ** 2)
                factor = max(factor, 1e-9)
                neg = d[i][d[i] < 0]
                if len(neg) > 0:
                    deltas.append((-neg.min()) / factor)
            dbeta_inc = max(deltas) if deltas else 0.0
            beta += dbeta_inc * 1.0001
            Dbeta = D + beta * (np.ones((n, n)) - np.eye(n))
            d, V = relational_update(U, Dbeta)
            d = np.clip(d, 0.0, None)

        # membership update from distances
        d = np.clip(d, 1e-12, None)
        # u_ik = 1 / sum_j (d_ik/d_jk)^(1/(m-1))
        power = 1.0 / (m - 1.0)
        Unew = np.zeros((c, n))
        for k in range(n):
            dk = d[:, k]
            ratios = (dk[:, None] / dk[None, :]) ** power   # (c,c)
            Unew[:, k] = 1.0 / ratios.sum(axis=1)
        Unew /= Unew.sum(axis=0, keepdims=True)

        change = np.abs(Unew - U).max()
        U = Unew
        if change < tol:
            if verbose:
                print(f"converged in {it+1} iters, beta={beta:.4f}")
            return U, beta, it + 1

    return U, beta, max_iter
