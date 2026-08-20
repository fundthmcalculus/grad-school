"""Shared TSP primitives: distances, candidate lists, constructions, reversal.

Everything both solver arms stand on lives here, so the comparison between the
baseline Lin-Kernighan and the FIS strategy engine differs only in *strategy* —
never in the speed of the underlying arithmetic. All hot code is nopython-jitted
and operates on coordinates directly (no O(n^2) matrix), which is what lets the
same code run at n=52 and n=85900.

Distances follow TSPLIB exactly: ``nint(euclidean)`` for EUC_2D, ``ceil`` for
CEIL_2D. A solver that optimised un-rounded distances would report lengths that
disagree with the published optima, so the rounding is inside the inner loop.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------


@njit(cache=True, inline="always")
def dist_raw(coords, a, b):
    """Un-rounded euclidean distance — used only to break ties that rounding created."""
    dx = coords[a, 0] - coords[b, 0]
    dy = coords[a, 1] - coords[b, 1]
    return np.sqrt(dx * dx + dy * dy)


@njit(cache=True, inline="always")
def dist(coords, a, b, ceil):
    """TSPLIB integer distance between cities a and b."""
    dx = coords[a, 0] - coords[b, 0]
    dy = coords[a, 1] - coords[b, 1]
    r = np.sqrt(dx * dx + dy * dy)
    if ceil:
        return np.ceil(r)
    return np.floor(r + 0.5)


@njit(cache=True)
def tour_length(tour, coords, ceil):
    """Closed-tour length under TSPLIB rounding."""
    n = tour.shape[0]
    total = 0.0
    for i in range(n):
        total += dist(coords, tour[i], tour[(i + 1) % n], ceil)
    return total


# ---------------------------------------------------------------------------
# candidate lists
# ---------------------------------------------------------------------------


@njit(cache=True)
def _cand_dist_kernel(coords, cand, ceil, out):
    n, k = cand.shape
    for i in range(n):
        for t in range(k):
            out[i, t] = dist(coords, i, cand[i, t], ceil)


def build_candidates(coords, k, ceil):
    """The k nearest neighbours of every city, ascending by distance.

    A k-d tree is used because it is the same C implementation for both arms and
    is a small fraction of either arm's runtime.

    Getting the tie handling right took three attempts, and it matters because the
    baseline is reported as a sweep over k — so anything that makes the k=8 list
    disagree with the first 8 of the k=24 list turns that sweep into a measurement of
    tie-break luck. TSPLIB instances are full of ties: exact ones (in pr1002, cities
    326 and 328 are both at distance 150 from 327) and, because distances are rounded
    to integers, many more that rounding creates. What is needed is
    (a) a deterministic key, since the tree's own order is arbitrary;
    (b) enough queried neighbours that a tie group straddling the k-th place is fully
        seen, since otherwise the returned *set* is k-dependent even if the key is not;
    (c) the same key the exact-scan fallback in :func:`nn_tour` uses, or the two paths
        disagree whenever the fallback fires.
    See the ordering comment below for the key, and ``test_invariants.py`` for the
    checks that hold all three.

    Returns (cand (n,k) int32, cand_d (n,k) float64) — the neighbour indices and
    the rounded distances to them, which is what the fuzzy features scale by.
    """
    n = coords.shape[0]
    k = min(k, n - 1)
    tree = cKDTree(coords)

    # Order by (euclidean distance, city index). Both are intrinsic to the pair, so the
    # k nearest under this key is well defined independently of how many neighbours were
    # queried — which is what makes the k=8 list a prefix of the k=24 list. Note this is
    # also consistent with the *rounded* metric the solver optimises and that
    # `nn_tour`'s exact-scan fallback compares, because rounding is monotone: a smaller
    # euclidean distance can never round to a larger integer. So the two paths agree.
    kq = min(n - 1, k + 16)
    dd, idx = tree.query(coords, k=kq + 1, workers=-1)
    # drop the point itself; for coincident points its index can land anywhere in the
    # tied block, so mask by index rather than assuming column 0
    dd = np.where(idx == np.arange(n)[:, None], np.inf, dd)
    order = np.lexsort((idx, dd), axis=1)
    dd_s = np.take_along_axis(dd, order, axis=1)
    cand = np.take_along_axis(idx, order, axis=1)[:, :k].astype(np.int32)

    # A tie group straddling the k-th place is the one case the window above cannot
    # settle: the tree returns *some* members of the group, not necessarily the
    # lowest-indexed ones, so which of them lands in the list would depend on kq.
    #
    # Widening the window until no row straddles is not a workable fix — these ties are
    # far more common than they sound (933 of 5915 rows in rl5915, whose coordinates are
    # grid-like), so the loop keeps doubling until it is querying every city, which cost
    # a factor of ~90 in wall clock when it was tried. Instead the contested rows are
    # settled individually and exactly: a ball of the k-th distance contains the whole
    # tie group by construction, so sorting that ball by the same key gives the right
    # answer in one query per row.
    if kq < n - 1:
        contested = np.flatnonzero(dd_s[:, k - 1] == dd_s[:, k])
        for i in contested:
            radius = float(dd_s[i, k - 1])
            ball = [
                j
                for j in tree.query_ball_point(coords[i], radius * (1 + 1e-9))
                if j != i
            ]
            if len(ball) < k:
                continue  # window already held everything within the radius
            ball = np.array(ball)
            d = np.hypot(coords[ball, 0] - coords[i, 0], coords[ball, 1] - coords[i, 1])
            cand[i] = ball[np.lexsort((ball, d))][:k]
    cand = np.ascontiguousarray(cand)
    cand_d = np.zeros(cand.shape, dtype=np.float64)
    _cand_dist_kernel(coords, cand, ceil, cand_d)
    return cand, cand_d


# ---------------------------------------------------------------------------
# scale statistics used to normalise every fuzzy input
# ---------------------------------------------------------------------------


@njit(cache=True)
def nn_stats(cand_d):
    """(nn1, mean_cand) per city: distance to the nearest neighbour, and the mean
    distance over the candidate list.

    Every fuzzy antecedent is a *ratio* against one of these, which is what makes
    one tuned rule base transfer across instances whose coordinates differ by
    five orders of magnitude.
    """
    n, k = cand_d.shape
    nn1 = np.empty(n, np.float64)
    mean_c = np.empty(n, np.float64)
    for i in range(n):
        nn1[i] = cand_d[i, 0]
        s = 0.0
        for t in range(k):
            s += cand_d[i, t]
        mean_c[i] = s / k
    # a coincident-point instance can have nn1 == 0; keep the ratios finite
    for i in range(n):
        if nn1[i] <= 0.0:
            nn1[i] = 1e-9
        if mean_c[i] <= 0.0:
            mean_c[i] = 1e-9
    return nn1, mean_c


# ---------------------------------------------------------------------------
# tour representation helpers
# ---------------------------------------------------------------------------


@njit(cache=True)
def make_pos(tour):
    n = tour.shape[0]
    pos = np.empty(n, np.int32)
    for i in range(n):
        pos[tour[i]] = i
    return pos


@njit(cache=True, inline="always")
def succ(tour, pos, n, city):
    return tour[(pos[city] + 1) % n]


@njit(cache=True, inline="always")
def pred(tour, pos, n, city):
    return tour[(pos[city] - 1 + n) % n]


@njit(cache=True)
def reverse_exact(tour, pos, n, i, j):
    """Reverse exactly the cyclic segment of positions i..j — never the complement.

    :func:`reverse` is free to flip the complement instead, which preserves the
    tour's edges but not the array layout. Block-swap surgery composes three
    reversals and reasons about *array contents*, so it needs the literal one.
    """
    inside = (j - i + n) % n + 1
    for s in range(inside // 2):
        a = (i + s) % n
        b = (j - s + n) % n
        ca = tour[a]
        cb = tour[b]
        tour[a] = cb
        tour[b] = ca
        pos[cb] = a
        pos[ca] = b
    return inside // 2


@njit(cache=True)
def block_swap(tour, pos, n, i, j, m, rev_a, rev_b):
    """Swap the two adjacent blocks A = positions i..j and B = positions j+1..m.

    Afterwards the span i..m holds B' then A', where A' is A reversed iff
    ``rev_a`` and B' is B reversed iff ``rev_b``. Uses the identity
    (A B)^R = B^R A^R, so pre-reversing a block cancels the outer reversal for it.

    This is the surgery behind an Or-opt move: relocating a short segment is a
    swap of that segment with the run of cities between it and its destination.
    """
    w = 0
    if not rev_a:
        w += reverse_exact(tour, pos, n, i, j)
    if not rev_b:
        w += reverse_exact(tour, pos, n, (j + 1) % n, m)
    w += reverse_exact(tour, pos, n, i, m)
    return w


@njit(cache=True, inline="always")
def span_len(n, i, j):
    """Number of positions in the cyclic span i..j inclusive."""
    return (j - i + n) % n + 1


@njit(cache=True)
def reverse(tour, pos, n, i, j):
    """Reverse the cyclic segment of positions i..j inclusive.

    Reverses whichever of the segment and its complement is shorter — the two
    give the same set of tour edges, so this is free correctness-wise and is what
    keeps a 2-opt move sub-linear on average instead of O(n).

    Returns the number of element swaps performed. That count is the solver's real
    memory-traffic cost and the single hardest part of its runtime to predict from
    move counts alone, so it is accumulated into the stats and used by the cost
    model in ``costmodel.py``.
    """
    inside = (j - i + n) % n + 1
    if 2 * inside > n:  # complement is shorter
        i, j = (j + 1) % n, (i - 1 + n) % n
        inside = n - inside
    for s in range(inside // 2):
        a = (i + s) % n
        b = (j - s + n) % n
        ca = tour[a]
        cb = tour[b]
        tour[a] = cb
        tour[b] = ca
        pos[cb] = a
        pos[ca] = b
    return inside // 2


# ---------------------------------------------------------------------------
# constructions
# ---------------------------------------------------------------------------


@njit(cache=True)
def nn_tour(coords, cand, ceil, start=0):
    """Nearest-neighbour tour, candidate list first and an exact scan as backup.

    The candidate list covers the common case in O(k); when every candidate of
    the current city is already visited the nearest unvisited city is found by
    scanning the compacted unvisited array, so the tour is a true
    nearest-neighbour tour and not a candidate-list approximation of one.
    """
    n = coords.shape[0]
    k = cand.shape[1]
    visited = np.zeros(n, np.uint8)
    tour = np.empty(n, np.int32)
    # unvisited[0:m] holds the cities still to place; where[c] indexes into it
    unvisited = np.empty(n, np.int32)
    where = np.empty(n, np.int32)
    for i in range(n):
        unvisited[i] = i
        where[i] = i
    m = n

    def_cur = start
    tour[0] = def_cur
    visited[def_cur] = 1
    # remove def_cur from the unvisited array
    last = unvisited[m - 1]
    p = where[def_cur]
    unvisited[p] = last
    where[last] = p
    m -= 1

    cur = def_cur
    for step in range(1, n):
        nxt = -1
        for t in range(k):
            c = cand[cur, t]
            if visited[c] == 0:
                nxt = c
                break
        if nxt < 0:  # exact fallback over what is left
            best = 1e300
            best_raw = 1e300
            for u in range(m):
                c = unvisited[u]
                d = dist(coords, cur, c, ceil)
                if d > best:
                    continue
                # Same three keys the candidate list is ordered by: rounded distance,
                # then true distance, then index. Without this the two paths break
                # rounding ties differently and the tour depends on k through which
                # steps happen to take this branch.
                r = dist_raw(coords, cur, c)
                if d < best or r < best_raw or (r == best_raw and c < nxt):
                    best = d
                    best_raw = r
                    nxt = c
        tour[step] = nxt
        visited[nxt] = 1
        last = unvisited[m - 1]
        p = where[nxt]
        unvisited[p] = last
        where[last] = p
        m -= 1
        cur = nxt
    return tour


@njit(cache=True)
def greedy_edge_tour(coords, cand, ceil):
    """Greedy-edge (Christofides' "greedy") construction over the candidate graph.

    Shortest edges first, accepting an edge when both ends have degree < 2 and it
    closes no premature subtour; leftover path ends are then joined by an exact
    nearest-feasible-end scan. Typically ~10-15%% over optimum versus ~25%% for
    nearest-neighbour, and it is the strongest construction available from a
    candidate list alone — so it is the honest baseline start, not a strawman.
    """
    n = coords.shape[0]
    k = cand.shape[1]
    # collect candidate edges (i<j deduplicated), sort by length
    cap = n * k
    eu = np.empty(cap, np.int32)
    ev = np.empty(cap, np.int32)
    ew = np.empty(cap, np.float64)
    m = 0
    for i in range(n):
        for t in range(k):
            j = cand[i, t]
            if j > i:
                eu[m] = i
                ev[m] = j
                ew[m] = dist(coords, i, j, ceil)
                m += 1
            elif j < i:
                # keep only if i is not in j's list, else it is a duplicate
                dup = False
                for s in range(k):
                    if cand[j, s] == i:
                        dup = True
                        break
                if not dup:
                    eu[m] = j
                    ev[m] = i
                    ew[m] = dist(coords, i, j, ceil)
                    m += 1
    order = np.argsort(ew[:m])

    deg = np.zeros(n, np.int32)
    link = np.full((n, 2), -1, np.int32)
    # union-find over path fragments
    parent = np.empty(n, np.int32)
    for i in range(n):
        parent[i] = i

    accepted = 0
    for oi in range(m):
        if accepted == n - 1:
            break
        e = order[oi]
        a = eu[e]
        b = ev[e]
        if deg[a] >= 2 or deg[b] >= 2:
            continue
        # find roots
        ra = a
        while parent[ra] != ra:
            ra = parent[ra]
        rb = b
        while parent[rb] != rb:
            rb = parent[rb]
        if ra == rb:
            continue  # would close a subtour early
        parent[ra] = rb
        link[a, deg[a]] = b
        link[b, deg[b]] = a
        deg[a] += 1
        deg[b] += 1
        accepted += 1

    # Join the leftover fragment ends. Only cities of degree < 2 can take part,
    # and after a candidate-list greedy pass there are few of them, so the
    # nearest-feasible-pair scan runs over that short list rather than over all
    # n cities.
    ends = np.empty(2 * (n - accepted) + 2, np.int32)
    n_ends = 0
    for a in range(n):
        if deg[a] < 2 and n_ends < ends.shape[0]:
            ends[n_ends] = a
            n_ends += 1

    while accepted < n - 1:
        best_d = 1e300
        ba = -1
        bb = -1
        for ia in range(n_ends):
            a = ends[ia]
            if deg[a] >= 2:
                continue
            ra = a
            while parent[ra] != ra:
                ra = parent[ra]
            for ib in range(n_ends):
                b = ends[ib]
                if b == a or deg[b] >= 2:
                    continue
                rb = b
                while parent[rb] != rb:
                    rb = parent[rb]
                if ra == rb:
                    continue
                d = dist(coords, a, b, ceil)
                if d < best_d:
                    best_d = d
                    ba = a
                    bb = b
        if ba < 0:
            break
        ra = ba
        while parent[ra] != ra:
            ra = parent[ra]
        rb = bb
        while parent[rb] != rb:
            rb = parent[rb]
        parent[ra] = rb
        link[ba, deg[ba]] = bb
        link[bb, deg[bb]] = ba
        deg[ba] += 1
        deg[bb] += 1
        accepted += 1

    # close the single remaining path into a tour
    ends = np.empty(2, np.int32)
    ne = 0
    for a in range(n):
        if deg[a] < 2 and ne < 2:
            ends[ne] = a
            ne += 1
    if ne == 2:
        link[ends[0], deg[ends[0]]] = ends[1]
        link[ends[1], deg[ends[1]]] = ends[0]
        deg[ends[0]] += 1
        deg[ends[1]] += 1

    # walk it
    tour = np.empty(n, np.int32)
    prev = -1
    cur = 0
    for i in range(n):
        tour[i] = cur
        nxt = link[cur, 0]
        if nxt == prev or nxt < 0:
            nxt = link[cur, 1]
        prev = cur
        cur = nxt
        if cur < 0:
            break
    return tour
