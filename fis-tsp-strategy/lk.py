"""A baseline Lin-Kernighan solver — the thing the FIS strategy engine must beat.

This is the standard LK step, implemented the way it is normally implemented on an
array tour (Johnson & McGeoch's "LK as a chain of 2-opt moves"):

  * break the tour edge (t1, t2) at an anchor city t1;
  * add y_i = (t2, u) for a candidate u of the current free end, subject to the
    sequential positive-gain criterion  G_{i-1} - |y_i| > 0;
  * break x_{i+1} = (u, t4) where t4 is the tour-neighbour of u on the far side,
    which is exactly a 2-opt move, and apply it — so the array always holds the
    tour you would get by closing up at this depth;
  * recurse to ``max_depth``, remembering the depth whose closing gain was best,
    and unwind the reversals back to it.

Backtracking is done at the first level over the top ``breadth`` candidates, and
the deeper levels take the best candidate they can see (LKH's ordering rule:
prefer the chain that breaks a long edge for a short one). Don't-look bits drive
a work queue so that converged regions are not rescanned.

``breadth``, ``deep_breadth`` and ``max_depth`` are the parameters the FIS engine
later sets per city instead of globally, and ``use_chain`` hands the
deepen-or-cut decision to a rule base instead of the fixed cut-off. Both arms call
this same module: the baseline is this file with ``use_chain`` false and constant
parameters, so the comparison isolates the strategy and not the arithmetic, and no
result can come from two subtly different LK implementations.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from fis import (
    CHAIN_MF_C as NO_MF_C,
    CHAIN_MF_S as NO_MF_S,
    NO_CHAIN_ANT,
    NO_CHAIN_CONS,
    fis_eval1,
)
from core import (
    block_swap,
    dist,
    make_pos,
    pred,
    reverse,
    span_len,
    succ,
    tour_length,
)

# Reported alongside the tour so the findings can say *why* one arm is faster.
# 0: accepted moves, 1: candidate evaluations, 2: city scans, 3: reversal elements
STAT_MOVES = 0
STAT_EVALS = 1
STAT_SCANS = 2
STAT_DEPTH = 3
STAT_CHAIN_CALLS = 4  # continuation decisions taken by the CHAIN rule base
N_STATS = 5


@njit(cache=True, inline="always")
def _free_end_dir(tour, pos, n, t1, t2):
    """1 if t2 is the successor of t1, -1 if the predecessor, 0 if neither.

    The shorter-side reversal in :func:`core.reverse` may reverse a segment's
    complement instead, which yields the same tour edges but can flip the array's
    orientation. So the chain re-reads the direction from the array at every level
    rather than assuming the one it started with.
    """
    if tour[(pos[t1] + 1) % n] == t2:
        return 1
    if tour[(pos[t1] - 1 + n) % n] == t2:
        return -1
    return 0


@njit(cache=True)
def _lk_chain(
    tour,
    pos,
    n,
    coords,
    cand,
    cand_d,
    ceil,
    t1,
    t2,
    u0,
    g0,
    scale,
    max_depth,
    deep_breadth,
    rev_i,
    rev_j,
    touched,
    stats,
    use_chain,
    ch_mf_c,
    ch_mf_s,
    ch_ant,
    ch_cons,
):
    """Run one LK gain chain anchored at (t1, t2) whose first added edge is
    (t2, u0), with ``g0`` the gain after breaking (t1,t2) and adding (t2,u0).

    Applies 2-opt moves as it descends, then unwinds to the best depth. Returns
    (gain, n_touched): a positive gain means the tour was improved and the first
    ``n_touched`` entries of ``touched`` are the cities whose edges changed.

    With ``use_chain`` the decision to descend one more level is taken by the CHAIN
    rule base from the chain's own gain trajectory, instead of running to the fixed
    ``max_depth``. Both arms share this one function, so the move semantics they
    explore are identical by construction and the comparison cannot be confounded by
    two subtly different LK implementations. ``scale`` is the length of the first
    broken edge, which every fuzzy input here is expressed as a fraction of.
    """
    k = cand.shape[1]
    g = g0
    cur_t2 = t2
    u = u0
    level = 0
    best_total = 0.0
    best_level = 0  # number of reversals to keep; 0 means "no improvement"
    n_touched = 0
    g_next = 0.0
    nxt_score = 0.0
    xc = np.empty(4, np.float64)

    while level < max_depth:
        d_now = _free_end_dir(tour, pos, n, t1, cur_t2)
        if d_now == 0:  # invariant lost — should not happen; bail out safely
            break
        if d_now > 0:
            t4 = pred(tour, pos, n, u)
        else:
            t4 = succ(tour, pos, n, u)
        if t4 == cur_t2 or t4 == t1 or u == t1 or u == cur_t2:
            break

        # apply the 2-opt: remove (t1,cur_t2) and (t4,u), add (cur_t2,u) and (t4,t1)
        if d_now > 0:
            ri = pos[cur_t2]
            rj = pos[t4]
        else:
            ri = pos[t4]
            rj = pos[cur_t2]
        reverse(tour, pos, n, ri, rj)
        rev_i[level] = ri
        rev_j[level] = rj

        if n_touched + 4 <= touched.shape[0]:
            touched[n_touched] = t1
            touched[n_touched + 1] = cur_t2
            touched[n_touched + 2] = t4
            touched[n_touched + 3] = u
            n_touched += 4

        g = g + dist(coords, t4, u, ceil)  # break x_{i+1} = (u, t4)
        total = g - dist(coords, t4, t1, ceil)  # gain if we close up here
        level += 1
        if total > best_total + 1e-9:
            best_total = total
            best_level = level  # number of reversals to keep

        # descend: the free end is now t4, and the next added edge is (t4, u_next)
        cur_t2 = t4
        nxt = -1
        nxt_score = -1.0e300
        nb = deep_breadth
        if nb > k:
            nb = k
        # the array does not change while we scan, so read the direction once
        d2 = _free_end_dir(tour, pos, n, t1, cur_t2)
        if d2 == 0:
            break
        for t in range(nb):
            c = cand[cur_t2, t]
            dc = cand_d[cur_t2, t]
            stats[STAT_EVALS] += 1
            if g - dc <= 1e-9:
                break  # candidates ascend, so no later one can pass either
            if c == t1 or c == cur_t2:
                continue
            if d2 > 0:
                c4 = pred(tour, pos, n, c)
            else:
                c4 = succ(tour, pos, n, c)
            if c4 == cur_t2 or c4 == t1:
                continue
            # LKH's ordering rule: favour breaking a long edge for a short one
            score = dist(coords, c4, c, ceil) - dc
            if score > nxt_score:
                nxt_score = score
                nxt = c
                g_next = g - dc
        if nxt < 0:
            break

        if use_chain:
            # the chain's own account of how it is going, all on one scale: the
            # length of the edge we broke to start it
            xc[0] = g_next / scale
            if xc[0] > 1.0:
                xc[0] = 1.0
            elif xc[0] < 0.0:
                xc[0] = 0.0
            xc[1] = level / max_depth
            if xc[1] > 1.0:
                xc[1] = 1.0
            xc[2] = best_total / scale
            if xc[2] > 1.0:
                xc[2] = 1.0
            elif xc[2] < 0.0:
                xc[2] = 0.0
            xc[3] = nxt_score / scale
            if xc[3] > 1.0:
                xc[3] = 1.0
            elif xc[3] < 0.0:
                xc[3] = 0.0
            stats[STAT_CHAIN_CALLS] += 1
            if fis_eval1(xc, ch_mf_c, ch_mf_s, ch_ant, ch_cons) < 0.5:
                break  # cut it here

        u = nxt
        g = g_next

    # unwind every reversal deeper than the best closing depth
    for lv in range(level - 1, best_level - 1, -1):
        reverse(tour, pos, n, rev_i[lv], rev_j[lv])
    stats[STAT_DEPTH] += level
    if best_level <= 0:
        return 0.0, 0
    # only the levels we kept actually changed edges, so only those cities need
    # re-activating — the undone levels' endpoints are back where they started
    keep = 4 * best_level
    if keep > n_touched:
        keep = n_touched
    return best_total, keep


@njit(cache=True)
def or_opt_city(
    tour, pos, n, coords, cand, cand_d, ceil, t1, breadth, max_seg, touched, stats
):
    """Or-opt: relocate the segment of 1..``max_seg`` cities starting at t1 so that
    its head sits next to one of its candidate neighbours, in either orientation.

    This is the move the 2-opt chain cannot reach. A chain of 2-opt steps only ever
    produces reconnections that reverse a segment; lifting a short run of cities
    out of one part of the tour and dropping it into another is a 3-opt move of the
    other kind, and it is where most of the remaining gap to a full LK lives.

    Returns (gain, n_touched).
    """
    k = cand.shape[1]
    b = breadth
    if b > k:
        b = k
    if n < 5:
        return 0.0, 0

    for L in range(1, max_seg + 1):
        if L > n - 3:
            break
        i = pos[t1]
        j = (i + L - 1) % n
        s0 = t1
        s1 = tour[j]
        p = tour[(i - 1 + n) % n]
        nx = tour[(j + 1) % n]
        if p == s1 or nx == s0:
            break  # the segment has swallowed the tour
        remove_gain = (
            dist(coords, p, s0, ceil)
            + dist(coords, s1, nx, ceil)
            - dist(coords, p, nx, ceil)
        )
        if remove_gain <= 1e-9:
            continue

        for t in range(b):
            c = cand[s0, t]
            d_c_s0 = cand_d[s0, t]
            stats[STAT_EVALS] += 1
            # the added edge (c,s0) must itself be shorter than what removing the
            # segment frees up; candidates ascend, so nothing later can qualify
            if d_c_s0 >= remove_gain - 1e-9:
                break
            if span_len(n, i, pos[c]) <= L:  # c lies inside the segment
                continue

            best_gain = 1e-9
            mode = 0  # 1: after c, forward orientation; 2: before c, reversed
            cn = succ(tour, pos, n, c)
            if cn != s0:  # else c == p and the move is a no-op
                add = (
                    d_c_s0
                    + dist(coords, s1, cn, ceil)
                    - dist(coords, c, cn, ceil)
                )
                if remove_gain - add > best_gain:
                    best_gain = remove_gain - add
                    mode = 1
            cp = pred(tour, pos, n, c)
            if cp != s1:  # else c == nx and the move is a no-op
                add = (
                    dist(coords, cp, s1, ceil)
                    + d_c_s0
                    - dist(coords, cp, c, ceil)
                )
                if remove_gain - add > best_gain:
                    best_gain = remove_gain - add
                    mode = 2
            if mode == 0:
                continue

            # Surgery. Relocating A = positions i..j to sit beside c is a swap of A
            # with the run of cities between the two sites; that run can be taken
            # forward or backward round the cycle, so take the shorter one.
            im1 = (i - 1 + n) % n
            if mode == 1:
                fwd = span_len(n, i, pos[c])
                bwd = span_len(n, pos[cn], j)
                if fwd <= bwd:
                    block_swap(tour, pos, n, i, j, pos[c], False, False)
                else:
                    block_swap(tour, pos, n, pos[cn], im1, j, False, False)
            else:
                fwd = span_len(n, i, pos[cp])
                bwd = span_len(n, pos[c], j)
                if fwd <= bwd:
                    block_swap(tour, pos, n, i, j, pos[cp], True, False)
                else:
                    block_swap(tour, pos, n, pos[c], im1, j, False, True)

            nt = 0
            if touched.shape[0] >= 6:
                touched[0] = p
                touched[1] = nx
                touched[2] = s0
                touched[3] = s1
                touched[4] = c
                touched[5] = cn if mode == 1 else cp
                nt = 6
            stats[STAT_MOVES] += 1
            return best_gain, nt
    return 0.0, 0


@njit(cache=True)
def improve_city(
    tour,
    pos,
    n,
    coords,
    cand,
    cand_d,
    ceil,
    t1,
    breadth,
    max_depth,
    deep_breadth,
    or_seg,
    rev_i,
    rev_j,
    touched,
    stats,
    use_chain,
    ch_mf_c,
    ch_mf_s,
    ch_ant,
    ch_cons,
):
    """Try to improve the tour with a chain anchored at t1, then by Or-opt.

    Both tour edges at t1 are tried, and the first level backtracks over the top
    ``breadth`` candidates. Returns (gain, n_touched); gain 0.0 means t1 is done
    and its don't-look bit can be set.
    """
    k = cand.shape[1]
    b = breadth
    if b > k:
        b = k
    stats[STAT_SCANS] += 1
    for d in range(2):
        if d == 0:
            t2 = succ(tour, pos, n, t1)
        else:
            t2 = pred(tour, pos, n, t1)
        d_t1t2 = dist(coords, t1, t2, ceil)
        for t in range(b):
            u0 = cand[t2, t]
            stats[STAT_EVALS] += 1
            g0 = d_t1t2 - cand_d[t2, t]
            if g0 <= 1e-9:
                break  # ascending candidates: the criterion fails from here on
            if u0 == t1 or u0 == t2:
                continue
            gain, nt = _lk_chain(
                tour,
                pos,
                n,
                coords,
                cand,
                cand_d,
                ceil,
                t1,
                t2,
                u0,
                g0,
                d_t1t2,
                max_depth,
                deep_breadth,
                rev_i,
                rev_j,
                touched,
                stats,
                use_chain,
                ch_mf_c,
                ch_mf_s,
                ch_ant,
                ch_cons,
            )
            if gain > 1e-9:
                stats[STAT_MOVES] += 1
                return gain, nt
    if or_seg > 0:
        return or_opt_city(
            tour, pos, n, coords, cand, cand_d, ceil, t1, b, or_seg, touched, stats
        )
    return 0.0, 0


@njit(cache=True)
def lk_solve(
    coords,
    cand,
    cand_d,
    ceil,
    tour_in,
    breadth=5,
    max_depth=6,
    deep_breadth=5,
    or_seg=3,
    max_moves=-1,
):
    """Run LK to a local optimum from ``tour_in``. Returns (tour, length, stats).

    Cities are processed from a FIFO work queue seeded in tour order; a city whose
    chain fails has its don't-look bit set, and a successful move re-activates the
    cities whose edges changed. Convergence is when the queue empties.
    """
    n = tour_in.shape[0]
    tour = tour_in.copy()
    pos = make_pos(tour)
    stats = np.zeros(N_STATS, np.int64)

    rev_i = np.empty(max_depth + 1, np.int64)
    rev_j = np.empty(max_depth + 1, np.int64)
    tsize = 4 * (max_depth + 1)
    if tsize < 8:
        tsize = 8
    touched = np.empty(tsize, np.int32)

    cap = n + 1
    queue = np.empty(cap, np.int32)
    in_queue = np.zeros(n, np.uint8)
    for i in range(n):
        queue[i] = tour[i]
        in_queue[tour[i]] = 1
    qh = 0
    qt = n
    qn = n

    while qn > 0:
        t1 = queue[qh]
        qh = (qh + 1) % cap
        qn -= 1
        in_queue[t1] = 0

        gain, nt = improve_city(
            tour,
            pos,
            n,
            coords,
            cand,
            cand_d,
            ceil,
            t1,
            breadth,
            max_depth,
            deep_breadth,
            or_seg,
            rev_i,
            rev_j,
            touched,
            stats,
            False,
            NO_MF_C,
            NO_MF_S,
            NO_CHAIN_ANT,
            NO_CHAIN_CONS,
        )
        if gain > 1e-9:
            for s in range(nt):
                c = touched[s]
                if in_queue[c] == 0 and qn < cap - 1:
                    in_queue[c] = 1
                    queue[qt] = c
                    qt = (qt + 1) % cap
                    qn += 1
            if max_moves > 0 and stats[STAT_MOVES] >= max_moves:
                break

    return tour, tour_length(tour, coords, ceil), stats
