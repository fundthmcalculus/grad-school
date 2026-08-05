"""The FIS strategy engine: a fuzzy next-city ranker, and fuzzy LK effort control.

Three places where a conventional solver uses a fixed rule, this one asks a fuzzy
inference system instead.

**Construction.** Nearest-neighbour construction answers "which city is next?" with
one number — the distance. :func:`fis_construct` answers it with a rule base over
four cues: how much worse than the nearest available option the candidate is,
whether it is about to be stranded, what coming back for it later would cost, and
whether it continues the direction of travel.

**Effort.** A conventional LK runs one (breadth, deep breadth, depth, Or-opt)
setting at every city. :func:`fis_lk_solve` reads four cheap features of each city
as it comes off the work queue and lets the rule base set those four parameters per
city.

**Chain continuation.** The deepen-or-cut decision inside each gain chain, taken by
the CHAIN rule base in ``lk._lk_chain`` from the chain's own gain trajectory rather
than at a fixed depth.

The point is not that fuzzy logic reaches moves a fixed LK cannot — the move
repertoire is identical, and both arms run the same ``lk`` code. It is that the
search's cost is wildly unevenly distributed across cities while its fixed
parameters are not, so a controller that can tell the expensive-and-worth-it cities
from the cheap-and-finished ones buys deep-search quality at shallow-search cost.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from core import dist, make_pos, nn_stats, pred, succ, tour_length
import fis as fis_mod
from fis import N_TERMS as N_TERMS_LOCAL, fis_eval, fis_eval1
from lk import N_STATS, improve_city

# extra diagnostics, appended after lk.py's stats (which occupy 0..N_STATS-1)
STAT_BREADTH_SUM = N_STATS  # summed breadth used, for the mean-breadth report
STAT_DEPTH_SUM = N_STATS + 1
STAT_FIS_CALLS = N_STATS + 2
STAT_FULL_ATTEMPTS = N_STATS + 3  # attempts at full effort (the verification pass)
STAT_DEFERRED = N_STATS + 4  # cheap attempts deferred rather than closed
N_STATS_FIS = N_STATS + 5


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
@njit(cache=True)
def fis_construct(
    coords,
    cand,
    cand_d,
    ceil,
    mean_c,
    tab,
    ant,
    cons,
    c_breadth=8,
    start=0,
):
    """Build a tour by asking the CONSTRUCT rule base which city to take next.

    Only the nearest ``c_breadth`` unvisited candidates are ranked — the fuzzy
    question is which of the plausible next cities to take, not whether to jump
    across the instance. When every candidate is already visited the nearest
    unvisited city is found by exact scan, exactly as the nearest-neighbour
    construction does, so the two differ only in the ranking rule.
    """
    n = coords.shape[0]
    k = cand.shape[1]
    kb = c_breadth
    if kb > k:
        kb = k

    visited = np.zeros(n, np.uint8)
    tour = np.empty(n, np.int32)
    unvisited = np.empty(n, np.int32)
    where = np.empty(n, np.int32)
    for i in range(n):
        unvisited[i] = i
        where[i] = i
    m = n
    n_in = ant.shape[1]
    x = np.empty(n_in, np.float64)
    mu = np.empty((n_in, 3), np.float64)

    cur = start
    tour[0] = cur
    visited[cur] = 1
    last = unvisited[m - 1]
    p = where[cur]
    unvisited[p] = last
    where[last] = p
    m -= 1

    prev = -1
    for step in range(1, n):
        # the best distance on offer this step: the yardstick every candidate's
        # excess is measured against, so the greedy choice always scores excess 0
        d_best = 1e300
        for t in range(kb):
            if visited[cand[cur, t]] == 0:
                d_best = cand_d[cur, t]
                break

        best = -1
        best_score = -1.0e300
        for t in range(kb):
            c = cand[cur, t]
            if visited[c] == 1:
                continue

            # how much worse than the best option available right now
            if d_best > 0.0:
                v = (cand_d[cur, t] - d_best) / d_best
            else:
                v = 0.0
            if v > 1.0:
                v = 1.0
            x[0] = v

            # stranding risk: how much of the candidate's own neighbourhood is gone,
            # and what it would cost to come back for it once we have moved on
            vis = 0
            d_back = -1.0
            for s in range(kb):
                cc = cand[c, s]
                if visited[cc] == 1:
                    vis += 1
                elif d_back < 0.0 and cc != cur:
                    d_back = cand_d[c, s]
            x[1] = vis / kb
            if d_back < 0.0:
                x[2] = 1.0  # nothing unvisited left near it at all
            else:
                v = d_back / mean_c[c]
                if v > 2.0:
                    v = 2.0
                x[2] = 0.5 * v

            # heading continuity
            if prev < 0:
                x[3] = 0.5
            else:
                ax = coords[cur, 0] - coords[prev, 0]
                ay = coords[cur, 1] - coords[prev, 1]
                bx = coords[c, 0] - coords[cur, 0]
                by = coords[c, 1] - coords[cur, 1]
                na = np.sqrt(ax * ax + ay * ay)
                nb = np.sqrt(bx * bx + by * by)
                if na <= 0.0 or nb <= 0.0:
                    x[3] = 0.5
                else:
                    cosang = (ax * bx + ay * by) / (na * nb)
                    if cosang > 1.0:
                        cosang = 1.0
                    elif cosang < -1.0:
                        cosang = -1.0
                    x[3] = 0.5 * (1.0 + cosang)

            score = fis_eval1(x, mu, tab, ant, cons)
            if score > best_score:
                best_score = score
                best = c

        if best < 0:  # nothing unvisited within reach — exact scan, as NN does
            bestd = 1e300
            for u in range(m):
                c = unvisited[u]
                d = dist(coords, cur, c, ceil)
                if d < bestd or (d == bestd and c < best):
                    bestd = d
                    best = c

        tour[step] = best
        visited[best] = 1
        last = unvisited[m - 1]
        p = where[best]
        unvisited[p] = last
        where[last] = p
        m -= 1
        prev = cur
        cur = best
    return tour


# ---------------------------------------------------------------------------
# fuzzy effort control over the LK queue
# ---------------------------------------------------------------------------
@njit(cache=True, inline="always")
def city_features(x, n_in, coords, cand, cand_d, ceil, tour, pos, n, k, t1, nn1, mean_c, pops):
    """Fill ``x`` with the EFFORT antecedents for city ``t1``. Returns the longer tour edge.

    Shared by the solver and by :func:`effort_scores`, which aims the perturbation in
    ``kick.py`` at the cities this scores highest. A second copy of these six formulas would be
    a second place for them to drift, and a rule base fitted against one definition and aimed
    with another would be quietly wrong rather than broken.

    x[0:5] are the five inputs that cleared AUC 0.74 in `experiments/features_probe.py`; x[5:8] are the
    middling band, computed only when ``n_in`` says the rule base references them, because the
    turn calculation needs two square roots and is the most expensive feature here.
    """
    s1 = succ(tour, pos, n, t1)
    p1 = pred(tour, pos, n, t1)
    d_s = dist(coords, t1, s1, ceil)
    d_p = dist(coords, t1, p1, ceil)
    d_long = d_s if d_s > d_p else d_p

    # The probe: one level of search run as a look-ahead over both directions of the broken
    # edge. Candidate distances ascend, so the loop breaks at the first candidate that fails.
    best_g1 = 0.0
    n_pass = 0
    for side in range(2):
        d_break = d_s if side == 0 else d_p
        t2 = s1 if side == 0 else p1
        for t in range(k):
            g1 = d_break - cand_d[t2, t]
            if g1 <= 1e-9:
                break
            n_pass += 1
            if g1 > best_g1:
                best_g1 = g1
    v = n_pass / (2.0 * k)
    if v > 1.0:
        v = 1.0
    x[0] = v
    v = best_g1 / d_long if d_long > 0.0 else 0.0
    if v > 1.0:
        v = 1.0
    x[2] = v

    r = 0
    for t in range(k):
        if cand_d[t1, t] >= d_long:
            break
        r += 1
    x[1] = r / k

    v = (0.5 * (d_s + d_p) / nn1[t1] - 1.0) * 0.5
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    x[3] = v

    tot = d_s + d_p
    x[4] = (d_s - d_p if d_s > d_p else d_p - d_s) / tot if tot > 0.0 else 0.0

    if n_in > 5:
        ax = coords[t1, 0] - coords[p1, 0]
        ay = coords[t1, 1] - coords[p1, 1]
        bx = coords[s1, 0] - coords[t1, 0]
        by = coords[s1, 1] - coords[t1, 1]
        na = np.sqrt(ax * ax + ay * ay)
        nb = np.sqrt(bx * bx + by * by)
        if na <= 0.0 or nb <= 0.0:
            x[5] = 0.5
        else:
            cosang = (ax * bx + ay * by) / (na * nb)
            if cosang > 1.0:
                cosang = 1.0
            elif cosang < -1.0:
                cosang = -1.0
            x[5] = 0.5 * (1.0 - cosang)
        x[6] = nn1[t1] / mean_c[t1]
        v = pops / (3.0 * n)
        if v > 1.0:
            v = 1.0
        x[7] = v
    return d_long


@njit(cache=True)
def effort_scores_kernel(coords, cand, cand_d, ceil, tour, pos, nn1, mean_c, tab, ant, cons):
    """The EFFORT base's depth output for every city, on the tour as it stands.

    Depth is the output the rule base uses to say "this city is worth working on", so it is the
    natural thing to aim a perturbation with.
    """
    n = tour.shape[0]
    k = cand.shape[1]
    n_in = ant.shape[1]
    x = np.empty(n_in, np.float64)
    mu = np.empty((n_in, N_TERMS_LOCAL), np.float64)
    out = np.empty(cons.shape[1], np.float64)
    scores = np.empty(n, np.float64)
    for t1 in range(n):
        city_features(
            x, n_in, coords, cand, cand_d, ceil, tour, pos, n, k, t1, nn1, mean_c, 0
        )
        fis_eval(x, mu, tab, ant, cons, out)
        scores[t1] = out[2]  # E_DEPTH
    return scores


def effort_scores(inst, cand, cand_d, tour, scale=None, tuned=None):
    """Per-city EFFORT depth scores, as a plain helper for ``kick.py``.

    ``tuned`` is a :class:`fis.Tuned` record; passing one aims the scores with the *fitted*
    rule base instead of the hand-written one, which matters wherever these scores are used
    to steer something rather than to describe it — kick targeting is the case that exists.
    Its own ``scale`` wins, since its consequents are only meaningful against that scale's
    antecedents.
    """
    if tuned is not None:
        return effort_scores_kernel(
            inst.coords, cand, cand_d, inst.ceil, tour, make_pos(tour),
            *nn_stats(cand_d), tuned.effort_tab, tuned.effort_ant, tuned.effort_cons,
        )
    scale = fis_mod.DEFAULT_SCALE if scale is None else scale
    ant, cons, _, _, tab = fis_mod.effort_base(scale)
    nn1, mean_c = nn_stats(cand_d)
    return effort_scores_kernel(
        inst.coords, cand, cand_d, inst.ceil, tour, make_pos(tour), nn1, mean_c, tab, ant, cons
    )


@njit(cache=True)
def fis_lk_solve(
    coords,
    cand,
    cand_d,
    ceil,
    tour_in,
    nn1,
    mean_c,
    tab,
    ant,
    cons,
    ch_tab,
    ch_ant,
    ch_cons,
    max_depth=10,
    or_max=3,
    min_breadth=1,
    defer=False,
    use_chain=True,
):
    """Run LK to a local optimum, with (breadth, depth, Or-opt) set per city by the
    EFFORT rule base. Returns (tour, length, stats).

    What the rule base is really allocating is **chain depth**. Measured on the
    baseline, widening the first level from 2 candidates to 32 costs almost nothing —
    the sequential gain criterion truncates those scans long before the cap applies —
    while deepening the chain from 4 levels to 10 costs 2.6x for about 0.4 points of
    tour quality. So depth is the budget, and the question the rule base answers is
    which cities deserve to spend it.

    ``defer=True`` adds a verification pass: a city that fails at reduced effort is
    not closed but re-searched at full effort, so the run cannot stop until every
    city has failed at full breadth and full depth — the baseline's exact stopping
    condition. It costs time and is off by default, but it is how the claim above is
    *checked* rather than asserted. If the cheap schedule had been quietly discarding
    moves, the verification pass would find them and return a shorter tour. Where it
    returns the identical tour, the reduced effort provably lost nothing: the cheap
    schedule landed on a full-effort local optimum for a fraction of the work.
    """
    n = tour_in.shape[0]
    k = cand.shape[1]
    tour = tour_in.copy()
    pos = make_pos(tour)
    stats = np.zeros(N_STATS_FIS, np.int64)

    rev_i = np.empty(max_depth + 2, np.int64)
    rev_j = np.empty(max_depth + 2, np.int64)
    tsize = 4 * (max_depth + 2)
    if tsize < 8:
        tsize = 8
    touched = np.empty(tsize, np.int32)

    fails = np.zeros(n, np.uint8)
    # a city is settled once a full-effort attempt has failed on it and nothing has
    # touched its neighbourhood since
    settled = np.zeros(n, np.uint8)
    n_e = ant.shape[1]
    n_c = ch_ant.shape[1]
    x = np.empty(n_e, np.float64)
    mu = np.empty((n_e, 3), np.float64)
    xc = np.empty(n_c, np.float64)
    mu_c = np.empty((n_c, 3), np.float64)
    out = np.empty(cons.shape[1], np.float64)

    cap = n + 1
    qa = np.empty(cap, np.int32)  # cheap pass, at the fuzzy-chosen effort
    qb = np.empty(cap, np.int32)  # deferred, awaiting a full-effort attempt
    in_a = np.zeros(n, np.uint8)
    in_b = np.zeros(n, np.uint8)
    for i in range(n):
        qa[i] = tour[i]
        in_a[tour[i]] = 1
    ah = 0
    at = n
    an = n
    bh = 0
    bt = 0
    bn = 0

    pops = 0

    while an > 0 or bn > 0:
        if an > 0:
            t1 = qa[ah]
            ah = (ah + 1) % cap
            an -= 1
            in_a[t1] = 0
            full = False
        else:
            t1 = qb[bh]
            bh = (bh + 1) % cap
            bn -= 1
            in_b[t1] = 0
            if settled[t1] == 1:
                continue
            full = True
        pops += 1

        if full:
            breadth = k
            deep = k
            depth = max_depth
            or_seg = or_max
            stats[STAT_FULL_ATTEMPTS] += 1
        else:
            city_features(
                x, n_e, coords, cand, cand_d, ceil, tour, pos, n, k, t1, nn1, mean_c, pops
            )

            fis_eval(x, mu, tab, ant, cons, out)
            stats[STAT_FIS_CALLS] += 1

            breadth = min_breadth + int(out[0] * (k - min_breadth) + 0.5)
            if breadth < min_breadth:
                breadth = min_breadth
            elif breadth > k:
                breadth = k
            deep = min_breadth + int(out[1] * (k - min_breadth) + 0.5)
            if deep < min_breadth:
                deep = min_breadth
            elif deep > k:
                deep = k
            depth = 2 + int(out[2] * (max_depth - 2) + 0.5)
            if depth < 2:
                depth = 2
            elif depth > max_depth:
                depth = max_depth
            or_seg = int(out[3] * or_max + 0.5)
            if or_seg < 0:
                or_seg = 0
            elif or_seg > or_max:
                or_seg = or_max
            if (
                breadth >= k
                and deep >= k
                and depth >= max_depth
                and or_seg >= or_max
            ):
                full = True  # the rule base asked for everything anyway
                stats[STAT_FULL_ATTEMPTS] += 1

        stats[STAT_BREADTH_SUM] += breadth
        stats[STAT_DEPTH_SUM] += depth

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
            depth,
            deep,
            or_seg,
            rev_i,
            rev_j,
            touched,
            stats,
            use_chain,
            ch_tab,
            ch_ant,
            ch_cons,
            xc,
            mu_c,
        )
        if gain > 1e-9:
            fails[t1] = 0
            settled[t1] = 0
            for s in range(nt):
                c = touched[s]
                settled[c] = 0  # its neighbourhood changed; it is worth a look again
                fails[c] = 0
                if in_a[c] == 0 and an < cap - 1:
                    in_a[c] = 1
                    qa[at] = c
                    at = (at + 1) % cap
                    an += 1
        elif full:
            settled[t1] = 1  # nothing here, and we looked at everything
        else:
            if fails[t1] < 255:
                fails[t1] += 1
            # a cheap look found nothing. Under `defer` that is not taken as
            # evidence the city is done, and it goes to the full-effort queue.
            if defer and in_b[t1] == 0 and bn < cap - 1:
                in_b[t1] = 1
                qb[bt] = t1
                bt = (bt + 1) % cap
                bn += 1
                stats[STAT_DEFERRED] += 1

    return tour, tour_length(tour, coords, ceil), stats


# ---------------------------------------------------------------------------
# end-to-end driver
# ---------------------------------------------------------------------------
def construct(inst, cand, cand_d, cons, c_breadth=8, start=0, tab=None):
    """The fuzzy next-city ranker, as a one-call helper.

    ``tab`` overrides the membership-function bank, which the optimiser fits along
    with the consequents; ``None`` keeps the hand-written one.
    """
    _, mean_c = nn_stats(cand_d)
    return fis_construct(
        inst.coords,
        cand,
        cand_d,
        inst.ceil,
        mean_c,
        fis_mod.CONSTRUCT_TAB if tab is None else tab,
        fis_mod.CONSTRUCT_ANT,
        cons,
        c_breadth,
        start,
    )


def local_search(
    inst,
    cand,
    cand_d,
    start_tour,
    effort_cons,
    chain_cons,
    max_depth=10,
    or_max=3,
    defer=False,
    use_chain=True,
    effort_tab=None,
    chain_tab=None,
    effort_ant=None,
    chain_ant=None,
):
    """Fuzzy-controlled LK from a given start tour, as a one-call helper.

    ``effort_tab`` / ``chain_tab`` override the membership-function banks, which the
    optimiser fits along with the consequents; ``None`` keeps the hand-written ones.
    """
    nn1, mean_c = nn_stats(cand_d)
    return fis_lk_solve(
        inst.coords,
        cand,
        cand_d,
        inst.ceil,
        start_tour,
        nn1,
        mean_c,
        fis_mod.EFFORT_TAB if effort_tab is None else effort_tab,
        fis_mod.EFFORT_ANT if effort_ant is None else effort_ant,
        effort_cons,
        fis_mod.CHAIN_TAB if chain_tab is None else chain_tab,
        fis_mod.CHAIN_ANT if chain_ant is None else chain_ant,
        chain_cons,
        max_depth,
        or_max,
        1,
        defer,
        use_chain,
    )
