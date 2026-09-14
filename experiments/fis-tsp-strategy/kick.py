"""A move the 2-opt chain cannot reach, and the iterated loop that makes it useful.

Everything measured so far shares one ceiling. The repertoire is sequential 2-opt chains plus
Or-opt relocation, and both are *improving* moves, so the search stops at a local optimum of
that neighbourhood and no amount of cleverness about effort allocation moves it. The rule
bases decide how much to spend reaching that optimum and where; they cannot decide to leave
it. Against LKH — which reaches the published optimum on essentially every test instance —
that ceiling is most of the remaining gap, and no feature or parameter can close it.

The classical answer is the **double bridge**: a non-sequential 4-opt move that reconnects
A-B-C-D as A-C-B-D. It matters because it is unreachable by any sequence of improving 2-opt
steps — it reverses no segment, so a 2-opt chain has no path to it — which is exactly why it
escapes the local optimum a 2-opt/Or-opt search converges to. It is applied as a
*perturbation*: on its own it almost always lengthens the tour, and the point is that
re-optimising afterwards recovers more than it lost, often enough to be worth the attempt.

Three things make this affordable at the sizes this engine targets:

* **The kick is local.** The textbook double bridge picks three uniform cut points, which
  makes the rebuild O(n) and the damage global. Here the three segments are drawn inside a
  bounded window, so the move costs O(window) and disturbs a region rather than the tour.
* **Re-optimisation is seeded, not restarted.** Only the eight cities whose edges changed go
  back on the work queue, so the local search after a kick costs a small multiple of a single
  city scan rather than a full convergence pass. This is what ``lk.lk_reopt`` exists for.
* **Rejection is a copy, not an undo.** Keeping an undo log through an arbitrary
  re-optimisation is complicated and easy to get subtly wrong; copying the best tour back is
  O(n) of memcpy, which against the cost of the re-optimisation itself is nothing.

Where the fuzzy engine re-enters: a kick has to be aimed, and the usual choice is uniform
random. The rule bases already estimate, per city, how much there is to find — so
``targeted=True`` aims kicks at the cities the EFFORT base scores highest instead. That is the
same payoff-prediction framing used to screen features, applied to perturbation rather than
effort.

Run:  python kick.py --demo        # frontier extension on one instance
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import njit

import fis
from core import build_candidates, greedy_edge_tour, make_pos, tour_length
from lk import N_STATS, lk_reopt
from tsplib import load, reference_length, validate_tour


@njit(cache=True)
def double_bridge(tour, pos, n, i, l1, l2, l3, touched):
    """Apply a localised double bridge at position ``i``: A B C -> A C B in place.

    The window starting at ``i`` is split into three runs of lengths l1, l2, l3; the first and
    third of those runs are exchanged, which is the double-bridge reconnection. No segment is
    reversed, which is precisely what puts this move outside the reach of any 2-opt chain.

    Writes the eight cities whose incident edges changed into ``touched`` and returns their
    count, so the caller can seed re-optimisation from them rather than from the whole tour.
    """
    total = l1 + l2 + l3
    if total + 2 > n:
        return 0
    buf = np.empty(total, tour.dtype)
    # The window holds three consecutive runs X Y Z. Writing them back as Z Y X exchanges the
    # outer two and leaves the middle in place, with every run's internal order preserved —
    # that is the double bridge. Preserving internal order is the whole point: a version that
    # wrote any run backwards would be a segment reversal, which the 2-opt chain can already
    # reach and would immediately undo, so the perturbation would accomplish nothing.
    w = 0
    for t in range(l3):  # third run first
        buf[w] = tour[(i + l1 + l2 + t) % n]
        w += 1
    for t in range(l2):  # middle run unchanged
        buf[w] = tour[(i + l1 + t) % n]
        w += 1
    for t in range(l1):  # first run last
        buf[w] = tour[(i + t) % n]
        w += 1
    for t in range(total):
        p = (i + t) % n
        tour[p] = buf[t]
        pos[buf[t]] = p

    nt = 0
    for p in (
        (i - 1 + n) % n,
        i,
        (i + l3 - 1) % n,
        (i + l3) % n,
        (i + l3 + l2 - 1) % n,
        (i + l3 + l2) % n,
        (i + total - 1) % n,
        (i + total) % n,
    ):
        c = tour[p]
        dup = False
        for s in range(nt):
            if touched[s] == c:
                dup = True
                break
        if not dup:
            touched[nt] = c
            nt += 1
    return nt


@njit(cache=True)
def iterated_lk(
    coords,
    cand,
    cand_d,
    ceil,
    tour_in,
    breadth,
    max_depth,
    deep_breadth,
    or_seg,
    n_kicks,
    window,
    seed,
    weights,
    use_chain,
    ch_tab,
    ch_ant,
    ch_cons,
    accept_equal=False,
    patience=0,
):
    """Local search, then ``n_kicks`` double-bridge kicks each followed by seeded re-optimisation.

    ``weights`` biases where kicks land: a cumulative distribution over cities, or an empty
    array for uniform. Returns (best tour, best length, stats).

    The default accept rule is strict improvement against the best tour seen, with rejection
    restoring that best. That keeps reported length monotone in the kick budget, so a frontier
    built by varying ``n_kicks`` is a genuine frontier rather than a sampling artefact.

    Two options exist because the curve *plateaus*, and the plateau has more than one possible
    cause. On pr2392 quadrupling the budget past 25 600 kicks buys 0.04 points, which is either
    the move repertoire running out of reachable improvements or the accept rule refusing to
    cross the plateau it is sitting on:

    * ``accept_equal`` keeps equal-length tours, letting the search drift sideways across a
      plateau to a different basin. This is the standard remedy and it costs the monotonicity
      guarantee above, so the two cannot be had together.
    * ``patience`` allows a *worsening* tour to be kept after that many consecutive rejections,
      a minimal record-to-record travel. Restores from best when it does improve, so it cannot
      lose the best tour found.

    If neither moves the plateau, the plateau is the repertoire, and no amount of perturbation
    scheduling will fix it — that is the diagnostic these exist to run.
    """
    n = tour_in.shape[0]
    tour = tour_in.copy()
    pos = make_pos(tour)
    stats = np.zeros(N_STATS, np.int64)

    seeds = np.empty(n, np.int32)
    for i in range(n):
        seeds[i] = tour[i]
    lk_reopt(
        tour,
        pos,
        n,
        coords,
        cand,
        cand_d,
        ceil,
        seeds,
        n,
        breadth,
        max_depth,
        deep_breadth,
        or_seg,
        stats,
        use_chain,
        ch_tab,
        ch_ant,
        ch_cons,
    )
    best = tour.copy()
    best_len = tour_length(tour, coords, ceil)

    if n_kicks <= 0 or n < 16:
        return best, best_len, stats

    np.random.seed(seed)
    since = 0
    kick_touched = np.empty(8, np.int32)
    lo = 1
    hi = window
    if 3 * hi + 2 > n:
        hi = (n - 2) // 3
    if hi < 2:
        return best, best_len, stats

    for _ in range(n_kicks):
        if weights.shape[0] == n:
            # aim the kick: sample a city from the supplied distribution, kick at its position
            r = np.random.random()
            a = 0
            b = n - 1
            while a < b:  # binary search in the cumulative weights
                m = (a + b) // 2
                if weights[m] < r:
                    a = m + 1
                else:
                    b = m
            i = pos[a]
        else:
            i = np.random.randint(0, n)
        l1 = lo + np.random.randint(0, hi)
        l2 = lo + np.random.randint(0, hi)
        l3 = lo + np.random.randint(0, hi)
        nt = double_bridge(tour, pos, n, i, l1, l2, l3, kick_touched)
        if nt == 0:
            continue
        lk_reopt(
            tour,
            pos,
            n,
            coords,
            cand,
            cand_d,
            ceil,
            kick_touched,
            nt,
            breadth,
            max_depth,
            deep_breadth,
            or_seg,
            stats,
            use_chain,
            ch_tab,
            ch_ant,
            ch_cons,
        )
        stats[N_STATS - 1] += 1
        length = tour_length(tour, coords, ceil)
        if length < best_len - 1e-9:
            best_len = length
            since = 0
            for t in range(n):
                best[t] = tour[t]
        elif accept_equal and length <= best_len + 1e-9:
            since = 0  # sideways move: keep the current tour, best is unchanged
        elif patience > 0 and since >= patience:
            since = 0  # keep a worse tour to escape; best is still recorded
        else:
            since += 1
            for t in range(n):
                tour[t] = best[t]
                pos[best[t]] = t
    return best, best_len, stats


def effort_weights(inst, cand, cand_d, tour, tuned=None):
    """A cumulative distribution over cities, weighted by how much the EFFORT rule base thinks
    there is to find at each.

    This is the fuzzy engine aiming the perturbation. The weight is the rule base's own depth
    output — the parameter it uses to say "this city is worth working on" — so a kick lands
    preferentially where the tour is judged weakest rather than uniformly. Pass a
    :class:`fis.Tuned` to aim with the fitted rule base rather than the hand-written one.

    The floor of ``1e-3`` above the minimum is deliberate: a city the rule base scores lowest
    must still be reachable, because the scores are computed once on the starting tour and a
    kick elsewhere can make a previously settled region worth revisiting. A distribution with
    exact zeros in it would make those cities permanently unkickable.
    """
    from fis_lk import effort_scores

    s = effort_scores(inst, cand, cand_d, tour, tuned=tuned)
    s = np.asarray(s, dtype=np.float64)
    s = s - s.min() + 1e-3
    c = np.cumsum(s)
    return np.ascontiguousarray(c / c[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="pr2392")
    ap.add_argument("--kicks", nargs="*", type=int, default=[0, 200, 1000, 4000, 16000])
    ap.add_argument("--window", type=int, default=24)
    args = ap.parse_args()

    inst = load(args.instance)
    cand, cand_d = build_candidates(inst.coords, 32, inst.ceil)
    start = greedy_edge_tour(inst.coords, cand, inst.ceil)
    none = np.empty(0, np.float64)
    print(f"{inst.name} n={inst.n}")
    print(f"{'kicks':>7s} {'gap':>8s} {'seconds':>9s}")
    for nk in args.kicks:
        iterated_lk(  # warm the JIT at this signature
            inst.coords,
            cand,
            cand_d,
            inst.ceil,
            start,
            8,
            6,
            16,
            3,
            1,
            args.window,
            1,
            none,
            False,
            fis.NO_CHAIN_TAB,
            fis.NO_CHAIN_ANT,
            fis.NO_CHAIN_CONS,
        )
        t0 = time.perf_counter()
        tour, length, stats = iterated_lk(
            inst.coords,
            cand,
            cand_d,
            inst.ceil,
            start,
            8,
            6,
            16,
            3,
            nk,
            args.window,
            1,
            none,
            False,
            fis.NO_CHAIN_TAB,
            fis.NO_CHAIN_ANT,
            fis.NO_CHAIN_CONS,
        )
        dt = time.perf_counter() - t0
        validate_tour(tour, inst.n)
        assert abs(length - reference_length(tour, inst)) < 1e-6
        print(f"{nk:7d} {inst.gap(length):7.3f}% {dt:8.3f}s")


if __name__ == "__main__":
    main()
