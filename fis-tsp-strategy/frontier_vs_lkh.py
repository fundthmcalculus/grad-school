"""Both solvers as time-quality curves, on the same instances, and where they cross.

Everything before this compared against a *swept LK* — the same move repertoire under fixed
parameters. That is the right control for asking whether adaptive effort helps, and the wrong
one for asking how good the solver is, because the whole family it was compared against shares
one ceiling. LKH does not share it.

LKH is not a single point either, so comparing against one LKH run is the same mistake in the
other direction. ``elkai`` exposes ``runs``, which is LKH's own effort dial, so LKH gets swept
too and the comparison is curve against curve. What that makes visible is a **crossover**
rather than a winner, and the crossover is the honest answer to "can we beat it".

Two caveats bound what this can show, and both are properties of `elkai` rather than of LKH:

* it takes no time limit, only a run count, so its cheapest available point is one full run.
  LKH has no arbitrarily-fast regime through this interface, and where that floor sits relative
  to our curve is most of the answer;
* it builds the problem as EUC_2D, so CEIL_2D instances would be mis-scored and are skipped.

Run:  python frontier_vs_lkh.py [--instances pr1002 d1291 pr2392] [--out lkh_frontier.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import fis
from core import build_candidates, greedy_edge_tour
from kick import iterated_lk
from lk import lk_solve
from tsplib import load, reference_length, validate_tour

HERE = Path(__file__).resolve().parent

# The fixed-parameter sweep, as the control that every earlier section used.
LK_GRID = [(32, 2, 4), (32, 3, 8), (32, 6, 8), (32, 6, 32), (32, 10, 32), (48, 10, 32)]

# Kick budgets for the iterated arm. Spread over two orders of magnitude, because the whole
# question is what the curve does as effort grows rather than where one point lands.
KICKS = [0, 100, 400, 1600, 6400, 25600, 102400]

# LKH's own effort dial.
LKH_RUNS = [1, 2, 5]

K = 32
OR_SEG = 3
WINDOW = 24


def _tmin(fn, reps=3):
    best_t = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return out, best_t


def lkh_curve(inst, runs_list, timeout=600):
    """(runs, gap, seconds) for LKH at each run count, in a subprocess with a timeout.

    Subprocessed because a native solver that hangs or dies takes the whole measurement with it
    otherwise, and because the matrix it needs is large enough to be worth reclaiming promptly.
    """
    import subprocess
    import sys

    out = []
    for runs in runs_list:
        # The *coordinates* API, not solve_int_matrix. Handing LKH a dense matrix disables
        # its own geometric preprocessing — alpha-nearness candidate sets built from a
        # minimum spanning tree — and it becomes drastically slower: one run on pr1002 does
        # not finish in 420s through the matrix interface and takes about a second through
        # this one. Measuring the matrix path would have understated LKH by orders of
        # magnitude, which is the opposite of the error worth making here.
        code = (
            "import time, numpy as np, elkai, sys;"
            "sys.path.insert(0, %r);" % str(HERE)
            + "from tsplib import load, reference_length, validate_tour;"
            f"inst = load({inst.name!r});"
            "coords = {str(i): (float(x), float(y)) for i, (x, y) in enumerate(inst.coords)};"
            "t0 = time.perf_counter();"
            f"order = elkai.Coordinates2D(coords).solve_tsp(runs={runs});"
            "dt = time.perf_counter() - t0;"
            "tour = np.array([int(v) for v in order], dtype=np.int32);"
            "tour = tour[:-1] if tour.shape[0] == inst.n + 1 else tour;"
            "validate_tour(tour, inst.n);"
            "L = reference_length(tour, inst);"
            "print(L, dt)"
        )
        try:
            r = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
            )
            if r.returncode != 0:
                out.append({"runs": runs, "gap": None, "s": None, "why": "failed"})
                continue
            length, dt = r.stdout.split()
            out.append(
                {"runs": runs, "gap": inst.gap(float(length)), "s": float(dt), "why": "ok"}
            )
        except subprocess.TimeoutExpired:
            out.append({"runs": runs, "gap": None, "s": None, "why": f"timeout>{timeout}s"})
            break  # more runs will only be slower
    return out


def measure(inst, targeted=False):
    cand, cand_d = build_candidates(inst.coords, K, inst.ceil)
    start = greedy_edge_tour(inst.coords, cand, inst.ceil)
    none = np.empty(0, np.float64)

    sweep = []
    for k, depth, deep in LK_GRID:
        c2, cd2 = (cand, cand_d) if k == K else build_candidates(inst.coords, k, inst.ceil)
        s2 = start if k == K else greedy_edge_tour(inst.coords, c2, inst.ceil)
        (res, dt) = _tmin(
            lambda c2=c2, cd2=cd2, s2=s2, d=depth, b=deep: lk_solve(
                inst.coords, c2, cd2, inst.ceil, s2, k, d, b, OR_SEG
            )
        )
        tour, length, _ = res
        validate_tour(tour, inst.n)
        sweep.append({"cfg": f"k{k}/d{depth}/b{deep}", "gap": inst.gap(length), "s": dt})

    iterated = []
    for nk in KICKS:
        (res, dt) = _tmin(
            lambda nk=nk: iterated_lk(
                inst.coords, cand, cand_d, inst.ceil, start,
                K, 6, 32, OR_SEG, nk, WINDOW, 12345, none,
                False, fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS,
            ),
            reps=2 if nk > 5000 else 3,
        )
        tour, length, _ = res
        validate_tour(tour, inst.n)
        assert abs(length - reference_length(tour, inst)) < 1e-6, "reported length disagrees"
        iterated.append({"kicks": nk, "gap": inst.gap(length), "s": dt})

    return {"sweep": sweep, "iterated": iterated}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=["pr1002", "d1291", "pr2392"])
    ap.add_argument("--lkh-timeout", type=int, default=600)
    ap.add_argument("--out", default=str(HERE / "lkh_frontier.json"))
    args = ap.parse_args()

    # warm every JIT signature before anything is timed
    warm = load("berlin52")
    wc, wcd = build_candidates(warm.coords, K, warm.ceil)
    wg = greedy_edge_tour(warm.coords, wc, warm.ceil)
    lk_solve(warm.coords, wc, wcd, warm.ceil, wg, K, 6, 32, OR_SEG)
    iterated_lk(
        warm.coords, wc, wcd, warm.ceil, wg, K, 6, 32, OR_SEG, 2, WINDOW, 1,
        np.empty(0, np.float64), False,
        fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS,
    )

    out = {}
    for name in args.instances:
        inst = load(name)
        print(f"\n{inst.name} n={inst.n}")
        m = measure(inst)
        print("  fixed-parameter sweep")
        for r in m["sweep"]:
            print(f"    {r['cfg']:>14s} {r['gap']:7.3f}% {r['s']:8.4f}s")
        print("  iterated (double-bridge kicks)")
        for r in m["iterated"]:
            print(f"    {r['kicks']:>10d} kicks {r['gap']:7.3f}% {r['s']:8.4f}s")
        print("  LKH")
        m["lkh"] = lkh_curve(inst, LKH_RUNS, args.lkh_timeout)
        for r in m["lkh"]:
            if r["gap"] is None:
                print(f"    {r['runs']:>10d} runs  {r['why']}")
            else:
                print(f"    {r['runs']:>10d} runs  {r['gap']:7.3f}% {r['s']:8.4f}s")
        out[name] = m
        Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
