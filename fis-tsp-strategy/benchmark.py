"""The reported comparison: the FIS strategy engine against the baseline LK.

Beating a single configuration of a tunable solver is not worth much — you can
always find one that is slow, or one that is weak. So the baseline is reported as a
*frontier*: the same LK swept across the parameters that trade its time against its
tour quality (candidate-list size, chain depth, deep breadth). A claim to have beaten
it means landing outside that frontier — strictly better quality *and* strictly less
wall clock than every baseline configuration measured.

Every tour is checked to be a permutation of the cities and re-scored from the
coordinates under TSPLIB rounding by ``tsplib.reference_length``, independently of
whatever the solver thought it had built. Times are the minimum of ``--reps`` runs,
after a warm-up call that pays all the JIT compilation.

Test instances are disjoint from the sets ``tune.py`` fits and validates on.

Run:  python benchmark.py [--reps 3] [--max-n 20000] [--tuned tuned.npz] [--lkh]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import fis
from core import build_candidates, greedy_edge_tour, nn_stats, nn_tour
from fis_lk import STAT_BREADTH_SUM, STAT_DEPTH_SUM
from fis_lk import construct as fis_build
from fis_lk import local_search as fis_ls
from lk import STAT_SCANS, lk_solve
from tsplib import load, reference_length, validate_tour

HERE = Path(__file__).resolve().parent

# Disjoint from tune.TRAIN and tune.VALID.
TEST = [
    "berlin52",
    "a280",
    "pr439",
    "d493",
    "rat783",
    "pcb1173",
    "rl1323",
    "fl1577",
    "d2103",
    "pr2392",
    "pcb3038",
    "fl3795",
    "fnl4461",
    "rl5915",
    "pla7397",
    "rl11849",
    "usa13509",
    "brd14051",
    "d15112",
    "d18512",
]

# The baseline frontier: (candidate list, chain depth, deep breadth). Swept because
# these are the three parameters that move its time/quality trade-off.
BASE_GRID = [
    (16, 4, 8),
    (24, 4, 8),
    (32, 4, 8),
    (32, 6, 8),
    (32, 6, 32),
    (32, 10, 12),
    (32, 10, 32),
    (48, 10, 32),
    (64, 10, 32),
]

FIS_K = 32
FIS_DEPTH = 10
FIS_OR = 3
FIS_C_BREADTH = 8


def _tmin(fn, reps):
    """(result, minimum wall clock). The minimum of a few runs estimates the real
    cost better than the mean, which only ever drifts up with interference."""
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _warm():
    """Pay every JIT compilation before anything is timed."""
    inst = load("berlin52")
    cand, cand_d = build_candidates(inst.coords, 32, inst.ceil)
    g = greedy_edge_tour(inst.coords, cand, inst.ceil)
    nn_tour(inst.coords, cand, inst.ceil, 0)
    nn_stats(cand_d)
    lk_solve(inst.coords, cand, cand_d, inst.ceil, g, 32, 10, 32, 3)
    fis_build(inst, cand, cand_d, fis.CONSTRUCT_CONS, 8)
    for defer in (False, True):
        for use_chain in (False, True):
            fis_ls(
                inst,
                cand,
                cand_d,
                g,
                fis.EFFORT_CONS,
                fis.CHAIN_CONS,
                10,
                3,
                defer,
                use_chain,
            )


def lkh_reference(inst, runs=1):
    """(gap, seconds) for LKH through ``elkai``, or None if unavailable.

    An external yardstick only — LKH is a mature C implementation with a much
    stronger candidate rule (alpha-nearness on a 1-tree) and 5-opt basic moves, and
    nothing here claims to beat it. Restricted to EUC_2D because elkai builds the
    problem as EUC_2D and a CEIL_2D instance would be scored under the wrong metric.
    """
    if inst.ceil:
        return None
    try:
        import elkai
    except ImportError:
        return None
    coords = {str(i): (float(x), float(y)) for i, (x, y) in enumerate(inst.coords)}
    t0 = time.perf_counter()
    order = elkai.Coordinates2D(coords).solve_tsp(runs=runs)
    dt = time.perf_counter() - t0
    tour = np.array([int(s) for s in order], dtype=np.int32)
    if tour.shape[0] == inst.n + 1:  # elkai returns a closed walk
        tour = tour[:-1]
    validate_tour(tour, inst.n)
    return inst.gap(reference_length(tour, inst)), dt


def run(names, reps=3, max_n=20000, tuned=None, with_lkh=False, lkh_max_n=3000):
    c_cons, e_cons, h_cons = fis.CONSTRUCT_CONS, fis.EFFORT_CONS, fis.CHAIN_CONS
    if tuned is not None:
        z = np.load(tuned)
        c_cons = np.ascontiguousarray(z["construct_cons"])
        e_cons = np.ascontiguousarray(z["effort_cons"])
        h_cons = np.ascontiguousarray(z["chain_cons"])
    rows = []

    for name in names:
        inst = load(name)
        if inst.n > max_n:
            continue
        row = {"name": name, "n": inst.n, "ewt": inst.ewt, "opt": inst.opt}

        # candidate lists are shared infrastructure; each arm is charged for the one
        # it uses, so the cost is inside every reported time
        cands = {}
        for k in sorted({g[0] for g in BASE_GRID} | {FIS_K}):
            _, t_c = _tmin(lambda k=k: build_candidates(inst.coords, k, inst.ceil), reps)
            cands[k] = build_candidates(inst.coords, k, inst.ceil) + (t_c,)

        # --- constructions
        cand, cand_d, t_cand = cands[FIS_K]
        nn1, mean_c = nn_stats(cand_d)
        (nn_t, t_nn) = _tmin(lambda: nn_tour(inst.coords, cand, inst.ceil, 0), reps)
        (gr_t, t_gr) = _tmin(lambda: greedy_edge_tour(inst.coords, cand, inst.ceil), reps)
        (fc_t, t_fc) = _tmin(
            lambda: fis_build(inst, cand, cand_d, c_cons, FIS_C_BREADTH), reps
        )
        for tag, tour, t in (
            ("nn", nn_t, t_nn),
            ("greedy", gr_t, t_gr),
            ("fis", fc_t, t_fc),
        ):
            validate_tour(tour, inst.n)
            row[f"construct_{tag}_gap"] = inst.gap(reference_length(tour, inst))
            row[f"construct_{tag}_s"] = t

        # --- baseline LK frontier, each from the greedy start that suits it best
        for k, depth, deep in BASE_GRID:
            ck, cdk, t_ck = cands[k]
            (start, t_start) = _tmin(
                lambda ck=ck: greedy_edge_tour(inst.coords, ck, inst.ceil), reps
            )
            (res, t_lk) = _tmin(
                lambda ck=ck, cdk=cdk, start=start, k=k, depth=depth, deep=deep: lk_solve(
                    inst.coords, ck, cdk, inst.ceil, start, k, depth, deep, 3
                ),
                reps,
            )
            tour, _, st = res
            validate_tour(tour, inst.n)
            row[f"lk_{k}_{depth}_{deep}_gap"] = inst.gap(reference_length(tour, inst))
            row[f"lk_{k}_{depth}_{deep}_s"] = t_lk + t_start + t_ck

        # --- the FIS arms. Each is (start tour, rule set, chain on/off, defer),
        # so the table separates the ranker from the effort controller from the
        # chain-continuation rules, and the tuned rule base from the hand-written one.
        arms = [
            ("fis_effort_greedy", gr_t, t_gr, e_cons, h_cons, False, False),
            ("fis_effort_chain_greedy", gr_t, t_gr, e_cons, h_cons, True, False),
            ("fis_full", fc_t, t_fc, e_cons, h_cons, False, False),
            ("fis_effort_nn", nn_t, t_nn, e_cons, h_cons, False, False),
            ("fis_defer", gr_t, t_gr, e_cons, h_cons, False, True),
            (
                "fis_effort_greedy_handwritten",
                gr_t,
                t_gr,
                fis.EFFORT_CONS,
                fis.CHAIN_CONS,
                False,
                False,
            ),
        ]
        for tag, start, t_start, ec, hc, use_chain, defer in arms:
            (res, t_run) = _tmin(
                lambda s=start, ec=ec, hc=hc, u=use_chain, d=defer: fis_ls(
                    inst, cand, cand_d, s, ec, hc, FIS_DEPTH, FIS_OR, d, u
                ),
                reps,
            )
            tour, _, st = res
            validate_tour(tour, inst.n)
            row[f"{tag}_gap"] = inst.gap(reference_length(tour, inst))
            row[f"{tag}_s"] = t_run + t_start + t_cand
            scans = max(int(st[STAT_SCANS]), 1)
            row[f"{tag}_mean_depth"] = float(st[STAT_DEPTH_SUM]) / scans
            row[f"{tag}_mean_breadth"] = float(st[STAT_BREADTH_SUM]) / scans

        # the fuzzy ranker feeding an unmodified LK: the other half of the ablation
        (res, t_run) = _tmin(
            lambda: lk_solve(inst.coords, cand, cand_d, inst.ceil, fc_t, FIS_K, 10, 32, 3),
            reps,
        )
        tour, _, _ = res
        validate_tour(tour, inst.n)
        row["fis_construct_lk_gap"] = inst.gap(reference_length(tour, inst))
        row["fis_construct_lk_s"] = t_run + t_fc + t_cand

        if with_lkh and inst.n <= lkh_max_n:
            ref = lkh_reference(inst)
            if ref is not None:
                row["lkh_gap"], row["lkh_s"] = ref

        rows.append(row)
        print(
            f"  {name:>9s} n={inst.n:6d}  "
            f"LK {row['lk_32_10_32_gap']:6.3f}%/{row['lk_32_10_32_s']:.4f}s   "
            f"FIS {row['fis_full_gap']:6.3f}%/{row['fis_full_s']:.4f}s   "
            f"FISeff {row['fis_effort_greedy_gap']:6.3f}%/{row['fis_effort_greedy_s']:.4f}s",
            flush=True,
        )
    return rows


def summarise(rows):
    """Mean gap and total seconds per arm, plus which arms are on the frontier."""
    keys = [k[:-4] for k in rows[0] if k.endswith("_gap")]
    out = {}
    for key in keys:
        gaps = [r[f"{key}_gap"] for r in rows if f"{key}_gap" in r]
        secs = [r[f"{key}_s"] for r in rows if f"{key}_s" in r]
        if not gaps:
            continue
        out[key] = {
            "mean_gap": float(np.mean(gaps)),
            "total_s": float(np.sum(secs)),
            "n_inst": len(gaps),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-n", type=int, default=20000)
    ap.add_argument("--tuned", default=str(HERE / "tuned.npz"))
    ap.add_argument("--lkh", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args()

    tuned = args.tuned if Path(args.tuned).exists() else None
    print(f"warming JIT...  (tuned rule base: {tuned})")
    _warm()
    names = args.names if args.names else TEST
    rows = run(names, args.reps, args.max_n, tuned, args.lkh)

    summary = summarise(rows)
    print(f"\n{'arm':>22s} {'mean gap':>9s} {'total s':>9s} {'n':>4s}")
    for key, v in sorted(summary.items(), key=lambda kv: kv[1]["mean_gap"]):
        print(f"{key:>22s} {v['mean_gap']:8.3f}% {v['total_s']:8.4f}s {v['n_inst']:4d}")

    Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
