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
    # The cheap end matters more than it looks. An earlier sweep started at depth 4 and
    # so had no cheap operating point at all, which flattered the fuzzy arm: it appeared
    # to dominate six of nine configurations, when what it had actually done was land in
    # a gap in the sweep. LK at depth 2-3 is both fast and respectable, and it is the
    # configuration the adaptive arm has to beat.
    (32, 2, 4),
    (32, 3, 8),
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
    c_tab = e_tab = h_tab = None  # None keeps the hand-written membership functions
    if tuned is not None:
        z = np.load(tuned)
        c_cons = np.ascontiguousarray(z["construct_cons"])
        e_cons = np.ascontiguousarray(z["effort_cons"])
        h_cons = np.ascontiguousarray(z["chain_cons"])
        # tune_opt.py fits the membership functions too and stores them compiled
        if "construct_tab" in z:
            c_tab = np.ascontiguousarray(z["construct_tab"])
            e_tab = np.ascontiguousarray(z["effort_tab"])
            h_tab = np.ascontiguousarray(z["chain_tab"])
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
            lambda: fis_build(inst, cand, cand_d, c_cons, FIS_C_BREADTH, 0, c_tab), reps
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
            ("fis_effort_greedy", gr_t, t_gr, e_cons, h_cons, e_tab, h_tab, False, False),
            ("fis_effort_chain_greedy", gr_t, t_gr, e_cons, h_cons, e_tab, h_tab, True, False),
            ("fis_full", fc_t, t_fc, e_cons, h_cons, e_tab, h_tab, True, False),
            ("fis_effort_nn", nn_t, t_nn, e_cons, h_cons, e_tab, h_tab, True, False),
            ("fis_defer", gr_t, t_gr, e_cons, h_cons, e_tab, h_tab, True, True),
            (
                "fis_effort_greedy_handwritten",
                gr_t,
                t_gr,
                fis.EFFORT_CONS,
                fis.CHAIN_CONS,
                None,
                None,
                False,
                False,
            ),
            (
                "fis_chain_greedy_handwritten",
                gr_t,
                t_gr,
                fis.EFFORT_CONS,
                fis.CHAIN_CONS,
                None,
                None,
                True,
                False,
            ),
        ]
        for tag, start, t_start, ec, hc, et, ht, use_chain, defer in arms:
            (res, t_run) = _tmin(
                lambda s=start, ec=ec, hc=hc, et=et, ht=ht, u=use_chain, d=defer: fis_ls(
                    inst, cand, cand_d, s, ec, hc, FIS_DEPTH, FIS_OR, d, u, et, ht
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


def _instance_frontier(row):
    """The baseline sweep's own frontier on one instance, as ascending-time arrays."""
    pts = []
    for k, depth, deep in BASE_GRID:
        key = f"lk_{k}_{depth}_{deep}"
        if f"{key}_gap" in row:
            pts.append((row[f"{key}_s"], row[f"{key}_gap"]))
    keep = [p for p in pts if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts)]
    keep.sort()
    t = np.array([p[0] for p in keep])
    g = np.array([p[1] for p in keep])
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] * (1 + 1e-9)
    return t, g


def summarise(rows, min_n=0):
    """Per-arm aggregates, including the frontier-relative ratio that is the honest one.

    Two numbers are reported and they answer different questions.

    ``mean_gap`` / ``total_s`` is the conventional pair, and it is the one that misleads:
    it puts an unweighted mean against a sum, so a handful of high-gap instances decide the
    quality axis while the largest instance decides the time axis. Under that pair the
    hand-written rule base appeared to dominate six of nine sweep configurations.

    ``mean_q`` is per-instance: the arm's tour length divided by the tour length the
    baseline sweep's own frontier reaches *at the same wall clock on that instance*,
    averaged over instances. Below 1.0 means the arm is outside LK's frontier — a shorter
    tour than every LK configuration that spends what it spent — which is the actual claim.
    It needs no weighting choice and cannot be carried by one instance.

    Lengths, not gaps. A ratio of gaps is undefined on the instances the baseline solves to
    optimality, where the gap is exactly 0 — and since ``gap = 100 (L/L* - 1)``, the length
    ratio is just ``(1 + gap/100) / (1 + bar/100)``, which is finite everywhere and reads
    directly as "how much longer is this tour than LK's at the same budget".
    """
    keys = [k[:-4] for k in rows[0] if k.endswith("_gap")]
    use = [r for r in rows if r["n"] >= min_n]
    out = {}
    for key in keys:
        gaps, secs, qs = [], [], []
        for r in use:
            if f"{key}_gap" not in r:
                continue
            gaps.append(r[f"{key}_gap"])
            secs.append(r[f"{key}_s"])
            t, g = _instance_frontier(r)
            if len(t):
                bar = float(np.interp(r[f"{key}_s"], t, g))
                qs.append((1.0 + r[f"{key}_gap"] / 100.0) / (1.0 + bar / 100.0))
        if not gaps:
            continue
        out[key] = {
            "mean_gap": float(np.mean(gaps)),
            "total_s": float(np.sum(secs)),
            "mean_q": float(np.mean(qs)) if qs else None,
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
    summary_big = summarise(rows, min_n=1000)
    print(f"\n{'arm':>22s} {'mean gap':>9s} {'total s':>9s} {'n':>4s}")
    print(f"\n{'arm':>32s} {'mean gap':>9s} {'total s':>9s} {'mean q':>7s} {'q n>=1k':>8s}")
    for key, v in sorted(
        summary.items(), key=lambda kv: kv[1]["mean_q"] if kv[1]["mean_q"] else 9e9
    ):
        qb = summary_big.get(key, {}).get("mean_q")
        q = v["mean_q"] if v["mean_q"] is not None else float("nan")
        qbv = qb if qb is not None else float("nan")
        secs = v["total_s"]
        gap = v["mean_gap"]
        print(f"{key:>32s} {gap:8.3f}% {secs:8.3f}s {q:7.4f} {qbv:8.4f}")
    print("\n(mean q < 1 means outside the baseline frontier at the same wall clock)")

    Path(args.out).write_text(
        json.dumps(
            {"rows": rows, "summary": summary, "summary_n_ge_1000": summary_big}, indent=1
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
