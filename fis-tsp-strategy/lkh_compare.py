"""Every arm this project has, and LKH, as time-quality curves on the same instances.

Everything in ``benchmark.py`` compares against a *swept LK* — the same move repertoire under
fixed parameters. That is the right control for asking whether adaptive effort helps, and the
wrong one for asking how good the solver is, because the whole family it is compared against
shares one ceiling. LKH does not share it.

LKH is not a single point either, so comparing against one LKH run is the same mistake in the
other direction. ``elkai`` exposes ``runs``, which is LKH's own effort dial, so LKH gets swept
too and the comparison is curve against curve. What that makes visible is a **crossover**
rather than a winner.

**Four arms of ours, and what separates them.** An earlier version of this script measured
only the two that contain no fuzzy inference at all, which meant the headline question — does
the FIS beat LKH anywhere — had no measurement behind it. The FIS now enters twice, and the
two entries are different claims:

* ``fis_ls`` — the fitted ``EFFORT`` + ``CHAIN`` local search, one point per instance. This is
  the arm ``benchmark.py`` reports, placed on the same axes as LKH for the first time.
* the **iterated 2x2** — once perturbation is in play the FIS can act in two independent
  places, and they are measured as a full factorial rather than bundled, because a bundle that
  moves the number cannot say which half moved it. ``EFFORT`` can *aim the kicks*, so a
  perturbation lands where the rule base judges there is something to find rather than
  uniformly at random; ``CHAIN`` can control how deep each seeded re-optimisation goes. That
  gives ``iterated`` (neither, the control), ``iterated_aim``, ``iterated_chain`` and
  ``iterated_fis`` (both). The loop, the budgets, the seed and the starting tour are identical
  across all four, so every difference is attributable to the inference.

The first draft of this bundled aiming and chain control into one arm. On the two instances
that measured, the bundle was faster and worse, and there was no way to tell whether the
speed came from the aiming and the loss from the chain or the other way round. The factorial
costs twice the runtime of our own arms, which is a rounding error against LKH's.

Two caveats bound what this can show, and both are properties of ``elkai`` rather than of LKH:

* it takes no time limit, only a run count, so its cheapest available point is one full run.
  LKH has no arbitrarily-fast regime through this interface, and where that floor sits relative
  to our curve is most of the answer;
* it builds the problem as EUC_2D, so CEIL_2D instances would be mis-scored and are skipped.

**Instances are test instances.** The previous default list led with ``pr1002`` and ``d1291``,
both of which ``tune_opt.py`` *fits* on — so the fitted arm would have been shown its own
training data in the one comparison that reaches outside this project. The default ladder is
now drawn from ``benchmark.TEST``, and any instance from the fitting pools is refused rather
than warned about, because a warning in a long run is a line nobody reads.

Run:  python lkh_compare.py [--instances rat783 pcb1173 pr2392] [--scale small]
      python lkh_compare.py --ladder            # the full size ladder; see --dry-run first
      python lkh_compare.py --dry-run           # what it would measure, and the LKH cost
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import fis
import paths
from core import build_candidates, greedy_edge_tour
from fis_lk import local_search as fis_ls_run
from kick import effort_weights, iterated_lk
from lk import lk_solve
from tsplib import load, reference_length, validate_tour

# The fixed-parameter sweep, as the control every earlier section used.
LK_GRID = [(32, 2, 4), (32, 3, 8), (32, 6, 8), (32, 6, 32), (32, 10, 32), (48, 10, 32)]

# Kick budgets for the iterated arms. Spread over three orders of magnitude, because the whole
# question is what the curve does as effort grows rather than where one point lands.
KICKS = [0, 100, 400, 1600, 6400, 25600, 102400]

# LKH's own effort dial.
LKH_RUNS = [1, 2, 5]

# A size ladder over test instances only, EUC_2D only (elkai mis-scores CEIL_2D, which is why
# pla7397 is absent despite fitting the size range). Roughly geometric in n, so that the trend
# in LKH's floor against n has evenly spaced support.
LADDER = ["rat783", "pcb1173", "rl1323", "pr2392", "pcb3038", "fnl4461", "rl5915"]

# The short default: the same ladder truncated where LKH is still affordable in minutes rather
# than tens of minutes. ``--ladder`` opts into the rest.
DEFAULT_INSTANCES = ["rat783", "pcb1173", "pr2392"]

K = 32
OR_SEG = 3
WINDOW = 24
FIS_DEPTH = 10

# Re-optimisation parameters for the iterated arms. Fixed across all four, so that the only
# differences between them are the two factors below.
IT_DEPTH = 6
IT_DEEP = 32

#: The iterated 2x2: (name, EFFORT aims the kicks, CHAIN controls re-optimisation depth).
IT_ARMS = (
    ("iterated", False, False),
    ("iterated_aim", True, False),
    ("iterated_chain", False, True),
    ("iterated_fis", True, True),
)


def _tmin(fn, reps=3):
    best_t = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return out, best_t


def _check_held_out(names):
    """Refuse any instance the rule base was fitted or selected on.

    Imported lazily: ``tune_opt`` pulls in the ``optimizers`` library and the cost model, and
    this script has to stay usable on a machine that only wants to measure.
    """
    import tune_opt

    fitted = set(tune_opt.TRAIN_REAL) | set(tune_opt.VALID_REAL)
    leaked = [n for n in names if n in fitted]
    if leaked:
        raise SystemExit(
            f"refusing to measure on {', '.join(leaked)}: the rule base was fitted or "
            f"selected on these, so a comparison against LKH here would be reporting "
            f"training performance. Pick from benchmark.TEST."
        )


def lkh_curve(inst, runs_list, timeout=1800):
    """(runs, gap, seconds) for LKH at each run count, in a subprocess with a timeout.

    Subprocessed because a native solver that hangs or dies takes the whole measurement with it
    otherwise, and because the memory it holds is worth reclaiming promptly.
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
            "sys.path.insert(0, %r);" % str(paths.ROOT)
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


def measure(inst, tuned, kicks=KICKS, reps=3):
    """Every arm of ours on one instance, each re-scored from the coordinates.

    ``tuned`` is a :class:`fis.Tuned`. Candidate-list construction and the starting tour are
    charged to every arm that uses them, so no arm is quietly given free setup.
    """
    (cand, cand_d), t_cand = _tmin(lambda: build_candidates(inst.coords, K, inst.ceil), reps)
    start, t_start = _tmin(lambda: greedy_edge_tour(inst.coords, cand, inst.ceil), reps)
    uniform = np.empty(0, np.float64)
    base = t_cand + t_start

    sweep = []
    for k, depth, deep in LK_GRID:
        if k == K:
            c2, cd2, s2, t2 = cand, cand_d, start, base
        else:
            (c2, cd2), tc2 = _tmin(
                lambda k=k: build_candidates(inst.coords, k, inst.ceil), reps
            )
            s2, ts2 = _tmin(lambda c2=c2: greedy_edge_tour(inst.coords, c2, inst.ceil), reps)
            t2 = tc2 + ts2
        (res, dt) = _tmin(
            lambda c2=c2, cd2=cd2, s2=s2, k=k, d=depth, b=deep: lk_solve(
                inst.coords, c2, cd2, inst.ceil, s2, k, d, b, OR_SEG
            ),
            reps,
        )
        tour, length, _ = res
        validate_tour(tour, inst.n)
        assert abs(length - reference_length(tour, inst)) < 1e-6, "reported length disagrees"
        sweep.append({"cfg": f"k{k}/d{depth}/b{deep}", "gap": inst.gap(length), "s": dt + t2})

    # --- the FIS local search, one point: the arm benchmark.py reports, on these axes.
    (res, dt) = _tmin(
        lambda: fis_ls_run(
            inst, cand, cand_d, start, tuned.effort_cons, tuned.chain_cons,
            FIS_DEPTH, OR_SEG, False, True,
            tuned.effort_tab, tuned.chain_tab, tuned.effort_ant, tuned.chain_ant,
        ),
        reps,
    )
    tour = res[0]
    validate_tour(tour, inst.n)
    fis_point = {"gap": inst.gap(reference_length(tour, inst)), "s": dt + base}

    # --- the iterated 2x2. Identical loop, budgets, seed and starting tour; the two factors
    #     are whether EFFORT aims the kicks and whether CHAIN runs inside re-optimisation.
    (weights, t_weights) = _tmin(
        lambda: effort_weights(inst, cand, cand_d, start, tuned=tuned), reps
    )

    def _iterated(nk, aim, chain):
        ch = (
            (True, tuned.chain_tab, tuned.chain_ant, tuned.chain_cons)
            if chain
            else (False, fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS)
        )
        return iterated_lk(
            inst.coords, cand, cand_d, inst.ceil, start,
            K, IT_DEPTH, IT_DEEP, OR_SEG, nk, WINDOW, 12345,
            weights if aim else uniform, *ch,
        )

    arms = {}
    for tag, aim, chain in IT_ARMS:
        arms[tag] = []
        for nk in kicks:
            (res, dt) = _tmin(
                lambda nk=nk, a=aim, c=chain: _iterated(nk, a, c),
                reps=2 if nk > 5000 else reps,
            )
            tour, length, _ = res
            validate_tour(tour, inst.n)
            assert abs(length - reference_length(tour, inst)) < 1e-6, "reported length disagrees"
            # The aim is computed once per solve, so an aiming arm carries its cost at every
            # budget rather than having it amortised away by the largest one.
            overhead = t_weights if aim else 0.0
            arms[tag].append({"kicks": nk, "gap": inst.gap(length), "s": dt + base + overhead})

    return {"n": inst.n, "sweep": sweep, "fis_ls": fis_point, **arms}


# --------------------------------------------------------------------------------------
# reading the result: where, if anywhere, are we not dominated by LKH
# --------------------------------------------------------------------------------------
OUR_ARMS = ("sweep", "fis_ls") + tuple(a[0] for a in IT_ARMS)


def _points(d, arm):
    """(time, gap) for one arm, however that arm stores its points."""
    if arm == "fis_ls":
        p = d.get("fis_ls")
        return [(p["s"], p["gap"])] if p else []
    return [(r["s"], r["gap"]) for r in d.get(arm, []) if r.get("gap") is not None]


def verdict(d):
    """Where each arm stands against LKH on one instance.

    Three distinct things are separated here, because blurring them is the easiest way to
    overclaim:

    * **strictly better** — a point of ours with a shorter tour *and* less wall clock than some
      measured LKH point. This is beating LKH, and nothing else is.
    * **uncontested** — a point of ours below LKH's cheapest available budget. LKH returns
      nothing at all down there, so such a point is non-dominated by construction rather than
      by merit, and it is a property of ``elkai``'s run-count interface as much as of LKH.
    * **dominated** — LKH reaches at least this quality in at most this time.
    """
    lkh = [(r["s"], r["gap"]) for r in d.get("lkh", []) if r.get("gap") is not None]
    out = {"lkh_floor": min((t for t, _ in lkh), default=None), "arms": {}}
    for arm in OUR_ARMS:
        pts = _points(d, arm)
        if not pts:
            continue
        better = [
            {"s": t, "gap": g, "vs_lkh_s": lt, "vs_lkh_gap": lg}
            for t, g in pts
            for lt, lg in lkh
            if g < lg - 1e-12 and t < lt
        ]
        under = [(t, g) for t, g in pts if out["lkh_floor"] is None or t < out["lkh_floor"]]
        out["arms"][arm] = {
            "strictly_better_than_some_lkh_point": bool(better),
            "best_strictly_better": min(better, key=lambda r: r["gap"]) if better else None,
            "best_gap_below_lkh_floor": min((g for _, g in under), default=None),
            "best_gap_overall": min(g for _, g in pts),
            "s_at_best_gap": min(pts, key=lambda p: p[1])[0],
        }
    return out


def matched_time_gap(d, arm, seconds):
    """What ``arm`` reaches at a given wall clock, by interpolating its own staircase.

    A comparison at the arm's *own* best budget is not a comparison — the arms do not spend
    the same time at the same kick count, and the fuzzy ones deliberately spend less. So the
    2x2 is read at a common budget, taken to be the control's dearest point, and an arm that
    never reaches that budget is reported as missing rather than extrapolated.
    """
    pts = sorted(_points(d, arm))
    if not pts or seconds < pts[0][0]:
        return None
    best = float("inf")
    for t, g in pts:
        if t > seconds:
            break
        best = min(best, g)
    return best if best < float("inf") else None


def scaling(data):
    """How the picture moves with n — the part that is a trend rather than a data point.

    Reported per instance rather than as a fitted exponent, so the reader sees the trend and
    its scatter instead of one number standing in for both.
    """
    rows = []
    for name, d in sorted(data.items(), key=lambda kv: kv[1].get("n", 0)):
        v = verdict(d)
        row = {"name": name, "n": d.get("n"), "lkh_floor": v["lkh_floor"]}
        lkh = [(r["s"], r["gap"]) for r in d.get("lkh", []) if r.get("gap") is not None]
        row["lkh_best_gap"] = min((g for _, g in lkh), default=None)
        # The common budget for the 2x2: what the control spends at its dearest point.
        ctrl = _points(d, "iterated")
        budget = max((t for t, _ in ctrl), default=None)
        row["matched_budget_s"] = budget
        for arm, a in v["arms"].items():
            row[f"{arm}_under_floor"] = a["best_gap_below_lkh_floor"]
            row[f"{arm}_best_gap"] = a["best_gap_overall"]
            row[f"{arm}_s_at_best"] = a["s_at_best_gap"]
            row[f"{arm}_beats_lkh"] = a["strictly_better_than_some_lkh_point"]
            if budget:
                row[f"{arm}_at_budget"] = matched_time_gap(d, arm, budget)
        rows.append(row)
    return rows


def report(data):
    """The two questions this script exists to answer, printed as two tables."""
    rows = scaling(data)
    it = [a[0] for a in IT_ARMS]

    print("\n--- against LKH -------------------------------------------------------------")
    print(f"{'instance':>10s} {'n':>6s} {'LKH floor':>10s} {'LKH best':>9s} "
          f"{'our best':>9s} {'at':>8s}   beats LKH?")
    for r in rows:
        fl = f"{r['lkh_floor']:9.1f}s" if r["lkh_floor"] else "        —"
        lb = f"{r['lkh_best_gap']:8.3f}%" if r["lkh_best_gap"] is not None else "       —"
        ours = [(r[f"{a}_best_gap"], a) for a in OUR_ARMS if r.get(f"{a}_best_gap") is not None]
        best, arm = min(ours) if ours else (None, "")
        ob = f"{best:8.3f}%" if best is not None else "       —"
        at = f"{r.get(f'{arm}_s_at_best', 0):7.2f}s" if arm else "      —"
        beats = [a for a in OUR_ARMS if r.get(f"{a}_beats_lkh")]
        print(f"{r['name']:>10s} {r['n']:6d} {fl} {lb} {ob} {at}"
              f"   {', '.join(beats) if beats else 'no arm'}")
    print("\n'beats LKH' means a shorter tour in strictly less wall clock than some measured")
    print("LKH point. Points below LKH's floor are uncontested, not better — elkai takes a run")
    print("count rather than a time limit, so LKH returns nothing at all down there.")

    print("\n--- what the FIS contributes to the iterated solver, at a matched budget -----")
    print(f"{'instance':>10s} {'n':>6s} {'budget':>8s} " + " ".join(f"{a:>15s}" for a in it))
    for r in rows:
        if not r.get("matched_budget_s"):
            continue
        cells = []
        for a in it:
            v = r.get(f"{a}_at_budget")
            cells.append(f"{v:14.3f}%" if v is not None else "        not run")
        print(f"{r['name']:>10s} {r['n']:6d} {r['matched_budget_s']:7.2f}s " + " ".join(cells))
    print("\nAll four run the same loop from the same tour with the same seed and budgets.")
    print("'aim' = EFFORT chooses where kicks land; 'chain' = CHAIN sets re-optimisation depth.")
    return rows


#: Fallback cost law, from the superseded run in ``results/legacy/``: floors of 3.8s at
#: n=1002, 54.6s at n=2392 and 165.5s at n=3038. Used only when nothing has been measured on
#: this machine yet, since LKH's absolute speed is hardware-dependent and these were not
#: measured here.
_FALLBACK_LAW = (3.5, 3038, 165.5, "the superseded run in results/legacy/")


def _lkh_cost_law(measured_path):
    """(exponent, reference n, reference seconds, provenance) for pricing an LKH run.

    Fitted to whatever floors have actually been measured on *this* machine, because LKH's
    absolute speed is hardware-dependent and an estimate carried over from other hardware is
    the kind of number that gets quoted long after it stopped being true. Two measured floors
    are enough for a log-log slope; below that it falls back to the recorded law and says so.
    """
    p = Path(measured_path)
    if not p.exists():
        return _FALLBACK_LAW
    data = json.loads(p.read_text())
    pts = []
    for d in data.values():
        lkh = [r["s"] for r in d.get("lkh", []) if r.get("s") is not None]
        if lkh and d.get("n"):
            pts.append((d["n"], min(lkh)))
    if len(pts) < 2:
        return _FALLBACK_LAW
    pts.sort()
    ns = np.log(np.array([p[0] for p in pts], float))
    ss = np.log(np.array([p[1] for p in pts], float))
    slope = float(np.polyfit(ns, ss, 1)[0])
    n_ref, s_ref = pts[-1]  # anchor at the largest measured point, not an extrapolated one
    return slope, n_ref, s_ref, f"{len(pts)} floors measured on this machine"


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--ladder", action="store_true", help=f"the full ladder: {' '.join(LADDER)}")
    ap.add_argument("--scale", default=fis.DEFAULT_SCALE,
                    choices=sorted(fis.EFFORT_RULES_BY_SCALE))
    ap.add_argument("--tuned", default=None, help="default: results/tuned_<scale>.npz")
    ap.add_argument("--hand-written", action="store_true",
                    help="use the hand-written rules instead of a fitted file")
    ap.add_argument("--kicks", nargs="*", type=int, default=KICKS)
    ap.add_argument("--lkh-runs", nargs="*", type=int, default=LKH_RUNS)
    ap.add_argument("--lkh-timeout", type=int, default=1800)
    ap.add_argument("--skip-lkh", action="store_true",
                    help="measure our arms only, keeping any LKH curve already on disk")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(paths.LKH_COMPARE))
    args = ap.parse_args()
    paths.ensure()

    names = args.instances or (LADDER if args.ladder else DEFAULT_INSTANCES)
    _check_held_out(names)

    if args.dry_run:
        print(f"would measure {len(names)} instances: {' '.join(names)}")
        print(f"our arms: {len(LK_GRID)} sweep configs, 1 FIS local search, "
              f"{len(IT_ARMS)} x {len(args.kicks)} kick budgets, x{args.reps} reps")
        print(f"LKH: run counts {args.lkh_runs}, timeout {args.lkh_timeout}s each\n")
        exponent, ref_n, ref_s, source = _lkh_cost_law(args.out)
        print(f"LKH's cost grows steeply — about n^{exponent:.1f}, from {source}. Extrapolating")
        print(f"from {ref_s:.1f}s at n={ref_n} (a rough guide; the tail of the ladder is where")
        print("the budget goes, and LKH's cost varies a lot with instance structure):")
        total = 0.0
        for name in names:
            n = load(name).n
            est = ref_s * (n / ref_n) ** exponent
            capped = sum(min(est * r, args.lkh_timeout) for r in args.lkh_runs)
            total += capped
            print(f"  {name:>10s} n={n:6d}  ~{est:8.0f}s per run, ~{capped / 60:7.1f} min "
                  f"for runs {args.lkh_runs}")
        print(f"\n  estimated LKH total ~{total / 60:.0f} min "
              f"(capped at the {args.lkh_timeout}s timeout)")
        return

    tuned = (
        fis.hand_written(args.scale)
        if args.hand_written
        else fis.load_tuned(args.tuned or paths.tuned(args.scale))
    )
    src = "hand-written" if args.hand_written else str(args.tuned or paths.tuned(args.scale))
    print(f"rule base: {src}  (scale {tuned.scale})")

    # warm every JIT signature before anything is timed
    warm = load("berlin52")
    wc, wcd = build_candidates(warm.coords, K, warm.ceil)
    wg = greedy_edge_tour(warm.coords, wc, warm.ceil)
    lk_solve(warm.coords, wc, wcd, warm.ceil, wg, K, IT_DEPTH, IT_DEEP, OR_SEG)
    fis_ls_run(warm, wc, wcd, wg, tuned.effort_cons, tuned.chain_cons, FIS_DEPTH, OR_SEG,
               False, True, tuned.effort_tab, tuned.chain_tab, tuned.effort_ant, tuned.chain_ant)
    ww = effort_weights(warm, wc, wcd, wg, tuned=tuned)
    for w, ch in (
        (np.empty(0, np.float64),
         (False, fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS)),
        (ww, (True, tuned.chain_tab, tuned.chain_ant, tuned.chain_cons)),
    ):
        iterated_lk(warm.coords, wc, wcd, warm.ceil, wg, K, IT_DEPTH, IT_DEEP, OR_SEG,
                    2, WINDOW, 1, w, *ch)

    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for name in names:
        inst = load(name)
        print(f"\n{inst.name} n={inst.n}", flush=True)
        m = measure(inst, tuned, args.kicks, args.reps)
        print("  fixed-parameter sweep")
        for r in m["sweep"]:
            print(f"    {r['cfg']:>14s} {r['gap']:7.3f}% {r['s']:8.4f}s")
        print(f"  FIS local search  {m['fis_ls']['gap']:7.3f}% {m['fis_ls']['s']:8.4f}s")
        for tag, _, _ in IT_ARMS:
            print(f"  {tag}")
            for r in m[tag]:
                print(f"    {r['kicks']:>10d} kicks {r['gap']:7.3f}% {r['s']:8.4f}s")
        if args.skip_lkh and name in out and "lkh" in out[name]:
            m["lkh"] = out[name]["lkh"]  # keep an expensive curve already measured
            print("  LKH: reusing the curve already on disk")
        else:
            print("  LKH", flush=True)
            m["lkh"] = lkh_curve(inst, args.lkh_runs, args.lkh_timeout)
            for r in m["lkh"]:
                if r["gap"] is None:
                    print(f"    {r['runs']:>10d} runs  {r['why']}")
                else:
                    print(f"    {r['runs']:>10d} runs  {r['gap']:7.3f}% {r['s']:8.4f}s")
        m["verdict"] = verdict(m)
        out[name] = m
        # written after each instance, so an interrupted ladder keeps everything it measured
        Path(args.out).write_text(json.dumps(out, indent=1))

    report(out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
