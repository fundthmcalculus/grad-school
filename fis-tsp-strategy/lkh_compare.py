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

**LKH is swept over its own parameters, not over elkai's one exposed knob.** This matters more
than anything else in the file, and it was wrong here for several revisions.
``elkai.Coordinates2D.solve_tsp`` accepts only ``runs`` and builds the parameter file
``"RUNS = n\nPROBLEM_FILE = :stdin:"``. That leaves ``MAX_TRIALS`` at LKH's default — the
problem *dimension* — so one "run" on a 5915-city instance is 5915 improvement trials. The
resulting 160 s cheapest-available-point was then reported as LKH's floor, and a speed
advantage was claimed against it. It was a default, not a floor.

The underlying ``_elkai.solve_problem(params, problem)`` takes the parameter file as a plain
string, so ``MAX_TRIALS``, ``TIME_LIMIT`` and ``CANDIDATE_SET_TYPE`` are all reachable. Capping
trials and switching to nearest-neighbour candidate sets moves LKH's cheap end by two orders of
magnitude: on rat783 it reaches 0.057% in 0.017 s. Those configurations are in the *default*
sweep rather than behind a flag, because they are the ones that make this comparison
unflattering and a flag is how such a thing quietly stops being run.

One caveat remains, and it is a genuine property of the interface: it builds the problem as
EUC_2D, so CEIL_2D instances would be mis-scored and are skipped.

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

# Above n ~ 10000 a fixed kick count stops meaning the same thing: 102 400 kicks is 17 per city
# at n = 5915 and 5.5 at n = 18512, so the dearest point gets relatively weaker exactly where
# the curve is most interesting. The large ladder therefore extends the budget rather than
# holding it, and drops the cheapest points, which are uninformative at this size.
KICKS_LARGE = [0, 1600, 6400, 25600, 102400, 409600]

# LKH's own effort dial.
LKH_RUNS = [1, 2, 5]

# LKH's *cheap end*, which the ``runs`` dial alone cannot reach.
#
# ``elkai.Coordinates2D.solve_tsp`` exposes only ``runs``, and it builds the parameter file
# ``"RUNS = n\nPROBLEM_FILE = :stdin:"`` — leaving ``MAX_TRIALS`` at LKH's default, which is the
# problem *dimension*. One "run" on a 5915-city instance is therefore 5915 improvement trials,
# and that, not any property of LKH, is what put its cheapest measurable point at 160 s.
#
# The underlying ``_elkai.solve_problem(params, problem)`` takes the parameter file as a plain
# string, so every LKH parameter is reachable. ``MAX_TRIALS`` is the real anytime dial and
# ``TIME_LIMIT`` is the direct one. Sweeping them is the difference between comparing against
# LKH and comparing against elkai's defaults — and an earlier version of this file did the
# latter while claiming the former.
#
# What remains irreducible is LKH's *preprocessing*: the alpha-nearness candidate set is built
# from a Held-Karp 1-tree before any trial runs, and no parameter here skips it. That cost is
# LKH's genuine floor, and it is the number this comparison should be reported against.
# And the preprocessing is optional too. ``CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR`` replaces the
# 1-tree construction with plain k-nearest — the same candidate rule this project's own solver
# uses — which removes the last thing holding LKH's floor up. On rat783 that combination reaches
# 0.057% in 0.017 s, which is better quality than anything here reaches at any budget, sooner
# than our cheapest point. These configurations are in the default sweep rather than behind a
# flag precisely because they are the ones that make the comparison unflattering.
_NN = {"CANDIDATE_SET_TYPE": "NEAREST-NEIGHBOR", "MAX_CANDIDATES": 5}

# ...at *our* candidate budget, which is the fair control. LKH's default MAX_CANDIDATES is 5;
# this project's solver builds 32-nearest lists. Handing LKH 5 nearest neighbours while we use
# 32 is not "LKH without alpha-nearness", it is a differently-handicapped solver — and on the
# rl* instances, whose coordinates are grid-like and tie-heavy, 5 nearest neighbours is
# catastrophic (18.4% on rl5915 against 0.45% with alpha-nearness). Matching the budget is what
# separates "alpha-nearness is what LKH needs here" from "five candidates is too few".
_NN32 = {"CANDIDATE_SET_TYPE": "NEAREST-NEIGHBOR", "MAX_CANDIDATES": 32}

LKH_CONFIGS = [
    # cheap end: k-nearest candidates at LKH's own default breadth, trials capped
    {"label": "nn5/trials=1", "MAX_TRIALS": 1, "RUNS": 1, **_NN},
    {"label": "nn5/trials=25", "MAX_TRIALS": 25, "RUNS": 1, **_NN},
    {"label": "nn5/trials=100", "MAX_TRIALS": 100, "RUNS": 1, **_NN},
    {"label": "nn5/trials=1000", "MAX_TRIALS": 1000, "RUNS": 1, **_NN},
    # the same, at this project's candidate budget of 32
    {"label": "nn32/trials=1", "MAX_TRIALS": 1, "RUNS": 1, **_NN32},
    {"label": "nn32/trials=25", "MAX_TRIALS": 25, "RUNS": 1, **_NN32},
    {"label": "nn32/trials=100", "MAX_TRIALS": 100, "RUNS": 1, **_NN32},
    {"label": "nn32/trials=1000", "MAX_TRIALS": 1000, "RUNS": 1, **_NN32},
    # LKH's own candidate rule (alpha-nearness on a 1-tree), trials capped
    {"label": "alpha/trials=1", "MAX_TRIALS": 1, "RUNS": 1},
    {"label": "alpha/trials=25", "MAX_TRIALS": 25, "RUNS": 1},
    {"label": "alpha/trials=100", "MAX_TRIALS": 100, "RUNS": 1},
    # what elkai's interface actually reaches: MAX_TRIALS left at the default of n
    {"label": "runs=1 (elkai default)", "RUNS": 1},
    {"label": "runs=2", "RUNS": 2},
    {"label": "runs=5", "RUNS": 5},
]

# Kept as a name for the two configs that were once opt-in, so ``--lkh-cheap-candidates``
# remains meaningful; they are now a subset of the default sweep.
LKH_CONFIGS_CHEAP_CANDIDATES = [c for c in LKH_CONFIGS if c["label"].startswith("nn/")]

# A size ladder over test instances only, EUC_2D only (elkai mis-scores CEIL_2D, which is why
# pla7397 is absent despite fitting the size range). Roughly geometric in n, so that the trend
# in LKH's floor against n has evenly spaced support.
LADDER = ["rat783", "pcb1173", "rl1323", "pr2392", "pcb3038", "fnl4461", "rl5915"]

# The instances LKH *cannot solve*, which is where any claim against it has to be made.
#
# Zheng et al. (VSR-LKH, AAAI 2021, Table 3) run LKH 10 times on all 111 symmetric TSPLIB
# instances and split them into 74 "easy" — LKH reaches the published optimum every time, with
# no runtime difference against their own method — and 37 "hard". Within the hard set there is
# a sharper subset: instances where LKH's success rate is **0/10**, so it never reaches the
# optimum in any run. Those are:
#
#     fl1577  rl1889  d2103  fl3795  rl5915  rl5934  brd14051  d15112  d18512  pla33810  pla85900
#
# Of those, rl1889 and rl5934 are in this project's *fitting* pools and cannot be used; pla33810
# and pla85900 are CEIL_2D, which elkai would mis-score. The remaining seven are all already in
# ``benchmark.TEST``, held out, and EUC_2D — so this ladder needs no new instances, only the
# recognition that the previous one was drawn almost entirely from the easy 74.
#
# This matters more than the size ladder. On the easy instances LKH is simply optimal and the
# only available claim is about the time axis; here there is a quality gap to compete for.
LADDER_HARD = ["fl1577", "d2103", "fl3795", "rl5915", "brd14051", "d15112", "d18512"]

# Hard for LKH but not hopeless — success 1-3 of 10 runs — and held out. rat575, vm1084, rl1304,
# fl1400, u1817 and u2152 also qualify but sit in the fitting pools.
LADDER_HARD_EXTRA = ["rl11849", "usa13509"]

# The rest of the test set that elkai can score, n = 11849…18512. Split out because LKH's cost
# here is tens of minutes per *run*, so this is opted into separately and normally with
# ``--lkh-runs 1``: LKH has returned the published optimum at one run on every instance
# measured, so further run counts buy curve shape at several times the price.
# pla7397 is absent despite fitting the range — it is CEIL_2D, which elkai would mis-score.
LADDER_XL = ["rl11849", "usa13509", "brd14051", "d15112", "d18512"]

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


_LKH_CHILD = r"""
import sys, time
sys.path.insert(0, {root!r})
import numpy as np
import elkai._elkai as _elkai
from tsplib import load, reference_length, validate_tour

inst = load({name!r})
# Built by hand rather than through elkai.Coordinates2D, which hard-codes the parameter file to
# "RUNS = n" and so cannot reach MAX_TRIALS, TIME_LIMIT or the candidate-set options. The
# problem file is the same EUC_2D coordinate section elkai would have written.
lines = ["TYPE : TSP", "DIMENSION : %d" % inst.n, "EDGE_WEIGHT_TYPE : EUC_2D",
         "NODE_COORD_SECTION"]
for i, (x, y) in enumerate(inst.coords):
    lines.append("%d %r %r" % (i + 1, float(x), float(y)))
problem = "\n".join(lines) + "\n"
params = {params!r}

t0 = time.perf_counter()
sol = _elkai.solve_problem(params, problem)
dt = time.perf_counter() - t0

tour = np.array([int(v) - 1 for v in sol], dtype=np.int32)   # LKH is one-indexed
if tour.shape[0] == inst.n + 1:
    tour = tour[:-1]
validate_tour(tour, inst.n)
print(reference_length(tour, inst), dt)
"""


def _lkh_configs(args):
    """The LKH parameter sets to measure, from the flags.

    ``--lkh-runs`` is kept as a legacy escape hatch reproducing the old behaviour — RUNS only,
    MAX_TRIALS left at LKH's default of n — because the numbers in FINDINGS before this change
    were produced that way and have to stay reproducible.
    """
    if args.lkh_runs:
        return [{"label": f"runs={r}", "RUNS": r} for r in args.lkh_runs]
    cfgs = list(LKH_CONFIGS)
    if args.lkh_cheap_candidates:
        cfgs += LKH_CONFIGS_CHEAP_CANDIDATES
    if args.lkh_labels:
        # Prefix match, so ``--lkh-labels nn32`` selects the whole family. Exists because the
        # sweep now spans three orders of magnitude in cost and a single decisive control
        # should not require paying for the configurations already measured.
        cfgs = [
            c for c in cfgs if any(c["label"].startswith(p) for p in args.lkh_labels)
        ]
        if not cfgs:
            raise SystemExit(
                f"no config label starts with any of {args.lkh_labels}; available: "
                + ", ".join(c["label"] for c in LKH_CONFIGS)
            )
    return cfgs


def _lkh_weights(args):
    """Rough cost of each config relative to one default run, for ``--dry-run`` pricing.

    MAX_TRIALS caps trials at that number against a default of n, so its cost is roughly the
    preprocessing plus a fraction of a run. Crude on purpose — the point is an order of
    magnitude, and the measured exponent already swings by a factor of six.
    """
    out = []
    for cfg in _lkh_configs(args):
        runs = cfg.get("RUNS", 1)
        out.append(0.15 * runs if "MAX_TRIALS" in cfg else float(runs))
    return out


def _params_text(cfg):
    """An LKH parameter file from a config dict, minus the display label."""
    body = "".join(f"{k} = {v}\n" for k, v in cfg.items() if k != "label")
    return body + "PROBLEM_FILE = :stdin:\n"


def lkh_curve(inst, configs, timeout=1800):
    """(label, gap, seconds) for LKH under each parameter set, subprocessed with a timeout.

    Subprocessed because a native solver that hangs or dies takes the whole measurement with it
    otherwise, and because the memory it holds is worth reclaiming promptly.

    A note on what is *not* varied: the coordinate interface is used rather than a distance
    matrix. Handing LKH a dense matrix disables its own geometric preprocessing — the
    alpha-nearness candidate set built from a Held-Karp 1-tree — and it becomes drastically
    slower: one run on pr1002 does not finish in 420 s through the matrix interface and takes
    about a second through this one. Measuring that path would have understated LKH by orders
    of magnitude, which is the opposite of the error worth making here.

    Configs are measured in the order given and a timeout does *not* stop the sweep, because
    these are no longer monotone in cost — ``trials=1`` is cheap and ``runs=5`` is dear, and a
    stall on one says nothing about the next.
    """
    import subprocess
    import sys

    out = []
    for cfg in configs:
        label = cfg.get("label", _params_text(cfg).replace("\n", " ").strip())
        code = _LKH_CHILD.format(
            root=str(paths.ROOT), name=inst.name, params=_params_text(cfg)
        )
        row = {"cfg": label, "params": {k: v for k, v in cfg.items() if k != "label"}}
        try:
            r = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if r.returncode != 0:
                out.append(
                    {
                        **row,
                        "gap": None,
                        "s": None,
                        "why": (r.stderr or "failed").strip()[-200:],
                    }
                )
                continue
            length, dt = r.stdout.split()
            out.append(
                {**row, "gap": inst.gap(float(length)), "s": float(dt), "why": "ok"}
            )
        except subprocess.TimeoutExpired:
            out.append({**row, "gap": None, "s": None, "why": f"timeout>{timeout}s"})
    return out


def measure(inst, tuned, kicks=KICKS, reps=3):
    """Every arm of ours on one instance, each re-scored from the coordinates.

    ``tuned`` is a :class:`fis.Tuned`. Candidate-list construction and the starting tour are
    charged to every arm that uses them, so no arm is quietly given free setup.
    """
    (cand, cand_d), t_cand = _tmin(
        lambda: build_candidates(inst.coords, K, inst.ceil), reps
    )
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
            s2, ts2 = _tmin(
                lambda c2=c2: greedy_edge_tour(inst.coords, c2, inst.ceil), reps
            )
            t2 = tc2 + ts2
        res, dt = _tmin(
            lambda c2=c2, cd2=cd2, s2=s2, k=k, d=depth, b=deep: lk_solve(
                inst.coords, c2, cd2, inst.ceil, s2, k, d, b, OR_SEG
            ),
            reps,
        )
        tour, length, _ = res
        validate_tour(tour, inst.n)
        assert (
            abs(length - reference_length(tour, inst)) < 1e-6
        ), "reported length disagrees"
        sweep.append(
            {"cfg": f"k{k}/d{depth}/b{deep}", "gap": inst.gap(length), "s": dt + t2}
        )

    # --- the FIS local search, one point: the arm benchmark.py reports, on these axes.
    res, dt = _tmin(
        lambda: fis_ls_run(
            inst,
            cand,
            cand_d,
            start,
            tuned.effort_cons,
            tuned.chain_cons,
            FIS_DEPTH,
            OR_SEG,
            False,
            True,
            tuned.effort_tab,
            tuned.chain_tab,
            tuned.effort_ant,
            tuned.chain_ant,
        ),
        reps,
    )
    tour = res[0]
    validate_tour(tour, inst.n)
    fis_point = {"gap": inst.gap(reference_length(tour, inst)), "s": dt + base}

    # --- the iterated 2x2. Identical loop, budgets, seed and starting tour; the two factors
    #     are whether EFFORT aims the kicks and whether CHAIN runs inside re-optimisation.
    weights, t_weights = _tmin(
        lambda: effort_weights(inst, cand, cand_d, start, tuned=tuned), reps
    )

    def _iterated(nk, aim, chain):
        ch = (
            (True, tuned.chain_tab, tuned.chain_ant, tuned.chain_cons)
            if chain
            else (False, fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS)
        )
        return iterated_lk(
            inst.coords,
            cand,
            cand_d,
            inst.ceil,
            start,
            K,
            IT_DEPTH,
            IT_DEEP,
            OR_SEG,
            nk,
            WINDOW,
            12345,
            weights if aim else uniform,
            *ch,
        )

    arms = {}
    for tag, aim, chain in IT_ARMS:
        arms[tag] = []
        for nk in kicks:
            res, dt = _tmin(
                lambda nk=nk, a=aim, c=chain: _iterated(nk, a, c),
                reps=2 if nk > 5000 else reps,
            )
            tour, length, _ = res
            validate_tour(tour, inst.n)
            assert (
                abs(length - reference_length(tour, inst)) < 1e-6
            ), "reported length disagrees"
            # The aim is computed once per solve, so an aiming arm carries its cost at every
            # budget rather than having it amortised away by the largest one.
            overhead = t_weights if aim else 0.0
            arms[tag].append(
                {"kicks": nk, "gap": inst.gap(length), "s": dt + base + overhead}
            )

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
        # Non-domination against LKH's *whole* measured set, not against some point of it.
        # "Better than some LKH point" is nearly free once LKH is swept over parameters — it
        # can be satisfied by beating a badly-configured one, which is not a result. A point
        # of ours survives only if no LKH configuration is at least as good on both axes.
        nd = [
            {"s": t, "gap": g}
            for t, g in pts
            if not any(lt <= t + 1e-12 and lg <= g + 1e-12 for lt, lg in lkh)
        ]
        under = [
            (t, g) for t, g in pts if out["lkh_floor"] is None or t < out["lkh_floor"]
        ]
        out["arms"][arm] = {
            "non_dominated_by_lkh": bool(nd),
            "n_non_dominated": len(nd),
            "best_non_dominated": min(nd, key=lambda r: r["gap"]) if nd else None,
            # the wall-clock window over which we are non-dominated, which is the honest
            # statement of "where, if anywhere" — a range, not a flag
            "non_dominated_window_s": (
                [min(r["s"] for r in nd), max(r["s"] for r in nd)] if nd else None
            ),
            "best_gap_below_lkh_floor": min((g for _, g in under), default=None),
            "best_gap_overall": min(g for _, g in pts),
            "s_at_best_gap": min(pts, key=lambda p: p[1])[0],
        }
    return out


#: Quality thresholds for the anytime question, in % over the published optimum. Chosen to
#: bracket what the iterated arms actually reach (0.3-1.5%) rather than to be round numbers.
THRESHOLDS = (2.0, 1.0, 0.5)


def time_to_within(d, arm, threshold):
    """The cheapest budget at which ``arm`` reaches within ``threshold`` % of the optimum.

    This is the question "how long until it is good enough", which is the one that matters for
    a solver being used rather than benchmarked, and it is not answerable from the best-tour
    column: an arm that reaches 0.4% in 30 s and one that reaches 0.4% in 3 s look identical
    there. Returns None if the arm never gets there at any measured budget — reported as a miss
    rather than extrapolated, since the curves flatten and extrapolating along a plateau would
    invent a crossing that does not happen.
    """
    reached = [t for t, g in _points(d, arm) if g <= threshold]
    return min(reached) if reached else None


def anytime(d):
    """Per-arm time-to-threshold, and how that compares with LKH's cheapest available budget.

    The speedup is against LKH's *floor*, not against LKH at matched quality, and the
    distinction is the whole caveat: LKH returns the optimum there, so reaching 1% in a
    fiftieth of that time is a statement about the time axis and not a claim to have matched it.
    """
    lkh = [(r["s"], r["gap"]) for r in d.get("lkh", []) if r.get("gap") is not None]
    floor = min((t for t, _ in lkh), default=None)
    out = {"lkh_floor": floor, "arms": {}}
    for arm in OUR_ARMS:
        if not _points(d, arm):
            continue
        rec = {}
        for th in THRESHOLDS:
            t = time_to_within(d, arm, th)
            rec[f"t_within_{th}"] = t
            rec[f"speedup_vs_lkh_floor_{th}"] = (floor / t) if (t and floor) else None
        out["arms"][arm] = rec
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
        at = anytime(d)
        for arm, rec in at["arms"].items():
            row.update({f"{arm}_{k}": val for k, val in rec.items()})
        for arm, a in v["arms"].items():
            row[f"{arm}_under_floor"] = a["best_gap_below_lkh_floor"]
            row[f"{arm}_best_gap"] = a["best_gap_overall"]
            row[f"{arm}_s_at_best"] = a["s_at_best_gap"]
            row[f"{arm}_nd"] = a["non_dominated_by_lkh"]
            row[f"{arm}_nd_window"] = a["non_dominated_window_s"]
            if budget:
                row[f"{arm}_at_budget"] = matched_time_gap(d, arm, budget)
        rows.append(row)
    return rows


def _window(row, arms):
    """The union of the non-dominated wall-clock windows across ``arms``, as a short string."""
    wins = [row.get(f"{a}_nd_window") for a in arms if row.get(f"{a}_nd_window")]
    if not wins:
        return "dominated everywhere"
    lo = min(w[0] for w in wins)
    hi = max(w[1] for w in wins)
    return f"{lo:.2f}-{hi:.2f}s  ({', '.join(arms)})"


def report(data):
    """The two questions this script exists to answer, printed as two tables."""
    rows = scaling(data)
    it = [a[0] for a in IT_ARMS]

    print(
        "\n--- against LKH -------------------------------------------------------------"
    )
    print(
        f"{'instance':>10s} {'n':>6s} {'LKH floor':>10s} {'LKH best':>9s} "
        f"{'our best':>9s} {'at':>8s}   non-dominated window"
    )
    for r in rows:
        fl = f"{r['lkh_floor']:9.1f}s" if r["lkh_floor"] else "        —"
        lb = (
            f"{r['lkh_best_gap']:8.3f}%"
            if r["lkh_best_gap"] is not None
            else "       —"
        )
        ours = [
            (r[f"{a}_best_gap"], a)
            for a in OUR_ARMS
            if r.get(f"{a}_best_gap") is not None
        ]
        best, arm = min(ours) if ours else (None, "")
        ob = f"{best:8.3f}%" if best is not None else "       —"
        at = f"{r.get(f'{arm}_s_at_best', 0):7.2f}s" if arm else "      —"
        beats = [a for a in OUR_ARMS if r.get(f"{a}_nd")]
        print(
            f"{r['name']:>10s} {r['n']:6d} {fl} {lb} {ob} {at}"
            f"   {_window(r, beats)}"
        )
    print(
        "\nA point is non-dominated when *no* measured LKH configuration matches it on both"
    )
    print(
        "axes. The window is the wall-clock range over which that holds, and it is the only"
    )
    print(
        "honest form of 'where, if anywhere, do we win' — 'better than some LKH point' is"
    )
    print(
        "nearly free once LKH is swept, since it can be satisfied by beating a bad config."
    )

    print(
        "\n--- what the FIS contributes to the iterated solver, at a matched budget -----"
    )
    print(
        f"{'instance':>10s} {'n':>6s} {'budget':>8s} "
        + " ".join(f"{a:>15s}" for a in it)
    )
    for r in rows:
        if not r.get("matched_budget_s"):
            continue
        cells = []
        for a in it:
            v = r.get(f"{a}_at_budget")
            cells.append(f"{v:14.3f}%" if v is not None else "        not run")
        print(
            f"{r['name']:>10s} {r['n']:6d} {r['matched_budget_s']:7.2f}s "
            + " ".join(cells)
        )
    print(
        "\nAll four run the same loop from the same tour with the same seed and budgets."
    )
    print(
        "'aim' = EFFORT chooses where kicks land; 'chain' = CHAIN sets re-optimisation depth."
    )

    print(
        "\n--- how fast is 'good enough': best arm's time to reach each threshold ---"
    )
    head = "  ".join(f"within {th}%".rjust(17) for th in THRESHOLDS)
    print(f"{'instance':>10s} {'n':>6s} {'LKH floor':>10s}  {head}")
    for r in rows:
        cells = []
        for th in THRESHOLDS:
            best = None
            for arm in OUR_ARMS:
                t = r.get(f"{arm}_t_within_{th}")
                if t is not None and (best is None or t < best[0]):
                    best = (t, arm)
            if best is None:
                cells.append("        never".rjust(17))
            elif r.get("lkh_floor"):
                cells.append(
                    f"{best[0]:8.2f}s ({r['lkh_floor'] / best[0]:5.0f}x)".rjust(17)
                )
            else:
                cells.append(f"{best[0]:8.2f}s".rjust(17))
        fl = f"{r['lkh_floor']:9.1f}s" if r.get("lkh_floor") else "        —"
        print(f"{r['name']:>10s} {r['n']:6d} {fl}  " + "  ".join(cells))
    print(
        "\n(Nx) is the ratio to LKH's cheapest measured budget on that instance. Since the"
    )
    print(
        "sweep now includes capped-trial configurations, that cheapest point is no longer the"
    )
    print(
        "optimum — on rl5915 it is 18.4%. The ratio is therefore only a statement about the"
    )
    print(
        "time axis; whether we are ahead at a budget is the non-dominated window above, and"
    )
    print("this table should not be read as a speedup against LKH at matched quality.")
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
    # Anchored on one *named* configuration rather than the cheapest measured, because the
    # sweep is now heterogeneous: taking the minimum would fit the law to nn5/trials=1, which
    # is three orders of magnitude cheaper than the dearest config and would under-price the
    # sweep as a whole by about that much.
    anchor = "runs=1 (elkai default)"
    pts = []
    for d in data.values():
        times = [
            r["s"]
            for r in d.get("lkh", [])
            if r.get("s") is not None and r.get("cfg") == anchor
        ]
        if times and d.get("n"):
            pts.append((d["n"], min(times)))
    if len(pts) < 2:
        return _FALLBACK_LAW
    pts.sort()
    ns = np.log(np.array([p[0] for p in pts], float))
    ss = np.log(np.array([p[1] for p in pts], float))
    slope = float(np.polyfit(ns, ss, 1)[0])
    n_ref, s_ref = pts[
        -1
    ]  # anchor at the largest measured point, not an extrapolated one
    return slope, n_ref, s_ref, f"{len(pts)} floors measured on this machine"


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument(
        "--ladder", action="store_true", help=f"the full ladder: {' '.join(LADDER)}"
    )
    ap.add_argument(
        "--ladder-hard",
        action="store_true",
        help="the held-out instances LKH never solves in 10 runs, per VSR-LKH "
        "Table 3: " + " ".join(LADDER_HARD),
    )
    ap.add_argument(
        "--ladder-xl",
        action="store_true",
        help=f"n=11849..18512: {' '.join(LADDER_XL)} (tens of minutes per LKH run; "
        f"implies the larger kick budgets unless --kicks is given)",
    )
    ap.add_argument(
        "--scale", default=fis.DEFAULT_SCALE, choices=sorted(fis.EFFORT_RULES_BY_SCALE)
    )
    ap.add_argument("--tuned", default=None, help="default: results/tuned_<scale>.npz")
    ap.add_argument(
        "--hand-written",
        action="store_true",
        help="use the hand-written rules instead of a fitted file",
    )
    ap.add_argument("--kicks", nargs="*", type=int, default=None)
    ap.add_argument(
        "--lkh-runs",
        nargs="*",
        type=int,
        default=None,
        help="legacy: sweep RUNS only, leaving MAX_TRIALS at LKH's default of n. "
        "That default is what put LKH's cheapest point minutes away; prefer "
        "the parameter sweep.",
    )
    ap.add_argument(
        "--lkh-labels",
        nargs="*",
        default=None,
        help="only measure configs whose label starts with one of these, e.g. "
        "--lkh-labels nn32 alpha",
    )
    ap.add_argument(
        "--lkh-cheap-candidates",
        action="store_true",
        help="also sweep NEAREST-NEIGHBOR candidate sets, which is the only knob "
        "that touches LKH's alpha-nearness preprocessing",
    )
    ap.add_argument("--lkh-timeout", type=int, default=1800)
    ap.add_argument(
        "--lkh-append",
        action="store_true",
        help="with --lkh-only: measure only configs not already on "
        "disk and keep the rest, instead of re-measuring all",
    )
    ap.add_argument(
        "--lkh-only",
        action="store_true",
        help="re-measure only LKH, keeping our arms already on disk",
    )
    ap.add_argument(
        "--skip-lkh",
        action="store_true",
        help="measure our arms only, keeping any LKH curve already on disk",
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(paths.LKH_COMPARE))
    args = ap.parse_args()
    paths.ensure()

    if args.ladder_hard:
        names = args.instances or LADDER_HARD
    elif args.ladder_xl:
        names = args.instances or LADDER_XL
    else:
        names = args.instances or (LADDER if args.ladder else DEFAULT_INSTANCES)
    if args.kicks is None:
        # the budget follows the size band, so that the dearest point means a comparable
        # number of kicks per city rather than a comparable number of kicks
        args.kicks = KICKS_LARGE if (args.ladder_xl or args.ladder_hard) else KICKS
    _check_held_out(names)

    if args.dry_run:
        print(f"would measure {len(names)} instances: {' '.join(names)}")
        print(
            f"our arms: {len(LK_GRID)} sweep configs, 1 FIS local search, "
            f"{len(IT_ARMS)} x {len(args.kicks)} kick budgets, x{args.reps} reps"
        )
        cfgs = _lkh_configs(args)
        print(
            f"LKH: {len(cfgs)} parameter sets ({', '.join(c['label'] for c in cfgs)}), "
            f"timeout {args.lkh_timeout}s each\n"
        )
        exponent, ref_n, ref_s, source = _lkh_cost_law(args.out)
        print(
            f"LKH's cost grows steeply — about n^{exponent:.1f}, from {source}. Extrapolating"
        )
        print(
            f"from {ref_s:.1f}s at n={ref_n} (a rough guide; the tail of the ladder is where"
        )
        print("the budget goes, and LKH's cost varies a lot with instance structure):")
        total = 0.0
        for name in names:
            n = load(name).n
            est = ref_s * (n / ref_n) ** exponent
            capped = sum(min(est * w, args.lkh_timeout) for w in _lkh_weights(args))
            total += capped
            print(
                f"  {name:>10s} n={n:6d}  ~{est:8.0f}s per run, ~{capped / 60:7.1f} min "
                f"over {len(_lkh_configs(args))} parameter sets"
            )
        print(
            f"\n  estimated LKH total ~{total / 60:.0f} min "
            f"(capped at the {args.lkh_timeout}s timeout)"
        )
        return

    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}

    if args.lkh_only:
        # Re-measure LKH against an unchanged set of our own arms. Worth having as its own
        # path because LKH's parameter sweep has been revised more than once while our arms
        # stayed fixed, and re-running a solver whose numbers are not in question wastes time
        # and perturbs a comparison that was measured in one sitting.
        missing = [n for n in names if n not in out]
        if missing:
            raise SystemExit(
                f"--lkh-only needs our arms already measured, and {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not in {args.out}"
            )
        for name in names:
            inst = load(name)
            print(f"\n{inst.name} n={inst.n}  (LKH only)", flush=True)
            cfgs = _lkh_configs(args)
            if args.lkh_append:
                # Measure only configurations not already on disk, and keep the rest. The
                # expensive end of the sweep (RUNS=1/2/5 at n=5915 is ~20 minutes) does not
                # change when a cheap configuration is added to the list, and re-running it
                # only adds machine-state noise to points that are already comparable.
                have = {r["cfg"] for r in out[name].get("lkh", [])}
                cfgs = [c for c in cfgs if c["label"] not in have]
                print(f"    appending {len(cfgs)} new config(s); keeping {len(have)}")
            fresh = lkh_curve(inst, cfgs, args.lkh_timeout)
            out[name]["lkh"] = (
                (out[name].get("lkh", []) + fresh) if args.lkh_append else fresh
            )
            for r in fresh:
                if r["gap"] is None:
                    print(f"    {r['cfg']:>24s}  {r['why']}")
                else:
                    print(f"    {r['cfg']:>24s}  {r['gap']:7.3f}% {r['s']:8.4f}s")
            out[name]["verdict"] = verdict(out[name])
            Path(args.out).write_text(json.dumps(out, indent=1))
        report(out)
        print(f"\nwrote {args.out}")
        return

    tuned = (
        fis.hand_written(args.scale)
        if args.hand_written
        else fis.load_tuned(args.tuned or paths.tuned(args.scale))
    )
    src = (
        "hand-written"
        if args.hand_written
        else str(args.tuned or paths.tuned(args.scale))
    )
    print(f"rule base: {src}  (scale {tuned.scale})")

    # warm every JIT signature before anything is timed
    warm = load("berlin52")
    wc, wcd = build_candidates(warm.coords, K, warm.ceil)
    wg = greedy_edge_tour(warm.coords, wc, warm.ceil)
    lk_solve(warm.coords, wc, wcd, warm.ceil, wg, K, IT_DEPTH, IT_DEEP, OR_SEG)
    fis_ls_run(
        warm,
        wc,
        wcd,
        wg,
        tuned.effort_cons,
        tuned.chain_cons,
        FIS_DEPTH,
        OR_SEG,
        False,
        True,
        tuned.effort_tab,
        tuned.chain_tab,
        tuned.effort_ant,
        tuned.chain_ant,
    )
    ww = effort_weights(warm, wc, wcd, wg, tuned=tuned)
    for w, ch in (
        (
            np.empty(0, np.float64),
            (False, fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS),
        ),
        (ww, (True, tuned.chain_tab, tuned.chain_ant, tuned.chain_cons)),
    ):
        iterated_lk(
            warm.coords,
            wc,
            wcd,
            warm.ceil,
            wg,
            K,
            IT_DEPTH,
            IT_DEEP,
            OR_SEG,
            2,
            WINDOW,
            1,
            w,
            *ch,
        )

    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for name in names:
        inst = load(name)
        print(f"\n{inst.name} n={inst.n}", flush=True)
        m = measure(inst, tuned, args.kicks, args.reps)
        print("  fixed-parameter sweep")
        for r in m["sweep"]:
            print(f"    {r['cfg']:>14s} {r['gap']:7.3f}% {r['s']:8.4f}s")
        print(
            f"  FIS local search  {m['fis_ls']['gap']:7.3f}% {m['fis_ls']['s']:8.4f}s"
        )
        for tag, _, _ in IT_ARMS:
            print(f"  {tag}")
            for r in m[tag]:
                print(f"    {r['kicks']:>10d} kicks {r['gap']:7.3f}% {r['s']:8.4f}s")
        if args.skip_lkh and name in out and "lkh" in out[name]:
            m["lkh"] = out[name]["lkh"]  # keep an expensive curve already measured
            print("  LKH: reusing the curve already on disk")
        else:
            print("  LKH", flush=True)
            m["lkh"] = lkh_curve(inst, _lkh_configs(args), args.lkh_timeout)
            for r in m["lkh"]:
                if r["gap"] is None:
                    print(f"    {r['cfg']:>24s}  {r['why']}")
                else:
                    print(f"    {r['cfg']:>24s}  {r['gap']:7.3f}% {r['s']:8.4f}s")
        m["verdict"] = verdict(m)
        out[name] = m
        # written after each instance, so an interrupted ladder keeps everything it measured
        Path(args.out).write_text(json.dumps(out, indent=1))

    report(out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
