"""Fit the rule bases with the `optimizers` library.

The genetic algorithm is the default and is what the reported results use. Drivers for
`ParticleSwarmOptimizer` and `AntColonyOptimizer` are kept and selectable with
``--optimizers``, since the objective is the same for all three; they are simply not part
of the reported comparison.

This replaces the hand-rolled (1+1) evolution strategy in ``tune.py`` and widens what
is fitted. Three things changed, in order of how much they matter:

**1. The objective is deterministic.** The old one timed the solver, which is noisy,
irreproducible, and — worst — unmeasurable under parallel evaluation, because
concurrent workers contend for cores. Cost now comes from ``costmodel.py``: a
non-negative least-squares fit of the solver's own work counters to measured wall
clock, accurate to ~3% relative and 0.998 rank correlation. The search optimises that;
``benchmark.py`` still reports real seconds.

**2. Membership functions are fitted too**, not just consequents — every term's centre and
width, which is what decides where one linguistic term stops and the next begins. Because the
bank is compiled to a lookup table (``fis.mf_table``) the functional form is free as well;
triangular beat gaussian on every run that compared them, so it is the default.

**3. Two stages, global then local.** The GA finds the basin; a derivative-free compass search
(``refine.py``) then descends inside it. The GA needs the second stage rather than more of
itself — its own final vector consistently scored worse on validation than the best vector it
passed through, which is what a search without a descent step looks like.

**4. The pool is 20 training and 14 validation instances**, of which 11 and 7 are synthetic
(``synth.py``). Overfitting has been the dominant failure mode throughout, and TSPLIB simply
does not contain enough instances at n >= 1000 outside the test set. Synthetic ones are usable
because the objective is a ratio of two tours on one instance, so no optimum is needed.
On top of that: a shrinkage prior toward the hand-written rules, a 87-parameter vector rather
than 170 (only what the reported arms use), and all selection routed through validation.

Every training instance is disjoint from ``benchmark.TEST``; the assertion at import
time enforces it rather than trusting the lists to stay right.

Run:  python tune_opt.py [--mf-kinds triangular] [--mf-scopes base input]
                        [--optimizers ga|pso|aco] [--generations 40] [--out tuned_opt.npz]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from optimizers import (
    AntColonyOptimizer,
    AntColonyOptimizerConfig,
    GeneticAlgorithmOptimizer,
    GeneticAlgorithmOptimizerConfig,
    ParticleSwarmOptimizer,
    ParticleSwarmOptimizerConfig,
    set_seed,
)
from optimizers.continuous.variables import InputContinuousVariable

import benchmark as bench_mod
import fis
import synth
from core import build_candidates, greedy_edge_tour, nn_stats
from costmodel import features_from_stats
from fis_lk import fis_lk_solve
from lk import lk_solve
from refine import compass_refine
from tsplib import load, reference_length, validate_tour

import paths

# --- instance splits -------------------------------------------------------
# Every fitting instance has n >= MIN_N. That is a deliberate restriction of scope, not a
# convenience: runtime below about a thousand cities is not what this engine is for, and
# including such instances actively distorted the fit. Two ways, both measured:
#
# * the per-city inference overhead is only ever decisive on small instances, so the
#   optimiser spent its budget fighting that instead of allocating effort — an earlier
#   split with 8 of 20 instances under n=200 produced better tours at 1.26x the cost;
# * and once the objective became frontier-relative, small instances had to be scored
#   against LK's *best* achievable gap (since their time is not being scored at all),
#   which is a far harsher bar than the frontier applies anywhere else. With 9 of 18
#   training instances under n=1000 that one term dominated the objective, holding it at
#   q = 1.46 and telling the search almost nothing about the regime that matters.
#
# Small instances are still reported — `benchmark.py` runs from n=52 up, and the findings
# give the size split — they are simply not fitted against.
#
# Families are spread across both splits on purpose: the clustered and structured sets
# (fl*, rl*, u*, dsj*) behave differently enough from the uniform ones that leaving a
# family out of training shows up immediately on test.
MIN_N = 1000

# TSPLIB instances at that size which are not in the test set. There are only sixteen, which
# is what motivated `synth.py`: nine training instances against a fitted vector of this size
# is the whole reason overfitting has dominated every attempt.
TRAIN_REAL = [
    "pr1002",
    "u1060",
    "d1291",
    "rl1304",
    "u1432",
    "d1655",
    "u1817",
    "rl1889",
    "rl5934",
]
VALID_REAL = ["dsj1000", "vm1084", "nrw1379", "fl1400", "vm1748", "u2152", "u2319"]

assert not (set(TRAIN_REAL) & set(VALID_REAL)), "train and validation overlap"
assert not (
    set(TRAIN_REAL) & set(bench_mod.TEST)
), "training instance is in the test set"
assert not (
    set(VALID_REAL) & set(bench_mod.TEST)
), "validation instance is in the test set"


def train_instances():
    """20 instances: 9 real and 11 synthetic, spanning four structural families."""
    return [load(n) for n in TRAIN_REAL] + synth.train_pool()


def valid_instances():
    """14 instances: 7 real and 7 synthetic, disjoint in seed and size from training."""
    return [load(n) for n in VALID_REAL] + synth.valid_pool()


K = 32
FIS_DEPTH = 10
OR_SEG = 3
C_BREADTH = 8

N_TERMS = fis.N_TERMS
W_LO, W_HI = 0.06, 0.60  # membership-function width range


class ParamSpace:
    """Flat [0,1]^d vector <-> the three rule bases' consequents and MF banks.

    Layout, per rule base: consequents, then MF centres, then MF widths. Centres are
    sorted within each input before use, so term 0 is always the leftmost and term 2 the
    rightmost. That costs nothing in expressiveness — the consequents are free to put any
    behaviour on any term — and it keeps LOW / MED / HIGH meaning what their names say,
    which is the whole reason for preferring a rule base to a black box.

    Input widths come from each rule base's own antecedent array rather than a constant,
    so adding an antecedent to ``fis.py`` needs no change here.
    """

    def __init__(self, kind="gaussian", mf_scope="base", scale=fis.DEFAULT_SCALE):
        self.kind = kind
        self.mf_scope = mf_scope
        self.scale = scale
        e_ant, e_cons, e_c, e_w, _ = fis.effort_base(scale)
        h_ant, h_cons, h_c, h_w, _ = fis.chain_base(scale)
        self.bases = {
            "effort": (e_ant, e_cons, e_c, e_w),
            "chain": (h_ant, h_cons, h_c, h_w),
        }
        self.blocks = []
        off = 0
        # CONSTRUCT is deliberately absent. Fuzzy construction is a measured failure (it
        # reaches 48% over optimum against greedy-edge's 17%), it appears in none of the
        # reported arms, and every parameter spent on it is a parameter the GA can overfit
        # with. Fitting only what is used takes the vector from 170 to 87.
        for name in ("effort", "chain"):
            ant, cons, _, _ = self.bases[name]
            n_cons = int(cons.size)
            n_in = int(ant.shape[1])
            # "base": one set of three terms shared by the rule base's inputs (3 centres
            # and 3 widths). "input": every input gets its own terms. The second is more
            # expressive and overfits measurably harder, so it is not the default.
            n_mf = n_in * N_TERMS if mf_scope == "input" else N_TERMS
            self.blocks.append(
                {
                    "name": name,
                    "cons_shape": cons.shape,
                    "n_in": n_in,
                    "cons": slice(off, off + n_cons),
                    "mfc": slice(off + n_cons, off + n_cons + n_mf),
                    "mfw": slice(off + n_cons + n_mf, off + n_cons + 2 * n_mf),
                }
            )
            off += n_cons + 2 * n_mf
        self.size = off

    def default(self):
        """The hand-written rule bases, as a vector — the point to beat."""
        theta = np.empty(self.size)
        for b in self.blocks:
            _, cons, mfc, mfw = self.bases[b["name"]]
            theta[b["cons"]] = cons.ravel()
            if self.mf_scope == "input":
                theta[b["mfc"]] = mfc.ravel()
                theta[b["mfw"]] = (mfw.ravel() - W_LO) / (W_HI - W_LO)
            else:
                theta[b["mfc"]] = mfc[0]
                theta[b["mfw"]] = (mfw[0] - W_LO) / (W_HI - W_LO)
        return np.clip(theta, 0.0, 1.0)

    def decode(self, theta):
        """(cons, membership table) per rule base, ready to hand to the solver."""
        theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
        out = {}
        for b in self.blocks:
            n_in = b["n_in"]
            cons = np.ascontiguousarray(theta[b["cons"]].reshape(b["cons_shape"]))
            if self.mf_scope == "input":
                centres = theta[b["mfc"]].reshape(n_in, N_TERMS)
                widths = theta[b["mfw"]].reshape(n_in, N_TERMS)
            else:
                centres = np.tile(theta[b["mfc"]], (n_in, 1))
                widths = np.tile(theta[b["mfw"]], (n_in, 1))
            centres = np.sort(centres, axis=1)
            widths = W_LO + widths * (W_HI - W_LO)
            out[b["name"]] = (cons, fis.mf_table(centres, widths, self.kind))
        return out


# --- what the objective actually asks ------------------------------------------
# The previous objective was
#
#     J = max(mean_gap / base_mean_gap, sum_cost / base_sum_cost)
#
# against one reference LK configuration, and it had three defects worth naming because
# each of them quietly changed what was being optimised.
#
# 1. *It mixed a mean with a sum.* Gap was averaged over instances, cost was summed, so a
#    summed cost is dominated by the largest instance while an unweighted mean gap is not.
#    The two halves of the objective described different populations of instances.
#
# 2. *The tie-break term was not small.* TIE_WEIGHT 0.15 on the mean of the two ratios
#    contributed ~0.15 to J, while the differences in the max term that it was meant to
#    break ties within were ~0.02. It could therefore prefer a candidate that was worse on
#    the binding axis — the opposite of what a Chebyshev objective is for.
#
# 3. *It compared against a single point, not the frontier.* The claim this work makes is
#    that adaptive effort lands outside LK's time-quality frontier. Scoring against one
#    configuration gives no credit for beating a different one, and a solver whose whole
#    selling point is picking its own operating point should not be judged at somebody
#    else's.
#
# What replaces it asks the frontier question directly, per instance:
#
#     q_i = (this candidate's tour length on instance i)
#           / (the tour length the swept baseline frontier reaches at the same cost)
#     J   = mean_i q_i
#
# q_i < 1 means "at the budget this candidate chose to spend, it beat every LK
# configuration that spends the same" — which is exactly the claim. Everything is
# per-instance, so there is no mean-versus-sum mismatch and no weight to choose; the
# baseline frontier supplies the exchange rate between time and quality that the old
# objective had to invent.
#
# The effort sweep, all at k=32. Fixing k is deliberate: the FIS arm allocates breadth
# *within* a k=32 candidate list, so a sweep at the same k is the apples-to-apples
# comparison, and it is the effort axis — depth and deep breadth — that the rule bases
# actually control.
SWEEP_K = 32
SWEEP = [(2, 4), (4, 8), (6, 8), (6, 32), (10, 12), (10, 32), (16, 32)]

OR_SEG_BASE = 3

# Give up on a candidate once one instance costs this multiple of the dearest baseline
# configuration on that instance. Past that the bar is pinned at LK's best gap and more
# spending cannot improve q_i, so finishing the measurement only burns time. A candidate
# that tells the rule bases to use full depth and breadth everywhere is several times the
# dearest sweep point, and the bound-seeking optimisers walk straight into that region.
ABORT_FACTOR = 2.0


class Objective:
    """Scores a rule-base vector against the swept LK effort frontier, per instance."""

    def __init__(self, instances, space, coef, verbose=False):
        self.space = space
        self.coef = coef
        self.items = []
        for inst in sorted(instances, key=lambda it: it.n):
            assert (
                inst.n >= MIN_N
            ), f"{inst.name}: n={inst.n} below the fitting floor {MIN_N}"
            cand, cand_d = build_candidates(inst.coords, SWEEP_K, inst.ceil)
            nn1, mean_c = nn_stats(cand_d)
            start = greedy_edge_tour(inst.coords, cand, inst.ceil)
            self.items.append((inst, cand, cand_d, nn1, mean_c, start))
        self.fronts = [self._front(it) for it in self.items]
        self.abort_cost = np.array([f["cost"][-1] for f in self.fronts]) * ABORT_FACTOR
        if verbose:
            for it, f in zip(self.items, self.fronts):
                print(
                    f"  {it[0].name:>18s} n={it[0].n:6d} frontier spans "
                    f"{1e3 * f['cost'][0]:7.1f}..{1e3 * f['cost'][-1]:7.1f}ms, "
                    f"length {f['length'][0]:.0f}..{f['length'][-1]:.0f}"
                )

    def _front(self, item):
        """The instance's baseline effort frontier as ascending-cost arrays.

        Only the non-dominated sweep points are kept, so the interpolant below is a
        genuine frontier rather than a curve through configurations that some other
        configuration already beats on both axes.
        """
        inst, cand, cand_d, _, _, start = item
        pts = []
        for depth, deep in SWEEP:
            tour, _, st = lk_solve(
                inst.coords,
                cand,
                cand_d,
                inst.ceil,
                start,
                SWEEP_K,
                depth,
                deep,
                OR_SEG_BASE,
            )
            validate_tour(tour, inst.n)
            # tour *lengths*, not gaps. The metric is a ratio of two tours on one instance,
            # so the optimum cancels and is never needed — which is what lets synthetic
            # instances, for which no optimum exists, into the training pool at all.
            pts.append(
                (
                    float(features_from_stats(st, inst.n) @ self.coef),
                    float(reference_length(tour, inst)),
                )
            )
        keep = [
            p
            for p in pts
            if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in pts)
        ]
        keep.sort()
        cost = np.array([p[0] for p in keep])
        length = np.array([p[1] for p in keep])
        # enforce strict monotonicity in cost so np.interp behaves
        for i in range(1, len(cost)):
            if cost[i] <= cost[i - 1]:
                cost[i] = cost[i - 1] * (1 + 1e-9)
        return {"cost": cost, "length": length, "best": float(min(p[1] for p in pts))}

    def bar(self, i, cost):
        """The tour length the baseline frontier achieves at this cost on instance i.

        Outside the swept range the nearer endpoint is used: cheaper than every baseline
        configuration means the baseline cannot operate there at all, so the weakest
        configuration's gap is the bar; dearer than all of them means the bar is the best
        gap the baseline can reach. Both are conservative — the candidate is never given
        credit for a budget the baseline was not measured at.
        """
        f = self.fronts[i]
        return float(np.interp(cost, f["cost"], f["length"]))

    def measure(self, theta, use_chain=True, abort=False):
        """(mean frontier ratio, per-instance ratios, total cost).

        The mean ratio is the number the search minimises; below 1.0 means the candidate
        beat the baseline frontier at its own chosen budget, averaged over instances.
        """
        d = self.space.decode(theta)
        e_cons, e_tab = d["effort"]
        h_cons, h_tab = d["chain"]
        ratios = []
        total = 0.0
        for i, (inst, cand, cand_d, nn1, mean_c, start) in enumerate(self.items):
            tour = start
            tour, _, st = fis_lk_solve(
                inst.coords,
                cand,
                cand_d,
                inst.ceil,
                tour,
                nn1,
                mean_c,
                e_tab,
                self.space.bases["effort"][0],
                e_cons,
                h_tab,
                self.space.bases["chain"][0],
                h_cons,
                FIS_DEPTH,
                OR_SEG,
                1,
                False,
                use_chain,
            )
            validate_tour(tour, inst.n)
            cost = float(features_from_stats(st, inst.n) @ self.coef)
            total += cost
            if abort and cost > self.abort_cost[i]:
                return float("nan"), None, cost / self.abort_cost[i]
            ratios.append(
                float(reference_length(tour, inst)) / max(self.bar(i, cost), 1e-9)
            )
        return float(np.mean(ratios)), np.array(ratios), total

    def report(self, theta, **kw):
        """(mean frontier ratio, mean gap over instances that have an optimum, total cost).

        For printing, never for search. The gap is reported only over the instances whose
        optimum is published — the synthetic ones have none — so it is a rough companion to
        the frontier ratio rather than the number anything is decided on.
        """
        mean_ratio, ratios, total = self.measure(theta, **kw)
        d = self.space.decode(theta)
        gaps = []
        for i, (inst, cand, cand_d, nn1, mean_c, start) in enumerate(self.items):
            if not inst.opt:
                continue
            tour, _, _ = fis_lk_solve(
                inst.coords,
                cand,
                cand_d,
                inst.ceil,
                start,
                nn1,
                mean_c,
                d["effort"][1],
                self.space.bases["effort"][0],
                d["effort"][0],
                d["chain"][1],
                self.space.bases["chain"][0],
                d["chain"][0],
                FIS_DEPTH,
                OR_SEG,
                1,
                False,
                kw.get("use_chain", True),
            )
            gaps.append(inst.gap(reference_length(tour, inst)))
        return mean_ratio, float(np.mean(gaps)) if gaps else float("nan"), total

    def scalar(self, theta):
        mean_ratio, _, over = self.measure(theta, abort=True)
        if np.isnan(mean_ratio):
            return ABORT_FACTOR + over
        return mean_ratio


class TrackedObjective:
    """The training objective, plus shrinkage, plus selection from a pool on validation.

    **Shrinkage.** A penalty on squared distance from the hand-written rule base. Those
    rules encode what the mechanism section of the findings measured about where the
    search's time goes, so treating them as a prior and charging the optimiser for leaving
    them is better motivated than a plain norm penalty.

    **Selection from a pool, on validation.** The search keeps the best ``pool_size``
    vectors it saw by *training* score; at the end each is scored on validation and the
    best-generalising one is returned. Scoring validation only at successive training-bests
    does not work: once the two scores decouple, every training improvement lies further
    along the overfitting path, so that is the only region validation ever judges. A
    684-evaluation run selected that way returned 4.21% where a 252-evaluation run of the
    same optimiser found 3.75%.

    Even this is not a clean estimate — selecting the best of ``pool_size`` candidates *on*
    validation makes validation part of the fitting procedure. Only the test set in
    ``benchmark.py`` is untouched, and it is the one that decides whether fitting helped.
    """

    def __init__(self, train, valid, theta_hand, shrink=0.0, pool_size=24):
        self.train = train
        self.valid = valid
        self.theta_hand = theta_hand
        self.shrink = shrink
        self.pool_size = pool_size
        self.pool = []
        self.n_calls = 0
        self.n_valid_calls = 0
        self.best_valid = valid.measure(theta_hand)[0]

    def __call__(self, theta):
        theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
        self.n_calls += 1
        j = self.train.scalar(theta)
        if not np.isfinite(j) or j >= ABORT_FACTOR:
            return j
        if self.shrink > 0.0:
            j += self.shrink * float(np.mean((theta - self.theta_hand) ** 2))
        if len(self.pool) < self.pool_size or j < self.pool[-1][0]:
            # skip near-duplicates, or one basin fills the pool and the validation pass
            # gets two dozen copies of the same answer to choose between
            if not any(float(np.max(np.abs(theta - t))) < 1e-3 for _, t in self.pool):
                self.pool.append((j, theta.copy()))
                self.pool.sort(key=lambda kv: kv[0])
                del self.pool[self.pool_size :]
        return j

    def select(self):
        """Score the pool on validation, return the best-generalising vector."""
        best = (float("inf"), self.theta_hand.copy())
        for _, theta in self.pool:
            v = self.valid.measure(theta)[0]
            self.n_valid_calls += 1
            if np.isfinite(v) and v < best[0]:
                best = (v, theta.copy())
        self.best_valid = best[0]
        return best[1]


# --- optimiser drivers -----------------------------------------------------
def _variables(space, seed_theta, perturbation):
    return [
        InputContinuousVariable(f"p{i}", 0.0, 1.0, float(seed_theta[i]), perturbation)
        for i in range(space.size)
    ]


def build_optimizer(kind, space, fcn, seed_theta, generations, population, jobs):
    variables = _variables(space, seed_theta, 0.0)
    common = dict(
        population_size=population,
        num_generations=generations,
        solution_archive_size=max(2 * population, 60),
        stop_after_iterations=max(8, generations // 3),
        n_jobs=jobs,
        joblib_prefer="threads",
    )
    if kind == "ga":
        cfg = GeneticAlgorithmOptimizerConfig(
            name="GA", mutation_rate=0.15, crossover_rate=0.8, **common
        )
        return GeneticAlgorithmOptimizer(config=cfg, variables=variables, fcn=fcn)
    if kind == "pso":
        cfg = ParticleSwarmOptimizerConfig(
            name="PSO",
            inertia=0.6,
            cognitive=1.4,
            social=1.4,
            velocity_clamp=0.3,
            **common,
        )
        return ParticleSwarmOptimizer(config=cfg, variables=variables, fcn=fcn)
    if kind == "aco":
        cfg = AntColonyOptimizerConfig(name="ACO", learning_rate=0.4, q=1.0, **common)
        return AntColonyOptimizer(config=cfg, variables=variables, fcn=fcn)
    raise ValueError(f"unknown optimizer {kind!r}")


def run_one(
    kind,
    mf_kind,
    mf_scope,
    generations,
    population,
    jobs,
    seed,
    shrink,
    log,
    polish_evals=2500,
    scale=fis.DEFAULT_SCALE,
    train_pool=None,
    valid_pool=None,
):
    """Fit with one optimiser / MF form / MF scope, selecting on the validation split.

    ``train_pool`` / ``valid_pool`` override the standard splits with explicit instance lists.
    They exist so that a transfer study can fit on one structural family and test on another
    *through this same function* — a second copy of the fitting procedure would drift from this
    one, and then any difference between a transfer run and the reported run would be
    uninterpretable.
    """
    set_seed(seed)
    space = ParamSpace(mf_kind, mf_scope, scale)
    coef = np.load(paths.COSTMODEL)["coef"]
    train = Objective(
        train_instances() if train_pool is None else train_pool, space, coef
    )
    valid = Objective(
        valid_instances() if valid_pool is None else valid_pool, space, coef
    )

    seed_theta = space.default()
    hand_tr = train.report(seed_theta)
    hand_va = valid.report(seed_theta)

    tracked = TrackedObjective(train, valid, seed_theta, shrink)
    t0 = time.perf_counter()
    opt = build_optimizer(
        kind, space, tracked, seed_theta, generations, population, jobs
    )
    result = opt.solve()
    dt = time.perf_counter() - t0

    # keep the pool member that generalises best, not the one with the best training score
    theta = tracked.select()

    # Then polish it with derivative-free stepwise refinement. The GA finds the basin and
    # then wanders inside it — its own final vector scored worse on validation than the best
    # vector it passed through — so a method that only ever accepts improvements is the
    # natural second stage. It descends on the *training* objective, so it can overfit too,
    # and its result is put back through the same validation selection rather than trusted.
    t_polish = time.perf_counter()
    polished, ptrain, pevals, psweeps = compass_refine(
        train.scalar, theta, max_evals=polish_evals
    )
    t_polish = time.perf_counter() - t_polish
    v_before = valid.measure(theta)[0]
    v_after = valid.measure(polished)[0]
    took_polish = np.isfinite(v_after) and v_after < v_before
    if took_polish:
        theta = polished

    tr = train.report(theta)
    va = valid.report(theta)
    raw = np.clip(np.asarray(result.solution_vector, dtype=np.float64), 0.0, 1.0)
    raw_va = valid.report(raw)
    rec = {
        "optimizer": kind,
        "mf_kind": mf_kind,
        "mf_scope": mf_scope,
        "scale": scale,
        "n_params": space.size,
        "seconds": dt,
        "generations": int(result.generations_completed),
        "stop_reason": str(result.stop_reason),
        "evaluations": tracked.n_calls,
        "polish_evals": int(pevals),
        "polish_sweeps": int(psweeps),
        "polish_seconds": t_polish,
        "polish_train": float(ptrain),
        "valid_before_polish": float(v_before),
        "valid_after_polish": float(v_after),
        "polish_kept": bool(took_polish),
        "train_ratio": tr[0],
        "train_gap": tr[1],
        "valid_ratio": va[0],
        "valid_gap": va[1],
        # what the optimiser's own final answer would have scored, i.e. what selecting
        # from a pool on validation bought over taking the training-best vector
        "untracked_valid_ratio": raw_va[0],
        "hand_train_ratio": hand_tr[0],
        "hand_train_gap": hand_tr[1],
        "hand_valid_ratio": hand_va[0],
        "hand_valid_gap": hand_va[1],
    }
    log.append(rec)
    print(
        f"  {kind:>4s}/{mf_kind:<10s} GA {dt:5.1f}s/{tracked.n_calls} evals  "
        f"polish {t_polish:5.1f}s/{pevals} evals ({psweeps} sweeps, "
        f"{'kept' if took_polish else 'rejected'}: valid {v_before:.4f} -> {v_after:.4f})",
        flush=True,
    )
    print(
        f"       train q={tr[0]:.4f}  valid q={va[0]:.4f}  "
        f"| hand-written valid q={hand_va[0]:.4f}  | GA-unpooled q={raw_va[0]:.4f}",
        flush=True,
    )
    return theta, rec, valid


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", nargs="*", default=["ga"])
    ap.add_argument("--mf-kinds", nargs="*", default=["triangular"])
    ap.add_argument("--mf-scopes", nargs="*", default=["base"])
    ap.add_argument("--shrink", type=float, default=0.3)
    ap.add_argument("--polish-evals", type=int, default=2500)
    ap.add_argument(
        "--scale", default=fis.DEFAULT_SCALE, choices=sorted(fis.EFFORT_RULES_BY_SCALE)
    )
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    # Both outputs are named after the scale, because the two scales are a reported
    # comparison rather than a default and a variant, and a fitted vector means something
    # different under each — the antecedent arrays it indexes are not the same.
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    paths.ensure()
    if args.out is None:
        args.out = str(paths.tuned(args.scale))
    if args.log is None:
        args.log = str(paths.tune_log(args.scale))

    n_tr = len(TRAIN_REAL) + len(synth.TRAIN_SPEC)
    n_va = len(VALID_REAL) + len(synth.VALID_SPEC)
    print(
        f"{n_tr} training ({len(TRAIN_REAL)} TSPLIB + {len(synth.TRAIN_SPEC)} synthetic) "
        f"and {n_va} validation instances, disjoint from the "
        f"{len(bench_mod.TEST)} test instances"
    )
    space0 = ParamSpace(mf_scope=args.mf_scopes[0], scale=args.scale)
    n_cons = sum(int(np.prod(b["cons_shape"])) for b in space0.blocks)
    print(
        f"fitting {space0.size} parameters ({n_cons} rule consequents + "
        f"{space0.size - n_cons} membership-function centres/widths)"
    )
    print(
        f"objective: mean over instances of (tour length / the swept k={SWEEP_K} LK "
        f"frontier's length at the same cost)"
    )
    print(
        f"           below 1.0 means outside the frontier; "
        f"fitted only on n >= {MIN_N}\n"
    )

    log = []
    best = None
    for mf_scope in args.mf_scopes:
        for mf_kind in args.mf_kinds:
            for kind in args.optimizers:
                theta, rec, valid = run_one(
                    kind,
                    mf_kind,
                    mf_scope,
                    args.generations,
                    args.population,
                    args.jobs,
                    args.seed,
                    args.shrink,
                    log,
                    args.polish_evals,
                    args.scale,
                )
                # selection across runs, on the validation split
                score = rec["valid_ratio"]
                if best is None or score < best[0]:
                    best = (score, theta, rec, mf_kind, mf_scope)

    score, theta, rec, mf_kind, mf_scope = best
    space = ParamSpace(mf_kind, mf_scope, rec["scale"])
    d = space.decode(theta)
    print(
        f"\nkept: {rec['optimizer']}/{mf_kind}/{mf_scope}/{rec['scale']}  "
        f"validation q={rec['valid_ratio']:.4f} ({rec['valid_gap']:.3f}%), "
        f"hand-written q={rec['hand_valid_ratio']:.4f} ({rec['hand_valid_gap']:.3f}%)"
    )
    np.savez(
        args.out,
        theta=theta,
        mf_kind=mf_kind,
        mf_scope=mf_scope,
        scale=rec["scale"],
        optimizer=rec["optimizer"],
        effort_cons=d["effort"][0],
        effort_tab=d["effort"][1],
        chain_cons=d["chain"][0],
        chain_tab=d["chain"][1],
        train=np.array(TRAIN_REAL),
        valid=np.array(VALID_REAL),
        **{k: v for k, v in rec.items() if isinstance(v, (int, float))},
    )
    Path(args.log).write_text(json.dumps(log, indent=1))
    print(f"wrote {args.out} and {args.log}")


if __name__ == "__main__":
    main()
