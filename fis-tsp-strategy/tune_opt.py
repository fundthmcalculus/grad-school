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

**2. More is fitted.** ``tune.py`` moved only the rule consequents. Here the
membership functions move too — every term's centre and width on every input — which
is what decides where one linguistic term stops and the next begins. That is 72 extra
parameters on top of the 110 consequents. Because the membership bank is
compiled to a lookup table (``fis.mf_table``), the functional form is a free choice as
well, so gaussian and triangular terms are both fitted and compared.

**3. The instance pool is larger and size-weighted.** 18 training and 13 validation
instances reaching n = 5934, against the previous 12 and 6, and deliberately weighted
toward the larger ones for the reason given at ``TRAIN``. Overfitting has been the
dominant failure mode of every attempt at this — the first reached 2.44% on 8 training
instances and 4.67% on unseen ones — so on top of more instances there is a shrinkage
term toward the hand-written rules and a validation split that the search never
optimises against and that all selection goes through.

Every training instance is disjoint from ``benchmark.TEST``; the assertion at import
time enforces it rather than trusting the lists to stay right.

Run:  python tune_opt.py [--mf-kinds gaussian triangular] [--mf-scopes base input]
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
from core import build_candidates, greedy_edge_tour, nn_stats
from costmodel import features_from_stats
from fis_lk import fis_construct, fis_lk_solve
from lk import lk_solve
from tsplib import load, reference_length, validate_tour

HERE = Path(__file__).resolve().parent

# --- instance splits -------------------------------------------------------
# Both splits are deliberately weighted toward the larger instances, and that choice
# is load-bearing rather than cosmetic.
#
# The objective compares a *mean* gap against a *summed* cost, which is how the results
# are reported — and a summed cost is dominated by the biggest instances while an
# unweighted mean gap is not. A split full of 100-city instances therefore makes the two
# halves of the objective describe different populations, and the search spends itself
# fighting the fixed per-city inference overhead, which is only ever decisive on small
# instances where this engine is not the right tool anyway. The first attempt at this
# used 8 of 20 training instances under n=200 and the optimiser duly went after
# overhead: 3.91% at 1.26x cost, better tours but more work.
#
# Families matter too: the clustered and structured sets (fl*, u*, rl*) behave
# differently enough from the uniform ones that leaving them out of training shows up
# immediately on test.
TRAIN = [
    "kroA100", "ch150", "d198", "tsp225",
    "lin318", "rd400", "pcb442", "rat575", "u724",
    "pr1002", "u1060", "d1291", "rl1304", "u1432", "d1655", "u1817", "rl1889", "rl5934",
]
VALID = [
    "kroB100", "kroB150", "rat195",
    "pr264", "fl417", "u574", "p654",
    "vm1084", "nrw1379", "fl1400", "vm1748", "u2152", "u2319",
]

assert not (set(TRAIN) & set(VALID)), "train and validation overlap"
assert not (set(TRAIN) & set(bench_mod.TEST)), "training instance is in the test set"
assert not (set(VALID) & set(bench_mod.TEST)), "validation instance is in the test set"

K = 32
FIS_DEPTH = 10
OR_SEG = 3
C_BREADTH = 8

# The baseline this fit is trying to dominate: the mid-frontier LK from benchmark.py.
BASE_DEPTH = 6
BASE_DEEP = 32

# How the two objectives are combined. The obvious framing — minimise tour length
# subject to cost <= 0.9x baseline — turns out to encode the wrong goal: it drives the
# search to spend its whole budget on the cost constraint and pay for it in quality,
# landing on a *different* point of the trade-off rather than a better one. Measured, it
# produced 4.35% at 0.82x cost against a baseline 3.95% at 1.0x: cheaper, but not better.
#
# What we actually want is domination on both axes at once, so the objective is the
# Chebyshev (max-ratio) form against the baseline as reference point:
#
#     J = max(gap / base_gap, cost / base_cost)
#
# J < 1 means strictly shorter tours *and* strictly less work than the baseline; the max
# means neither axis can be traded away to buy the other, because improving only the
# slack one does not move J at all. TIE_WEIGHT adds a small pull on the mean of the two
# ratios so that, among vectors with the same worst axis, the one that also improves the
# other is preferred.
TIE_WEIGHT = 0.15

# Early abandoning. The cost of *evaluating* a candidate depends on the candidate: a
# vector that tells the rule bases to use full breadth and full depth at every city
# makes the local search several times slower than the baseline, and the bound-seeking
# optimisers walk straight into that region — PSO drives particles onto the bounds, and
# clipping them to 1.0 is exactly "maximum effort everywhere". Measured, one PSO
# generation cost 99s against the GA's 12.5s, which would have made the full comparison
# a seven-hour run.
#
# Such a candidate is hopeless anyway: it cannot win on cost, so its tour quality does
# not matter. So the evaluation walks the instances in increasing size and gives up as
# soon as accumulated cost passes ABORT_FACTOR x the baseline's over the same prefix,
# returning a score that still ranks abandoned candidates by how far over they went, so
# the search keeps a gradient to descend rather than a flat penalty plateau.
ABORT_FACTOR = 2.0

N_IN = 4
N_TERMS = fis.N_TERMS
W_LO, W_HI = 0.06, 0.60  # membership-function width range


# --- parameter vector ------------------------------------------------------
class ParamSpace:
    """Flat [0,1]^d vector <-> the three rule bases' consequents and MF banks.

    Layout, per rule base: consequents, then MF centres, then MF widths. Centres are
    sorted within each input before use, so term 0 is always the leftmost and term 2
    the rightmost. That costs nothing in expressiveness — the consequents are free to
    put any behaviour on any term — and it keeps LOW / MED / HIGH meaning what their
    names say, which is the whole reason for preferring a rule base to a black box.
    """

    def __init__(self, kind="gaussian", mf_scope="base"):
        self.kind = kind
        self.mf_scope = mf_scope
        self.blocks = []
        off = 0
        for name, cons_shape in (
            ("construct", fis.CONSTRUCT_CONS.shape),
            ("effort", fis.EFFORT_CONS.shape),
            ("chain", fis.CHAIN_CONS.shape),
        ):
            n_cons = int(np.prod(cons_shape))
            # "base": one set of three terms shared by the rule base's four inputs
            # (6 numbers). "input": every input gets its own terms (24). The second is
            # more expressive and overfits measurably harder, so it is not the default.
            n_mf = N_IN * N_TERMS if mf_scope == "input" else N_TERMS
            self.blocks.append(
                {
                    "name": name,
                    "cons_shape": cons_shape,
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
        defaults = {
            "construct": (fis.CONSTRUCT_CONS, fis.CONSTRUCT_MF_C, fis.CONSTRUCT_MF_S),
            "effort": (fis.EFFORT_CONS, fis.EFFORT_MF_C, fis.EFFORT_MF_S),
            "chain": (fis.CHAIN_CONS, fis.CHAIN_MF_C, fis.CHAIN_MF_S),
        }
        for b in self.blocks:
            cons, mfc, mfw = defaults[b["name"]]
            theta[b["cons"]] = cons.ravel()
            if self.mf_scope == "input":
                theta[b["mfc"]] = mfc.ravel()
                theta[b["mfw"]] = (mfw.ravel() - W_LO) / (W_HI - W_LO)
            else:
                theta[b["mfc"]] = mfc[0]
                theta[b["mfw"]] = (mfw[0] - W_LO) / (W_HI - W_LO)
        return np.clip(theta, 0.0, 1.0)

    def decode(self, theta):
        """(cons, table) per rule base, ready to hand to the solver."""
        theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
        out = {}
        for b in self.blocks:
            cons = np.ascontiguousarray(theta[b["cons"]].reshape(b["cons_shape"]))
            if self.mf_scope == "input":
                centres = theta[b["mfc"]].reshape(N_IN, N_TERMS)
                widths = theta[b["mfw"]].reshape(N_IN, N_TERMS)
            else:
                centres = np.tile(theta[b["mfc"]], (N_IN, 1))
                widths = np.tile(theta[b["mfw"]], (N_IN, 1))
            centres = np.sort(centres, axis=1)
            widths = W_LO + widths * (W_HI - W_LO)
            out[b["name"]] = (cons, fis.mf_table(centres, widths, self.kind))
        return out


# --- objective -------------------------------------------------------------
class Objective:
    """Mean tour gap and predicted cost over an instance set, plus the scalar the
    optimisers minimise."""

    def __init__(self, names, space, coef, verbose=False):
        self.space = space
        self.coef = coef
        self.items = []
        for name in sorted(names, key=lambda nm: load(nm).n):
            inst = load(name)
            cand, cand_d = build_candidates(inst.coords, K, inst.ceil)
            nn1, mean_c = nn_stats(cand_d)
            start = greedy_edge_tour(inst.coords, cand, inst.ceil)
            self.items.append((inst, cand, cand_d, nn1, mean_c, start))
        base_costs = self._baseline_costs()
        self.base_cost_prefix = np.cumsum(base_costs)
        self.base_gap, self.base_cost = self._baseline()
        if verbose:
            print(
                f"  baseline LK k{K}/d{BASE_DEPTH}/b{BASE_DEEP}: "
                f"{self.base_gap:.3f}% gap, {1e3 * self.base_cost:.1f}ms predicted"
            )

    def _baseline_costs(self):
        """Predicted baseline cost per instance, in the same order as ``items``."""
        out = []
        for inst, cand, cand_d, _, _, start in self.items:
            _, _, st = lk_solve(
                inst.coords, cand, cand_d, inst.ceil, start, K, BASE_DEPTH, BASE_DEEP, OR_SEG
            )
            out.append(float(features_from_stats(st, inst.n) @ self.coef))
        return out

    def _baseline(self):
        gaps = []
        cost = 0.0
        for inst, cand, cand_d, _, _, start in self.items:
            tour, _, st = lk_solve(
                inst.coords, cand, cand_d, inst.ceil, start, K, BASE_DEPTH, BASE_DEEP, OR_SEG
            )
            validate_tour(tour, inst.n)
            gaps.append(inst.gap(reference_length(tour, inst)))
            cost += float(features_from_stats(st, inst.n) @ self.coef)
        return float(np.mean(gaps)), cost

    def measure(self, theta, use_chain=True, construct=True, abort_factor=None):
        """(mean gap, predicted cost) for one rule-base vector.

        With ``abort_factor`` the walk stops early once accumulated cost exceeds that
        multiple of the baseline's cost over the same instances, returning
        ``(nan, cost ratio so far)``; the caller turns that into a ranked penalty.
        Without it every instance is evaluated, which is what reporting uses.
        """
        d = self.space.decode(theta)
        c_cons, c_tab = d["construct"]
        e_cons, e_tab = d["effort"]
        h_cons, h_tab = d["chain"]
        gaps = []
        cost = 0.0
        for i, (inst, cand, cand_d, nn1, mean_c, start) in enumerate(self.items):
            if construct:
                tour = fis_construct(
                    inst.coords, cand, cand_d, inst.ceil, mean_c,
                    c_tab, fis.CONSTRUCT_ANT, c_cons, C_BREADTH, 0,
                )
            else:
                tour = start
            tour, _, st = fis_lk_solve(
                inst.coords, cand, cand_d, inst.ceil, tour, nn1, mean_c,
                e_tab, fis.EFFORT_ANT, e_cons,
                h_tab, fis.CHAIN_ANT, h_cons,
                FIS_DEPTH, OR_SEG, 1, False, use_chain,
            )
            validate_tour(tour, inst.n)
            gaps.append(inst.gap(reference_length(tour, inst)))
            cost += float(features_from_stats(st, inst.n) @ self.coef)
            if abort_factor is not None and cost > abort_factor * self.base_cost_prefix[i]:
                return float("nan"), cost / self.base_cost_prefix[i]
        return float(np.mean(gaps)), cost

    def ratios(self, theta, **kw):
        """(gap ratio, cost ratio) against the baseline. Both < 1 means domination."""
        gap, cost = self.measure(theta, **kw)
        return gap / self.base_gap, cost / self.base_cost

    def scalar(self, theta):
        gap, cost = self.measure(theta, abort_factor=ABORT_FACTOR)
        if np.isnan(gap):  # abandoned: rank by how far over budget it ran
            return ABORT_FACTOR + cost
        gr, cr = gap / self.base_gap, cost / self.base_cost
        return max(gr, cr) + TIE_WEIGHT * 0.5 * (gr + cr)


class TrackedObjective:
    """The training objective, plus shrinkage, plus validation-tracked selection.

    Two things are bolted on, both aimed at the one failure mode that has dominated
    every attempt at this: overfitting. Fitting 182 parameters on 20 instances, a
    25-generation GA run reached 2.54% on its training set and 4.03% on unseen
    instances — worse than the 3.95% baseline it was supposed to beat, and worse than
    a 24-evaluation run of the same optimiser had managed.

    **Shrinkage.** The objective carries a penalty on the squared distance from the
    hand-written rule base. Those rules are not an arbitrary starting point — they
    encode what §3 of the findings measured about where the search's time goes — so
    treating them as a prior and charging the optimiser for leaving them is a
    better-motivated regulariser than a plain norm penalty.

    **Selection from a pool, on validation.** The search keeps an archive of the best
    ``pool_size`` vectors it saw by *training* score; at the end every one of them is
    scored on validation and the winner is the one that generalises best. Selection
    therefore happens on data the search never optimised against.

    Scoring validation only at successive training-bests — the obvious cheaper version —
    does not work, and the way it fails is instructive: once training score and
    validation score decouple, every training improvement lies further along the
    overfitting path, so that is the only part of the space validation ever gets to
    judge. Measured, a 684-evaluation run selected that way returned 4.21% on validation
    where a 252-evaluation run of the same optimiser had found 3.75% — more search,
    worse answer. The pool sees the whole good-on-training region instead.
    """

    def __init__(self, train, valid, theta_hand, shrink=0.0, pool_size=24):
        self.train = train
        self.valid = valid
        self.theta_hand = theta_hand
        self.shrink = shrink
        self.pool_size = pool_size
        self.pool = []  # (train score, theta), best first
        self.n_calls = 0
        self.n_valid_calls = 0
        self.best_valid = valid.measure(theta_hand)

    def __call__(self, theta):
        theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
        self.n_calls += 1
        gap, cost = self.train.measure(theta, abort_factor=ABORT_FACTOR)
        if np.isnan(gap):
            # hopeless on cost, and cheap to have found out; never worth pooling
            return ABORT_FACTOR + cost
        gr = gap / self.train.base_gap
        cr = cost / self.train.base_cost
        j = max(gr, cr) + TIE_WEIGHT * 0.5 * (gr + cr)
        if self.shrink > 0.0:
            j += self.shrink * float(np.mean((theta - self.theta_hand) ** 2))
        if len(self.pool) < self.pool_size or j < self.pool[-1][0]:
            # keep it only if it is not a near-duplicate of something already pooled;
            # otherwise one good basin crowds out every other and the validation pass
            # gets two dozen copies of the same answer to choose between
            if not any(float(np.max(np.abs(theta - t))) < 1e-3 for _, t in self.pool):
                self.pool.append((j, theta.copy()))
                self.pool.sort(key=lambda kv: kv[0])
                del self.pool[self.pool_size:]
        return j

    def select(self):
        """Score the whole pool on validation, return the best-generalising vector."""
        best = (float("inf"), self.theta_hand.copy(), self.best_valid)
        for _, theta in self.pool:
            v_gap, v_cost = self.valid.measure(theta)
            self.n_valid_calls += 1
            v_gr = v_gap / self.valid.base_gap
            v_cr = v_cost / self.valid.base_cost
            score = max(v_gr, v_cr) + TIE_WEIGHT * 0.5 * (v_gr + v_cr)
            if score < best[0]:
                best = (score, theta.copy(), (v_gap, v_cost))
        self.best_valid = best[2]
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
            name="PSO", inertia=0.6, cognitive=1.4, social=1.4, velocity_clamp=0.3, **common
        )
        return ParticleSwarmOptimizer(config=cfg, variables=variables, fcn=fcn)
    if kind == "aco":
        cfg = AntColonyOptimizerConfig(name="ACO", learning_rate=0.4, q=1.0, **common)
        return AntColonyOptimizer(config=cfg, variables=variables, fcn=fcn)
    raise ValueError(f"unknown optimizer {kind!r}")


def run_one(kind, mf_kind, mf_scope, generations, population, jobs, seed, shrink, log):
    """Fit with one optimiser / MF form / MF scope, selecting on the validation split."""
    set_seed(seed)
    space = ParamSpace(mf_kind, mf_scope)
    coef = np.load(HERE / "costmodel.npz")["coef"]
    train = Objective(TRAIN, space, coef)
    valid = Objective(VALID, space, coef)

    seed_theta = space.default()
    hand_tr = train.measure(seed_theta)
    tracked = TrackedObjective(train, valid, seed_theta, shrink)
    hand_va = tracked.best_valid

    t0 = time.perf_counter()
    opt = build_optimizer(kind, space, tracked, seed_theta, generations, population, jobs)
    result = opt.solve()
    dt = time.perf_counter() - t0

    # the vector we keep is the pool member that generalises best, not the one with
    # the best training score
    theta = tracked.select()
    tr = train.measure(theta)
    va = valid.measure(theta)
    raw = np.clip(np.asarray(result.solution_vector, dtype=np.float64), 0.0, 1.0)
    raw_va = valid.measure(raw)
    rec = {
        "optimizer": kind,
        "mf_kind": mf_kind,
        "mf_scope": mf_scope,
        "n_params": space.size,
        "seconds": dt,
        "generations": int(result.generations_completed),
        "stop_reason": str(result.stop_reason),
        "evaluations": tracked.n_calls,
        "valid_evaluations": tracked.n_valid_calls,
        "train_gap": tr[0],
        "train_cost_ratio": tr[1] / train.base_cost,
        "valid_gap": va[0],
        "valid_cost_ratio": va[1] / valid.base_cost,
        "valid_base_gap": valid.base_gap,
        "train_base_gap": train.base_gap,
        # what the optimiser's own final answer would have scored, i.e. what
        # validation-tracked selection bought over taking the training-best vector
        "untracked_valid_gap": raw_va[0],
        "untracked_valid_cost_ratio": raw_va[1] / valid.base_cost,
        "hand_train_gap": hand_tr[0],
        "hand_train_cost_ratio": hand_tr[1] / train.base_cost,
        "hand_valid_gap": hand_va[0],
        "hand_valid_cost_ratio": hand_va[1] / valid.base_cost,
    }
    log.append(rec)
    print(
        f"  {kind:>4s}/{mf_kind:<10s}/{mf_scope:<5s} {dt:6.1f}s "
        f"evals={tracked.n_calls:4d}  train {tr[0]:.3f}% @{rec['train_cost_ratio']:.2f}x  "
        f"valid {va[0]:.3f}% @{rec['valid_cost_ratio']:.2f}x  "
        f"(untracked would be {raw_va[0]:.3f}%; baseline {valid.base_gap:.3f}%)",
        flush=True,
    )
    return theta, rec, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", nargs="*", default=["ga"])
    ap.add_argument("--mf-kinds", nargs="*", default=["gaussian", "triangular"])
    ap.add_argument("--mf-scopes", nargs="*", default=["base"])
    ap.add_argument("--shrink", type=float, default=0.3)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "tuned_opt.npz"))
    ap.add_argument("--log", default=str(HERE / "tune_opt_log.json"))
    args = ap.parse_args()

    print(f"{len(TRAIN)} training and {len(VALID)} validation instances, "
          f"disjoint from the {len(bench_mod.TEST)} test instances")
    space0 = ParamSpace(mf_scope=args.mf_scopes[0])
    n_cons = sum(int(np.prod(b["cons_shape"])) for b in space0.blocks)
    print(f"fitting {space0.size} parameters ({n_cons} rule consequents + "
          f"{space0.size - n_cons} membership-function centres/widths)")
    print(f"objective: minimise max(gap, cost) relative to baseline LK "
          f"k{K}/d{BASE_DEPTH}/b{BASE_DEEP}; below 1.0 on both means domination\n")

    log = []
    best = None
    for mf_scope in args.mf_scopes:
        for mf_kind in args.mf_kinds:
            for kind in args.optimizers:
                theta, rec, valid = run_one(
                    kind, mf_kind, mf_scope, args.generations, args.population,
                    args.jobs, args.seed, args.shrink, log,
                )
                # selection across runs, on the validation split, under the same
                # constraint the training objective used
                gr = rec["valid_gap"] / rec["valid_base_gap"]
                cr = rec["valid_cost_ratio"]
                score = max(gr, cr) + TIE_WEIGHT * 0.5 * (gr + cr)
                if best is None or score < best[0]:
                    best = (score, theta, rec, mf_kind, mf_scope)

    score, theta, rec, mf_kind, mf_scope = best
    space = ParamSpace(mf_kind, mf_scope)
    d = space.decode(theta)
    print(f"\nkept: {rec['optimizer']}/{mf_kind}/{mf_scope}  "
          f"validation {rec['valid_gap']:.3f}% @ {rec['valid_cost_ratio']:.2f}x baseline "
          f"(baseline {rec['valid_base_gap']:.3f}%)")
    np.savez(
        args.out,
        theta=theta,
        mf_kind=mf_kind,
        mf_scope=mf_scope,
        optimizer=rec["optimizer"],
        construct_cons=d["construct"][0],
        construct_tab=d["construct"][1],
        effort_cons=d["effort"][0],
        effort_tab=d["effort"][1],
        chain_cons=d["chain"][0],
        chain_tab=d["chain"][1],
        train=np.array(TRAIN),
        valid=np.array(VALID),
        **{k: v for k, v in rec.items() if isinstance(v, (int, float))},
    )
    Path(args.log).write_text(json.dumps(log, indent=1))
    print(f"wrote {args.out} and {args.log}")


if __name__ == "__main__":
    main()
