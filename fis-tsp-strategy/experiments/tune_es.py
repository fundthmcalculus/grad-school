"""Fit the two rule bases' consequents against the objective we actually care about.

The antecedents, the membership functions and the rule structure are all fixed by
hand — they are the interpretable part, and they stay readable. What gets fitted is
the singleton each rule points at: 18 desirabilities for the construction ranker and
19x3 parameter settings for the effort controller, 75 numbers in one vector.

The objective is the pair of things the engine is supposed to win on, with the time
side written as a constraint rather than a second term to trade against:

    J(theta) = mean gap over the training instances
             + PENALTY * max(0, our_time / baseline_time - TIME_TARGET)

so the search minimises tour length among the rule bases that are at least
1/TIME_TARGET faster than the tuned baseline LK. Timing is the noisiest input to
any tuner, so every instance is run twice per evaluation and the *minimum* is kept
— for wall-clock, the minimum of a few runs estimates the underlying cost far
better than the mean, which only ever drifts upward with interference.

Training instances are disjoint from the ones reported in ``benchmark.py``.

Run:  python tune.py [--seconds 600] [--out tuned.npz]
"""

from __future__ import annotations

import argparse
import time

# ``experiments/`` sits one level below the modules it imports, so the project root goes
# on sys.path before any of them. ``paths`` also owns every output location, so an
# experiment writes into the same results/ tree as the reported pipeline.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402

paths.on_path()

import numpy as np  # noqa: E402

import fis  # noqa: E402
from core import build_candidates, greedy_edge_tour, nn_stats
from fis_lk import construct as fis_build
from fis_lk import local_search as fis_ls
from lk import lk_solve
from tsplib import load, reference_length, validate_tour

# Instances used for fitting, and a held-out set used only to *choose* which of the
# fitted vectors to keep. Fitting on eight small instances and reporting on the rest
# overfits badly — the first attempt reached 2.44% on its training set and 4.67% on
# unseen instances, worse than the baseline it was supposed to beat. The fix is both
# a wider size range to fit on and a validation set the search cannot see.
TRAIN = [
    "kroA100",
    "eil101",
    "ch150",
    "d198",
    "lin318",
    "pcb442",
    "rat575",
    "u724",
    "pr1002",
    "d1291",
    "u1817",
    "pr2392",
]
VALID = [
    "kroB150",
    "rd400",
    "p654",
    "vm1084",
    "nrw1379",
    "u2152",
]

# The baseline LK configuration, chosen by its own sweep over
# (k, breadth, depth, deep breadth, Or-opt, start) — see FINDINGS.md.
K = 32
DEPTH = 10
OR_SEG = 3
C_BREADTH = 8

TIME_TARGET = 0.90  # must be at least ~10% faster than the baseline
PENALTY = 8.0


class Bench:
    """Training instances with their candidate lists and the baseline to beat."""

    def __init__(self, names=TRAIN, k=K, reps=2):
        self.reps = reps
        self.items = []
        for name in names:
            inst = load(name)
            cand, cand_d = build_candidates(inst.coords, k, inst.ceil)
            nn1, mean_c = nn_stats(cand_d)
            start = greedy_edge_tour(inst.coords, cand, inst.ceil)
            self.items.append((inst, cand, cand_d, nn1, mean_c, start))
        self.base_gap, self.base_time = self._baseline()

    def _baseline(self):
        gaps = []
        total = 0.0
        for inst, cand, cand_d, _, _, start in self.items:
            best = float("inf")
            for _ in range(self.reps):
                t0 = time.perf_counter()
                tour, _, _ = lk_solve(
                    inst.coords, cand, cand_d, inst.ceil, start, K, DEPTH, K, OR_SEG
                )
                best = min(best, time.perf_counter() - t0)
            validate_tour(tour, inst.n)
            gaps.append(inst.gap(reference_length(tour, inst)))
            total += best
        return float(np.mean(gaps)), total

    def evaluate(self, theta, reps=None):
        """(mean gap, total seconds) for the FIS arm under ``theta``."""
        reps = self.reps if reps is None else reps
        c_cons, e_cons, h_cons = fis.rules_from_theta(theta)
        gaps = []
        total = 0.0
        for inst, cand, cand_d, nn1, mean_c, _ in self.items:
            best = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                tour = fis_build(inst, cand, cand_d, c_cons, C_BREADTH)
                tour, _, _ = fis_ls(
                    inst, cand, cand_d, tour, e_cons, h_cons, DEPTH, OR_SEG
                )
                best = min(best, time.perf_counter() - t0)
            validate_tour(tour, inst.n)
            gaps.append(inst.gap(reference_length(tour, inst)))
            total += best
        return float(np.mean(gaps)), total

    def objective(self, theta):
        gap, t = self.evaluate(theta)
        ratio = t / self.base_time
        return gap + PENALTY * max(0.0, ratio - TIME_TARGET), gap, t


def tune_construction(bench, valid, seconds=120.0, seed=0, verbose=True):
    """Fit the CONSTRUCT consequents against the length of the tour they build.

    Fitting the ranker end-to-end — on the tour length *after* the local search has
    finished with it — turns out to be the wrong signal. The local search absorbs
    most of what the construction does, so the gradient the ranker sees is weak and
    mostly noise, and an end-to-end fit happily lets construction quality rot (it
    drifted to 24.0% mean gap, worse than the 21.9% of plain nearest-neighbour) while
    the effort consequents pick up the slack.

    Its own tour length is the honest objective for it, and it is a far cheaper one:
    no local search runs, so this stage does thousands of evaluations in seconds.
    Returns the fitted consequent table.
    """
    rng = np.random.default_rng(seed)

    def build_gap(cons, items):
        cons = np.ascontiguousarray(cons.clip(0.0, 1.0))
        gaps = []
        for inst, cand, cand_d, _, _, _ in items:
            tour = fis_build(inst, cand, cand_d, cons, C_BREADTH)
            validate_tour(tour, inst.n)
            gaps.append(inst.gap(reference_length(tour, inst)))
        return float(np.mean(gaps))

    cons = fis.CONSTRUCT_CONS.copy()
    best = build_gap(cons, bench.items)
    keep = cons.copy()
    keep_v = build_gap(cons, valid.items)
    if verbose:
        print(f"  construction, hand-written: train {best:.2f}%  valid {keep_v:.2f}%")

    sigma = 0.25
    flat = cons.ravel()
    n_dim = flat.size
    t_end = time.perf_counter() + seconds
    wins = window = it = 0
    while time.perf_counter() < t_end:
        it += 1
        window += 1
        trial = flat.copy()
        n_mut = min(1 + int(abs(rng.normal(0.0, 0.25 * n_dim))), n_dim)
        idx = rng.choice(n_dim, size=n_mut, replace=False)
        trial[idx] += rng.normal(0.0, sigma, size=n_mut)
        np.clip(trial, 0.0, 1.0, out=trial)
        g = build_gap(trial.reshape(cons.shape), bench.items)
        if g < best:
            flat, best = trial, g
            wins += 1
            gv = build_gap(trial.reshape(cons.shape), valid.items)
            if gv < keep_v:
                keep_v = gv
                keep = trial.reshape(cons.shape).copy()
                if verbose:
                    print(
                        f"  it {it:5d}  construction train {g:.2f}%  valid {gv:.2f}%  <- kept"
                    )
        if window >= 25:
            sigma = (
                min(sigma * 1.5, 0.6)
                if wins / window > 0.2
                else max(sigma / 1.5, 0.004)
            )
            wins = window = 0
    if verbose:
        print(
            f"  construction fitted: train {best:.2f}%  valid {keep_v:.2f}%  ({it} evals)\n"
        )
    return np.ascontiguousarray(keep.clip(0.0, 1.0))


def tune(seconds=600.0, seed=0, verbose=True, construct_seconds=120.0):
    """(1+1) evolution strategy with the 1/5th success rule on the step size.

    A plain hill climber is the right tool here: 75 bounded parameters, a cheap but
    *noisy* objective, and no gradient. The 1/5th rule is what keeps it moving —
    step size grows while progress is easy and collapses to fine local polish once
    it is not.
    """
    rng = np.random.default_rng(seed)
    bench = Bench()
    valid = Bench(names=VALID)
    if verbose:
        print(
            f"train    baseline LK: {bench.base_gap:.3f}% mean gap, {bench.base_time:.3f}s"
        )
        print(
            f"validate baseline LK: {valid.base_gap:.3f}% mean gap, {valid.base_time:.3f}s"
        )
        print(
            f"objective: minimise gap subject to time <= {TIME_TARGET:.2f} x baseline\n"
        )

    # stage 1: the ranker, against its own tour length
    theta = fis.DEFAULT_THETA.copy()
    n_c = fis.CONSTRUCT_CONS.size
    theta[:n_c] = tune_construction(
        bench, valid, construct_seconds, seed, verbose
    ).ravel()

    # stage 2: the effort controller, end-to-end, with the ranker held fixed
    effort_idx = np.arange(n_c, theta.size)  # effort + chain consequents
    best_j, best_gap, _ = bench.objective(theta)
    # model selection happens on the validation set, so a training gain that does
    # not transfer is not kept
    best_vj, best_vgap, best_vt = valid.objective(theta)
    keep = theta.copy()
    keep_vj = best_vj
    if verbose:
        print(
            f"  hand-written rules: train J={best_j:.4f} gap={best_gap:.3f}%  "
            f"valid gap={best_vgap:.3f}% ({best_vt / valid.base_time:.2f}x)"
        )

    sigma = 0.25
    t_end = time.perf_counter() + seconds
    it = 0
    wins = 0
    window = 0
    while time.perf_counter() < t_end:
        it += 1
        window += 1
        # perturb a random subset: full-vector steps stall early, single-coordinate
        # steps waste evaluations once the vector is roughly right
        cand_theta = theta.copy()
        n_mut = 1 + int(abs(rng.normal(0.0, 0.25 * effort_idx.size)))
        n_mut = min(n_mut, effort_idx.size)
        idx = rng.choice(effort_idx, size=n_mut, replace=False)
        cand_theta[idx] += rng.normal(0.0, sigma, size=n_mut)
        np.clip(cand_theta, 0.0, 1.0, out=cand_theta)

        j, gap, t = bench.objective(cand_theta)
        if j < best_j:
            theta, best_j, best_gap = cand_theta, j, gap
            wins += 1
            vj, vgap, vt = valid.objective(theta)
            tag = ""
            if vj < keep_vj:  # transfers — this is the one we would ship
                keep = theta.copy()
                keep_vj = vj
                best_vgap, best_vt = vgap, vt
                tag = "  <- kept"
            if verbose:
                print(
                    f"  it {it:5d} sigma={sigma:.3f}  train gap={gap:.3f}% "
                    f"({t / bench.base_time:.2f}x)  valid gap={vgap:.3f}% "
                    f"({vt / valid.base_time:.2f}x){tag}"
                )
        if window >= 25:  # 1/5th success rule
            rate = wins / window
            sigma = min(sigma * 1.5, 0.6) if rate > 0.2 else max(sigma / 1.5, 0.004)
            wins = 0
            window = 0

    if verbose:
        print(
            f"\n{it} evaluations. kept vector: validation gap {best_vgap:.3f}% "
            f"(baseline {valid.base_gap:.3f}%), time {best_vt / valid.base_time:.2f}x baseline"
        )
    return keep, bench, valid, best_vgap, best_vt


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--construct-seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(paths.LEGACY / "tuned_es.npz"))
    args = ap.parse_args()

    theta, bench, valid, gap, t = tune(
        args.seconds, args.seed, construct_seconds=args.construct_seconds
    )
    c_cons, e_cons, h_cons = fis.rules_from_theta(theta)
    np.savez(
        args.out,
        theta=theta,
        construct_cons=c_cons,
        effort_cons=e_cons,
        chain_cons=h_cons,
        train=np.array(TRAIN),
        valid=np.array(VALID),
        k=K,
        depth=DEPTH,
        or_seg=OR_SEG,
        c_breadth=C_BREADTH,
        valid_gap=gap,
        valid_time=t,
        base_gap=valid.base_gap,
        base_time=valid.base_time,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
