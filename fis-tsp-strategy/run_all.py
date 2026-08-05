"""Reproduce every reported artifact, in the order the dependencies require.

The order is not a convention, it is a constraint, and each edge exists for a reason worth
knowing before skipping a stage:

1. ``test_invariants`` first. Four of the bugs recorded in FINDINGS.md §10 produced *plausible*
   numbers rather than crashes, so a benchmark run on broken code looks exactly like a benchmark
   run on working code. Correctness is checked before anything is measured.
2. ``costmodel`` before ``tune_opt``. The tuner's objective is the fitted cost proxy rather than
   wall clock, so that the search is deterministic and not corrupted by its own CPU contention.
   Its coefficients must be re-fitted after any change to the solver's hot path, or the tuner is
   optimising against a model of code that no longer exists.
3. ``tune_opt`` before ``benchmark``. The benchmark reports a fitted rule base and falls back to
   the hand-written one if the file is missing — silently, which is the sort of quiet
   substitution this pipeline exists to prevent, so the stage runs rather than being assumed.
4. Figures last, since each reads a results file rather than re-measuring.

Both rule-base scales are built. They are a reported comparison, not a default and a variant:
``small`` is better over the whole test set and ``large`` is better above n ~ 5000 (FINDINGS
§3b), so a run that produced only one of them could not state the result.

The stages differ in cost by three orders of magnitude, and the expensive one is not ours —
LKH's cost grows as roughly n^3.5, so ``lkh-compare`` over the full size ladder dominates
everything else combined. It is therefore its own stage, off by default at full ladder, and
``--dry-run`` prices it before it is started.

Run:  python run_all.py                       # everything except the full LKH ladder
      python run_all.py --stages bench figs    # just these
      python run_all.py --list                 # what the stages are
      python run_all.py --ladder               # include the full LKH size ladder (hours)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import paths

PY = sys.executable


def _stages(args):
    """(name, argv, what it writes) in dependency order."""
    quick = ["--generations", "6", "--population", "12"] if args.quick else []
    reps = ["--reps", str(args.reps)]
    ladder = ["--ladder"] if args.ladder else []
    return [
        ("tests", [PY, "test_invariants.py"], "nothing — it either passes or stops the run"),
        ("costmodel", [PY, "costmodel.py"], paths.COSTMODEL),
        ("tune-small", [PY, "tune_opt.py", "--scale", "small", *quick], paths.tuned("small")),
        ("tune-large", [PY, "tune_opt.py", "--scale", "large", *quick], paths.tuned("large")),
        ("bench-small", [PY, "benchmark.py", "--scale", "small", *reps],
         paths.benchmark("small")),
        ("bench-large", [PY, "benchmark.py", "--scale", "large", *reps],
         paths.benchmark("large")),
        ("lkh-ref", [PY, "lkh_reference.py"], paths.LKH_REFERENCE),
        ("lkh-compare", [PY, "lkh_compare.py", *ladder, *reps], paths.LKH_COMPARE),
        ("figs", [PY, "figures.py"], paths.FIGURES / "fis_tsp_pareto.png"),
        ("figs-tuning", [PY, "figures_tuning.py"], paths.FIGURES / "fis_tsp_tuning.png"),
        ("figs-lkh", [PY, "figures_lkh.py"], paths.FIGURES / "fis_tsp_vs_lkh.png"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=None, help="default: all of them")
    ap.add_argument("--skip", nargs="*", default=[], help="stage names to leave out")
    ap.add_argument("--ladder", action="store_true",
                    help="run lkh-compare over the full size ladder — hours, see --dry-run")
    ap.add_argument("--quick", action="store_true",
                    help="a small GA budget: proves the pipeline runs, not the result")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths.ensure()

    stages = _stages(args)
    if args.list or args.dry_run:
        for name, argv, writes in stages:
            print(f"  {name:<12s} {' '.join(argv[1:]):<46s} -> {writes}")
        if args.dry_run:
            print("\npricing the LKH stage:")
            subprocess.run([PY, "lkh_compare.py", "--dry-run", *(["--ladder"] if args.ladder else [])],
                           cwd=str(paths.ROOT))
        return

    chosen = [s for s in stages if (args.stages is None or s[0] in args.stages)
              and s[0] not in args.skip]
    if args.stages:
        unknown = set(args.stages) - {s[0] for s in stages}
        if unknown:
            raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}")

    t_all = time.perf_counter()
    for i, (name, argv, writes) in enumerate(chosen, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(chosen)}] {name}: {' '.join(argv[1:])}\n{'=' * 78}",
              flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(argv, cwd=str(paths.ROOT))
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            # Stop rather than continue: every later stage reads what this one writes, so
            # carrying on would report the previous run's artifacts as if they were this one's.
            raise SystemExit(f"\n{name} failed after {dt:.1f}s (exit {r.returncode})")
        print(f"\n-- {name} ok in {dt:.1f}s -> {writes}", flush=True)
    print(f"\nall {len(chosen)} stages ok in {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
