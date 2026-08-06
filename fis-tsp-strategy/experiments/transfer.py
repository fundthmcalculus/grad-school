"""Does one fitted rule base transfer, or does each instance family need its own?

This is the load-bearing claim and it had never been measured directly. The reported result is
that a rule base fitted once, on a pool of nine TSPLIB and eleven synthetic instances, works on
a held-out test set spanning n = 52…18512 and four structural families. That is a *test score*.
It becomes a *generalisation* result only against a contrast: what would fitting per family
have bought? If refitting on the family you are about to be tested on barely helps, the rule
base transfers, and the scale-free-ratio design is doing what it was built to do.

So: fit on one structural family alone, test on all of them. The diagonal of the resulting
matrix is the home-field score, the off-diagonal is transfer, and the gap between them is the
cost of not knowing your instance family in advance.

**Why this is affordable at all.** The objective is a ratio of two tours on *one* instance —
the arm's length over what the swept LK frontier reaches at the same cost — so the optimum
cancels and synthetic instances need none. `synth.py` generates as many as patience allows in
four families:

* ``uniform`` — points uniform in a square, the easy case;
* ``clustered`` — Gaussian blobs, where candidate lists inside a blob are useless and the
  interesting edges are the few between blobs (this is what makes fl* and vm* hard);
* ``grid`` — jittered lattice, massively tie-heavy, where the rl* instances live;
* ``mixed`` — half dense blobs, half sparse uniform, so one instance contains two regimes and
  no single global effort setting is right.

Fitting reuses ``tune_opt.run_one`` rather than reimplementing it, so a transfer run and the
reported run differ only in their instance pools.

Run:  python experiments/transfer.py                 # the full matrix, ~15-25 min
      python experiments/transfer.py --quick         # a small GA budget, for wiring
      python experiments/transfer.py --families uniform clustered
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402

paths.on_path()

import numpy as np  # noqa: E402

import fis  # noqa: E402
import synth  # noqa: E402
import tune_opt as T  # noqa: E402

FAMILIES = ("uniform", "clustered", "grid", "mixed")

#: Instances per family, generated fresh here rather than reused from ``synth.TRAIN_SPEC`` so
#: that every family gets the same count and size spread — otherwise a family's transfer score
#: would partly measure how many instances it happened to have.
SIZES_FIT = (1300, 2700, 5300)
SIZES_TEST = (1500, 1900, 3100, 4100)


def _pool(family, sizes, seed0):
    return synth.pool([(family, n, seed0 + i) for i, n in enumerate(sizes)])


def _tsplib_pool():
    """The held-out TSPLIB instances above the fitting floor, as a fifth 'family'.

    Drawn from ``benchmark.TEST`` so nothing here has been fitted on, and capped in size
    because the frontier sweep inside ``Objective`` re-solves each instance seven times.
    """
    from tsplib import load

    import benchmark

    names = [n for n in benchmark.TEST if 1000 <= load(n).n <= 4000]
    return [load(n) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--polish-evals", type=int, default=400)
    ap.add_argument("--scale", default=fis.DEFAULT_SCALE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(paths.RESULTS / "transfer.json"))
    args = ap.parse_args()
    paths.utf8_stdout()
    paths.ensure()
    if args.quick:
        args.generations, args.population, args.polish_evals = 3, 8, 50

    coef = np.load(paths.COSTMODEL)["coef"]
    space = T.ParamSpace("triangular", "base", args.scale)

    # test pools: one per family, plus held-out TSPLIB. Built once and shared by every fitted
    # vector, so a row and a column of the matrix are directly comparable.
    print("building test pools...", flush=True)
    tests = {f: T.Objective(_pool(f, SIZES_TEST, 900 + 10 * i), space, coef)
             for i, f in enumerate(args.families)}
    tests["tsplib"] = T.Objective(_tsplib_pool(), space, coef)
    for name, obj in tests.items():
        print(f"  {name:>10s}: {len(obj.items)} instances "
              f"({', '.join(str(it[0].n) for it in obj.items)})")

    rows = {}

    # reference rows, fitted on nothing and on everything
    print("\nreference rule bases", flush=True)
    rows["hand-written"] = {t: tests[t].report(space.default())[0] for t in tests}
    shipped = paths.tuned(args.scale)
    if shipped.exists():
        z = np.load(shipped)
        if "theta" in z:
            rows["shipped (all families)"] = {
                t: tests[t].report(np.asarray(z["theta"], dtype=np.float64))[0] for t in tests
            }

    for i, fam in enumerate(args.families):
        print(f"\nfitting on {fam} only...", flush=True)
        t0 = time.perf_counter()
        theta, rec, _ = T.run_one(
            "ga", "triangular", "base", args.generations, args.population, 1,
            args.seed, 0.3, [], args.polish_evals, args.scale,
            train_pool=_pool(fam, SIZES_FIT, 100 + 10 * i),
            valid_pool=_pool(fam, SIZES_TEST, 500 + 10 * i),
        )
        print(f"  fitted in {time.perf_counter() - t0:.0f}s "
              f"(train q {rec['train_ratio']:.4f}, valid q {rec['valid_ratio']:.4f})")
        rows[f"fitted on {fam}"] = {t: tests[t].report(theta)[0] for t in tests}

    cols = list(tests)
    print(f"\n{'':>26s} " + " ".join(f"{c:>10s}" for c in cols))
    for name, r in rows.items():
        print(f"{name:>26s} " + " ".join(f"{r[c]:10.4f}" for c in cols))
    print("\nq = arm's tour length / what the swept LK frontier reaches at the same cost.")
    print("Lower is better; 1.0 is the frontier. Diagonal cells are home-field (fitted and")
    print("tested on the same family, different instances); off-diagonal cells are transfer.")

    # the number the claim rests on: how much worse is transfer than home field
    print()
    for fam in args.families:
        row = rows.get(f"fitted on {fam}")
        if not row:
            continue
        home = row[fam]
        away = [row[c] for c in args.families if c != fam]
        print(f"  fitted on {fam:>10s}: home {home:.4f}, mean away {np.mean(away):.4f} "
              f"(+{np.mean(away) - home:+.4f}), TSPLIB {row['tsplib']:.4f}")

    Path(args.out).write_text(json.dumps(
        {"rows": rows, "columns": cols, "sizes_fit": SIZES_FIT, "sizes_test": SIZES_TEST,
         "generations": args.generations, "population": args.population}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
