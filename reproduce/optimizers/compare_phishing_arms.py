#!/usr/bin/env python3
"""Is any arm actually separable from any other on PhiUSIIL?

    uv run --with numpy python reproduce/optimizers/compare_phishing_arms.py \
        --archive opt-phishing-hot10-2026-08-03 --init hot

The five-seed run put Powell at 5/5 seeds with a ±0.0000 spread, which is the
first arm-vs-arm ordering anywhere in this project that looked separable. Five
seeds is not enough to say so, hence the ten-seed run this script reads.

Two tests per pair, because they fail in different ways:

**Effect size relative to spread.** `|mean(d)| / sd(d)` over the paired per-seed
differences. Below 1.0 the difference is smaller than its own scatter and the
ordering is not a result — that is the bar the rest of this study has been held
to, and it is deliberately the same bar.

**Sign test.** How many seeds favour one arm, as an exact two-sided binomial
probability. This catches the case the effect size misses: a difference that is
tiny but reliable in *direction*. On a saturated dataset that is the more likely
shape, since a model can be consistently better by two test errors out of 48,000
and have almost no variance to divide by.

**The seed count sets a floor on the sign test, and the floor can be above the
threshold.** With n paired seeds the smallest attainable two-sided p is `2/2**n`:
at five seeds that is 0.0625, so a *perfect* 5–0 sweep cannot clear 0.05 and the
test is structurally incapable of certifying anything. At ten seeds the floor is
0.002. The script prints the floor, because "p = 0.062, not separable" invites
the reading that the arms are equivalent when the truth is that the experiment
was too small to answer.

Both are reported, and a pair is only called separable when the effect size
clears 1.0 **and** the sign test clears 0.05. Neither alone is sufficient: the
first can be inflated by a single outlier seed, the second is indifferent to
magnitude and will happily certify a difference nobody should care about.

Accuracy is also reported as an error count, because `+0.0004` on a 48,000-row
split is nineteen test errors and reads like nothing.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
OUTPUTS = os.path.join(ROOT, "reproduce", "outputs")

#: Arms whose numbers are not a measurement of the named method. Reported, but
#: never called separable -- see Addendum 7 and fundthmcalculus/optimizers#101.
SUSPECT = {"opt-pso": "optimizers#101: velocities collapse, swarm restarts each "
                      "generation -- this row is random sampling"}


def _sign_test(d):
    """Two-sided exact binomial p for the signs of `d`, ties dropped."""
    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    n = pos + neg
    if n == 0:
        return 1.0, 0, 0
    k = max(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail), pos, neg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default="opt-phishing-hot10-2026-08-03")
    ap.add_argument("--init", default="hot")
    ap.add_argument("--metric", default="acc",
                    help="column to compare; `acc` (higher better) or `obj` "
                         "(lower better)")
    args = ap.parse_args()

    path = os.path.join(OUTPUTS, args.archive, "table_opt_phishing_seeds.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no seeds CSV at {os.path.relpath(path, ROOT)}")
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["init"] == args.init]
    if not rows:
        raise SystemExit(f"no rows for init={args.init!r}")

    higher_better = args.metric == "acc"
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[int(r["seed"])] = float(r[args.metric])
    n_test = int(rows[0].get("n_test") or 0)
    seeds = sorted(set.intersection(*(set(v) for v in by.values())))
    arms = [a for a in by if a != "none"]

    floor = 2.0 / 2 ** len(seeds)
    print(f"archive: {args.archive}   init: {args.init}   metric: {args.metric}")
    print(f"seeds:   {len(seeds)} common to every arm")
    print(f"sign-test floor at {len(seeds)} seeds: p >= {floor:.4f}"
          + ("  -- ABOVE 0.05, so no pair can clear the threshold however clean "
             "the sweep; this run cannot certify separability"
             if floor > 0.05 else "  (a clean sweep can clear 0.05)"))
    print()

    ref = by.get("none", {})
    print(f"{'arm':<16}{'mean':>10}{'s.d.':>10}{'errors':>10}"
          f"{'vs construction (paired)':>28}{'seeds':>8}")
    for arm in sorted(arms, key=lambda a: -np.mean([by[a][s] for s in seeds])
                      if higher_better else np.mean([by[a][s] for s in seeds])):
        v = np.array([by[arm][s] for s in seeds])
        d = np.array([by[arm][s] - ref[s] for s in seeds]) if ref else np.zeros(0)
        err = (1.0 - v.mean()) * n_test if higher_better and n_test else None
        won = int(np.sum(d > 0)) if higher_better else int(np.sum(d < 0))
        flag = "  <- see note" if arm in SUSPECT else ""
        print(f"{arm:<16}{v.mean():>10.5f}{v.std():>10.5f}"
              f"{('—' if err is None else f'{err:.0f}'):>10}"
              f"{f'{d.mean():+.5f} ± {d.std():.5f}':>28}"
              f"{f'{won}/{len(seeds)}':>8}{flag}")

    print("\npairwise, paired over seeds "
          "(effect = |mean|/s.d.; sign test = exact two-sided binomial)")
    print(f"{'pair':<30}{'mean Δ':>12}{'s.d.':>10}{'effect':>9}"
          f"{'signs':>9}{'p':>8}   verdict")
    verdicts = []
    for a, b in itertools.combinations(sorted(arms), 2):
        d = np.array([by[a][s] - by[b][s] for s in seeds])
        sd = float(d.std())
        effect = abs(float(d.mean())) / sd if sd > 0 else float("inf")
        p, pos, neg = _sign_test(d)
        suspect = a in SUSPECT or b in SUSPECT
        sep = (effect > 1.0) and (p < 0.05) and not suspect
        if suspect:
            verdict = "not measured (see note)"
        elif sep:
            verdict = "SEPARABLE"
        elif p < 0.05:
            verdict = "consistent in sign, small"
        else:
            verdict = "not separable"
        verdicts.append((sep, effect, p, a, b, d))
        print(f"{f'{a} − {b}':<30}{d.mean():>+12.5f}{sd:>10.5f}"
              f"{effect:>9.2f}{f'{pos}/{pos + neg}':>9}{p:>8.3f}   {verdict}")

    sep = [v for v in verdicts if v[0]]
    print()
    if not sep and floor > 0.05:
        print(f"No pair clears both bars — but at {len(seeds)} seeds none could. "
              f"Re-read this as 'the run was too small', not 'the arms are the "
              f"same'.\nThe pairs to watch are those with a large effect and a "
              f"clean sweep:")
        for _s, effect, p, a, b, _d in sorted(verdicts, key=lambda v: -v[1])[:4]:
            if a in SUSPECT or b in SUSPECT:
                continue
            print(f"  {a} − {b}: effect {effect:.2f}, p = {p:.3f} (floor {floor:.4f})")
    elif sep:
        print(f"{len(sep)} pair(s) clear both bars:")
        for _s, effect, p, a, b, d in sorted(sep, key=lambda v: -v[1]):
            n = abs(float(d.mean())) * n_test if n_test else float("nan")
            print(f"  {a} − {b}: effect {effect:.2f}, p = {p:.3f}"
                  + (f", {n:.0f} test errors" if n_test else ""))
    else:
        print("No pair clears both bars, and the seed count was sufficient for "
              "one to. The ordering is not a result.")
    if any(a in SUSPECT for a in arms):
        print()
        for arm, why in SUSPECT.items():
            if arm in arms:
                print(f"note: {arm} — {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
