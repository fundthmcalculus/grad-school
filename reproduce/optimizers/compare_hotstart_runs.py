#!/usr/bin/env python3
"""Did the library fix move any conclusion in the hot-start study?

    uv run python reproduce/optimizers/compare_hotstart_runs.py \
        --before opt-hotcold-2026-08-02 --after opt-hotcold-kmbic-2026-08-03

The study's claims are all *paired*: same seed, same split, same box, one thing
changed. Re-running it against a different library is another such pairing, so
it is checked the same way rather than by eyeballing two tables side by side.

Four things are compared, in the order the report makes them:

1. **The starting point** — the construction's own R² and CV MSE at zero
   evaluations, per seed. This is the only place the fix *should* show, because
   the hot start is the construction.
2. **Evaluations to match the heuristic** from a cold start. The report's
   headline "what the construction is worth, in iterations".
3. **Held-out R² at full budget**, per arm and init.
4. **Whether any arm-vs-arm ordering became separable.** The report's strongest
   negative claim is that none of them is; a re-run is a chance for that to
   fail, so it is re-tested rather than restated.

Seeds present in only one run are dropped, and the count is printed — a paired
statistic over a mismatched set is not paired.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
OUTPUTS = os.path.join(ROOT, "reproduce", "outputs")


def _rows(label, basename):
    path = os.path.join(OUTPUTS, label, f"{basename}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing: {os.path.relpath(path, ROOT)}")
    with open(path) as f:
        return list(csv.DictReader(f))


def _index(rows, *fields):
    out = {}
    for r in rows:
        out[tuple(r[f] for f in fields)] = r
    return out


def _paired(before, after, key_fields, value, where=lambda r: True):
    """(delta array, n) over keys present in both runs and passing `where`."""
    b, a = _index(before, *key_fields), _index(after, *key_fields)
    keys = [k for k in b if k in a and where(b[k]) and where(a[k])]
    deltas = []
    for k in sorted(keys):
        try:
            deltas.append(float(a[k][value]) - float(b[k][value]))
        except (ValueError, KeyError):
            continue
    return np.array(deltas), len(keys)


def _fmt(deltas, unit="", scale=1.0):
    if len(deltas) == 0:
        return "no paired rows"
    d = deltas * scale
    won = int((d > 0).sum())
    return (f"{d.mean():+.4f} ± {d.std():.4f}{unit}  "
            f"(after higher on {won}/{len(d)} seeds)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="opt-hotcold-2026-08-02")
    ap.add_argument("--after", default="opt-hotcold-kmbic-2026-08-03")
    args = ap.parse_args()

    b_seeds = _rows(args.before, "table_opt_hotstart_seeds")
    a_seeds = _rows(args.after, "table_opt_hotstart_seeds")
    print(f"before: {args.before}   ({len(b_seeds)} rows)")
    print(f"after:  {args.after}   ({len(a_seeds)} rows)")

    # -- 1. the starting point ---------------------------------------------- #
    print("\n== 1. the construction itself, before any optimization ==")
    print("   (the `none` arm, hot init -- this is the only place the fix must show)")
    only_none_hot = lambda r: r["arm"] == "none" and r["init"] == "hot"  # noqa: E731
    for field, unit, scale in (("r2", " R²", 1.0), ("cv_mse", " CV MSE", 1.0)):
        d, n = _paired(b_seeds, a_seeds, ("arm", "init", "seed"), field,
                       where=only_none_hot)
        print(f"   {field:<8} {_fmt(d, unit, scale)}   [{n} paired seeds]")

    # -- 2. what the construction is worth, in iterations -------------------- #
    print("\n== 2. cold-start evaluations to match the construction ==")
    cold = lambda r: r["init"] == "cold" and r["arm"] != "none"  # noqa: E731
    b_i, a_i = _index(b_seeds, "arm", "init", "seed"), _index(a_seeds, "arm", "init", "seed")
    arms = sorted({r["arm"] for r in a_seeds if cold(r)})
    for arm in arms:
        pairs = [(b_i[k], a_i[k]) for k in b_i
                 if k in a_i and k[0] == arm and k[1] == "cold"]
        if not pairs:
            continue
        bv = np.array([float(x["evals_to_heuristic"]) for x, _ in pairs])
        av = np.array([float(y["evals_to_heuristic"]) for _, y in pairs])
        print(f"   {arm:<15} before {bv.mean():7.0f} ± {bv.std():<6.0f}   "
              f"after {av.mean():7.0f} ± {av.std():<6.0f}   "
              f"paired {av.mean() - bv.mean():+.0f}")

    # -- 3. held-out R² at full budget --------------------------------------- #
    print("\n== 3. held-out R² at full budget, per arm ==")
    for init in ("hot", "cold"):
        print(f"   -- {init} --")
        for arm in sorted({r["arm"] for r in a_seeds}):
            d, n = _paired(b_seeds, a_seeds, ("arm", "init", "seed"), "r2",
                           where=lambda r, _a=arm, _i=init: r["arm"] == _a and r["init"] == _i)
            if n:
                print(f"     {arm:<15} {_fmt(d, ' R²')}")

    # -- 4. is any arm ordering separable now? ------------------------------- #
    print("\n== 4. arm-vs-arm, after the fix (hot) ==")
    print("   The report's strongest negative claim is that no ordering survives")
    print("   pairing. A |mean| smaller than its own s.d. is 'still not separable'.")
    by_seed = defaultdict(dict)
    for r in a_seeds:
        if r["init"] == "hot" and r["arm"] != "none":
            by_seed[r["seed"]][r["arm"]] = float(r["r2"])
    arms = sorted({a for v in by_seed.values() for a in v})
    worst = None
    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            d = np.array([v[x] - v[y] for v in by_seed.values() if x in v and y in v])
            if len(d) < 2:
                continue
            ratio = abs(d.mean()) / (d.std() or np.inf)
            if worst is None or ratio > worst[0]:
                worst = (ratio, x, y, d)
    if worst:
        ratio, x, y, d = worst
        verdict = "SEPARABLE -- re-read section 3" if ratio > 1.0 else "not separable"
        print(f"   widest pair: {x} - {y} = {d.mean():+.4f} ± {d.std():.4f} "
              f"({int((d > 0).sum())}/{len(d)} seeds) -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
