"""LKH reference numbers, measured separately and with a wall-clock cap.

LKH is the external yardstick, not a competitor this work claims to beat: it is a
mature C implementation with a much stronger candidate rule (alpha-nearness from a
held 1-tree rather than plain k-nearest) and 5-opt basic moves. What it is good for
here is showing where both arms sit in absolute terms.

It lives in its own script because it cannot be trusted to terminate promptly. On the
clustered ``fl*`` instances a single LKH run can spin for many minutes, so each
instance is run in a subprocess with a hard timeout and a miss is recorded as a miss
rather than being allowed to stall the benchmark.

Run:  python lkh_reference.py [--max-n 3000] [--timeout 120] [--out lkh.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

_CHILD = r"""
import json, sys, time
sys.path.insert(0, {here!r})
import numpy as np
import elkai
from tsplib import load, reference_length, validate_tour

inst = load({name!r})
coords = {{str(i): (float(x), float(y)) for i, (x, y) in enumerate(inst.coords)}}
t0 = time.perf_counter()
order = elkai.Coordinates2D(coords).solve_tsp(runs=1)
dt = time.perf_counter() - t0
tour = np.array([int(s) for s in order], dtype=np.int32)
if tour.shape[0] == inst.n + 1:
    tour = tour[:-1]
validate_tour(tour, inst.n)
print(json.dumps({{"gap": inst.gap(reference_length(tour, inst)), "s": dt}}))
"""


def lkh_one(name, timeout):
    """(gap, seconds) or None if LKH did not finish inside ``timeout``."""
    code = _CHILD.format(here=str(HERE), name=name)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(HERE),
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    for line in reversed(proc.stdout.strip().splitlines()):
        try:
            d = json.loads(line)
            return d["gap"], d["s"]
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def main():
    import benchmark
    from tsplib import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=3000)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default=str(HERE / "lkh.json"))
    args = ap.parse_args()

    out = {}
    for name in benchmark.TEST:
        inst = load(name)
        if inst.n > args.max_n or inst.ceil:
            continue  # elkai builds the problem as EUC_2D; CEIL_2D would be mis-scored
        t0 = time.perf_counter()
        res = lkh_one(name, args.timeout)
        if res is None:
            print(f"  {name:>9s} n={inst.n:6d}  no result within {args.timeout:.0f}s", flush=True)
            out[name] = {"n": inst.n, "gap": None, "s": None}
            continue
        gap, dt = res
        print(
            f"  {name:>9s} n={inst.n:6d}  gap {gap:6.3f}%  {dt:7.3f}s "
            f"(wall incl. startup {time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
        out[name] = {"n": inst.n, "gap": gap, "s": dt}

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
