"""Run a grid of ``flm.train`` jobs, one CPU thread each, resumably.

    .venv/bin/python -m flm.sweep scaling --workers 4
    .venv/bin/python -m flm.sweep scaling --list      # print the grid, run nothing

A job whose JSON already exists is skipped, so an interrupted sweep restarts where
it stopped. Grids are defined in GRIDS below; each entry is (name, argv).
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs"

# Arms of the main comparison: (label, extra argv). shortconv is applied to every arm alike.
ARMS = {
    "softmax-mlp": ["--mixer", "softmax", "--ffn", "mlp"],
    "linear-mlp": ["--mixer", "linear", "--ffn", "mlp"],
    "gla-mlp": ["--mixer", "linear", "--ffn", "mlp", "--decay", "data"],
    "delta-mlp": ["--mixer", "delta", "--ffn", "mlp", "--decay", "data"],
    "gru": ["--mixer", "gru", "--ffn", "none"],
    "flm": ["--mixer", "gauss", "--ffn", "tsk"],
    "frlm-acc": ["--mixer", "fuzzy", "--ffn", "tsk", "--decay", "data"],
    "frlm-delta": ["--mixer", "fuzzydelta", "--ffn", "tsk", "--decay", "data"],
}


def scaling_grid(widths, layers, seeds, chars, arms=ARMS, extra=()):
    jobs = []
    for (arm, argv), d, L, s in itertools.product(arms.items(), widths, layers, seeds):
        name = f"{arm}_d{d}_L{L}_s{s}"
        jobs.append(
            (
                name,
                argv
                + [
                    "--d",
                    str(d),
                    "--layers",
                    str(L),
                    "--seed",
                    str(s),
                    "--chars",
                    str(chars),
                    *extra,
                ],
            )
        )
    return jobs


GRIDS = {}


def register(name, jobs):
    GRIDS[name] = jobs


def run_job(sweep, name, argv):
    out = OUT / sweep / f"{name}.json"
    if out.exists():
        return name, "skip", 0.0
    log = OUT / sweep / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(
            [
                sys.executable,
                "-m",
                "flm.train",
                *argv,
                "--threads",
                "1",
                "--out",
                str(out),
            ],
            cwd=HERE,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    return (
        name,
        ("ok" if rc == 0 and out.exists() else f"FAIL rc={rc}"),
        time.time() - t,
    )


def main():
    import flm.grids  # noqa: F401  (registers grids into the *imported* flm.sweep)
    from flm.sweep import GRIDS

    p = argparse.ArgumentParser()
    p.add_argument("sweep", choices=sorted(GRIDS))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    jobs = GRIDS[a.sweep]
    if a.list:
        for n, argv in jobs:
            print(n, " ".join(argv))
        print(len(jobs), "jobs")
        return
    todo = [(n, v) for n, v in jobs if not (OUT / a.sweep / f"{n}.json").exists()]
    print(
        f"{a.sweep}: {len(jobs)} jobs, {len(todo)} to run, {a.workers} workers",
        flush=True,
    )
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, a.sweep, *j) for j in todo]
        for i, f in enumerate(as_completed(futs), 1):
            name, status, dt = f.result()
            print(f"[{i}/{len(todo)}] {status} {name} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
