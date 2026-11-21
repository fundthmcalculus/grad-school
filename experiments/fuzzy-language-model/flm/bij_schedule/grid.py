"""Rules against width grid (see PLAN_BIJECTIVE.md): ci, learned dictionary, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1.  python flm/bij_schedule/grid.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
HAVE = {(64, 124), (32, 328), (16, 706)}
CELLS = [(d, r) for d in (64, 32, 24, 16) for r in (124, 328, 706)] + [(24, 456)]
JOBS = [(d, r, s) for s in (0, 1) for d, r in CELLS if (d, r) not in HAVE]


def run(j):
    d, r, s = j
    env = dict(os.environ, D=str(d), FFN_R=str(r), FFN_R0=str(round(r * 0.7)))
    subprocess.run(
        [
            BASH,
            "flm/bij_schedule/run_full.sh",
            "ci",
            "0.01",
            "30",
            f"outputs/grid/d{d}_r{r}_s{s}",
            "--precision",
            "fp32",
            "--seed",
            str(s),
        ],
        env=env,
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, JOBS))
