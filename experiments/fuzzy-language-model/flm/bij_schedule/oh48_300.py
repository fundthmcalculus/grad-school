"""oh48 at 300M characters: learned, frozen, as initial value; seeds 0-2.  python flm/bij_schedule/oh48_300.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
ARMS = {"learned": None, "frozen": {}, "init": {"FIXED_TRAIN": "1"}}
JOBS = [(a, s) for s in (0, 1, 2) for a in ARMS]


def run(j):
    a, s = j
    env = dict(os.environ, D="48", FFN_R="196", FFN_R0="137")
    if ARMS[a] is not None:
        env.update(
            FIXED_C="outputs/embspace/oh48.npz",
            RUN_MODULE="flm.run_fixed_embedding",
            **ARMS[a],
        )
    subprocess.run(
        [
            BASH,
            "flm/bij_schedule/run_full.sh",
            "ci",
            "0.01",
            "300",
            f"outputs/oh48_300/{a}_s{s}",
            "--precision",
            "fp32",
            "--seed",
            str(s),
        ],
        env=env,
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, JOBS))
