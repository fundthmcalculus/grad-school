"""Canonical-dictionary test at width 64: ci, 100K, 300M characters, seeds 0-2, learned vs constructed initial value.  python flm/bij_schedule/canon300.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
ARMS = {
    "learned": {},
    "canonInit": {"FIXED_C": "outputs/embspace/canon64.npz", "FIXED_TRAIN": "1"},
}
JOBS = [(a, s) for s in (0, 1, 2) for a in ARMS]


def run(j):
    a, s = j
    env = dict(os.environ, D="64", FFN_R="125", FFN_R0="88", **ARMS[a])
    if a != "learned":
        env["RUN_MODULE"] = "flm.run_fixed_embedding"
    subprocess.run(
        [
            BASH,
            "flm/bij_schedule/run_full.sh",
            "ci",
            "0.01",
            "300",
            f"outputs/canon300/{a}_s{s}",
            "--precision",
            "fp32",
            "--seed",
            str(s),
        ],
        env=env,
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, JOBS))
