"""Canonical-dictionary test at width 64: ci, 100K, 30M characters of TinyStories, fp32, lr 0.01, seeds 0 and 1.  Arms: learned (control), constructed frozen, constructed as initial value, random frozen.  python flm/bij_schedule/canon.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
ARMS = {
    "learned": {},
    "canonFrozen": {"FIXED_C": "outputs/embspace/canon64.npz"},
    "canonInit": {"FIXED_C": "outputs/embspace/canon64.npz", "FIXED_TRAIN": "1"},
    "randFrozen": {"FIXED_C": "outputs/embspace/rand64.npz"},
}
JOBS = [(a, s) for s in (0, 1) for a in ARMS]


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
            "30",
            f"outputs/canon/{a}_s{s}",
            "--precision",
            "fp32",
            "--seed",
            str(s),
        ],
        env=env,
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, JOBS))
