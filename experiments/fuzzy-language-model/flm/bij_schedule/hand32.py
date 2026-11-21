"""Hand-built 32-wide dictionary (see PLAN_BIJECTIVE.md): ci, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1.  python flm/bij_schedule/hand32.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
ARMS = {"hand32Frozen": {}, "hand32Init": {"FIXED_TRAIN": "1"}}
JOBS = [(a, r, s) for s in (0, 1) for r in (328, 706) for a in ARMS]


def run(j):
    a, r, s = j
    env = dict(
        os.environ,
        D="32",
        FFN_R=str(r),
        FFN_R0=str(round(r * 0.7)),
        FIXED_C="outputs/embspace/hand32.npz",
        RUN_MODULE="flm.run_fixed_embedding",
        **ARMS[a],
    )
    subprocess.run(
        [
            BASH,
            "flm/bij_schedule/run_full.sh",
            "ci",
            "0.01",
            "30",
            f"outputs/hand32/{a}_r{r}_s{s}",
            "--precision",
            "fp32",
            "--seed",
            str(s),
        ],
        env=env,
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, JOBS))
