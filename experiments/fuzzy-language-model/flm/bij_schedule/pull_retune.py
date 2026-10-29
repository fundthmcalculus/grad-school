"""Pull-back strength retune at 300M, Cosmopedia seed 0 (see PLAN_BIJECTIVE.md).  python flm/bij_schedule/pull_retune.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
BASE = dict(D="48", FFN_R="196", FFN_R0="137", FIXED_C="outputs/embspace/oh48.npz", RUN_MODULE="flm.run_fixed_embedding", FIXED_TRAIN="1", DATA_ROOT="C:/personal/grad-school/data/cosmopedia")


def run(k):
    subprocess.run(
        [BASH, "flm/bij_schedule/run_full.sh", "ci", "0.01", "300", f"outputs/pull_retune/k{k}_s0", "--precision", "fp32", "--seed", "0"],
        env=dict(os.environ, FIXED_PULL=k, **BASE),
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, ("0.0001", "0.00003", "0.00001")))
