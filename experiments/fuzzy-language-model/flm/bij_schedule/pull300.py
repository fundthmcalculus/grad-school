"""oh48 with the weak pull-back at 300M: Cosmopedia seeds 0 and 1, Pile seed 0 (see PLAN_BIJECTIVE.md).  python flm/bij_schedule/pull300.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
BASE = dict(D="48", FFN_R="196", FFN_R0="137", FIXED_C="outputs/embspace/oh48.npz", RUN_MODULE="flm.run_fixed_embedding", FIXED_TRAIN="1", FIXED_PULL="0.001")
J = [
    ("cosmo", 0, "C:/personal/grad-school/data/cosmopedia"),
    ("cosmo", 1, "C:/personal/grad-school/data/cosmopedia"),
    ("pile", 0, "C:/personal/grad-school/data/pile_chars"),
]


def run(j):
    name, s, root = j
    subprocess.run(
        [BASH, "flm/bij_schedule/run_full.sh", "ci", "0.01", "300", f"outputs/pull300/{name}_s{s}", "--precision", "fp32", "--seed", str(s)],
        env=dict(os.environ, DATA_ROOT=root, **BASE),
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, J))
