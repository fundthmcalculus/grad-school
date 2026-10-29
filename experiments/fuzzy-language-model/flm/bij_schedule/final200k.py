"""The final model at 200K parameters on Cosmopedia, 300M characters, seed 0 (see PLAN_BIJECTIVE.md).  python flm/bij_schedule/final200k.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
BASE = dict(D="48", FFN_R="448", FFN_R0="314", DATA_ROOT="C:/personal/grad-school/data/cosmopedia")
FIX = dict(FIXED_C="outputs/embspace/oh48.npz", RUN_MODULE="flm.run_fixed_embedding")
J = [
    ("learned", dict(BASE)),
    ("frozen", dict(BASE, **FIX)),
    ("init", dict(BASE, FIXED_TRAIN="1", **FIX)),
    ("pull0.0001", dict(BASE, FIXED_TRAIN="1", FIXED_PULL="0.0001", **FIX)),
]


def run(j):
    name, e = j
    subprocess.run(
        [BASH, "flm/bij_schedule/run_full.sh", "ci", "0.01", "300", f"outputs/final200k/{name}_s0", "--precision", "fp32", "--seed", "0"],
        env=dict(os.environ, **e),
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, J))
