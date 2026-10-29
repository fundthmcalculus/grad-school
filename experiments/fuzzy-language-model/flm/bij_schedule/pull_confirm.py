"""Pull-back confirmation (see PLAN_BIJECTIVE.md).  python flm/bij_schedule/pull_confirm.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
BASE = dict(D="48", FFN_R="196", FFN_R0="137", FIXED_C="outputs/embspace/oh48.npz", RUN_MODULE="flm.run_fixed_embedding", FIXED_TRAIN="1")
COSMO, PILE = "C:/personal/grad-school/data/cosmopedia", "C:/personal/grad-school/data/pile_chars"
J = [("0.0001", 1, COSMO, "cosmo"), ("0.00003", 1, COSMO, "cosmo"), ("0.0001", 0, PILE, "pile")]


def run(j):
    k, s, root, name = j
    subprocess.run(
        [BASH, "flm/bij_schedule/run_full.sh", "ci", "0.01", "300", f"outputs/pull_confirm/{name}_k{k}_s{s}", "--precision", "fp32", "--seed", str(s)],
        env=dict(os.environ, FIXED_PULL=k, DATA_ROOT=root, **BASE),
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, J))
