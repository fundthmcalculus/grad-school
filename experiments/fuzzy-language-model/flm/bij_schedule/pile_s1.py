"""oh48 on the Pile, seed 1: learned, frozen, trainable start (see PLAN_BIJECTIVE.md).  python flm/bij_schedule/pile_s1.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
PILE = "C:/personal/grad-school/data/pile_chars"
OH = dict(D="48", FFN_R="196", FFN_R0="137", DATA_ROOT=PILE)
FIX = dict(FIXED_C="outputs/embspace/oh48.npz", RUN_MODULE="flm.run_fixed_embedding")
J = [("learned", dict(OH)), ("frozen", dict(OH, **FIX)), ("init", dict(OH, FIXED_TRAIN="1", **FIX))]


def run(j):
    name, e = j
    subprocess.run(
        [BASH, "flm/bij_schedule/run_full.sh", "ci", "0.01", "300", f"outputs/pile_s1/{name}_s1", "--precision", "fp32", "--seed", "1"],
        env=dict(os.environ, **e),
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, J))
