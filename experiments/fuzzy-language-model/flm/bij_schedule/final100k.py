"""The full invertible fuzzy model at 100K parameters on Cosmopedia v2, 300M characters (see PLAN_BIJECTIVE.md).  python flm/bij_schedule/final100k.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
ROOT = "C:/personal/grad-school/data/cosmopedia"
OH = dict(D="48", FFN_R="196", FFN_R0="137")
FIX = dict(FIXED_C="outputs/embspace/oh48.npz", RUN_MODULE="flm.run_fixed_embedding")
J = []
for s in (0, 1):
    J.append(("ci", "learned", s, dict(OH)))
    J.append(("ci", "frozen", s, dict(OH, **FIX)))
    J.append(("ci", "init", s, dict(OH, FIXED_TRAIN="1", **FIX)))
J.append(("tied", "tied", 0, dict(D="56", FFN_R="186", FFN_R0="130")))


def run(j):
    arm, name, s, e = j
    subprocess.run(
        [BASH, "flm/bij_schedule/run_full.sh", arm, "0.01", "300", f"outputs/final100k/{name}_s{s}", "--precision", "fp32", "--seed", str(s)],
        env=dict(os.environ, DATA_ROOT=ROOT, **e),
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, J))
