"""Equal-distance-letter dictionaries (see PLAN_BIJECTIVE.md): ci, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1.  python flm/bij_schedule/ohsq.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
SHAPE = {"oh48": (48, 196, 137), "squeeze32": (32, 328, 230)}
JOBS = [("learned48", "oh48", None, s) for s in (0, 1)]
for s in (0, 1):
    for name in SHAPE:
        JOBS.append((f"{name}Frozen", name, {}, s))
        JOBS.append((f"{name}Init", name, {"FIXED_TRAIN": "1"}, s))


def run(j):
    tag, name, extra, s = j
    d, r, r0 = SHAPE[name]
    env = dict(os.environ, D=str(d), FFN_R=str(r), FFN_R0=str(r0))
    if extra is not None:
        env.update(
            FIXED_C=f"outputs/embspace/{name}.npz",
            RUN_MODULE="flm.run_fixed_embedding",
            **extra,
        )
    subprocess.run(
        [
            BASH,
            "flm/bij_schedule/run_full.sh",
            "ci",
            "0.01",
            "30",
            f"outputs/ohsq/{tag}_s{s}",
            "--precision",
            "fp32",
            "--seed",
            str(s),
        ],
        env=env,
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, JOBS))
