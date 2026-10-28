"""Separation and width tests for the constructed dictionary (see PLAN_BIJECTIVE.md): ci, 100K, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1.  python flm/bij_schedule/width.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
SHAPE = {64: (124, 87), 56: (156, 109), 32: (328, 230), 16: (706, 494), 8: (1443, 1010)}
JOBS = []
for s in (0, 1):
    for v in ("dlin64", "dlin64b", "kind64", "soft64"):
        JOBS.append((v, 64, f"outputs/embspace/{v}.npz", s))
    for k in (56, 32, 16, 8):
        JOBS.append((f"pca{k}", k, f"outputs/embspace/canonpca{k}.npz", s))
        JOBS.append((f"learned{k}", k, None, s))


def run(j):
    name, d, npz, s = j
    r, r0 = SHAPE[d]
    env = dict(os.environ, D=str(d), FFN_R=str(r), FFN_R0=str(r0))
    if npz:
        env.update(FIXED_C=npz, FIXED_TRAIN="1", RUN_MODULE="flm.run_fixed_embedding")
    subprocess.run(
        [
            BASH,
            "flm/bij_schedule/run_full.sh",
            "ci",
            "0.01",
            "30",
            f"outputs/width/{name}_s{s}",
            "--precision",
            "fp32",
            "--seed",
            str(s),
        ],
        env=env,
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, JOBS))
