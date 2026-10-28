"""Precision study: 100K-parameter ci and tied, 30M characters of TinyStories, fp32 / tf32 / bf16 / fp16 autocast, seeds 0 and 1, two jobs at a time.  Run from the experiment folder: python flm/bij_schedule/precision.py"""

import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
JOBS = [
    (a, p, s)
    for s in (0, 1)
    for p in ("fp32", "tf32", "bf16", "fp16")
    for a in ("ci", "tied")
]


def run(j):
    a, p, s = j
    out = f"outputs/precision/{p}_{a}_s{s}"
    subprocess.run(
        [
            BASH,
            "flm/bij_schedule/run_full.sh",
            a,
            "0.01",
            "30",
            out,
            "--precision",
            p,
            "--seed",
            str(s),
        ]
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, JOBS))
