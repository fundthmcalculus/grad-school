"""Close the single-seed gaps (see PLAN_BIJECTIVE.md).  python flm/bij_schedule/gaps.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
OH = dict(
    D="48",
    FFN_R="196",
    FFN_R0="137",
    FIXED_C="outputs/embspace/oh48.npz",
    RUN_MODULE="flm.run_fixed_embedding",
    FIXED_TRAIN="1",
)
COSMO, PILE = (
    "C:/personal/grad-school/data/cosmopedia",
    "C:/personal/grad-school/data/pile_chars",
)
J = [
    (
        "ci",
        "tied_cosmo_s1",
        1,
        dict(D="56", FFN_R="186", FFN_R0="130", DATA_ROOT=COSMO, BIJ="tied"),
    ),
    ("ci", "pile_k0.00003_s0", 0, dict(OH, FIXED_PULL="0.00003", DATA_ROOT=PILE)),
    ("ci", "pile_k0.0001_s1", 1, dict(OH, FIXED_PULL="0.0001", DATA_ROOT=PILE)),
]


def run(j):
    arm, name, s, e = j
    a = "tied" if e.get("BIJ") == "tied" else "ci"
    subprocess.run(
        [
            BASH,
            "flm/bij_schedule/run_full.sh",
            a,
            "0.01",
            "300",
            f"outputs/gaps/{name}",
            "--precision",
            "fp32",
            "--seed",
            str(s),
        ],
        env=dict(os.environ, **e),
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, J))
