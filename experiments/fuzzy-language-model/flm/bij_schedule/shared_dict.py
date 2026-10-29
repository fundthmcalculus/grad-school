"""Shared-dictionary test: ci at 100K, 30M characters of TinyStories, fp32, lr 0.01, seeds 0 and 1 (the control, a learned dictionary, is the precision study's fp32 ci runs: same settings).  Arms: a frozen random Gaussian dictionary; the Pile model's codebook frozen; codebook, F and G frozen from the Pile model; the same from the TinyStories 2.2B model.  Two jobs at a time: python flm/bij_schedule/shared_dict.py"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASH = "C:/Program Files/Git/bin/bash.exe"
PILE = "outputs/pile/ci_300M/ci_lr0.01_300M.pt"
TS = "outputs/full/ci_2200M/ci_lr0.01_1900M.pt"
ARMS = {
    "randC": {"FIXED_C": "outputs/embspace/rand56.npz"},
    "pileC": {"FIXED_C": "outputs/embspace/ci_pile.npz"},
    "pileFG": {"FIXED_FG": PILE},
    "tsFG": {"FIXED_FG": TS},
}
JOBS = [(a, s) for s in (0, 1) for a in ARMS]


def run(j):
    a, s = j
    env = dict(os.environ, RUN_MODULE="flm.run_fixed_embedding", **ARMS[a])
    subprocess.run(
        [
            BASH,
            "flm/bij_schedule/run_full.sh",
            "ci",
            "0.01",
            "30",
            f"outputs/shared/{a}_s{s}",
            "--precision",
            "fp32",
            "--seed",
            str(s),
        ],
        env=env,
    )


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, JOBS))
