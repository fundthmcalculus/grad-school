"""CPU screen of the RFLM variants (PLAN_RFLM.md): 96K config, 4M chars, seeds 0-1, paired with v19-pilot's base_s0/s1.

Usage: python -m flm.run_rflm [lanes]   (jobs run in ``lanes`` parallel lanes, 6 threads each)
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flm.grids_v3 import D

OUT = Path("outputs/v20-rflm")
VARIANTS = {
    "fixed": ["--mixer", "fuzzydelta_static", "--decay", "fixed"],
    "cpread": ["--mixer", "fuzzydelta_cpread"],
    "cpboth": ["--mixer", "fuzzydelta_cpboth"],
}


def argv(tag: str, seed: int) -> list[str]:
    a = list(D)
    over = VARIANTS[tag]
    for i in range(0, len(over), 2):  # replace, don't duplicate, the base's flag
        j = a.index(over[i])
        a[j + 1] = over[i + 1]
    a += ["--d", "48", "--layers", "2", "--seed", str(seed), "--chars", "4e6"]
    a += ["--lr", "6e-2", "--shortconv", "8", "--tok", "bpe1024", "--ctx", "512"]
    a += ["--batch", "16", "--rules", "32", "--warmup", "100", "--device", "cpu"]
    a += ["--threads", "6", "--out", str(OUT / f"{tag}_s{seed}.json")]
    return a


def run(job):
    tag, seed = job
    log = OUT / "logs" / f"{tag}_s{seed}.log"
    with open(log, "w") as f:
        r = subprocess.run(
            [sys.executable, "-m", "flm.run_variant"] + argv(tag, seed),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print(tag, seed, "exit", r.returncode, flush=True)


if __name__ == "__main__":
    lanes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(t, s) for s in (0, 1) for t in VARIANTS]
    with ThreadPoolExecutor(lanes) as ex:
        list(ex.map(run, jobs))
