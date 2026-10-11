"""GPU test of the consequent-first (TRIBBLE-order) init, PLAN_BACKWARD.md stage 1b.

96K BPE1024 D, 30M characters, seeds 0-4, evaluated every 100 steps and at step 0; one cell at a time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from flm.grids_v3 import D

OUT = Path("outputs/v22-tribble")
VARIANTS = ["random", "data", "cf", "cfls", "assocls", "succcf"]


def argv(v, seed):
    a = list(D) + ["--d", "48", "--layers", "2", "--seed", str(seed), "--chars", "30e6"]
    a += ["--lr", "6e-2", "--shortconv", "8", "--tok", "bpe1024", "--ctx", "512"]
    a += ["--batch", "16", "--rules", "32", "--warmup", "100", "--device", "cuda"]
    a += [
        "--eval-every",
        "100",
        "--eval-at-start",
        "--out",
        str(OUT / f"{v}_s{seed}.json"),
    ]
    if v != "random":
        a += ["--rule-init", "data"]
    return a


if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in range(5):
        for v in VARIANTS:
            if (OUT / f"{v}_s{seed}.json").exists():
                continue
            mod = ["-m", "flm.train"] if v in ("random", "data") else ["-m", "flm.run_conseq", v]
            with open(OUT / "logs" / f"{v}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable] + mod + argv(v, seed),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            print(v, seed, "exit", r.returncode, flush=True)
