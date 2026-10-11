"""Diagnosis of the consequent-first init's harm (PLAN_BACKWARD.md, `v24-diag`): 30M characters, seeds 0-2 (+ saved seed-0 pair).

cfg1   : cf with the per-layer output gain capped at 1 (the construction chose 1,024 every time)
cfw300 : cf with warm-up 300 instead of 100 (reference for warm-up alone: v19-optA30 wu300 seeds 0-2)
random-save / cf-save : seed 0 with --save-model, to compare rule usage after training
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v24-diag")
CELLS = [("cfg1", s, []) for s in range(3)] + [
    ("cfw300", s, ["--warmup", "300"]) for s in range(3)
]
CELLS += [("random-save", 0, ["--save-model"]), ("cf-save", 0, ["--save-model"])]

if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for name, seed, extra in CELLS:
        out = OUT / f"{name}_s{seed}.json"
        if out.exists():
            continue
        v = {"random-save": "random", "cf-save": "cf"}.get(name, name)
        a = argv(v, seed)
        a[a.index("--out") + 1] = str(out)
        a += extra  # a later --warmup overrides the earlier one
        mod = ["-m", "flm.train"] if v == "random" else ["-m", "flm.run_conseq", v]
        with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
            r = subprocess.run(
                [sys.executable] + mod + a, stdout=f, stderr=subprocess.STDOUT
            )
        print(name, seed, "exit", r.returncode, flush=True)
