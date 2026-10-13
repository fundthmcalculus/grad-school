"""GPU screen for the repetition-aware loss (PLAN_NEXT.md item 4): 96K, 30M characters, seeds 0-4, one cell at a time.
Baseline = the random-init cells of v22-tribble (same config and seeds)."""

import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v25-rep")
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in range(5):
        for v in ("unlikelihood", "weight"):
            out = OUT / f"{v}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            if seed == 0:
                a += ["--save-model"]
            with open(OUT / "logs" / f"{v}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_rep", v] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            print(v, seed, "exit", r.returncode, flush=True)
