"""GPU screen of the coverage hybrid init (PLAN_BACKWARD.md, `v27-cover`): 30M characters, seeds 0-4, one cell at a time.
Arms: hyb18g1 (25% built consequent-first, rest random), cov25 (25% built, rest D^2-spread over the layer's inputs),
cov0 (all rules D^2-spread, no clustering).  References at the same size/seeds: v22-tribble random and data."""

import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v27-cover")
ARMS = ["cov0", "cov25", "hyb18g1"]
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in range(5):
        for v in ARMS:
            out = OUT / f"{v}_s{seed}.json"
            if out.exists():
                continue
            a = argv("cf", seed)
            a[a.index("--out") + 1] = str(out)
            if seed == 0:
                a += ["--save-model"]
            with open(OUT / "logs" / f"{v}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_conseq", v] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            print(v, seed, "exit", r.returncode, flush=True)
