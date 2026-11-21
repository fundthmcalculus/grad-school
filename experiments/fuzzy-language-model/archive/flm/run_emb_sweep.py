"""GPU screen of the embedding configurations (`v28-emb`): 96K, 30M characters, seeds 0-4; baseline = v22-tribble random."""

import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v28-emb")
ARMS = ["ppmi", "ppmip", "std05", "std20", "nowd"]
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in range(5):
        for v in ARMS:
            out = OUT / f"{v}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            with open(OUT / "logs" / f"{v}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_emb", v] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            print(v, seed, "exit", r.returncode, flush=True)
