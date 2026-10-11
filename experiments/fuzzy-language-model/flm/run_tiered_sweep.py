"""GPU screen of the frequency-tiered embedding (`v30-tiered`): 30M characters, seeds 0-2; baseline = v22-tribble random (96,416 params)."""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v30-tiered")
ARMS = {  # name: (head tokens, tail width, ffn mult)
    "t256e32m15": (256, 32, "1.5"),
    "t256e32m20": (256, 32, "2.0"),
    "t512e24m20": (512, 24, "2.0"),
    "t512e24m25": (512, 24, "2.5"),
    "t768e16m25": (768, 16, "2.5"),
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in range(3):
        for name, (nh, e, m) in ARMS.items():
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = m
            env = dict(os.environ, TIER_NH=str(nh), TIER_E=str(e), PYTHONPATH=".")
            with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_tiered"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
