"""GPU screen (`v33-tiered-d`): spend the tiered embedding's saved parameters on a wider d_model; 30M chars, seeds 0-4 vs v30-tiered t768e16m225."""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v33-tiered-d")
CHARS = os.environ.get("TD_CHARS", "30e6")
ARMS = {  # name: (d, head tokens, tail width, ffn mult)
    "d56h448e16m15": (56, 448, 16, "1.5"),
    "d56h384e16m15": (56, 384, 16, "1.5"),
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in os.environ.get("TD_SEEDS", "0,1,2,3,4").split(",")]
    for seed in seeds:
        for name, (d, nh, e, m) in ARMS.items():
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = m
            a[a.index("--d") + 1] = str(d)
            a[a.index("--chars") + 1] = CHARS
            env = dict(os.environ, TIER_NH=str(nh), TIER_E=str(e), PYTHONPATH=".")
            with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_tiered"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
