"""300M-character confirmation of the best tiered-embedding arm (`v31-tiered300m`): t768e16m225 (768 head tokens, tail 16, FFN mult 2.25), seeds 0-1.
References (same config except the embedding/FFN, warm-up 100): v12-long512 0.9976 (seed 0) / 1.0053 (seed 1).
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v31-tiered300m")
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in range(2):
        out = OUT / f"t768e16m225_300M_s{seed}.json"
        if out.exists():
            continue
        a = argv("random", seed)
        a[a.index("--out") + 1] = str(out)
        a[a.index("--chars") + 1] = "300e6"
        a[a.index("--ffn-mult") + 1] = "2.25"
        for flag in ("--eval-every", "--eval-at-start"):
            i = a.index(flag)
            del a[i : i + (2 if flag == "--eval-every" else 1)]
        a += ["--save-model"]
        env = dict(os.environ, TIER_NH="768", TIER_E="16", PYTHONPATH=".")
        with open(OUT / "logs" / f"s{seed}.log", "w") as f:
            r = subprocess.run(
                [sys.executable, "-m", "flm.run_tiered"] + a,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        print(seed, "exit", r.returncode, flush=True)
