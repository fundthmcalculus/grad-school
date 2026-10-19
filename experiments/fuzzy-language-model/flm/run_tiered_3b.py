"""3B-character run (the _all stream, about 1.4 epochs; `v38-tiered3b`) of the best tiered-embedding recipe (as `v35-tiered1b`): t768e16m225 (98,280 params), warm-up 100, lr 6e-2, seed 0.
References at 1B chars (same config, plain embedding): v17-1b seed 0 / seed 1; 3B chars: 0.97767.  Needs the _all stream.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v38-tiered3b")
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = OUT / f"t768e16m225_3000M_s{seed}.json"
    a = argv("random", seed)
    a[a.index("--out") + 1] = str(out)
    a[a.index("--chars") + 1] = "3000e6"
    a[a.index("--ffn-mult") + 1] = "2.25"
    for flag in ("--eval-every", "--eval-at-start"):
        i = a.index(flag)
        del a[i : i + (2 if flag == "--eval-every" else 1)]
    a += ["--eval-every", "5000", "--save-model"]
    env = dict(
        os.environ,
        TIER_NH="768",
        TIER_E="16",
        PYTHONPATH=".",
        FLM_TRAIN_SUFFIX="_all",
    )
    with open(OUT / f"s{seed}.log", "w") as f:
        r = subprocess.run(
            [sys.executable, "-m", "flm.run_tiered"] + a,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(seed, "exit", r.returncode, flush=True)
