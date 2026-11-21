"""GPU screens of fuzzy-inference replacements for the mixer's linear q/k/v and output projections (`v48-fuzzyproj`, PLAN_FUZZYLOCAL.md stage 3b):
30M characters, seeds 0-4, BPE3072 tiered recipe with the convolution kept (baseline mean 1.0547).  ``python -m flm.run_fuzzy_proj [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v48-fuzzyproj")
ARMS = {  # name: (env, train-arg overrides)
    "proj_tsk0": ({"FUZZ_PROJ": "tsk0"}, {}),
    "proj_tskl": ({"FUZZ_PROJ": "tskl"}, {}),
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(ARMS)
    for seed in range(5):
        for name in arms:
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            envx, over = ARMS[name]
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = "1.5"
            a[a.index("--tok") + 1] = "bpe3072"
            for k, v in over.items():
                a[a.index(k) + 1] = v
            env = dict(
                os.environ,
                TIER_NH="384",
                TIER_E="12",
                TIER_FREQ="outputs/freq_bpe3072.npz",
                PYTHONPATH=".",
                **envx,
            )
            with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_allfuzzy"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
