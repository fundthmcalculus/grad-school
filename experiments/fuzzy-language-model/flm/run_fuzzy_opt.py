"""Optimiser screens for the all-fuzzy model (`v55-fuzzyopt`, PLAN_FUZZYLOCAL.md): the `e2e_tail` recipe at its best 30M settings (lr 6e-2, warm-up 600), 30M characters,
seeds 0-4, paired with `lr6_wu600` of `v53-fuzzylr` (mean 1.1207).  One setting changed per arm: weight decay (default 0.01), Adam beta2 (default 0.98), gradient clip (default 1.0).

``python -m flm.run_fuzzy_opt [arm ...]``
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

OUT = Path("outputs/v55-fuzzyopt")
GRID = {  # name: extra train args
    "wd003": ["--wd", "0.003"],
    "wd03": ["--wd", "0.03"],
    "wd10": ["--wd", "0.1"],
    "b2_95": ["--beta2", "0.95"],
    "b2_999": ["--beta2", "0.999"],
    "clip03": ["--clip", "0.3"],
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(GRID)
    envx, over, mult = E2E["e2e_tail"]
    for seed in range(5):
        for name in arms:
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = mult
            a[a.index("--tok") + 1] = "bpe3072"
            a[a.index("--warmup") + 1] = "600"
            for k, v in over.items():
                if k in a:
                    a[a.index(k) + 1] = v
                else:
                    a += [k, v]
            a += GRID[name]
            env = {
                **os.environ,
                "TIER_NH": "384",
                "TIER_E": "12",
                "TIER_FREQ": "outputs/freq_bpe3072.npz",
                "PYTHONPATH": ".",
                **envx,
            }
            with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_allfuzzy"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
