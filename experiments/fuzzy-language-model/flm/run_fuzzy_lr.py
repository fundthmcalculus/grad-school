"""Learning-rate and warm-up sweep for the all-fuzzy model (`v53-fuzzylr`, PLAN_FUZZYLOCAL.md): the 2-block all-fuzzy `e2e_tail` recipe, 30M characters, seeds 0-4,
paired with `e2e_tail` at the hybrid's settings (lr 6e-2, warm-up 100: mean 1.1368).  Every all-fuzzy run so far used the hybrid recipe's optimiser settings.

``python -m flm.run_fuzzy_lr [arm ...]``; arms are named lr<value x 100>_wu<warm-up>.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

OUT = Path("outputs/v53-fuzzylr")
GRID = {  # name: (lr, warm-up)
    "lr3_wu100": ("3e-2", "100"),
    "lr10_wu100": ("1e-1", "100"),
    "lr15_wu100": ("1.5e-1", "100"),
    "lr6_wu300": ("6e-2", "300"),
    "lr10_wu300": ("1e-1", "300"),
    # longer warm-ups (the 30M runs have 953 steps: 450 / 600 / 900 are 47% / 63% / 94% of the run; the 300M runs have 9,522)
    "lr6_wu450": ("6e-2", "450"),
    "lr6_wu600": ("6e-2", "600"),
    "lr6_wu900": ("6e-2", "900"),
    "lr6_wu1000": ("6e-2", "1000"),
    "lr6_wu3000": ("6e-2", "3000"),
}
CH = os.environ.get(
    "LR_CHARS", "30"
)  # millions of characters: 30 = the five-seed screen, 300 = the two-seed confirmation
SEEDS = [int(x) for x in os.environ.get("LR_SEEDS", "0,1,2,3,4").split(",")]
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(GRID)
    envx, over, mult = E2E["e2e_tail"]
    for seed in SEEDS:
        for name in arms:
            lr, wu = GRID[name]
            out = OUT / (
                f"{name}_s{seed}.json" if CH == "30" else f"{name}_{CH}M_s{seed}.json"
            )
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = mult
            a[a.index("--tok") + 1] = "bpe3072"
            a[a.index("--lr") + 1] = lr
            a[a.index("--warmup") + 1] = wu
            a[a.index("--chars") + 1] = f"{CH}e6"
            if CH != "30":
                for flag in ("--eval-every", "--eval-at-start"):
                    i = a.index(flag)
                    del a[i : i + (2 if flag == "--eval-every" else 1)]
                a += ["--eval-every", "5000", "--save-model"]
            for k, v in over.items():
                if k in a:
                    a[a.index(k) + 1] = v
                else:
                    a += [k, v]
            env = {
                **os.environ,
                "TIER_NH": "384",
                "TIER_E": "12",
                "TIER_FREQ": "outputs/freq_bpe3072.npz",
                "PYTHONPATH": ".",
                **envx,
            }
            with open(OUT / "logs" / f"{name}_{CH}M_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_allfuzzy"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
