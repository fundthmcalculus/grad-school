"""Separate input and output prototypes in the end-to-end model (`v61-e2euntie`, PLAN_FUZZYLOCAL.md stage 9): 30M characters, seeds 0-4, lr 6e-2, warm-up 600, paired with `lr6_wu600` (1.1207).
Arms: `e2e_all_tanh` (both token tables fuzzy, signed memberships, no prior: the reference), `e2e_all_untied` (the same with separate input and output prototype matrices for the head table,
+2,304 parameters paid for by 42 / 28 feed-forward rules instead of 46 / 31: 99,674 parameters).  ``python -m flm.run_fuzzy_e2e4 [arm ...]``
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

OUT = Path("outputs/v61-e2euntie")
GRID = {  # name: (e2e arm, extra environment)
    "e2e_all_tanh": ("e2e_all", {"FUZZ_TOK_FORM": "tanh"}),
    "e2e_all_untied": (
        "e2e_all",
        {
            "FUZZ_TOK_FORM": "tanh",
            "FUZZ_TOK_UNTIE": "1",
            "FUZZ_FFN_R": "42",
            "FUZZ_FFN_R0": "28",
        },
    ),
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(GRID)
    for seed in range(5):
        for name in arms:
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            envx, over, mult = E2E[GRID[name][0]]
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
            env = {
                **os.environ,
                "TIER_NH": "384",
                "TIER_E": "12",
                "TIER_FREQ": "outputs/freq_bpe3072.npz",
                "PYTHONPATH": ".",
                **envx,
                **GRID[name][1],
            }
            with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_allfuzzy"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
