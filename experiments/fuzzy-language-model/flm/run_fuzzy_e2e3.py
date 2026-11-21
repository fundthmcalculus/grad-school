"""Head repairs carried into the end-to-end all-fuzzy model (`v58-e2eembed`, PLAN_FUZZYLOCAL.md stage 7): 30M characters, seeds 0-4, lr 6e-2, warm-up 600, paired with `lr6_wu600` of `v53-fuzzylr`
(`e2e_tail`, mean 1.1207).  Arms: `e2e_prior` (the e2e_tail recipe plus the log-unigram prior), `e2e_all_wu600` (both token tables fuzzy, no repair: the reference), `e2e_all_prior`, `e2e_all_tp` (signed
memberships plus the prior on the head table).  ``python -m flm.run_fuzzy_e2e3 [arm ...]``
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

OUT = Path("outputs/v58-e2eembed")
GRID = {  # name: (e2e arm, extra environment)
    "e2e_prior": ("e2e_tail", {"FUZZ_TOK_PRIOR": "1"}),
    "e2e_all_wu600": ("e2e_all", {}),
    "e2e_all_prior": ("e2e_all", {"FUZZ_TOK_PRIOR": "1"}),
    "e2e_all_tp": ("e2e_all", {"FUZZ_TOK_FORM": "tanh", "FUZZ_TOK_PRIOR": "1"}),
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
