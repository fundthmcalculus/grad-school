"""Projection variants of the all-fuzzy model (`v62-proj`, PLAN_FUZZYLOCAL.md stage 10): the mixer's q/k/v and output projections are rule-gated low-rank maps (`TSKLinear`, 8 rules over
the k-number projection, ranks 32 / 20), the largest single block cost at 30M (+0.028).  30M characters, seeds 0-4, `e2e_tail`, lr 6e-2, warm-up 600, paired with `lr6_wu600` (1.1207).
Arms (each pays for its extra parameters with fewer layer-1 feed-forward rules, so each has a trim-only control): `proj_raw` (memberships on the raw input instead of on the projection),
`proj_r16` (16 rules), `proj_bias` (a constant offset per rule: the full first-order consequent), `proj_rank40` (ranks 40 / 28), and the controls `ffn_r41` and `ffn_r40_27` (only the feed-forward trims).
``python -m flm.run_fuzzy_proj2 [arm ...]``
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

OUT = Path("outputs/v62-proj")
GRID = {  # name: (e2e arm, extra environment)
    "proj_raw": ("e2e_tail", {"FUZZ_PROJ_ANTE": "raw", "FUZZ_FFN_R": "49"}),
    "proj_r16": ("e2e_tail", {"FUZZ_PROJ_R": "16", "FUZZ_FFN_R": "44"}),
    "proj_bias": ("e2e_tail", {"FUZZ_PROJ_BIAS": "1", "FUZZ_FFN_R": "41"}),
    "proj_rank40": (
        "e2e_tail",
        {"FUZZ_PROJ_RANKS": "40,28", "FUZZ_FFN_R": "40", "FUZZ_FFN_R0": "27"},
    ),
    "ffn_r41": ("e2e_tail", {"FUZZ_FFN_R": "41"}),
    "ffn_r40_27": ("e2e_tail", {"FUZZ_FFN_R": "40", "FUZZ_FFN_R0": "27"}),
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
