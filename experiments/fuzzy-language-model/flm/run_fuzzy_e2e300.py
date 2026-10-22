"""The stage-7 winners at 300M characters (`v59-e2e300`, PLAN_FUZZYLOCAL.md stage 7): lr 6e-2, warm-up 1000 (the 300M optimum, `v53-fuzzylr`), seeds 0-1, checkpoints saved; paired with `e2e_tail` at
warm-up 1000 (0.98929, two seeds) and the hybrid recipe (0.9589).  Arms (`e2e_all_untied`, separate input and output head prototypes with 42 / 28 FFN rules, added after stage 9; `e2e_all_tanh`, signed memberships without the prior, added after the first four runs showed the prior hurts at 300M): `e2e_prior` (fuzzy blocks and tail table, crisp head, log-unigram prior), `e2e_all_tp` (head table fuzzy too: signed memberships plus the prior).
``python -m flm.run_fuzzy_e2e300 [arm ...]``
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

OUT = Path("outputs/v59-e2e300")
GRID = {  # name: (e2e arm, extra environment)
    "e2e_prior": ("e2e_tail", {"FUZZ_TOK_PRIOR": "1"}),
    "e2e_all_tp": ("e2e_all", {"FUZZ_TOK_FORM": "tanh", "FUZZ_TOK_PRIOR": "1"}),
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
    for seed in range(2):
        for name in arms:
            out = OUT / f"{name}_300M_s{seed}.json"
            if out.exists():
                continue
            envx, over, mult = E2E[GRID[name][0]]
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = mult
            a[a.index("--tok") + 1] = "bpe3072"
            a[a.index("--warmup") + 1] = "1000"
            a[a.index("--chars") + 1] = "300e6"
            for flag in ("--eval-every", "--eval-at-start"):
                k = a.index(flag)
                del a[k : k + (2 if flag == "--eval-every" else 1)]
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
                **GRID[name][1],
            }
            with open(OUT / "logs" / f"{name}_300M_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_allfuzzy"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
