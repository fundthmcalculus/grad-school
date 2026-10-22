"""GPU screens of the review's untried head ideas (`v60-embed4`, PLAN_FUZZYLOCAL.md stage 8): 30M characters, seeds 0-4, the stage-5 setting, signed-membership head (`fall_tanh` 1.1059, FFN x1.35).
Arms: `fall_tanh_untied` (separate input and output prototype matrices, +2,304 parameters, FFN x1.2), `fall_tanh_mos2` (a mixture of two softmaxes, rank-8 component shifts, FFN x1.25),
`fall_tanh_pq4` (product-quantised head: four groups of twelve prototypes, prototype matrix 576 instead of 2,304 parameters, FFN x1.5).  ``python -m flm.run_fuzzy_embed4 [arm ...]``
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v60-embed4")
ARMS = {  # name: (env, ffn mult)
    "fall_tanh_untied": (
        {"FUZZ_EMB": "all", "FUZZ_TOK_FORM": "tanh", "FUZZ_TOK_UNTIE": "1"},
        "1.2",
    ),
    "fall_tanh_mos2": (
        {"FUZZ_EMB": "all", "FUZZ_TOK_FORM": "tanh", "FUZZ_TOK_MOS": "2"},
        "1.25",
    ),
    "fall_tanh_pq4": (
        {"FUZZ_EMB": "all", "FUZZ_TOK_FORM": "tanh", "FUZZ_TOK_PQ": "4"},
        "1.5",
    ),
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(ARMS)
    for seed in range(5):
        for name in arms:
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            envx, mult = ARMS[name]
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = mult
            a[a.index("--tok") + 1] = "bpe3072"
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
