"""GPU screens of fuzzy embedding / output-head variants (`v54-embed`, PLAN_FUZZYLOCAL.md stage 6): 30M characters, seeds 0-4, the stage-5 setting (BPE3072 tiered
hybrid recipe, convolution kept, warm-up 100 lr 6e-2: reference 1.0547; emb_ftail +0.040, emb_fall +0.130 from `v51-fuzzytokens`).
Arms: crisp_prior (control: the recipe plus the log-unigram prior), ftail_tanh / ftail_prior (tail table with signed memberships / with the prior),
fall_tanh / fall_prior / fall_res4 (head table: signed memberships / prior / softmax plus a crisp rank-4 residual; FFN x1.25 for the residual to stay under the cap).
``python -m flm.run_fuzzy_embed [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v54-embed")
ARMS = {  # name: (env, ffn mult)
    "crisp_prior": ({"FUZZ_EMB": "crisp", "FUZZ_TOK_PRIOR": "1"}, "1.5"),
    "ftail_tanh": ({"FUZZ_EMB": "tail", "FUZZ_TOK_FORM": "tanh"}, "1.5"),
    "ftail_prior": ({"FUZZ_EMB": "tail", "FUZZ_TOK_PRIOR": "1"}, "1.5"),
    "fall_prior": ({"FUZZ_EMB": "all", "FUZZ_TOK_PRIOR": "1"}, "1.35"),
    "fall_tanh": ({"FUZZ_EMB": "all", "FUZZ_TOK_FORM": "tanh"}, "1.35"),
    "fall_res4": ({"FUZZ_EMB": "all", "FUZZ_TOK_RES": "4"}, "1.25"),
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
