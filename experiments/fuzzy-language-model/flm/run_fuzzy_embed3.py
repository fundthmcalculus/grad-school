"""GPU screens of stacked head repairs (`v57-embed3`, PLAN_FUZZYLOCAL.md stage 6c): 30M characters, seeds 0-4, the stage-5 setting, FFN x1.35.  Signed memberships (`fall_tanh`
1.1059) and the log-unigram prior (`fall_tanh_prior` 1.0934) are the two repairs that work; the soft tier gate (`fall_tier` 1.1579, 49 parameters) works on a different thing, so it is
stacked on each: `fall_tanh_tier`, `fall_tanh_prior_tier`.  ``python -m flm.run_fuzzy_embed3 [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v57-embed3")
ARMS = {  # name: (env, ffn mult)
    "fall_tanh_prior_tier": (
        {
            "FUZZ_EMB": "all",
            "FUZZ_TOK_FORM": "tanh",
            "FUZZ_TOK_PRIOR": "1",
            "FUZZ_TOK_TIER": "1",
        },
        "1.35",
    ),
    "fall_tanh_tier": (
        {"FUZZ_EMB": "all", "FUZZ_TOK_FORM": "tanh", "FUZZ_TOK_TIER": "1"},
        "1.35",
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
