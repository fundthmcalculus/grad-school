"""GPU screens of head-logit repairs from the literature review (`v56-embed2`, PLAN_FUZZYLOCAL.md stage 6b): 30M characters, seeds 0-4, the stage-5 setting
(reference 1.0547; `emb_fall` +0.130, the fuzzy head table with convex memberships), FFN x1.35.  Arms add to `emb_fall`: a learned per-token output bias initialised at the
log unigram (`fall_bias`, +384 parameters), a learned output gain and per-token amplitude (`fall_gain`, +385), a soft tier gate P(tier | h) * P(token | tier) (`fall_tier`, +49),
and combinations; `fall_tanh_prior` / `fall_tanh_bias` stack the signed-membership head of `v54-embed` (`fall_tanh`, +0.079 over `emb_fall`) with the log-unigram prior (`fall_prior`, +0.067) or the learned per-token bias.  ``python -m flm.run_fuzzy_embed2 [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v56-embed2")
ARMS = {  # name: (env, ffn mult)
    "fall_tanh_prior": (
        {"FUZZ_EMB": "all", "FUZZ_TOK_FORM": "tanh", "FUZZ_TOK_PRIOR": "1"},
        "1.35",
    ),
    "fall_tanh_bias": (
        {"FUZZ_EMB": "all", "FUZZ_TOK_FORM": "tanh", "FUZZ_TOK_BIAS": "head"},
        "1.35",
    ),
    "fall_bias": ({"FUZZ_EMB": "all", "FUZZ_TOK_BIAS": "head"}, "1.35"),
    "fall_gain": ({"FUZZ_EMB": "all", "FUZZ_TOK_GAIN": "1"}, "1.35"),
    "fall_tier": ({"FUZZ_EMB": "all", "FUZZ_TOK_TIER": "1"}, "1.35"),
    "fall_bias_gain": (
        {"FUZZ_EMB": "all", "FUZZ_TOK_BIAS": "head", "FUZZ_TOK_GAIN": "1"},
        "1.35",
    ),
    "fall_all3": (
        {
            "FUZZ_EMB": "all",
            "FUZZ_TOK_BIAS": "head",
            "FUZZ_TOK_GAIN": "1",
            "FUZZ_TOK_TIER": "1",
        },
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
