"""GPU screens of fuzzified token tables and tokenizer (`v51-fuzzytokens`, PLAN_FUZZYLOCAL.md stage 5): 30M characters, seeds 0-4, BPE3072 tiered recipe (baseline 1.0547).
Arms: tok_dr05 / tok_dr15 (training stream segmented with BPE dropout 0.05 / 0.15: stochastic, graded segmentations; flm.prep_bpe_dropout),
emb_ftail (tail tokens as fuzzy sets over the 12 prototypes of the up-projection, defuzzified output scores), emb_fall (head tokens too, 48 prototypes;
FFN x1.35 to stay under the cap).  ``python -m flm.run_fuzzy_tokens [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v51-fuzzytokens")
ARMS = {  # name: (env, train-arg overrides, ffn mult)
    "tok_dr05": ({"FLM_TRAIN_SUFFIX": "_dr05"}, {}, "1.5"),
    "tok_dr15": ({"FLM_TRAIN_SUFFIX": "_dr15"}, {}, "1.5"),
    "emb_ftail": ({"FUZZ_EMB": "tail"}, {}, "1.5"),
    "emb_fall": ({"FUZZ_EMB": "all"}, {}, "1.35"),
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(ARMS)
    for seed in range(5):
        for name in arms:
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            envx, over, mult = ARMS[name]
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = mult
            a[a.index("--tok") + 1] = "bpe3072"
            for k, v in over.items():
                if k in a:
                    a[a.index(k) + 1] = v
                else:
                    a += [k, v]
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
