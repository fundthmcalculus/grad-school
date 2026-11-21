"""GPU screens of fuzzy replacements for RMSNorm and the sigmoid gates (`v45-fuzzystage`, PLAN_FUZZYLOCAL.md stages 2 and 4): 30M characters,
seeds 0-4, on the BPE3072 tiered recipe (head 384, tail 12, FFN x1.5, convolution kept; baseline mean 1.0547).  ``python -m flm.run_fuzzy_stage [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v45-fuzzystage")
ARMS = {  # name: env overrides
    "fnorm": {"FUZZ_NORM": "fuzzy"},
    "scaleonly": {"FUZZ_NORM": "scale"},
    "fgate": {"FUZZ_GATE": "1"},
    "fnorm_fgate": {"FUZZ_NORM": "fuzzy", "FUZZ_GATE": "1"},
    "fgate12": {
        "FUZZ_GATE": "1",
        "FUZZ_GATE_R": "12",
        "MULT": "1.3",
    },  # stage 4b: twelve rules, FFN trimmed to stay under the cap
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(ARMS)
    for seed in range(5):
        for name in arms:
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = ARMS[name].get("MULT", "1.5")
            a[a.index("--tok") + 1] = "bpe3072"
            env = dict(
                os.environ,
                TIER_NH="384",
                TIER_E="12",
                TIER_FREQ="outputs/freq_bpe3072.npz",
                PYTHONPATH=".",
                **{k: v for k, v in ARMS[name].items() if k != "MULT"},
            )
            with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_allfuzzy"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
