"""The best all-fuzzy model at 1B characters (`v63-e2e1b`, PLAN_FUZZYLOCAL.md): `e2e_tail` (every block component fuzzy, fuzzy tail token table, crisp head), lr 6e-2, warm-up 1000 (the 300M setting), the BPE3072 `_1b` stream, checkpoint saved.
References: the hybrid recipe at 1B (`v42-tiervocab1b`) 0.94274 / 0.94777, mean 0.9453; this model at 300M 0.98701 / 0.99157, mean 0.98929.
``python -m flm.run_fuzzy_e2e1b [seed ...]``
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

OUT = Path("outputs/v63-e2e1b")
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in sys.argv[1:]] or [0]:
        out = OUT / f"e2e_tail_1000M_s{seed}.json"
        if out.exists():
            continue
        envx, over, mult = E2E["e2e_tail"]
        a = argv("random", seed)
        a[a.index("--out") + 1] = str(out)
        a[a.index("--ffn-mult") + 1] = mult
        a[a.index("--tok") + 1] = "bpe3072"
        a[a.index("--warmup") + 1] = "1000"
        a[a.index("--chars") + 1] = "1000e6"
        for flag in ("--eval-every", "--eval-at-start"):
            k = a.index(flag)
            del a[k : k + (2 if flag == "--eval-every" else 1)]
        a += ["--eval-every", "5000", "--save-model"]
        for k, v in over.items():
            a[a.index(k) + 1] = v
        env = {
            **os.environ,
            "TIER_NH": "384",
            "TIER_E": "12",
            "TIER_FREQ": "outputs/freq_bpe3072.npz",
            "PYTHONPATH": ".",
            "FLM_TRAIN_SUFFIX": "_1b",
            **envx,
        }
        with open(OUT / "logs" / f"s{seed}.log", "w") as f:
            r = subprocess.run(
                [sys.executable, "-m", "flm.run_allfuzzy"] + a,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        print(seed, "exit", r.returncode, flush=True)
