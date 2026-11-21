"""GPU screen of replacing the causal convolution by fuzzy local rules (`v44-fuzzylocal`, PLAN_FUZZYLOCAL.md): 30M characters, seeds 0-4.

Base: the BPE3072 tiered recipe (head 384, tail 12, FFN x1.5, 98,528 params, mean 1.0547 at 30M, shortconv 8).  Arms drop the convolution
(--shortconv 0); the layer-0 TSK feed-forward antecedent then reads the last k tokens (--ante-taps k): "IF token t-1 is A AND token t is B ...".
``python -m flm.run_fuzzy_local [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v44-fuzzylocal")
ARMS = {  # name: (extra flags, ffn mult)
    "noconv": ({"--shortconv": "0"}, "1.5"),  # 97,760 params
    "noconv_t3": ({"--shortconv": "0", "--ante-taps": "3"}, "1.5"),  # 99,296
    "noconv_t4": ({"--shortconv": "0", "--ante-taps": "4"}, "1.45"),
    "noconv_t8": ({"--shortconv": "0", "--ante-taps": "8"}, "1.2"),
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(ARMS)
    for seed in range(5):
        for name in arms:
            extra, m = ARMS[name]
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = m
            a[a.index("--tok") + 1] = "bpe3072"
            for k, v in extra.items():
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
            )
            with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_tiered"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
