"""1B-character run of the BPE3072 tiered arm (`v42-tiervocab1b`, the _1b stream, about 1.3B characters encoded): head 384, tail width 12, FFN x1.5 (98,528 params), seeds 0-1.
References: BPE1024 tiered at 1B characters (`v35-tiered1b`) 0.96796 / 0.96129; BPE3072 tiered at 300M 0.95758 / 0.96015.  ``python -m flm.run_tiervocab_1b [seed ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v42-tiervocab1b")
V, NH, E, M = 3072, 384, 12, "1.5"
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in sys.argv[1:]] or [0, 1]:
        out = OUT / f"v{V}h{NH}e{E}_1000M_s{seed}.json"
        if out.exists():
            continue
        a = argv("random", seed)
        a[a.index("--out") + 1] = str(out)
        a[a.index("--chars") + 1] = "1000e6"
        a[a.index("--ffn-mult") + 1] = M
        a[a.index("--tok") + 1] = f"bpe{V}"
        for flag in ("--eval-every", "--eval-at-start"):
            i = a.index(flag)
            del a[i : i + (2 if flag == "--eval-every" else 1)]
        a += ["--eval-every", "5000", "--save-model"]
        env = dict(
            os.environ,
            TIER_NH=str(NH),
            TIER_E=str(E),
            TIER_FREQ=f"outputs/freq_bpe{V}.npz",
            PYTHONPATH=".",
            FLM_TRAIN_SUFFIX="_1b",
        )
        with open(OUT / "logs" / f"s{seed}.log", "w") as f:
            r = subprocess.run(
                [sys.executable, "-m", "flm.run_tiered"] + a,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        print(seed, "exit", r.returncode, flush=True)
