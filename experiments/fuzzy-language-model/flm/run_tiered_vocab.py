"""GPU screen of larger vocabularies under the tiered embedding (`v40-tiervocab`): 30M characters, seeds 0-4, baseline = t768e16m225 on BPE1024
(`v30-tiered`, 98,280 params, mean 1.0965).  The tiered embedding makes a bigger vocabulary cheap (tail tokens cost E=16 numbers, not d=48),
which the plain-embedding vocabulary ladder (`v7-vocab`: BPE1536 d40 -0.0086 at 30M, 2 seeds) could not exploit.
Frequencies come from the token stream itself (outputs/freq_bpe<V>.npz).  ``python -m flm.run_tiered_vocab [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v40-tiervocab")
ARMS = {  # name: (vocab, head tokens, tail width, ffn mult)
    "v1536h640e16m20": (1536, 640, 16, "2.0"),  # 99,280 params
    "v1536h512e16m225": (
        1536,
        512,
        16,
        "2.25",
    ),  # 98,280: iso-parameter with t768e16m225 on BPE1024
    "v2048h512e16m15": (2048, 512, 16, "1.5"),  # 97,184
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(ARMS)
    for seed in range(5):
        for name in arms:
            v, nh, e, m = ARMS[name]
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = m
            a[a.index("--tok") + 1] = f"bpe{v}"
            env = dict(
                os.environ,
                TIER_NH=str(nh),
                TIER_E=str(e),
                TIER_FREQ=f"outputs/freq_bpe{v}.npz",
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
