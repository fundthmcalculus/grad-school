"""Run one `flm.run_tiered_vocab` arm at a longer budget (`v43-tiervocab-arm`): ``python -m flm.run_tiervocab_arm <arm> <chars in M> [seed ...]``.

300M characters reads the default stream (600M characters encoded); 1000M needs the _1b stream (flm.prep_bpe_long) of that vocabulary.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tiered_vocab import ARMS
from flm.run_tribble import argv

OUT = Path("outputs/v43-tiervocab-arm")
if __name__ == "__main__":
    name, mchars = sys.argv[1], int(sys.argv[2])
    seeds = [int(s) for s in sys.argv[3:]] or [0, 1]
    v, nh, e, m = ARMS[name]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        out = OUT / f"{name}_{mchars}M_s{seed}.json"
        if out.exists():
            continue
        a = argv("random", seed)
        a[a.index("--out") + 1] = str(out)
        a[a.index("--chars") + 1] = f"{mchars}e6"
        a[a.index("--ffn-mult") + 1] = m
        a[a.index("--tok") + 1] = f"bpe{v}"
        for flag in ("--eval-every", "--eval-at-start"):
            i = a.index(flag)
            del a[i : i + (2 if flag == "--eval-every" else 1)]
        a += ["--eval-every", "5000", "--save-model"]
        env = dict(
            os.environ,
            TIER_NH=str(nh),
            TIER_E=str(e),
            TIER_FREQ=f"outputs/freq_bpe{v}.npz",
            PYTHONPATH=".",
        )
        if mchars > 300:
            env["FLM_TRAIN_SUFFIX"] = "_1b"
        with open(OUT / "logs" / f"{name}_{mchars}M_s{seed}.log", "w") as f:
            r = subprocess.run(
                [sys.executable, "-m", "flm.run_tiered"] + a,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        print(name, mchars, seed, "exit", r.returncode, flush=True)
