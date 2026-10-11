"""Size ladder between 96K and 1.32M (`v32-ladder`; user: "Raise the cap and keep moving"): which size first reaches the coherence of the
1.32M x 1B model?  ``python -m flm.run_ladder <name> <chars>`` ; names from flm.grids_v9.SIZES; lr 2e-2 (the d192 optimum), warm-up 300, seed 0,
stream: _1b when chars > 300e6 else the default stream."""

import os
import subprocess
import sys
from pathlib import Path

from flm.grids_v9 import SIZES

OUT = Path("outputs/v32-ladder")
if __name__ == "__main__":
    name, chars = sys.argv[1], float(sys.argv[2])
    d, L, h, r = SIZES[name]
    tag = f"{name}_{int(chars / 1e6)}M_s0"
    a = [
        "--mixer",
        "fuzzydelta",
        "--decay",
        "data",
        "--exp-norm",
        "mean",
        "--ffn-exp-norm",
        "mean",
        "--heads",
        str(h),
    ]
    a += [
        "--ffn-mult",
        "1.5",
        "--ffn",
        "tsk",
        "--ffn-order",
        "1",
        "--ffn-norm",
        "none",
        "--tsk-dim",
        "16",
        "--tok",
        "bpe1024",
    ]
    a += [
        "--warmup",
        "300",
        "--ctx",
        "512",
        "--batch",
        "16",
        "--d",
        str(d),
        "--layers",
        str(L),
        "--rules",
        str(r),
        "--seed",
        "0",
    ]
    a += [
        "--chars",
        f"{chars:g}",
        "--lr",
        "2e-2",
        "--shortconv",
        "8",
        "--device",
        "cuda",
        "--save-model",
        "--eval-every",
        "2000",
    ]
    a += ["--out", str(OUT / f"{tag}.json")]
    env = dict(os.environ, PYTHONPATH=".")
    if chars > 300e6:
        env["FLM_TRAIN_SUFFIX"] = "_1b"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{tag}.log", "w") as f:
        r = subprocess.run(
            [sys.executable, "-m", "flm.train"] + a,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(tag, "exit", r.returncode, flush=True)
