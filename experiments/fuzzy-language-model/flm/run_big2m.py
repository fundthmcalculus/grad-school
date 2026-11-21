"""~2.2M-parameter model (d256 x 4 layers, `flm.grids_v9.SIZES["d256-L4"]`, 2,223,936 params; user: "raise the model size a bit, say 2M") on TinyStories
and Cosmopedia (`v37-big2m`).  ``python -m flm.run_big2m <corpus: ts|cosmo> <chars>``; recipe of flm.run_ladder (lr 2e-2, warm-up 300, seed 0).
TinyStories above 300M chars uses the _1b stream; Cosmopedia reads GRAD_SCHOOL_DATA=<repo>/data/cosmopedia and the first ``chars`` of its stream.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.grids_v9 import SIZES

OUT = Path("outputs/v37-big2m")
if __name__ == "__main__":
    corpus, chars = sys.argv[1], float(sys.argv[2])
    d, L, h, r = SIZES["d256-L4"]
    tag = f"{corpus}_d256-L4_{int(chars / 1e6)}M_s0"
    a = "--mixer fuzzydelta --decay data --exp-norm mean --ffn-exp-norm mean".split()
    a += ["--heads", str(h), "--ffn-mult", "1.5", "--ffn", "tsk", "--ffn-order", "1"]
    a += ["--ffn-norm", "none", "--tsk-dim", "16", "--tok", "bpe1024"]
    a += ["--warmup", "300", "--ctx", "512", "--batch", "16", "--d", str(d)]
    a += ["--layers", str(L), "--rules", str(r), "--seed", "0", "--chars", f"{chars:g}"]
    a += ["--lr", "2e-2", "--shortconv", "8", "--device", "cuda", "--save-model"]
    a += ["--eval-every", "2000", "--out", str(OUT / f"{tag}.json")]
    env = dict(os.environ, PYTHONPATH=".")
    if corpus == "cosmo":
        env["GRAD_SCHOOL_DATA"] = str(Path("../../data/cosmopedia").resolve())
    elif chars > 300e6:
        env["FLM_TRAIN_SUFFIX"] = "_1b"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{tag}.log", "w") as f:
        p = subprocess.run(
            [sys.executable, "-m", "flm.train"] + a,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(tag, "exit", p.returncode, flush=True)
