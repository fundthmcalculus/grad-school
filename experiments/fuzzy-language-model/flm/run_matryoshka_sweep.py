"""GPU screen of the Matryoshka loss (`v34-mrl`): 30M characters, seeds 0-4; baseline = v22-tribble random (96,416 params)."""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v34-mrl")
CHARS = os.environ.get("MRL_CHARS", "30e6")
ARMS = {  # name: (prefix widths, weights)
    "m1": ("8,16,32,48", "1,1,1,1"),
    "m2": ("8,16,32,48", "0.25,0.25,0.25,1"),
    "m6": (
        "8,16,32,48",
        "0.25,0.25,0.25,1",
    ),  # m2 + nested dropout on the input embedding (MRL_IN=1)
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = os.environ.get("MRL_ARMS", ",".join(ARMS)).split(",")
    seeds = [int(s) for s in os.environ.get("MRL_SEEDS", "0,1,2,3,4").split(",")]
    for seed in seeds:
        for name, (ks, w) in ARMS.items():
            if name not in arms:
                continue
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--chars") + 1] = CHARS
            a += ["--save-model"]
            env = dict(
                os.environ,
                MRL_KS=ks,
                MRL_W=w,
                MRL_IN="1" if name == "m6" else "0",
                PYTHONPATH=".",
            )
            with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_matryoshka"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
