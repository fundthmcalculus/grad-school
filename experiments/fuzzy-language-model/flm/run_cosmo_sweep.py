"""GPU screen on the Cosmopedia corpus (`v36-cosmo`, PLAN_CORPUS.md C1/C2): 30M characters, seeds 0-4, BPE1024 retrained on the corpus.
Arms: ref (96,416 params) and tiered t768e16m225 (98,280).  Reads GRAD_SCHOOL_DATA=<repo>/data/cosmopedia (see flm.prep_cosmopedia).
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v36-cosmo")
CHARS = os.environ.get("COSMO_CHARS", "30e6")
DATA = str(Path("../../data/cosmopedia").resolve())
ARMS = (
    "ref",
    "tiered",
)  # "mixed": ref architecture on data/mixed (flm.prep_mixed), saved for flm.eval_other
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in os.environ.get("COSMO_SEEDS", "0,1,2,3,4").split(",")]
    arms = os.environ.get("COSMO_ARMS", ",".join(ARMS)).split(",")
    for seed in seeds:
        for name in arms:
            tag = f"{name}_{int(float(CHARS) / 1e6)}M_s{seed}"
            out = OUT / f"{tag}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--chars") + 1] = CHARS
            env = dict(os.environ, GRAD_SCHOOL_DATA=DATA, PYTHONPATH=".")
            if name == "mixed":
                env["GRAD_SCHOOL_DATA"] = str(Path("../../data/mixed").resolve())
                a += ["--save-model"]
            if name == "tiered":
                a[a.index("--ffn-mult") + 1] = "2.25"
                env.update(
                    TIER_NH="768",
                    TIER_E="16",
                    TIER_FREQ="outputs/cosmo_freq_bpe1024.npz",
                )
                mod = "flm.run_tiered"
            else:
                mod = "flm.train"
            if float(CHARS) > 100e6:
                for flag in ("--eval-every", "--eval-at-start"):
                    i = a.index(flag)
                    del a[i : i + (2 if flag == "--eval-every" else 1)]
                a += ["--eval-every", "5000", "--save-model"]
            with open(OUT / "logs" / f"{tag}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", mod] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(tag, "exit", r.returncode, flush=True)
