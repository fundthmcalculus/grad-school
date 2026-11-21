"""GPU screens of data-driven initialisation of the fuzzy rule bases (`v50-fuzzyinit`, PLAN_FUZZYLOCAL.md; FUZZ_INIT=data places centres at sampled first-batch inputs and widths at the data spread): 30M characters, seeds 0-4, BPE3072 tiered recipe (baseline 1.0547).
Arms: ffn_raw1_di (stage 3a raw-input FFN), fgate_di (stage 4 gates), proj_tskl_di (stage 3b projections), c_full2_di (the whole c_full2 composition),
each compared with its un-initialised twin from v47 / v45 / v48 / v49.  ``python -m flm.run_fuzzy_init [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v50-fuzzyinit")
ARMS = {  # name: (env, train-arg overrides, ffn mult)
    "ffn_raw1_di": (
        {"FUZZ_FFN": "raw", "FUZZ_FFN_R": "52", "FUZZ_INIT": "data"},
        {},
        "1.5",
    ),
    "fgate_di": ({"FUZZ_GATE": "1", "FUZZ_INIT": "data"}, {}, "1.5"),
    "proj_tskl_di": ({"FUZZ_PROJ": "tskl", "FUZZ_INIT": "data"}, {}, "1.5"),
    "c_full2_di": (
        {
            "FUZZ_NORM": "fuzzy",
            "FUZZ_GATE": "1",
            "FUZZ_PROJ": "tskl",
            "FUZZ_FFN": "raw",
            "FUZZ_FFN_R": "52",
            "FUZZ_FFN_TAPS": "2",
            "FUZZ_FFN_R0": "35",
            "FUZZ_INIT": "data",
        },
        {"--shortconv": "0"},
        "1.5",
    ),
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(ARMS)
    for seed in range(5):
        for name in arms:
            out = OUT / f"{name}_s{seed}.json"
            if out.exists():
                continue
            envx, over, mult = ARMS[name]
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = mult
            a[a.index("--tok") + 1] = "bpe3072"
            for k, v in over.items():
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
                **envx,
            )
            with open(OUT / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_allfuzzy"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
