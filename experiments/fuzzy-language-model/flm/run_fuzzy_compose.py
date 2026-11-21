"""GPU screens of COMPOSED fuzzy replacements (`v49-compose`, PLAN_FUZZYLOCAL.md): 30M characters, seeds 0-4, BPE3072 tiered recipe (baseline 1.0547).
Arms (each replacement was measured alone in stages 1-4):
  c_partial  convolution -> 4-token layer-0 rules (TapLinear antecedent) + RMSNorm -> TSK gain controller
  c_mid      c_partial + sigmoid gates -> 6-rule TSK + mixer q/k/v/out -> rule-gated low-rank TSK maps
  c_full     everything: layer-0 FFN = raw-input rules over the last 4 tokens (no projection, no convolution), other layers raw-input FFN,
             TSK gain controllers, 6-rule TSK gates, rule-gated low-rank TSK projections (the FFN slope is kept: it is a first-order consequent)
  c_full2    as c_full with the layer-0 rules over the last 2 tokens (35 rules)
``python -m flm.run_fuzzy_compose [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v49-compose")
ARMS = {  # name: (env, train-arg overrides, ffn mult)
    "c_partial": (
        {"FUZZ_NORM": "fuzzy"},
        {"--shortconv": "0", "--ante-taps": "4"},
        "1.45",
    ),
    "c_mid": (
        {"FUZZ_NORM": "fuzzy", "FUZZ_GATE": "1", "FUZZ_PROJ": "tskl"},
        {"--shortconv": "0", "--ante-taps": "4"},
        "1.45",
    ),
    "c_full": (
        {
            "FUZZ_NORM": "fuzzy",
            "FUZZ_GATE": "1",
            "FUZZ_PROJ": "tskl",
            "FUZZ_FFN": "raw",
            "FUZZ_FFN_R": "52",
            "FUZZ_FFN_TAPS": "4",
            "FUZZ_FFN_R0": "21",
        },
        {"--shortconv": "0"},
        "1.5",
    ),
    "c_full2": (
        {
            "FUZZ_NORM": "fuzzy",
            "FUZZ_GATE": "1",
            "FUZZ_PROJ": "tskl",
            "FUZZ_FFN": "raw",
            "FUZZ_FFN_R": "52",
            "FUZZ_FFN_TAPS": "2",
            "FUZZ_FFN_R0": "35",
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
