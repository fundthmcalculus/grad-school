"""END-TO-END fuzzy model (`v52-e2e`, PLAN_FUZZYLOCAL.md): every replacement together, including the fuzzy token tables: 30M characters, seeds 0-4, BPE3072 tiered recipe (baseline 1.0547).
Arms (30M characters, seeds 0-4, vs the BPE3072 tiered recipe 1.0547 and vs c_full2 1.1096): e2e_tail = c_full2 (no convolution; raw-input FFN rules, layer 0 over the last
2 tokens; TSK gain controllers; TSK gates; rule-gated low-rank TSK projections) + fuzzy tail token table (99,019 params); e2e_all = the same with the head tokens fuzzy too
(FFN rules trimmed to 46 / 31 to pay for the 48x48 prototype matrix).  ``python -m flm.run_fuzzy_e2e [arm ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v52-e2e")
CH = os.environ.get(
    "E2E_CHARS", "30"
)  # millions of characters: 30 = the five-seed screen, 300 = the two-seed confirmation
SEEDS = [int(x) for x in os.environ.get("E2E_SEEDS", "0,1,2,3,4").split(",")]
BASE = {
    "FUZZ_NORM": "fuzzy",
    "FUZZ_GATE": "1",
    "FUZZ_PROJ": "tskl",
    "FUZZ_FFN": "raw",
    "FUZZ_FFN_TAPS": "2",
}
ARMS = {  # name: (env, train-arg overrides, ffn mult)
    "c_full2": (
        {**BASE, "FUZZ_FFN_R": "52", "FUZZ_FFN_R0": "35"},
        {"--shortconv": "0"},
        "1.5",
    ),  # block layers only, crisp token tables
    "e2e_tail": (
        {**BASE, "FUZZ_FFN_R": "52", "FUZZ_FFN_R0": "35", "FUZZ_EMB": "tail"},
        {"--shortconv": "0"},
        "1.5",
    ),
    # 3-block all-fuzzy models (PLAN_FUZZYLOCAL.md): the third block is paid for with narrower width or fewer rules / lower projection ranks
    "b3_lean": (
        {
            **BASE,
            "FUZZ_FFN_R": "31",
            "FUZZ_FFN_R0": "21",
            "FUZZ_EMB": "tail",
            "FUZZ_PROJ_RANKS": "24,16",
        },
        {"--shortconv": "0", "--layers": "3", "--rules": "16"},
        "1.5",
    ),
    "b3_d40": (
        {
            **BASE,
            "FUZZ_FFN_R": "44",
            "FUZZ_FFN_R0": "30",
            "FUZZ_EMB": "tail",
            "FUZZ_PROJ_RANKS": "24,16",
        },
        {"--shortconv": "0", "--layers": "3", "--d": "40"},
        "1.5",
    ),
    "b3_smallemb": (
        {
            **BASE,
            "FUZZ_FFN_R": "42",
            "FUZZ_FFN_R0": "30",
            "FUZZ_EMB": "tail",
            "FUZZ_PROJ_RANKS": "24,16",
            "TIER_NH": "320",
            "TIER_E": "10",
        },
        {"--shortconv": "0", "--layers": "3", "--rules": "16"},
        "1.5",
    ),
    "e2e_all": (
        {**BASE, "FUZZ_FFN_R": "46", "FUZZ_FFN_R0": "31", "FUZZ_EMB": "all"},
        {"--shortconv": "0"},
        "1.5",
    ),
}
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    arms = sys.argv[1:] or list(ARMS)
    for seed in SEEDS:
        for name in arms:
            out = OUT / (
                f"{name}_s{seed}.json" if CH == "30" else f"{name}_{CH}M_s{seed}.json"
            )
            if out.exists():
                continue
            envx, over, mult = ARMS[name]
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--ffn-mult") + 1] = mult
            a[a.index("--chars") + 1] = f"{CH}e6"
            if CH != "30":
                for flag in ("--eval-every", "--eval-at-start"):
                    i = a.index(flag)
                    del a[i : i + (2 if flag == "--eval-every" else 1)]
                a += ["--eval-every", "5000", "--save-model"]
            a[a.index("--tok") + 1] = "bpe3072"
            for k, v in over.items():
                if k in a:
                    a[a.index(k) + 1] = v
                else:
                    a += [k, v]
            env = {
                **os.environ,
                "TIER_NH": "384",
                "TIER_E": "12",
                "TIER_FREQ": "outputs/freq_bpe3072.npz",
                "PYTHONPATH": ".",
                **envx,  # an arm may override the defaults above (b3_smallemb changes the embedding)
            }
            with open(OUT / "logs" / f"{name}_{CH}M_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_allfuzzy"] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
