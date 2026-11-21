"""300M-character confirmation of the fuzzy local layer (`v46-fuzzylocal300m`): BPE3072 tiered recipe, convolution removed, 4-tap layer-0 antecedent,
FFN x1.45, seeds 0-1.  References (same recipe WITH the convolution, `v41-tiervocab300m`): 0.95758 / 0.96015.  ``python -m flm.run_fuzzy_local_300m [seed ...]``.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v46-fuzzylocal300m")
if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in sys.argv[1:]] or [0, 1]:
        out = OUT / f"noconv_t4_300M_s{seed}.json"
        if out.exists():
            continue
        a = argv("random", seed)
        a[a.index("--out") + 1] = str(out)
        a[a.index("--chars") + 1] = "300e6"
        a[a.index("--ffn-mult") + 1] = "1.45"
        a[a.index("--tok") + 1] = "bpe3072"
        a[a.index("--shortconv") + 1] = "0"
        a += ["--ante-taps", "4"]
        for flag in ("--eval-every", "--eval-at-start"):
            i = a.index(flag)
            del a[i : i + (2 if flag == "--eval-every" else 1)]
        a += ["--eval-every", "5000", "--save-model"]
        env = dict(
            os.environ,
            TIER_NH="384",
            TIER_E="12",
            TIER_FREQ="outputs/freq_bpe3072.npz",
            PYTHONPATH=".",
        )
        with open(OUT / "logs" / f"s{seed}.log", "w") as f:
            r = subprocess.run(
                [sys.executable, "-m", "flm.run_tiered"] + a,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        print(seed, "exit", r.returncode, flush=True)
