"""CPU screen of hybrid consequent-first init (PLAN_BACKWARD.md, `v26-hybrid`): ``python -m flm.run_hybrid_cpu <cf|hyb4|hyb12|hyb24> <seed>``.
4M characters, 4 threads, same config as v19-pilot (paired with outputs/v19-pilot/base_s<seed>.json).
"""

import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

OUT = Path("outputs/v26-hybrid")

if __name__ == "__main__":
    v, seed = sys.argv[1], int(sys.argv[2])
    out = OUT / f"{v}_s{seed}.json"
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    a = argv("cf", seed)
    a[a.index("--out") + 1] = str(out)
    a[a.index("--chars") + 1] = "4e6"
    a[a.index("--device") + 1] = "cpu"
    for flag in ("--eval-every", "--eval-at-start"):
        i = a.index(flag)
        del a[i : i + (2 if flag == "--eval-every" else 1)]
    a += ["--threads", "4", "--save-model"]
    with open(OUT / "logs" / f"{v}_s{seed}.log", "w") as f:
        r = subprocess.run(
            [sys.executable, "-m", "flm.run_conseq", v] + a,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print(v, seed, "exit", r.returncode)
