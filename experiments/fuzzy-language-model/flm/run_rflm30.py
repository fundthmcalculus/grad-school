"""GPU 30M-character test of the RFLM variants (`v39-rflm30`, PLAN_NEXT.md item 3), seeds 0-4, paired with `v22-tribble`'s `random` cells.

The variant mixers live in archive/flm/rflm_variants.py (not moved back into flm/): this driver loads that file by path, which
registers them into flm.models.MIXERS, then calls flm.train.main in-process.  ``python -m flm.run_rflm30 <variant> <seed>`` runs one
cell; ``python -m flm.run_rflm30`` runs every missing cell, one at a time.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv as base_argv

OUT = Path("outputs/v39-rflm30")
VARIANTS = {
    "fixed": ["--mixer", "fuzzydelta_static", "--decay", "fixed"],
    "cpread": ["--mixer", "fuzzydelta_cpread"],
    "cpboth": ["--mixer", "fuzzydelta_cpboth"],
}


def argv(tag: str, seed: int) -> list[str]:
    a = base_argv("random", seed)
    a[a.index("--out") + 1] = str(OUT / f"{tag}_s{seed}.json")
    over = VARIANTS[tag]
    for i in range(0, len(over), 2):
        a[a.index(over[i]) + 1] = over[i + 1]
    return a


def register():
    p = Path(__file__).resolve().parent.parent / "archive" / "flm" / "rflm_variants.py"
    spec = importlib.util.spec_from_file_location("rflm_variants", p)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        import flm.train as T

        register()
        T.main(argv(sys.argv[1], int(sys.argv[2])))
        sys.exit(0)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for seed in range(5):
        for tag in VARIANTS:
            if (OUT / f"{tag}_s{seed}.json").exists():
                continue
            with open(OUT / "logs" / f"{tag}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", "flm.run_rflm30", tag, str(seed)],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={**__import__("os").environ, "PYTHONPATH": "."},
                )
            print(tag, seed, "exit", r.returncode, flush=True)
