"""Fuzzy model D against its neural reference at 1K-10K parameters (`v54-small`, PLAN_FUZZYLOCAL.md): CPU, 30M characters, byte-level text.

Fills the fuzzy (blue) line of the 'Fuzzy LM Under 100K' artifact below its single 97.5K point, and the neural line below its 6K point.  Phase 1 screens two learning rates at seed 0 for every arm and size; phase 2 runs seeds 1-2 at each cell's better learning rate.

``python -m flm.run_small_cpu [workers]``
"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("outputs/v54-small")
LRS = ("3e-2", "6e-2")
# arm -> widths; parameter counts at two layers: fuzzy d5 1,399 / d20 11,256; neural d6 1,298 / d14 4,810 / d24 12,080
SIZES = {"fuzzy": (5, 8, 12, 18), "neural": (6, 10, 14, 22)}
DM = [
    "--mixer",
    "fuzzydelta",
    "--decay",
    "data",
    "--exp-norm",
    "mean",
    "--ffn-exp-norm",
    "mean",
]
FFN = ["--ffn", "tsk", "--ffn-order", "1", "--ffn-norm", "none"]


def argv(arm, d, lr, seed, out):
    if arm == "fuzzy":  # model D's recipe, heads and antecedent width shrunk with d
        h = 1 if d <= 8 else 2
        a = DM + FFN + ["--tsk-dim", str(min(16, d // 2)), "--heads", str(h)]
        a += ["--rules", "16", "--ffn-mult", "1.5", "--shortconv", "8"]
    else:  # the neural delta-rule + MLP reference at its CPU-tuned settings
        a = ["--mixer", "delta", "--decay", "data", "--ffn", "mlp", "--shortconv", "4"]
    a += ["--d", str(d), "--layers", "2", "--lr", lr, "--seed", str(seed)]
    return a + [
        "--chars",
        "30e6",
        "--device",
        "cpu",
        "--threads",
        "1",
        "--out",
        str(out),
    ]


def run(job):
    arm, d, lr, seed = job
    name = f"{arm}_d{d}_lr{lr}_s{seed}"
    out = OUT / f"{name}.json"
    if out.exists():
        return
    with open(OUT / "logs" / f"{name}.log", "w") as f:
        r = subprocess.run(
            [sys.executable, "-m", "flm.train"] + argv(arm, d, lr, seed, out),
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**__import__("os").environ, "PYTHONPATH": "."},
        )
    print(name, "exit", r.returncode, flush=True)


def best_lr(arm, d):
    got = {}
    for lr in LRS:
        p = OUT / f"{arm}_d{d}_lr{lr}_s0.json"
        if p.exists():
            got[lr] = json.loads(p.read_text())["val_bpc"]
    return min(got, key=got.get)


if __name__ == "__main__":
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cells = [(a, d) for a, ds in SIZES.items() for d in ds]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, [(a, d, lr, 0) for a, d in cells for lr in LRS]))
        list(ex.map(run, [(a, d, best_lr(a, d), s) for a, d in cells for s in (1, 2)]))
