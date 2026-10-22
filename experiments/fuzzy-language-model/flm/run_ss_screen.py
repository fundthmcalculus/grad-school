"""Hybrid recipe and all-fuzzy model on SimpleStories (`PLAN_NEXTROUND.md` track B2/B3; data at GRAD_SCHOOL_DATA=<repo>/data/simplestories, BPE3072 and its frequency file from `flm.prep_simplestories_bpe`).

``python -m flm.run_ss_screen 30 [arm ...]``   30M characters, seeds 0-4 (`v67-ss30`)
``python -m flm.run_ss_screen 300 [arm ...]``  300M characters, seeds 0-1, checkpoints saved (`v68-ss300`)
Warm-up screen arms ``fz_wu300`` / ``fz_wu1000`` (all-fuzzy, lr 6e-2, warm-up 300 / 1000 at 30M).  Arms: ``hyb_lr6`` / ``hyb_lr3`` (the hybrid recipe, `flm.run_tiered`, warm-up 100, lr 6e-2 / 3e-2) and ``fz_lr6`` / ``fz_lr3`` (`e2e_tail`, `flm.run_allfuzzy`, warm-up 600 at 30M and 1000 at 300M).  The TinyStories-tuned
recipes are transferred, not re-tuned; both models get the same two learning rates.
"""

import os
import subprocess
import sys
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

DATA = str(Path(__file__).resolve().parents[3] / "data" / "simplestories")
ARMS = {
    "hyb_lr6": ("hyb", "6e-2"),
    "hyb_lr3": ("hyb", "3e-2"),
    "fz_lr6": ("fz", "6e-2"),
    "fz_lr3": ("fz", "3e-2"),
    "fz_wu300": ("fz", "6e-2", "300"),  # 30M warm-up screen (PLAN_NEXTROUND.md 3.4.5)
    "fz_wu1000": ("fz", "6e-2", "1000"),
}
if __name__ == "__main__":
    scale = sys.argv[1]
    arms = sys.argv[2:] or list(ARMS)
    big = scale == "300"
    out_dir = Path("outputs/v68-ss300" if big else "outputs/v67-ss30")
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    seeds = [
        int(s)
        for s in os.environ.get("SEEDS", "0,1" if big else "0,1,2,3,4").split(",")
    ]
    for seed in seeds:
        for name in arms:
            kind, lr, *wu = ARMS[name]
            out = out_dir / f"{name}_s{seed}.json"
            if out.exists():
                continue
            a = argv("random", seed)
            a[a.index("--out") + 1] = str(out)
            a[a.index("--tok") + 1] = "bpe3072"
            a[a.index("--lr") + 1] = lr
            envx = {}
            if kind == "fz":
                envx, over, mult = E2E["e2e_tail"]
                a[a.index("--ffn-mult") + 1] = mult
                a[a.index("--warmup") + 1] = wu[0] if wu else ("1000" if big else "600")
                for k, v in over.items():
                    a[a.index(k) + 1] = v
            else:
                a[a.index("--ffn-mult") + 1] = "1.5"
            if big:
                a[a.index("--chars") + 1] = "300e6"
                for flag in ("--eval-every", "--eval-at-start"):
                    i = a.index(flag)
                    del a[i : i + (2 if flag == "--eval-every" else 1)]
                a += ["--eval-every", "5000", "--save-model"]
            env = {
                **os.environ,
                "GRAD_SCHOOL_DATA": DATA,
                "TIER_NH": "384",
                "TIER_E": "12",
                "TIER_FREQ": "outputs/freq_simplestories_bpe3072.npz",
                "PYTHONPATH": ".",
                **envx,
            }
            mod = "flm.run_allfuzzy" if kind == "fz" else "flm.run_tiered"
            with open(out_dir / "logs" / f"{name}_s{seed}.log", "w") as f:
                r = subprocess.run(
                    [sys.executable, "-m", mod] + a,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            print(name, seed, "exit", r.returncode, flush=True)
