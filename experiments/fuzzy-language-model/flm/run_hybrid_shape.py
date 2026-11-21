"""The hybrid recipe at the embedding shapes of `PLAN_SHAPE.md` section 20 (arms and fitted rule counts in report/hybrid_arms.json, made by ``python -m flm.hybrid_budget --table``).

``python -m flm.run_hybrid_shape [arm ...]``                30M characters, seeds 0-2, warm-up ``$HYB_WU`` (default 100) -> outputs/v71-hybshape/<arm>[_wu<N>]_s<seed>.json
``HYB_SCALE=300 python -m flm.run_hybrid_shape [arm ...]`` 300M characters, seeds 0-1, checkpoints saved -> outputs/v72-hybshape300/
``SMOKE=1`` shrinks every run to 0.4M characters (outputs/smoke-hybshape).  Needs the tokenizer streams and frequency file of the arm's vocabulary (outputs/freq_bpe<V>.npz).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from flm.run_tribble import argv

ROOT_DATA = Path(__file__).resolve().parents[3] / "data"
CORPUS = os.environ.get(
    "HYB_CORPUS", ""
)  # "simplestories": that corpus's tokenizer and frequency files (PLAN_ANALYSIS.md item 3)
FREQ = lambda v: f"outputs/freq_{CORPUS + '_' if CORPUS else ''}bpe{v}.npz"
# optional overrides of the training schedule (PLAN_ANALYSIS.md): learning rate, final-lr floor as a fraction of the peak (train.py's default is 0.1), characters in millions
OV = {k: os.environ.get("HYB_" + k.upper()) for k in ("lr", "floor", "chars")}
EXTRA = "".join(f"_{k}{v}" for k, v in OV.items() if v)


def cell(name, arm, seed, out_dir, smoke, big, wu):
    suffix = ("" if wu == "100" else f"_wu{wu}") + EXTRA
    out = out_dir / f"{name}{suffix}_s{seed}.json"
    a = argv("random", seed)
    a[a.index("--out") + 1] = str(out)
    a[a.index("--tok") + 1] = f"bpe{arm['V']}"
    a[a.index("--d") + 1] = str(arm["d"])
    a[a.index("--ffn-mult") + 1] = str(arm["ffn_mult"])
    a[a.index("--warmup") + 1] = wu
    if big:
        a[a.index("--chars") + 1] = "300e6"
        for flag in ("--eval-every", "--eval-at-start"):
            i = a.index(flag)
            del a[i : i + (2 if flag == "--eval-every" else 1)]
        a += ["--eval-every", "5000", "--save-model"]
    if OV["lr"]:
        a[a.index("--lr") + 1] = OV["lr"]
    if OV["floor"]:
        a += ["--lr-floor", OV["floor"]]
    if OV["chars"]:
        a[a.index("--chars") + 1] = f"{OV['chars']}e6"
    if smoke:
        a[a.index("--chars") + 1] = "0.4e6"
    env = {
        **os.environ,
        "TIER_NH": str(arm["NH"]),
        "TIER_E": str(arm["E"]),
        "TIER_FREQ": FREQ(arm["V"]),
        **({"GRAD_SCHOOL_DATA": str(ROOT_DATA / CORPUS)} if CORPUS else {}),
        "PYTHONPATH": ".",
    }
    with open(out_dir / "logs" / f"{name}{suffix}_s{seed}.log", "w") as f:
        r = subprocess.run(
            [sys.executable, "-m", "flm.run_tiered"] + a,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(name, suffix, seed, "exit", r.returncode, flush=True)


if __name__ == "__main__":
    smoke = os.environ.get("SMOKE") == "1"
    big = os.environ.get("HYB_SCALE") == "300"
    wu = os.environ.get("HYB_WU", "100")
    out_dir = Path(
        "outputs/smoke-hybshape"
        if smoke
        else (
            "outputs/v75-ssshape"
            if CORPUS
            else ("outputs/v72-hybshape300" if big else "outputs/v71-hybshape")
        )
    )
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    table = json.load(open("report/hybrid_arms.json"))
    names = sys.argv[1:] or list(table)
    seeds = [
        int(s) for s in os.environ.get("SEEDS", "0,1" if big else "0,1,2").split(",")
    ]
    for s in seeds:
        for n in names:
            suffix = ("" if wu == "100" else f"_wu{wu}") + EXTRA
            if not (out_dir / f"{n}{suffix}_s{s}.json").exists():
                cell(n, table[n], s, out_dir, smoke, big, wu)
