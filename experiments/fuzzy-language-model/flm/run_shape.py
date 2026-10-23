"""Shape screens of the all-fuzzy model (`PLAN_SHAPE.md`; arms and fitted rule counts in report/shape_arms.json, made by ``python -m flm.shape_budget --table``).

``python -m flm.run_shape [arm ...]`` runs the arms (default: all) at 30M characters, lr 6e-2, warm-up 600, seeds ``$SEEDS`` (default 0,1,2), into ``outputs/v69-shape/<arm>_s<seed>.json``; seeds outer, arms inner, so a partial run is balanced.
An arm whose tokenizer stream or frequency file is not ready yet (the vocabulary arms, while ``flm.prep_shape_vocab`` runs) is skipped and retried on the next pass.  ``SMOKE=1`` shrinks every run to 0.4M characters (outputs/smoke-shape).  ``SHAPE_SCALE=300`` runs the 300M-character confirmation (warm-up 1000, seeds 0-1, checkpoints saved) into ``outputs/v70-shape300``.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

ROOT_DATA = Path(__file__).resolve().parents[3] / "data"
CORPUS = os.environ.get(
    "SHAPE_CORPUS", ""
)  # "simplestories": the token streams and frequency files of that corpus (PLAN_ANALYSIS.md item 3)
DATA = ROOT_DATA / (CORPUS or "tinystories") / ("tinystories" if CORPUS else "")
FREQ = lambda v: f"outputs/freq_{CORPUS + '_' if CORPUS else ''}bpe{v}.npz"
# optional overrides of the training schedule (PLAN_ANALYSIS.md): learning rate, final-lr floor as a fraction of the peak (train.py's default is 0.1), characters in millions
OV = {k: os.environ.get("SHAPE_" + k.upper()) for k in ("lr", "floor", "chars")}
SUFFIX = "".join(f"_{k}{v}" for k, v in OV.items() if v)
# PLAN_ROUND3.md: extra train.py flags for one arm of a screen (SHAPE_EXTRA="--batch 16"), named by SHAPE_TAG in the output files
EXTRA = os.environ.get("SHAPE_EXTRA", "").split()
if os.environ.get("SHAPE_TAG"):
    SUFFIX += "_" + os.environ["SHAPE_TAG"]


def ready(v):
    return (
        (DATA / f"train_bpe{v}.bin").exists()
        and (DATA / f"valid_bpe{v}.bin").exists()
        and Path(FREQ(v)).exists()
    )


def cell(name, arm, seed, out_dir, smoke, big=False):
    out = out_dir / f"{name}{SUFFIX}_s{seed}.json"
    envx, over, mult = E2E["e2e_tail"]
    a = argv("random", seed)
    a[a.index("--out") + 1] = str(out)
    a[a.index("--ffn-mult") + 1] = mult
    a[a.index("--tok") + 1] = f"bpe{arm['V']}"
    a[a.index("--warmup") + 1] = "1000" if big else "600"
    a[a.index("--d") + 1] = str(arm["d"])
    if (
        big
    ):  # the 300M confirmation: warm-up 1000, evaluation every 5000 steps, checkpoints saved
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
    if arm.get("mixrules"):  # mixer rules per head (random_shape.py)
        a[a.index("--rules") + 1] = str(arm["mixrules"])
    if os.environ.get(
        "SHAPE_DEVICE"
    ):  # cpu: a smoke run on a machine whose GPU is busy
        a[a.index("--device") + 1] = os.environ["SHAPE_DEVICE"]
    a += EXTRA
    if smoke:
        a[a.index("--chars") + 1] = "0.4e6"
    for k, v in over.items():
        a[a.index(k) + 1] = v
    env = {
        **os.environ,
        "TIER_NH": str(arm["NH"]),
        "TIER_E": str(arm["E"]),
        "TIER_FREQ": FREQ(arm["V"]),
        **({"GRAD_SCHOOL_DATA": str(ROOT_DATA / CORPUS)} if CORPUS else {}),
        "PYTHONPATH": ".",
        **envx,
        "FUZZ_FFN_R": str(arm["rules"]),
        "FUZZ_FFN_R0": str(arm["rules0"]),
    }
    mod = "flm.run_allfuzzy"
    if arm.get("hr"):
        env["FUZZ_TOK_HEADRANK"] = str(arm["hr"])
    if arm.get("mid"):
        env["FUZZ_TOK_MID"] = ",".join(str(x) for x in arm["mid"])
    if arm.get("ortho"):
        env["FUZZ_ORTHO"], mod = str(arm["ortho"]), "flm.run_embed_rev"
    if arm.get("ortho_scaled"):
        env["FUZZ_ORTHO_SCALED"] = "1"
    if arm.get("cycle"):
        env["FUZZ_CYCLE"], mod = str(arm["cycle"]), "flm.run_embed_rev"
    with open(out_dir / "logs" / f"{name}{SUFFIX}_s{seed}.log", "w") as f:
        r = subprocess.run(
            [sys.executable, "-m", mod] + a,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(name, seed, "exit", r.returncode, flush=True)


if __name__ == "__main__":
    smoke = os.environ.get("SMOKE") == "1"
    big = os.environ.get("SHAPE_SCALE") == "300"
    out_dir = Path(
        "outputs/smoke-shape"
        if smoke
        else (
            "outputs/v75-ssshape"
            if CORPUS
            else ("outputs/v70-shape300" if big else "outputs/v69-shape")
        )
    )
    if os.environ.get("SHAPE_OUT"):  # random_shape.py: its own results directory
        out_dir = Path(os.environ["SHAPE_OUT"])
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    table = json.load(open(os.environ.get("SHAPE_ARMS", "report/shape_arms.json")))
    names = sys.argv[1:] or list(table)
    seeds = [
        int(s) for s in os.environ.get("SEEDS", "0,1" if big else "0,1,2").split(",")
    ]
    todo = [
        (n, s)
        for s in seeds
        for n in names
        if not (out_dir / f"{n}{SUFFIX}_s{s}.json").exists()
    ]
    waited = 0
    while todo:
        ran = False
        for n, s in list(todo):
            if not ready(table[n]["V"]):
                continue
            cell(n, table[n], s, out_dir, smoke, big)
            todo.remove((n, s))
            ran = True
        if todo and not ran:
            if waited > 7200:
                print("giving up on", sorted({n for n, _ in todo}), flush=True)
                break
            time.sleep(60)
            waited += 60
