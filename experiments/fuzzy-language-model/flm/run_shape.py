"""Shape screens of the all-fuzzy model (`PLAN_SHAPE.md`; arms and fitted rule counts in report/shape_arms.json, made by ``python -m flm.shape_budget --table``).

``python -m flm.run_shape [arm ...]`` runs the arms (default: all) at 30M characters, lr 6e-2, warm-up 600, seeds ``$SEEDS`` (default 0,1,2), into ``outputs/v69-shape/<arm>_s<seed>.json``; seeds outer, arms inner, so a partial run is balanced.
An arm whose tokenizer stream or frequency file is not ready yet (the vocabulary arms, while ``flm.prep_shape_vocab`` runs) is skipped and retried on the next pass.  ``SMOKE=1`` shrinks every run to 0.4M characters (outputs/smoke-shape).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flm.run_fuzzy_e2e import ARMS as E2E
from flm.run_tribble import argv

DATA = Path(__file__).resolve().parents[3] / "data" / "tinystories"


def ready(v):
    return (
        (DATA / f"train_bpe{v}.bin").exists()
        and (DATA / f"valid_bpe{v}.bin").exists()
        and Path(f"outputs/freq_bpe{v}.npz").exists()
    )


def cell(name, arm, seed, out_dir, smoke):
    out = out_dir / f"{name}_s{seed}.json"
    envx, over, mult = E2E["e2e_tail"]
    a = argv("random", seed)
    a[a.index("--out") + 1] = str(out)
    a[a.index("--ffn-mult") + 1] = mult
    a[a.index("--tok") + 1] = f"bpe{arm['V']}"
    a[a.index("--warmup") + 1] = "600"
    a[a.index("--d") + 1] = str(arm["d"])
    if smoke:
        a[a.index("--chars") + 1] = "0.4e6"
    for k, v in over.items():
        a[a.index(k) + 1] = v
    env = {
        **os.environ,
        "TIER_NH": str(arm["NH"]),
        "TIER_E": str(arm["E"]),
        "TIER_FREQ": f"outputs/freq_bpe{arm['V']}.npz",
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
    with open(out_dir / "logs" / f"{name}_s{seed}.log", "w") as f:
        r = subprocess.run(
            [sys.executable, "-m", mod] + a,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(name, seed, "exit", r.returncode, flush=True)


if __name__ == "__main__":
    smoke = os.environ.get("SMOKE") == "1"
    out_dir = Path("outputs/smoke-shape" if smoke else "outputs/v69-shape")
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    table = json.load(open("report/shape_arms.json"))
    names = sys.argv[1:] or list(table)
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
    todo = [
        (n, s)
        for s in seeds
        for n in names
        if not (out_dir / f"{n}_s{s}.json").exists()
    ]
    waited = 0
    while todo:
        ran = False
        for n, s in list(todo):
            if not ready(table[n]["V"]):
                continue
            cell(n, table[n], s, out_dir, smoke)
            todo.remove((n, s))
            ran = True
        if todo and not ran:
            if waited > 7200:
                print("giving up on", sorted({n for n, _ in todo}), flush=True)
                break
            time.sleep(60)
            waited += 60
