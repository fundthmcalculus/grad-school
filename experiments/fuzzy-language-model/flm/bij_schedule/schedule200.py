"""200K parameters (owner: "a 200K parameter training set"): five arms; phase A lr sweep {0.0025, 0.005, 0.01, 0.04} seed 0; phase B seeds 1-2 at each arm's best rate. Merges its choices into chosen_lr.json."""

import json, subprocess, time, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASH = "C:/Program Files/Git/bin/bash.exe"
O = Path("outputs/bij10k")
OUT = O / "sizes"
ARMS = ["tied", "ident", "z", "c", "ci"]
LRS = ["0.0025", "0.005", "0.01", "0.04"]
SIZE = "200k"


def result(arm, lr, seed):
    for p in OUT.glob(f"{SIZE}_{arm}_lr{lr}_s{seed}.json"):
        return json.load(open(p))


def run(job):
    arm, lr, seed = job
    if result(arm, lr, seed) is None:
        r = subprocess.run(
            [BASH, str(O / "run_size.sh"), SIZE, arm, str(seed), lr, "30", str(OUT)],
            capture_output=True,
            text=True,
        )
        print(
            time.strftime("%H:%M"),
            SIZE,
            arm,
            lr,
            seed,
            r.stdout.strip().split(" ")[-4:] if r.stdout else r.stderr[-200:],
            flush=True,
        )


def phase(jobs, name):
    todo = [j for j in jobs if result(*j) is None]
    print(f"== {name}: {len(todo)} of {len(jobs)} to run", flush=True)
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "2"))) as ex:
        list(ex.map(run, todo))


phase([(a, lr, 0) for lr in LRS for a in ARMS], "200K phase A: lr sweep")
chosen = {a: min((result(a, lr, 0)["val_bpc"], lr) for lr in LRS)[1] for a in ARMS}
f = O / "chosen_lr.json"
d = json.load(open(f)) if f.exists() else {}
d.update({f"{SIZE}|{a}": chosen[a] for a in ARMS})
json.dump(d, open(f, "w"), indent=1)
print("chosen", chosen, flush=True)
phase([(a, chosen[a], s) for s in (1, 2) for a in ARMS], "200K phase B: seeds 1-2")
print("schedule200 done", flush=True)
