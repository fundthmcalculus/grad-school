"""The whole schedule (owner: "build the entire schedule and rip it"), two parallel streams, every cell skipped if its result file exists.
Phase 1  lr sweep: sizes 10K, 40K, 100K x arms {tied, ident, z, c, ci} x lr {0.01, 0.02, 0.04, 0.08, 0.16}, seed 0, 30M characters.
Phase 2  sizes 20K and 70K at each arm's best swept lr (geometric neighbour rule, snapped to the grid), seed 0.
Phase 3  seeds 1 and 2 at every size and arm, at that arm's chosen lr (error bars).
Chosen lr per (size, arm) goes to outputs/bij10k/chosen_lr.json after phase 1 and 2.
"""
import json, math, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
BASH = "C:/Program Files/Git/bin/bash.exe"
O = Path("outputs/bij10k"); OUT = O / "sizes"; OUT.mkdir(exist_ok=True)
ARMS = ["tied", "ident", "z", "c", "ci"]
LRS = ["0.01", "0.05", "0.1"]  # the sweep (owner: fewer samples)
TRIED = ["0.01", "0.02", "0.04", "0.05", "0.08", "0.1", "0.16"]  # every rate with results, new or earlier
SWEPT = ["10k", "40k", "100k"]
ALL = ["10k", "20k", "40k", "70k", "100k"]
N = {"10k": 10, "20k": 20, "40k": 40, "70k": 70, "100k": 100}

def result(size, arm, lr, seed):
    for p in list(OUT.glob(f"{size}_{arm}_lr{lr}_s{seed}.json")):
        return json.load(open(p))
    if size in ("10k", "20k"):  # the earlier 10K / 20K full runs sit in final/ and final20/
        d = O / ("final" if size == "10k" else "final20")
        for p in d.glob(f"{arm}_lr{lr}_s{seed}.json"):
            return json.load(open(p))
    return None

def run(job):
    size, arm, lr, seed = job
    if result(size, arm, lr, seed) is not None:
        return
    r = subprocess.run([BASH, str(O / "run_size.sh"), size, arm, str(seed), lr, "30", str(OUT)], capture_output=True, text=True)
    print(time.strftime("%H:%M"), size, arm, lr, seed, r.stdout.strip().split(" ")[-4:] if r.stdout else r.stderr[-200:], flush=True)

def phase(jobs, name):
    todo = [j for j in jobs if result(*j) is None]
    print(f"== {name}: {len(todo)} of {len(jobs)} to run", flush=True)
    with ThreadPoolExecutor(int(__import__("os").environ.get("WORKERS", "1"))) as ex:
        list(ex.map(run, todo))

def best(size, arm):
    rows = [(result(size, arm, lr, 0)["val_bpc"], lr) for lr in TRIED if result(size, arm, lr, 0)]
    return min(rows)[1] if rows else None

def snap(x):
    return min(TRIED, key=lambda l: abs(math.log(float(l)) - math.log(x)))

chosen = {}
phase([(s, a, lr, 0) for s in SWEPT for a in ARMS for lr in LRS], "phase 1: lr sweep")
for s in SWEPT:
    for a in ARMS:
        chosen[f"{s}|{a}"] = best(s, a)
for a in ARMS:  # neighbour rule for the unswept sizes
    chosen[f"20k|{a}"] = snap(math.sqrt(float(chosen[f"10k|{a}"]) * float(chosen[f"40k|{a}"])))
    chosen[f"70k|{a}"] = chosen[f"100k|{a}"]
json.dump(chosen, open(O / "chosen_lr.json", "w"), indent=1)
print("chosen", chosen, flush=True)
phase([(s, a, chosen[f"{s}|{a}"], 0) for s in ("20k", "70k") for a in ARMS], "phase 2: unswept sizes")
phase([(s, a, chosen[f"{s}|{a}"], seed) for seed in (1, 2) for s in ALL for a in ARMS], "phase 3: seeds 1-2")
print("schedule done", flush=True)
