"""Collect the invertible-meaning-layer runs into row files for the status dashboard: ``python -m flm.bij_status <out_dir>/runs.json`` (also writes ``curves.json`` and ``sweep.json`` beside it).

``runs.json``: one row per finished run at its (size, arm)'s chosen lr (the lr with the lowest seed-0 validation among those run): ``size`` (nominal parameters in thousands), ``arm`` (tied / ident / z / c / ci, see ``flm/run_bijective.py``), ``seed``, ``params`` (measured), ``lr``, ``val`` (bits per character at 30M characters), ``wall`` (training seconds).
``sweep.json``: every seed-0 run at every lr (``size, arm, lr, val``).  ``curves.json``: validation at every evaluation of every ``runs.json`` run at seed 0 (``size, arm, seed, step, total, val``).
Sources: ``outputs/bij10k/final`` (10K), ``final20`` (20K) and ``sizes`` (every size).
"""

import json
import re
import sys
from pathlib import Path

O = Path("outputs/bij10k")
ARMS = "tied|ident|z|c|ci"


def found():
    """(size, arm, seed, lr, json path) of every finished run, de-duplicated."""
    seen, out = {}, []
    for d, size in ((O / "final", 10), (O / "final20", 20)):
        for p in sorted(d.glob("*.json")):
            m = re.match(rf"({ARMS})_lr([\d.e-]+)_s(\d+)\.json", p.name)
            if m:
                seen[(size, m[1], int(m[3]), float(m[2]))] = p
    for p in sorted((O / "sizes").glob("*.json")):
        m = re.match(rf"(\d+)k_({ARMS})_lr([\d.e-]+)_s(\d+)\.json", p.name)
        if m:
            seen[(int(m[1]), m[2], int(m[4]), float(m[3]))] = p
    return [(k[0], k[1], k[2], k[3], p) for k, p in seen.items()]


def chosen(runs):
    best = {}
    for size, arm, seed, lr, p in runs:
        if seed == 0:
            v = json.loads(p.read_text())["val_bpc"]
            if (size, arm) not in best or v < best[(size, arm)][0]:
                best[(size, arm)] = (v, lr)
    return {k: v[1] for k, v in best.items()}


def rows():
    runs = found()
    ch = chosen(runs)
    res = []
    for size, arm, seed, lr, p in runs:
        if ch.get((size, arm)) == lr:
            j = json.loads(p.read_text())
            res.append(dict(size=size, arm=arm, seed=seed, params=j["params_total"], lr=lr, val=round(j["val_bpc"], 5), wall=round(j["train_seconds"], 1)))
    return sorted(res, key=lambda r: (r["size"], r["arm"], r["seed"]))


def sweep():
    res = []
    for size, arm, seed, lr, p in found():
        if seed == 0:
            res.append(dict(size=size, arm=arm, lr=lr, val=round(json.loads(p.read_text())["val_bpc"], 5)))
    return sorted(res, key=lambda r: (r["size"], r["arm"], r["lr"]))


def curves():
    """Validation against step for every chosen-lr seed-0 run, from its log (one line per evaluation): ``size, arm, seed, step, total, val``."""
    runs = found()
    ch = chosen(runs)
    res = []
    for size, arm, seed, lr, p in runs:
        log = p.with_suffix(".log")
        if seed == 0 and ch.get((size, arm)) == lr and log.exists():
            for m in re.finditer(r"^step (\d+)/(\d+) train [\d.]+ val ([\d.]+)", log.read_text(), re.M):
                res.append(dict(size=size, arm=arm, seed=seed, step=int(m[1]), total=int(m[2]), val=float(m[3])))
    return res


if __name__ == "__main__":
    r = rows()
    Path(sys.argv[1]).write_text(json.dumps(r))
    c, w = curves(), sweep()
    Path(sys.argv[1]).with_name("curves.json").write_text(json.dumps(c))
    Path(sys.argv[1]).with_name("sweep.json").write_text(json.dumps(w))
    print(len(r), "runs,", len(c), "curve points,", len(w), "sweep points")
