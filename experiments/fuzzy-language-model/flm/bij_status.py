"""Collect the invertible-meaning-layer runs into one row file for the status dashboard: ``python -m flm.bij_status <out.json>``.

One row per finished run: ``size`` (nominal parameters in thousands), ``arm`` (tied / ident / z / c, see ``flm/run_bijective.py``), ``seed``, ``params`` (measured), ``lr``, ``val`` (bits per character at 30M characters), ``wall`` (training seconds).
Sources: ``outputs/bij10k/final`` (10K), ``final20`` (20K) and ``sizes`` (5K, 40K, 80K; seed 0 first).
"""

import json
import re
import sys
from pathlib import Path

O = Path("outputs/bij10k")


def rows():
    out = []
    for d, size in ((O / "final", 10), (O / "final20", 20)):
        for p in sorted(d.glob("*.json")):
            m = re.match(r"(tied|ident|z|c)_lr[\d.e-]+_s(\d+)\.json", p.name)
            if m:
                out.append((size, m[1], int(m[2]), p))
    for p in sorted((O / "sizes").glob("*.json")):
        m = re.match(r"(\d+)k_(tied|ident|z|c)_lr[\d.e-]+_s(\d+)\.json", p.name)
        if m:
            out.append((int(m[1]), m[2], int(m[3]), p))
    res = []
    for size, arm, seed, p in out:
        j = json.loads(p.read_text())
        res.append(dict(size=size, arm=arm, seed=seed, params=j["params_total"], lr=j["args"]["lr"], val=round(j["val_bpc"], 5), wall=round(j["train_seconds"], 1)))
    return sorted(res, key=lambda r: (r["size"], r["arm"], r["seed"]))


if __name__ == "__main__":
    r = rows()
    Path(sys.argv[1]).write_text(json.dumps(r))
    print(len(r), "runs")
