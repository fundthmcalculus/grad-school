"""Collect the goal-track run records into report/data.json for the HTML report.

python -m flm.report_data
"""

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"


def recs(sweep):
    return {f.stem: json.loads(f.read_text()) for f in (OUT / sweep).glob("*.json")}


def group(sweep, pat=r"_s\d+$"):
    g = defaultdict(list)
    for n, r in recs(sweep).items():
        if r["val_bpc"] == r["val_bpc"]:
            g[re.sub(pat, "", n)].append(r)
    return {
        k: {
            "bpc": st.mean(x["val_bpc"] for x in v),
            "sd": st.pstdev([x["val_bpc"] for x in v]) if len(v) > 1 else None,
            "n": len(v),
            "params": v[0]["params_total"],
            "seeds": [x["val_bpc"] for x in v],
        }
        for k, v in g.items()
    }


d = {}
# 30M-char scaling curves from the GPU qualification run (3 seeds)
sg = group("scaling-gpu", r"_s\d+$")
curves = defaultdict(list)
for k, v in sg.items():
    arm = k.split("_d")[0]
    curves[arm].append({"params": v["params"], "bpc": v["bpc"], "sd": v["sd"]})
d["scaling30"] = {a: sorted(p, key=lambda x: x["params"]) for a, p in curves.items()}
d["attrib"] = {k: v["bpc"] for k, v in group("v3-attrib", r"_lr\S+$").items() if False}
# attribution: best of the two lrs per cell, seed 0
att = {}
for n, r in recs("v3-attrib").items():
    m, f = n.split("_")[0].rsplit("-", 1)
    key = f"{m}|{f}"
    if r["val_bpc"] == r["val_bpc"] and (
        key not in att or r["val_bpc"] < att[key]["bpc"]
    ):
        att[key] = {"bpc": r["val_bpc"], "params": r["params_total"]}
d["attrib"] = att
d["ffn3"] = group("v3-ffn3")
d["mix"] = group("v3-mix")
d["heads"] = group("v3-heads")
d["fill"] = group("v3-fill")
d["final"] = group("v3-final")
d["ref"] = group("v3-ref")
d["lr"] = group("v3-lr", r"_s\d+$")
# lr 3e-2 cells already exist from the final / ref sweeps
d["lr_existing"] = {
    "D|3e-2": d["final"].get("D_30M"),
    "delta-mlp-d72-h8|3e-2": d["ref"].get("delta-mlp-d72-h8_30M"),
}
# upstream lrwidth (H10a): d=96 seed 0, tuned lr x1 / div3 / div10
lw = {}
for n, r in recs("lrwidth").items():
    arm, tag = n.split("_")[0], n.split("_")[1]
    lw.setdefault(arm, {})[tag] = {
        "bpc": r["val_bpc"],
        "lr": r["args"]["lr"],
        "device": r.get("device"),
    }
for arm in lw:
    f = OUT / "scaling" / f"{arm}_d96_L2_s0.json"
    if f.exists():
        r = json.loads(f.read_text())
        lw[arm]["x1"] = {
            "bpc": r["val_bpc"],
            "lr": r["args"]["lr"],
            "device": r.get("device"),
        }
d["lrwidth"] = lw
(OUT.parent / "report").mkdir(exist_ok=True)
(OUT.parent / "report" / "data.json").write_text(json.dumps(d, indent=1))
print("wrote report/data.json:", {k: len(v) for k, v in d.items()})
