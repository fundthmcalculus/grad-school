"""Collect the 14M extension runs into four JSON row files for the status dashboard: ``python -m flm.ext_status <out_dir>``.

Every number comes from a file the runs write (result JSON, log, checkpoint folder); the registered probabilities in ``PREDICTIONS`` are the ones typed into ``PLAN_M14_EXTEND.md`` before the runs.
Files: ``runs.json`` (one row per run), ``curves.json`` (``run, kind, chars_m, value``: ``val`` in bits per character, ``train`` in nats per token), ``checkpoints.json`` (one row per file on disk), ``predictions.json`` (registered claims scored against the results).
x is cumulative training characters in millions: the base run covers 0-300, each extension 300-600.
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("outputs")
STEPS = 9043
CHARS_PER_RUN = 300.0
RUNS = [  # id, label, directory, learning rate, offset (millions of characters before this run starts)
    ("base", "300M, lr 1e-2 (fp32)", OUT / "v82-m14-300", 1e-2, 0.0),
    ("ext1", "+300M, lr 1e-3 (bf16)", OUT / "v90-m14-ext", 1e-3, 300.0),
    ("ext3", "+300M, lr 3e-3 (bf16)", OUT / "v91-m14-ext3", 3e-3, 300.0),
    ("ext6", "+300M, lr 6e-3 (bf16)", OUT / "v92-m14-ext6", 6e-3, 300.0),
    ("ext10", "+300M, lr 1e-2 (bf16)", OUT / "v93-m14-ext10", 1e-2, 300.0),
]
START_VAL = 0.6683  # the weights both extensions start from: the base run's final validation


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_evals(d):
    out = []
    for p in sorted((d / "logs").glob("*.log")):
        for m in re.finditer(r"^step (\d+)/\d+ train ([\d.]+) val ([\d.]+)", p.read_text(), re.M):
            out.append((int(m[1]), float(m[2]), float(m[3])))
    return out


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, curves, ckpts = [], [], []
    for rid, label, d, lr, off in RUNS:
        res = next(iter(sorted(d.glob("*.json"))), None) if d.exists() else None
        ev = log_evals(d) if d.exists() else []
        files = sorted((d / "ckpt").glob("step_*.ckpt"), key=lambda p: int(re.findall(r"\d+", p.name)[0])) if (d / "ckpt").exists() else []
        step = int(re.findall(r"\d+", files[-1].name)[0]) if files else (ev[-1][0] if ev else 0)
        row = dict(run=rid, label=label, lr=lr, status="not started", step=0, steps=STEPS, val=None, val_fp32=None, start_val=None if rid == "base" else START_VAL, seconds=None, tok_per_s=None, peak_gb=None, updated=None)
        if res and res.exists():
            j = json.loads(res.read_text())
            row.update(status="done", step=j["steps"], val=j["val_bpc"], val_fp32=j.get("val_bpc_fp32"), seconds=j["train_seconds"], tok_per_s=j["chars_per_sec"], peak_gb=(j["peak_mem_mb"] / 1024) if j.get("peak_mem_mb") else None, updated=iso(res.stat().st_mtime))
            for s, v in j["evals"]:
                curves.append(dict(run=rid, kind="val", chars_m=off + CHARS_PER_RUN * s / STEPS, value=v))
            for s, v in j["train_curve"]:
                if v < 4.0:
                    curves.append(dict(run=rid, kind="train", chars_m=off + CHARS_PER_RUN * s / STEPS, value=v))
        elif d.exists() and (files or ev):
            row.update(status="running", step=step, updated=iso(max([p.stat().st_mtime for p in files] + [0]) or time.time()))
            for s, tr, v in ev:
                curves.append(dict(run=rid, kind="val", chars_m=off + CHARS_PER_RUN * s / STEPS, value=v))
            if ev:
                row["val"] = ev[-1][2]
        if rid != "base":
            curves.append(dict(run=rid, kind="val", chars_m=off, value=START_VAL))
        runs.append(row)
        for p in files:
            ckpts.append(dict(run=rid, file=p.name, step=int(re.findall(r"\d+", p.name)[0]), mb=round(p.stat().st_size / 1e6), saved=iso(p.stat().st_mtime), kind="resume (model, optimiser, schedule)"))
        for p in sorted(d.glob("*.pt")) if d.exists() else []:
            ckpts.append(dict(run=rid, file=p.name, step=row["steps"] if row["status"] == "done" else None, mb=round(p.stat().st_size / 1e6), saved=iso(p.stat().st_mtime), kind="final weights only"))
    byid = {r["run"]: r for r in runs}
    e1, e3, b = byid["ext1"], byid["ext3"], byid["base"]
    P = []

    def scored(run, claim, prob, test, value_fmt):
        r = byid[run]
        if r["status"] != "done":
            P.append(dict(run=run, claim=claim, p=prob, outcome="pending", value=None, note=""))
        else:
            ok, val = test(r)
            P.append(dict(run=run, claim=claim, p=prob, outcome="right" if ok else "wrong", value=value_fmt(val), note=""))

    for lim, p in ((0.6683, 0.95), (0.65, 0.80), (0.62, 0.50), (0.60, 0.20)):
        scored("ext1", f"ends below {lim}", p, lambda r, lim=lim: (r["val"] < lim, r["val"]), lambda v: f"{v:.4f}")
    scored("ext1", "bf16 at least 1.15x the fp32 base run's speed", 0.65, lambda r: (r["tok_per_s"] / b["tok_per_s"] >= 1.15, r["tok_per_s"] / b["tok_per_s"]), lambda v: f"{v:.2f}x")
    scored("ext1", "run takes at most 36 minutes", 0.65, lambda r: (r["seconds"] <= 2160, r["seconds"] / 60), lambda v: f"{v:.0f} min")
    scored("ext1", "bf16 and fp32 validation differ by under 0.005", 0.85, lambda r: (abs(r["val"] - r["val_fp32"]) < 0.005, abs(r["val"] - r["val_fp32"])), lambda v: f"{v:.4f}")
    scored("ext3", "lr 3e-3 ends below the lr 1e-3 run", 0.55, lambda r: (e1["val"] is not None and r["val"] < e1["val"], r["val"]), lambda v: f"{v:.4f}")
    scored("ext3", "lr 3e-3 ends below 0.650", 0.30, lambda r: (r["val"] < 0.65, r["val"]), lambda v: f"{v:.4f}")
    scored("ext6", "lr 6e-3 ends below the lr 3e-3 run", 0.50, lambda r: (e3["val"] is not None and r["val"] < e3["val"], r["val"]), lambda v: f"{v:.4f}")
    scored("ext6", "lr 6e-3 ends below 0.640", 0.20, lambda r: (r["val"] < 0.64, r["val"]), lambda v: f"{v:.4f}")
    scored("ext6", "lr 6e-3 ends above the lr 1e-3 result", 0.15, lambda r: (e1["val"] is not None and r["val"] > e1["val"], r["val"]), lambda v: f"{v:.4f}")
    e6 = byid["ext6"]
    scored("ext10", "lr 1e-2 ends below the lr 6e-3 run", 0.40, lambda r: (e6["val"] is not None and r["val"] < e6["val"], r["val"]), lambda v: f"{v:.4f}")
    scored("ext10", "lr 1e-2 ends below 0.635", 0.15, lambda r: (r["val"] < 0.635, r["val"]), lambda v: f"{v:.4f}")
    scored("ext10", "lr 1e-2 ends above the lr 3e-3 result", 0.25, lambda r: (e3["val"] is not None and r["val"] > e3["val"], r["val"]), lambda v: f"{v:.4f}")
    scored("ext3", "lr 3e-3 ends above the starting 0.6683", 0.10, lambda r: (r["val"] > START_VAL, r["val"]), lambda v: f"{v:.4f}")
    stamp = dict(updated=iso(time.time()))
    for name, rows in (("runs", runs), ("curves", curves), ("checkpoints", ckpts), ("predictions", P)):
        (out / f"{name}.json").write_text(json.dumps(rows))
    print({n: len(r) for n, r in (("runs", runs), ("curves", curves), ("checkpoints", ckpts), ("predictions", P))}, stamp)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/ext_status")
