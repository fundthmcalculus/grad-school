"""Export the CSV datasets behind the published results dashboard.

    .venv/bin/python -m flm.dashboard_export OUT_DIR

Writes scaling.csv, smallest.csv, gap.csv, h9.csv, tune.csv and state.csv from the run
records under outputs/ (scaling, scaling2, h9, tune). Every number on the dashboard
comes from one of these files, so this script is how those numbers trace to runs.
"""

from __future__ import annotations

import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from flm.analyze import OUT, load, smallest_reaching, summarize
from flm.state import state_floats

KEEP = [
    "softmax-mlp",
    "linear-mlp",
    "gla-mlp",
    "delta-mlp",
    "gru",
    "flm",
    "frlm-acc",
    "frlm-delta",
    "frlm-acc-htsk",
    "frlm-delta-htsk",
]
LABEL = {
    "softmax-mlp": "Softmax attention",
    "linear-mlp": "Linear attention",
    "gla-mlp": "Gated linear attention",
    "delta-mlp": "DeltaNet",
    "gru": "GRU",
    "flm": "FLM (Gaussian-TSK attention)",
    "frlm-acc": "FRLM accumulating",
    "frlm-delta": "FRLM delta",
    "frlm-acc-htsk": "FRLM accumulating + HTSK",
    "frlm-delta-htsk": "FRLM delta + HTSK",
}
FUZZY = {"flm", "frlm-acc", "frlm-delta", "frlm-acc-htsk", "frlm-delta-htsk"}
NEURAL_LINEAR = ["linear-mlp", "gla-mlp", "delta-mlp", "gru"]
H9_NAMES = {
    ("sum", "sum"): "Classic everywhere",
    ("mean", "sum"): "HTSK on recurrent mixer only",
    ("sum", "mean"): "HTSK on feed-forward only",
    ("mean", "mean"): "HTSK on both",
}


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = [s for s in summarize(load(["scaling", "scaling2"])) if s["arm"] in KEEP]

    with open(out / "scaling.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "key",
                "arm",
                "model",
                "fuzzy",
                "cost",
                "width",
                "params",
                "params_nonemb",
                "seeds",
                "bpc_mean",
                "bpc_std",
            ]
        )
        for s in summary:
            cost = "quadratic" if s["arm"] in ("softmax-mlp", "flm") else "linear"
            w.writerow(
                [
                    f"{s['arm']}|{s['d']}",
                    s["arm"],
                    LABEL[s["arm"]],
                    int(s["arm"] in FUZZY),
                    cost,
                    s["d"],
                    s["params"],
                ]
                + [
                    s["params_nonemb"],
                    s["n_seeds"],
                    round(s["bpc_mean"], 4),
                    round(s["bpc_std"], 4),
                ]
            )

    with open(out / "smallest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "model", "threshold", "params", "bound"])
        for t in (2.0, 1.8, 1.6, 1.4):
            res = smallest_reaching(summary, t)
            for arm in KEEP:
                p, kind = res[arm]
                bound = {"interp": "interpolated", "below-grid": "upper bound"}.get(
                    kind, "not reached"
                )
                w.writerow([arm, LABEL[arm], t, "" if p is None else round(p), bound])

    def curve(arm):
        return sorted((s["params"], s["bpc_mean"]) for s in summary if s["arm"] == arm)

    def interp(arm, p):
        pts = curve(arm)
        if p < pts[0][0] or p > pts[-1][0]:
            return None  # no extrapolation
        for (p0, b0), (p1, b1) in zip(pts, pts[1:]):
            if p0 <= p <= p1:
                f = (math.log(p) - math.log(p0)) / (math.log(p1) - math.log(p0))
                return b0 + f * (b1 - b0)
        return None

    with open(out / "gap.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "key",
                "arm",
                "model",
                "width",
                "params",
                "bpc",
                "best_neural",
                "best_neural_bpc",
                "gap",
            ]
        )
        for arm in ("frlm-acc", "frlm-delta", "frlm-acc-htsk", "frlm-delta-htsk"):
            for s in sorted(
                (s for s in summary if s["arm"] == arm), key=lambda s: s["params"]
            ):
                c = [
                    (interp(a, s["params"]), a)
                    for a in NEURAL_LINEAR
                    if interp(a, s["params"]) is not None
                ]
                if not c:
                    continue
                nb, na = min(c)
                w.writerow(
                    [
                        f"{arm}|{s['d']}",
                        arm,
                        LABEL[arm],
                        s["d"],
                        s["params"],
                        round(s["bpc_mean"], 4),
                    ]
                    + [LABEL[na], round(nb, 4), round(s["bpc_mean"] - nb, 4)]
                )

    cells = collections.defaultdict(list)
    for f in (OUT / "h9").glob("*.json"):
        arm, mx, ff, _ = f.stem.rsplit("_", 3)
        cells[(arm, mx[2:], ff[2:])].append(json.loads(f.read_text())["val_bpc"])
    with open(out / "h9.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "key",
                "arm",
                "model",
                "variant",
                "mixer_norm",
                "ffn_norm",
                "seeds",
                "bpc_mean",
                "bpc_std",
            ]
        )
        for (arm, mx, ff), v in sorted(cells.items()):
            std = round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else ""
            w.writerow(
                [
                    f"{arm}|{mx}|{ff}",
                    arm,
                    LABEL[arm],
                    H9_NAMES[(mx, ff)],
                    mx,
                    ff,
                    len(v),
                    round(float(np.mean(v)), 4),
                    std,
                ]
            )

    with open(out / "tune.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "arm", "model", "lr", "shortconv", "bpc"])
        for f in sorted((OUT / "tune").glob("*.json")):
            arm, lr, sc = f.stem.rsplit("_", 2)
            if arm in KEEP:
                bpc = json.loads(f.read_text())["val_bpc"]
                w.writerow(
                    [
                        f"{arm}|{lr[2:]}|{sc[2:]}",
                        arm,
                        LABEL[arm],
                        float(lr[2:]),
                        int(sc[2:]),
                        bpc,
                    ]
                )

    with open(out / "state.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "arm", "model", "context", "state_floats"])
        for arm, mixer in (
            ("softmax-mlp", "softmax"),
            ("delta-mlp", "delta"),
            ("frlm-delta-htsk", "fuzzydelta"),
            ("gru", "gru"),
        ):
            for t in (16, 64, 256, 1024, 4096, 16384, 65536):
                floats = state_floats(
                    mixer, 16, 2, 2, 16, t, shortconv=0 if mixer == "gru" else 4
                )
                w.writerow([f"{arm}|{t}", arm, LABEL[arm], t, floats])
    print(f"wrote 6 CSVs to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
