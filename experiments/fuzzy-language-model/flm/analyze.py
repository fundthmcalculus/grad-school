"""Aggregate a sweep: mean +/- std val BPC per (arm, width, layers) across seeds.

    .venv/bin/python -m flm.analyze scaling [headline ...] [--name merged] [--threshold 2.0 1.8 1.6]

Writes outputs/<sweep>/summary.{csv,md} and outputs/<sweep>/bpc_vs_params.png.
Every row states its seed count; a cell with fewer seeds than the sweep's maximum
is flagged so a thin cell is never mistaken for a full one.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "outputs"

# Reference categorical palette (light mode), fixed order -- never cycled.
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
# arms drawn in the plot: 8 slots of the validated palette, fixed order (never cycled).
# The saturated (sum) FRLMs stay in the tables; their HTSK versions are plotted.
PLOT_ARMS = [
    "softmax-mlp",
    "linear-mlp",
    "gla-mlp",
    "delta-mlp",
    "gru",
    "flm",
    "frlm-acc-htsk",
    "frlm-delta-htsk",
]
ARM_ORDER = [
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
NAME_RE = re.compile(r"^(?P<arm>.+)_d(?P<d>\d+)_L(?P<L>\d+)_s(?P<s>\d+)$")


def load(sweeps):
    rows = []
    files = [f for sw in sweeps for f in sorted((OUT / sw).glob("*.json"))]
    for f in files:
        m = NAME_RE.match(f.stem)
        if not m:
            continue
        r = json.loads(f.read_text())
        rows.append(
            dict(
                arm=m["arm"],
                d=int(m["d"]),
                L=int(m["L"]),
                seed=int(m["s"]),
                params=r["params_total"],
                params_nonemb=r["params_nonemb"],
                bpc=r["val_bpc"],
                secs=r["train_seconds"],
                cps=r["chars_per_sec"],
                device=r.get("device", "cpu"),
            )
        )
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        # device is part of the key: CPU and GPU runs are never pooled into one cell
        groups[(r["arm"], r["d"], r["L"], r["device"])].append(r)
    out = []
    for (arm, d, L, device), g in groups.items():
        b = np.array([x["bpc"] for x in g], dtype=float)
        ok = np.isfinite(b)
        out.append(
            dict(
                arm=arm,
                d=d,
                L=L,
                device=device,
                params=g[0]["params"],
                params_nonemb=g[0]["params_nonemb"],
                n_seeds=len(g),
                n_diverged=int((~ok).sum()),
                bpc_mean=float(b[ok].mean()) if ok.any() else math.nan,
                bpc_std=float(b[ok].std(ddof=1)) if ok.sum() > 1 else math.nan,
                secs_mean=float(np.mean([x["secs"] for x in g])),
                cps_mean=float(np.mean([x["cps"] for x in g])),
            )
        )
    key = {a: i for i, a in enumerate(ARM_ORDER)}
    out.sort(key=lambda r: (key.get(r["arm"], 99), r["L"], r["d"]))
    return out


def smallest_reaching(summary, threshold):
    """Per arm: (params, kind) for the first crossing of mean BPC <= threshold, scanning sizes
    upward and interpolating in log-params. kind is "interp" for a crossing between two grid
    sizes, "below-grid" if the smallest size already reaches the threshold (so params is only
    an upper bound), or None if no size reaches it."""
    res = {}
    for arm in {s["arm"] for s in summary}:
        pts = sorted(
            (s["params"], s["bpc_mean"])
            for s in summary
            if s["arm"] == arm and np.isfinite(s["bpc_mean"])
        )
        res[arm] = (None, None)
        if pts and pts[0][1] <= threshold:
            res[arm] = (pts[0][0], "below-grid")
            continue
        for (p0, b0), (p1, b1) in zip(pts, pts[1:]):
            if b0 > threshold >= b1:
                f = (b0 - threshold) / (b0 - b1)
                res[arm] = (
                    math.exp(math.log(p0) + f * (math.log(p1) - math.log(p0))),
                    "interp",
                )
                break
    return res


def write_tables(sweep, summary, thresholds):
    d = OUT / sweep
    cols = [
        "arm",
        "d",
        "L",
        "params",
        "params_nonemb",
        "n_seeds",
        "n_diverged",
        "bpc_mean",
        "bpc_std",
        "secs_mean",
        "cps_mean",
    ]
    with open(d / "summary.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for s in summary:
            fh.write(",".join(str(s[c]) for c in cols) + "\n")
    max_seeds = max(s["n_seeds"] for s in summary)
    lines = [
        f"# Sweep `{sweep}` — validation bits per character (mean ± std over seeds)\n",
        "| arm | d | L | params (total) | params (non-emb) | seeds | val BPC | train s | chars/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        flag = " ⚠" if s["n_seeds"] < max_seeds else ""
        div = f" ({s['n_diverged']} diverged)" if s["n_diverged"] else ""
        lines.append(
            f"| {s['arm']} | {s['d']} | {s['L']} | {s['params']:,} | {s['params_nonemb']:,} | {s['n_seeds']}{flag}{div} "
            f"| {s['bpc_mean']:.4f} ± {s['bpc_std']:.4f} | {s['secs_mean']:.0f} | {s['cps_mean']:,.0f} |"
        )
    if thresholds:
        res = {t: smallest_reaching(summary, t) for t in thresholds}
        arms = sorted(
            {x["arm"] for x in summary},
            key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99,
        )
        lines += [
            "",
            "## Smallest model reaching a mean val BPC threshold (total params)",
            "",
            "Log-interpolated between adjacent grid sizes at the first crossing. `≤ N` means the",
            "smallest grid size already reaches the threshold, so N is only an upper bound.",
            "",
            "| arm | " + " | ".join(f"BPC ≤ {t}" for t in thresholds) + " |",
            "|---|" + "---|" * len(thresholds),
        ]

        def cell(v):
            p, kind = v
            if p is None:
                return "not reached"
            return f"≤ {p:,.0f}" if kind == "below-grid" else f"{p:,.0f}"

        for arm in arms:
            lines.append(
                f"| {arm} | " + " | ".join(cell(res[t][arm]) for t in thresholds) + " |"
            )
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    return d / "summary.md"


def plot(sweep, summary, nonemb=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    arms = [a for a in PLOT_ARMS if any(s["arm"] == a for s in summary)]
    pkey = "params_nonemb" if nonemb else "params"
    for i, arm in enumerate(arms):
        pts = sorted(
            (s[pkey], s["bpc_mean"], s["bpc_std"])
            for s in summary
            if s["arm"] == arm
            and s["L"] == min(x["L"] for x in summary if x["arm"] == arm)
        )
        pts = [p for p in pts if np.isfinite(p[1])]
        if not pts:
            continue
        x, y, e = map(np.array, zip(*pts))
        c = PALETTE[i]
        ax.plot(x, y, "-", color=c, lw=2, marker="o", ms=5, label=arm)
        ax.fill_between(
            x, y - np.nan_to_num(e), y + np.nan_to_num(e), color=c, alpha=0.15, lw=0
        )
        # no end-of-line labels: 8 series converge at the right edge and collide;
        # the legend carries identity
    ax.set_xscale("log")
    ax.set_xlabel(("non-embedding" if nonemb else "total") + " parameters")
    ax.set_ylabel("validation bits per character")
    ax.set_title(
        f"TinyStories char-level LM, CPU-trained ({sweep}): mean ± std over seeds",
        fontsize=10,
        color="#0b0b0b",
    )
    ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = OUT / sweep / ("bpc_vs_params_nonemb.png" if nonemb else "bpc_vs_params.png")
    fig.savefig(path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "sweeps", nargs="+", help="one or more sweep dirs under outputs/, merged"
    )
    p.add_argument(
        "--name",
        default=None,
        help="output dir for the merged summary (default: first sweep)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        nargs="*",
        default=[],
        help="BPC thresholds for the smallest-model table",
    )
    a = p.parse_args()
    name = a.name or a.sweeps[0]
    (OUT / name).mkdir(parents=True, exist_ok=True)
    summary = summarize(load(a.sweeps))
    print(write_tables(name, summary, a.threshold).read_text())
    print(plot(name, summary), plot(name, summary, nonemb=True))


if __name__ == "__main__":
    main()
