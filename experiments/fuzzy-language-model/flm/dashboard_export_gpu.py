"""Export the datasets behind the GPU-phase results dashboard from the run records.

    .venv/bin/python -m flm.dashboard_export_gpu OUT_DIR

Writes gpu_runs.csv (one row per run), gpu_curves.csv (validation curves of the headline runs),
gpu_samples.csv (the saved 300-token sample of selected runs) and gaps.csv (the open fairness gaps).
Every number on the dashboard comes from these files; the aggregations are dashboard calculations.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"

GRIDS = [
    "v3-attrib",
    "v3-ffn",
    "v3-ffn2",
    "v3-ffn3",
    "v3-mix",
    "v3-heads",
    "v3-fill",
    "v3-final",
    "v3-ref",
    "v3-big",
    "v3-post",
    "v5-bpe-probe",
    "v5-bpe30",
    "v5-sub100k",
    "v5-spend",
    "v5-final100k",
]
CURVE_GRIDS = ["v3-final", "v3-ref", "v3-big", "v5-bpe30", "v5-sub100k", "v5-final100k"]

# (grid, run) -> label for the generated-text page
SAMPLES = [
    ("v3-final", "D_100M_s0", "Fuzzy D, character model, 100M chars"),
    ("v3-final", "D_30M_s0", "Fuzzy D, character model, 30M chars"),
    (
        "v3-ref",
        "delta-mlp-d72-h8_30M_s0",
        "Neural delta-mlp (93K), character model, 30M chars",
    ),
    ("v3-big", "D-d96_30M_s0", "Fuzzy D d = 96 (162K), character model, 30M chars"),
    ("v5-bpe30", "D-d80-bpe512_30M_s0", "Fuzzy D, BPE512 d = 80 (150K), 30M chars"),
    (
        "v5-final100k",
        "D-d48-bpe1024_30M_s0",
        "Fuzzy D, BPE1024 d = 48 (93K), 30M chars",
    ),
    (
        "v5-final100k",
        "D-d48-bpe1024_100M_s0",
        "Fuzzy D, BPE1024 d = 48 (93K), 100M chars",
    ),
]

GAPS = [
    (
        "Character level: D vs the size-near reference",
        "3 seeds; the reference has 4.6% fewer parameters and was tuned less (heads and one lr)",
        "not citable",
        "v4-ref-tune, v3-lr, v4-final10 (PLAN_GPU.md)",
        "Ready and tested; not started (GPU session's user must approve the compute)",
    ),
    (
        "Subwords: a neural control on the same tokenizer",
        "Until a neural BPE1024 model of the same size is tuned equally, the subword headline is a statement about subwords, not fuzziness",
        "open",
        "v6-d-lr, v6-ref-tune, v6-final10 (flm/grids_v6.py)",
        "Grids written and arg-parsed; the GPU session will smoke-run one cell of each, then needs approval",
    ),
    (
        "D's learning rate on subwords",
        "Only 3e-2 was ever run; the neural control gets a 4-lr search",
        "open",
        "v6-d-lr",
        "Part of the subword control set",
    ),
    (
        "Ten seeds",
        "Every cell is 1 to 3 seeds; AGENTS.md makes ten the protocol",
        "open",
        "v4-final10, v6-final10",
        "Not started",
    ),
    (
        "Learning-rate study (v3-lr) and H10a",
        "The 1e-2 cells were lost to a crash from a stale train.py; the 2e-2 cells were killed for VRAM contention",
        "incomplete",
        "resume v3-lr, then lrwidth",
        "Not resumed",
    ),
    (
        "Coherence",
        "Bits per character is not coherence; low-temperature text loops",
        "open",
        "a coherence grader, longer context, larger models",
        "No grader exists",
    ),
    (
        "n-gram rule initialisation on D",
        "A null at d = 16 on the old FFN (CPU screen); untested on D's first-order FFN",
        "untested",
        "v4-taps (21 cells, ~1 GPU-h)",
        "Not started",
    ),
]


def arm_of(run: str) -> str:
    return re.sub(r"_s\d+$", "", run)


def seed_of(run: str) -> str:
    m = re.search(r"_s(\d+)$", run)
    return m.group(1) if m else ""


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    recs = {}
    for g in GRIDS:
        for f in sorted((OUT / g).glob("*.json")):
            recs[(g, f.stem)] = json.loads(f.read_text())

    cols = [
        "grid",
        "run",
        "arm",
        "seed",
        "tok",
        "params",
        "chars_m",
        "val_bpc",
        "val_bpc_ema",
        "chars_per_token",
    ]
    cols += [
        "steps",
        "cps",
        "lr",
        "d",
        "heads",
        "layers",
        "mixer",
        "ffn",
        "ffn_order",
        "ffn_norm",
        "shortconv",
    ]
    with open(out / "gpu_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for (g, run), r in recs.items():
            a = r["args"]
            bpc = r["val_bpc"]
            ema = r.get("val_bpc_ema")
            w.writerow(
                [
                    g,
                    run,
                    arm_of(run),
                    seed_of(run),
                    a.get("tok") or "char",
                    r["params_total"],
                ]
                + [round(r["chars_seen"] / 1e6, 1), "" if bpc != bpc else round(bpc, 5)]
                + [
                    "" if ema is None or ema != ema else round(ema, 5),
                    r.get("chars_per_token") or 1.0,
                ]
                + [
                    r["steps"],
                    r["chars_per_sec"],
                    a["lr"],
                    a["d"],
                    a["heads"],
                    a["layers"],
                    a["mixer"],
                    a["ffn"],
                ]
                + [a.get("ffn_order", 0), a.get("ffn_norm", "softmax"), a["shortconv"]]
            )

    with open(out / "gpu_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["key", "grid", "run", "arm", "seed", "tok", "params", "chars_m", "val_bpc"]
        )
        for (g, run), r in recs.items():
            if g not in CURVE_GRIDS or len(r["evals"]) < 3:
                continue
            tok = r["args"].get("tok") or "char"
            for step, bpc in r["evals"]:
                chars = step / r["steps"] * r["chars_seen"] / 1e6
                w.writerow(
                    [
                        f"{g}|{run}|{step}",
                        g,
                        run,
                        arm_of(run),
                        seed_of(run),
                        tok,
                        r["params_total"],
                        round(chars, 2),
                        round(bpc, 5),
                    ]
                )

    with open(out / "gpu_samples.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "label", "params", "tok", "val_bpc", "sample"])
        for g, run, label in SAMPLES:
            r = recs[(g, run)]
            text = (
                r["sample"].replace("\x1f", " ⏎⏎ ").replace("\n", " ↵ ").strip()[:520]
            )
            w.writerow(
                [
                    f"{g}|{run}",
                    label,
                    r["params_total"],
                    r["args"].get("tok") or "char",
                    round(r["val_bpc"], 4),
                    text,
                ]
            )

    with open(out / "gaps.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "gap", "why", "status", "how", "state"])
        for i, row in enumerate(GAPS):
            w.writerow([f"gap{i}", *row])
    print(f"wrote 4 CSVs ({len(recs)} runs) to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
