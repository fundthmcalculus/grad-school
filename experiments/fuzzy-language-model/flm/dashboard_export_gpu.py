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
    "v5-long",
    "v6-d-lr",
    "v6-d-lr-edge",
    "v6-ref-tune",
    "v6-final10",
]
CURVE_GRIDS = [
    "v3-final",
    "v3-ref",
    "v3-big",
    "v5-bpe30",
    "v5-sub100k",
    "v5-final100k",
    "v5-long",
]

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
    (
        "v5-long",
        "D-d48-bpe1024_300M_lr3e-2_s0",
        "Fuzzy D, BPE1024 d = 48 (93K), 300M chars",
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
        "Without it the subword headline is a statement about subwords, not fuzziness",
        "supported",
        "done at ten seeds and 30M characters (v6-final10); other tokenizers, sizes and budgets are the next gap",
        "Tuned fuzzy D beats both tuned neural references by 0.027 and 0.030 bits per character (ten seeds each, more than 2 standard errors); D's worst seed beats the references' best. Caveats: one tokenizer, one size, one budget; the references were chosen at 10M characters, seed 0, which favours them; D runs about 1.5 times slower per character",
    ),
    (
        "D's learning rate on subwords",
        "The 30M, 100M and 300M subword runs used lr 3e-2, while the 10M optimum is 6e-2",
        "closed at 30M",
        "an lr check at 100M and 300M, if those numbers are to be quoted",
        "At 30M it does not matter: D at 6e-2 (ten seeds) scores 1.1291, D at 3e-2 (two seeds) 1.1292",
    ),
    (
        "Ten seeds",
        "AGENTS.md makes ten the protocol; the subword comparison has them, nothing else does",
        "partly closed",
        "v4-final10 for the character-level claim",
        "Done for subwords (v6-final10); the character-level comparison has three seeds",
    ),
    (
        "Generality of the subword result",
        "One tokenizer (BPE1024), one model size (about 93K parameters) and one budget (30M characters)",
        "open",
        "other vocabularies, sizes and budgets with their own tuned references",
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
    rulebook(out)
    print(f"wrote 5 CSVs ({len(recs)} runs) to {out}")


RULEBOOK_CKPT = OUT / "v5-final100k" / "D-d48-bpe1024_100M_s0.pt"


def rulebook(out: Path, windows: int = 400, per_layer: int = 15) -> None:
    """The first-layer and second-layer FFN rules of the best sub-100K model, as rows (needs the BPE
    tokenizer and a rebuilt validation token stream: see flm/bpe.py)."""
    from flm.rules import ffn_rule_rows, load_model, make_codec

    model, tok = load_model(RULEBOOK_CKPT)
    codec = make_codec(tok)
    rows = ffn_rule_rows(
        model,
        codec.windows(windows, model.cfg.max_len),
        codec,
        top=3,
        max_rules=per_layer,
    )
    with open(out / "rulebook.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "key",
                "layer",
                "rule",
                "wins_pct",
                "firing",
                "gain",
                "gain_sd",
                "fires_on",
                "examples",
                "raises",
                "raises_neg",
            ]
        )
        for r in rows:
            fmt = lambda x, n: "" if x is None else round(x, n)  # noqa: E731
            w.writerow(
                [
                    f"L{r['layer']}R{r['rule']}",
                    r["layer"],
                    r["rule"],
                    fmt(100 * r["wins"], 2),
                    fmt(r["firing"], 3),
                ]
                + [
                    fmt(r["gain"], 2),
                    fmt(r["gain_sd"], 2),
                    " · ".join(r["fires_on"]),
                    " ¦ ".join(r["examples"]),
                ]
                + [r["raises"], r["raises_neg"] or ""]
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
