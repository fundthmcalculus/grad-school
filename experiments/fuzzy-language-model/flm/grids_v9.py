"""Scale D until the stories cohere, and keep it readable (PLAN_COHERENCE.md).

The sub-100K D reaches plot scores of 1.9/5 even at 300M characters: grammar and local consistency
are there, a plot is not. This is the size ladder that asks where coherence appears, with the same
architecture (delta-form fuzzy recurrent mixer, first-order TSK FFN) so every size stays readable with
the same rulebook tool.

    v9-lr    d 192, 4 layers (1.32M), BPE1024, ctx 512, 10M chars, lr 1e-2 .. 6e-2, seed 0      (4 cells)
    v9-size  six sizes 0.25M .. 7M parameters at 100M chars, ctx 512, lr from v9-lr, seed 0      (6 cells)

Parameter counts (BPE1024, tied embedding, shortconv 8, ffn-mult 1.5): d96/L2 251,168; d128/L4 653,248;
d192/L4 1,323,904; d256/L4 2,223,936; d256/L6/32 rules 3,253,856; d384/L6/12 heads/32 rules 6,963,600.
Checkpoints are saved (the coherence and probe tools read them). Written and arg-parsed on CPU, not
run: run one cell first and read steps/s before launching the rest.
"""

from __future__ import annotations

import json

from flm.grids_v3 import D, cell
from flm.sweep import OUT, register

BPE = ["--tok", "bpe1024", "--warmup", "200", "--save-model", "--ctx", "512"]
BPE += ["--batch", "16"]
LRS = ("1e-2", "2e-2", "4e-2", "6e-2")
# name -> (d, layers, heads, rules)
SIZES = {
    "d96-L2": (96, 2, 8, 16),
    "d128-L4": (128, 4, 8, 16),
    "d192-L4": (192, 4, 8, 16),
    "d256-L4": (256, 4, 8, 16),
    "d256-L6-r32": (256, 6, 8, 32),
    "d384-L6-h12-r32": (384, 6, 12, 32),
}


def size_args(name: str) -> list[str]:
    _, _, h, r = SIZES[name]  # width and depth go through cell()
    return ["--heads", str(h), "--rules", str(r)]


register(
    "v9-lr",
    [
        cell(
            f"d192-L4-lr{lr}_10M_s0",
            D + BPE + size_args("d192-L4"),
            d=192,
            L=4,
            sc="8",
            lr=lr,
            chars="10e6",
        )
        for lr in LRS
    ],
)


def best_lr(default: str = "3e-2") -> str:
    """Lowest val BPC among the finished v9-lr cells (needs at least two)."""
    got = {}
    for f in (OUT / "v9-lr").glob("d192-L4-lr*_10M_s0.json"):
        r = json.loads(f.read_text())
        if r["val_bpc"] == r["val_bpc"]:
            got[r["args"]["lr"]] = r["val_bpc"]
    if len(got) < 2:
        return default
    return f"{float(min(got, key=got.get)):g}"


register(
    "v9-size",
    [
        cell(
            f"{n}_100M_s0",
            D + BPE + size_args(n),
            d=SIZES[n][0],
            L=SIZES[n][1],
            sc="8",
            lr=best_lr(),
            chars="100e6",
        )
        for n in SIZES
    ],
)
