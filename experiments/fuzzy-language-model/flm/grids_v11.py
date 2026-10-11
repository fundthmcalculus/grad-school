"""Hybrid: the recurrent fuzzy mixer for memory plus one Fibonacci-lag attention layer (PLAN_FIBONACCI.md, stage 2).

v10 showed that Fibonacci attention beats dense attention (0.03 bpc) but trails D by 0.07, and that the
attention models do not recall a stated name at 10M characters while D does. The hybrid asks whether the two
are complementary: D's fixed-state memory in one layer, exact sparse retrieval at Fibonacci lags in the other.

    hybDF  layer 0 = D's delta-form fuzzy mixer, layer 1 = fibgauss (local 8)
    hybFD  layer 0 = fibgauss, layer 1 = D mixer
    lr 3e-2 and 6e-2, BPE1024, d 48, ctx 512, 10M characters, seed 0              (4 cells)

D's mixer keeps its 8 heads, the attention layer uses 4 (``--attn-heads 4``); ffn-mult 1.75 (~94,144 parameters,
D is 93,344). Run with ``--workers 1`` on a GPU.
"""

from __future__ import annotations

from flm.grids_v3 import FFN_STAR, cell
from flm.sweep import register

BPE = ["--tok", "bpe1024", "--warmup", "100", "--save-model", "--ctx", "512"]
BPE += ["--batch", "16"]
BASE = ["--decay", "data", "--exp-norm", "mean", "--ffn-exp-norm", "mean"]
BASE += ["--heads", "8", "--attn-heads", "4", "--ffn-mult", "1.75"] + FFN_STAR
BASE += ["--fib-local", "8"]
ARMS = {
    "hybDF": ["--layer-mixers", "fuzzydelta,fibgauss"],
    "hybFD": ["--layer-mixers", "fibgauss,fuzzydelta"],
}
register(
    "v11-hybrid",
    [
        cell(
            f"{n}-lr{lr}_10M_s0",
            ["--mixer", "fuzzydelta"] + BASE + extra + BPE,
            d=48,
            sc="8",
            lr=lr,
            chars="10e6",
        )
        for n, extra in ARMS.items()
        for lr in ("3e-2", "6e-2")
    ],
)

# The same cells under another name for the CPU host (records in outputs/v11-hybrid-cpu, never the GPU's).
register("v11-hybrid-cpu", list(__import__("flm.sweep").sweep.GRIDS["v11-hybrid"]))
