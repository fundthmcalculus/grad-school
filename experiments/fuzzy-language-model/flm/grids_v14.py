"""State curve within the 165K cap: extend v13-state (PLAN_FIBONACCI.md, stage 3).

v13-state: r64 1.1113 (102.6K), r128 1.0979 (114.8K), two seeds each, steps of 0.0012 and 0.0134. Two questions:
are those steps real (seeds 2-3 of r64 and r128), and does the curve keep falling (r192 ~127K, r256 ~139K parameters,
seeds 0-1)? D, BPE1024, d 48, ctx 512, batch 16, lr 6e-2, 30M characters, checkpoints saved.   (8 cells)
Run with ``--workers 1``.
"""

from __future__ import annotations

from flm.grids_v3 import D, cell
from flm.sweep import register

BPE = ["--tok", "bpe1024", "--warmup", "100", "--save-model", "--ctx", "512"]
BPE += ["--batch", "16"]
CELLS = [(64, 2), (64, 3), (128, 2), (128, 3), (192, 0), (192, 1), (256, 0), (256, 1)]
register(
    "v14-state2",
    [
        cell(
            f"D-ctx512-r{r}_30M_s{s}",
            D + BPE + ["--rules", str(r)],
            d=48,
            sc="8",
            lr="6e-2",
            chars="30e6",
            seed=s,
        )
        for r, s in CELLS
    ],
)
