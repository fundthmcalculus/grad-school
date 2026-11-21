"""More recurrent state at story size: 64 and 128 rules per head (PLAN_FIBONACCI.md, stage 3).

``v8-rules`` found 32 rules per head at ctx 512 beats 16 by 0.011 bits per character (1.1125 vs 1.1232, 30M
characters, two seeds). If the gain is state capacity, it should continue. D, BPE1024, d 48, ctx 512, batch 16,
lr 6e-2, 30M characters, seeds 0-1, checkpoints saved:

    r64   102,560 parameters      r128  114,848 parameters                     (4 cells)

Compare with ``v8-rules`` (r32, 96,416) and ``v8-ctx`` ctx 512 (r16, 93,344). Run with ``--workers 1``.
"""

from __future__ import annotations

from flm.grids_v3 import D, cell
from flm.sweep import register

BPE = ["--tok", "bpe1024", "--warmup", "100", "--save-model", "--ctx", "512"]
BPE += ["--batch", "16"]
register(
    "v13-state",
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
        for r in (64, 128)
        for s in (0, 1)
    ],
)
