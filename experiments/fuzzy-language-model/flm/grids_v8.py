"""Does D use more context? Scale the training context of the BPE1024 D model (PLAN_CONTEXT.md).

    v8-ctx    D, BPE1024, d 48, lr 6e-2, ctx 256 / 512 / 1024 / 2048 at a constant 8,192 tokens
              per step (batch 32 / 16 / 8 / 4), 30M characters, seeds 0-1, checkpoints saved   (8 cells)
    v8-rules  the same at ctx 1024 with 32 rules per head instead of 16 (more state to remember
              with; the parameter count rises, it is recorded)                                  (2 cells)

The ctx-256 cells reproduce ``v6-final10``'s D (30M characters, lr 6e-2) up to the seed set, so they
are the baseline and cost nothing new to interpret. Written and arg-parsed on CPU; not run on GPU.
Run one cell first and check steps/s: the delta-rule scan is sequential in time, so ctx 2048 may be
slow; if it is, drop that arm rather than shrinking the budget.
"""

from __future__ import annotations

from flm.grids_v3 import D, cell
from flm.sweep import register

BPE = ["--tok", "bpe1024", "--warmup", "100", "--save-model"]
CTXS = (256, 512, 1024, 2048)


def ctx_args(ctx: int) -> list[str]:
    return ["--ctx", str(ctx), "--batch", str(8192 // ctx)]


register(
    "v8-ctx",
    [
        cell(
            f"D-ctx{c}_30M_s{s}",
            D + BPE + ctx_args(c),
            d=48,
            sc="8",
            lr="6e-2",
            chars="30e6",
            seed=s,
        )
        for c in CTXS
        for s in (0, 1)
    ],
)
register(
    "v8-rules",
    [
        cell(
            f"D-ctx1024-r32_30M_s{s}",
            D + BPE + ctx_args(1024) + ["--rules", "32"],
            d=48,
            sc="8",
            lr="6e-2",
            chars="30e6",
            seed=s,
        )
        for s in (0, 1)
    ],
)
