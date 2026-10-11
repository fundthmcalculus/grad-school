"""Fibonacci attention at story size: remember mostly locally, a few tokens far back (PLAN_FIBONACCI.md).

Every arm: BPE1024, d 48, 2 layers, ctx 512 (about one story), D's first-order TSK FFN, 10M characters,
seed 0, lr 2e-2 and 6e-2. Attention arms use 4 heads and ffn-mult 2.0 (94,944 parameters; D is 93,344).

    D            the recurrent fuzzy mixer (the current headline), the baseline
    dense-gauss  Gaussian-TSK attention over all 512 positions (quadratic)
    fibgauss-L8  the same, but each query sees lags 0-7 and then only the Fibonacci lags 8, 13, 21, ..., 377
    fibgauss-L4 / -L16   the dense local window shrunk / grown
    fibsoftmax-L8        the same lag set with ordinary softmax attention (is the fuzzy form needed?)

    12 cells. The mask form is exact but not fast (it is still a dense T x T product); an O(T log T) gather
    implementation is for after the question is answered. Written and arg-parsed on CPU, not run on GPU.
"""

from __future__ import annotations

from flm.grids_v3 import D, FFN_STAR, cell
from flm.sweep import register

BPE = ["--tok", "bpe1024", "--warmup", "100", "--save-model", "--ctx", "512"]
BPE += ["--batch", "16"]
LRS = ("2e-2", "6e-2")


def att(mixer: str, local: int = 8) -> list[str]:
    return (
        ["--mixer", mixer, "--heads", "4", "--ffn-mult", "2.0"]
        + FFN_STAR
        + ["--fib-local", str(local)]
    )


ARMS = {
    "D": D,
    "dense-gauss": att("gauss"),
    "fibgauss-L8": att("fibgauss", 8),
    "fibgauss-L4": att("fibgauss", 4),
    "fibgauss-L16": att("fibgauss", 16),
    "fibsoftmax-L8": att("fibsoftmax", 8),
}
register(
    "v10-fib",
    [
        cell(f"{n}-lr{lr}_10M_s0", argv + BPE, d=48, sc="8", lr=lr, chars="10e6")
        for n, argv in ARMS.items()
        for lr in LRS
    ],
)
