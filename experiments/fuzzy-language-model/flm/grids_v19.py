"""CPU tuning of the ~96K-parameter model (BPE1024 D, d 48, 2 layers, ctx 512 / batch 16, 32 rules, lr 6e-2).

Plan: PLAN_TUNE96K.md. Run with ``--device cpu --threads 4`` (about 5.4K chars/s per job at 4 threads).
Screens are SHORT (4M chars) and 3 seeds: they rank settings, they do not set a citable number; a winner is
confirmed at 30M chars on 5 seeds before it is claimed.
"""

from __future__ import annotations

from flm.grids_v3 import D, cell
from flm.sweep import register

TAIL = ["--tok", "bpe1024", "--ctx", "512", "--batch", "16", "--rules", "32"]
CHARS = "4e6"
SEEDS = (0, 1, 2)


def _cell(tag, seed, extra=(), d=48, L=2, lr="6e-2", warmup="100", chars=CHARS):
    name, argv = cell(
        f"{tag}_s{seed}", D, d=d, L=L, sc="8", chars=chars, seed=seed, lr=lr
    )
    return name, argv + TAIL + ["--warmup", warmup] + list(extra)


def _grid(variants, **kw):
    """variants: {tag: (extra argv, cell kwargs)}"""
    return [
        _cell(f"{t}", s, ex, **{**kw, **ck})
        for t, (ex, ck) in variants.items()
        for s in SEEDS
    ]


# pilot: the baseline alone, 3 seeds. Sizes the noise and the per-job cost before anything else runs.
register("v19-pilot", _grid({"base": ([], {})}))

# stage A: optimiser and schedule, one knob at a time (each differs from base in exactly one setting)
register(
    "v19-optA",
    _grid(
        {
            "base": ([], {}),
            "floor0": (["--lr-floor", "0.0"], {}),
            "floor3": (["--lr-floor", "0.3"], {}),
            "b2-95": (["--beta2", "0.95"], {}),
            "b2-999": (["--beta2", "0.999"], {}),
            "clip03": (["--clip", "0.3"], {}),
            "clip0": (["--clip", "0"], {}),
            "wu30": ([], {"warmup": "30"}),
            "wu300": ([], {"warmup": "300"}),
            "lr4": ([], {"lr": "4e-2"}),
            "lr9": ([], {"lr": "9e-2"}),
            "wd0": (["--wd", "0.0"], {}),
            "wd1": (["--wd", "0.1"], {}),
        }
    ),
)


# small CPU screen (user-approved to run beside the GPU job): four optimiser knobs, seeds 0-1, paired with the pilot's base_s0/s1
_SMALL = {
    "floor0": (["--lr-floor", "0.0"], {}),
    "b2-95": (["--beta2", "0.95"], {}),
    "lr4": ([], {"lr": "4e-2"}),
    "lr9": ([], {"lr": "9e-2"}),
}
register(
    "v19-small",
    [_cell(t, s, ex, **ck) for t, (ex, ck) in _SMALL.items() for s in (0, 1)],
)
