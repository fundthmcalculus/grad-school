"""Subword fairness controls: a neural reference on the same tokenizer at D's parameter count.

The GPU session's best sub-100K model is the fuzzy D at BPE1024, d = 48 (93,344 params; 1.129 BPC at
30M chars). That is a statement about *subwords* until a neural model gets the same tokenizer at the
same size and the same tuning effort. These grids supply that control, cheaply (the neural arms are
~2x faster than D):

    v6-ref-tune    6 neural configs x 4 lrs, 10M chars, seed 0           (24 cells)
    v6-d-lr        D at the same 4 lrs, 10M chars, seeds 0-1             (8 cells; D's lr was
                   only ever 3e-2 on subwords, so it is tuned here with the same effort)
    v6-final10     D at its best lr vs the two best references, 30M chars, seeds 0-9 (30 cells)

``v6-final10`` reads the winners of the other two from committed records; an empty grid means a
prerequisite has not run. The MLP width of each neural config is padded so its parameter count
matches D's (vocabulary 1024 counted in the tied embedding): flm.match_mults.match(..., vocab=1024).
Written and arg-parsed on CPU by the CPU session; **not run end to end** (it needs the BPE stream), so
run one cell first.
"""

from __future__ import annotations

import json

from flm.grids_v3 import D, cell
from flm.ref_space import NEURAL
from flm.sweep import OUT, register

BPE = ["--tok", "bpe1024", "--warmup", "100"]
LRS = ("2e-2", "3e-2", "4e-2", "6e-2")
# (mixer config, short-conv) -> MLP width multiplier giving 93,376-93,392 params (D: 93,344)
MULT = {
    ("delta-h4", 4): 2.64,
    ("delta-h4", 8): 2.6,
    ("delta-h8", 4): 2.56,
    ("delta-h8", 8): 2.52,
    ("gla-h8", 4): 2.64,
    ("gla-h8", 8): 2.6,
}
REF_JOBS = {f"{n}-sc{sc}-lr{lr}": (n, sc, lr) for (n, sc) in MULT for lr in LRS}


def ref_cell(
    label: str, n: str, sc: int, lr: str, chars: str, seed: int, save: bool = False
):
    argv = (
        NEURAL[n]
        + ["--ffn-mult", str(MULT[(n, sc)])]
        + BPE
        + (["--save-model"] if save else [])
    )
    return cell(label, argv, d=48, sc=str(sc), lr=lr, chars=chars, seed=seed)


register("v6-ref-tune", [ref_cell(j, *m, "10e6", 0) for j, m in REF_JOBS.items()])
register(
    "v6-d-lr",
    [
        cell(f"D-lr{lr}_10M_s{s}", D + BPE, d=48, sc="8", lr=lr, chars="10e6", seed=s)
        for lr in LRS
        for s in (0, 1)
    ],
)


def best_refs(k: int) -> list[str]:
    scored = []
    for job in REF_JOBS:
        f = OUT / "v6-ref-tune" / f"{job}.json"
        if f.exists():
            bpc = json.loads(f.read_text())["val_bpc"]
            if bpc == bpc:
                scored.append((bpc, job))
    return [j for _, j in sorted(scored)[:k]]


def best_d_lr(default: str = "3e-2") -> str:
    """D's lr on subwords: lowest mean val BPC over seeds 0-1 at 10M chars; needs >= 2 complete lrs."""
    by_lr: dict[float, dict[int, float]] = {}
    for f in (OUT / "v6-d-lr").glob("D-lr*_10M_s*.json"):
        r = json.loads(f.read_text())
        if r["val_bpc"] == r["val_bpc"]:
            by_lr.setdefault(float(r["args"]["lr"]), {})[int(r["args"]["seed"])] = r[
                "val_bpc"
            ]
    full = {lr: v for lr, v in by_lr.items() if {0, 1} <= set(v)}
    return (
        default
        if len(full) < 2
        else f"{min(full, key=lambda lr: full[lr][0] + full[lr][1]):g}"
    )


def _final10():
    refs = best_refs(2)
    if not refs:
        return []
    lr = best_d_lr()
    jobs = []
    for s in range(10):
        jobs.append(
            cell(
                f"D-lr{lr}_30M_s{s}",
                D + BPE + (["--save-model"] if s == 0 else []),
                d=48,
                sc="8",
                lr=lr,
                chars="30e6",
                seed=s,
            )
        )
        for i, job in enumerate(refs, 1):
            jobs.append(
                ref_cell(
                    f"ref{i}-{job}_30M_s{s}", *REF_JOBS[job], "30e6", s, save=s == 0
                )
            )
    return jobs


register("v6-final10", _final10())
