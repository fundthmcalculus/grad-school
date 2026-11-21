"""Grids for PLAN_GPU.md: make the goal-track claim citable, then probe what scales it.

All cells are CUDA (``cell()`` from ``flm.grids_v3`` appends ``--device cuda``; the sweep's
own ``--device`` flag still wins). Three grids select their settings from committed run
records, so each step reads the winners of the step before it; **an empty grid means its
prerequisite has not run** (``--list`` prints ``0 jobs``), never that there is nothing to do.

    v4-ref-tune   equal-effort search for the neural reference (64 cells, seed 0, 10M chars)
    v4-final10    D vs the two best tuned references vs the old reference, ten seeds, 30M chars
    v4-scale      does the lead hold at 36K / 63K / 187K parameters (36 cells, 30M chars)
    v4-taps       n-gram init and a 2-tap antecedent on D (21 cells, 10M chars, 3 seeds)
"""

from __future__ import annotations

import json
from pathlib import Path

from flm.grids_v3 import REF, D, cell
from flm.sweep import OUT, register

from flm.ref_space import NEURAL, REF_LRS, SHORTCONVS  # noqa: E402

MM = json.loads((Path(__file__).resolve().parent / "matched_mults.json").read_text())


def neural_cfg(name: str, sc: int, d: int = 72, L: int = 2):
    """(argv, cell kwargs) for a neural reference whose MLP is sized to match D's parameters."""
    mult = MM["mult"][f"{name}|sc{sc}|d{d}|L{L}"]
    return NEURAL[name] + ["--ffn-mult", str(mult)], dict(d=d, L=L, sc=str(sc))


def half(lr: str) -> str:
    return f"{float(lr) / 2:g}"


# ---------------------------------------------------------------- v4-ref-tune
REF_META: dict[str, dict] = {}  # job name -> {name, sc, lr, L, d}
for _name in NEURAL:
    for _sc in SHORTCONVS:
        for _lr in REF_LRS:
            REF_META[f"{_name}-sc{_sc}-lr{_lr}"] = dict(
                name=_name, sc=_sc, lr=_lr, L=2, d=72
            )
for _sc in (
    4,
    8,
):  # depth was searched for the fuzzy model, so the reference gets 3 layers too
    for _lr in ("3e-2", "4e-2"):
        REF_META[f"delta-h8-L3-sc{_sc}-lr{_lr}"] = dict(
            name="delta-h8", sc=_sc, lr=_lr, L=3, d=56
        )


def ref_cell(job: str, chars: str, seed: int, label: str | None = None, **over):
    m = {**REF_META[job], **over}
    argv, kw = neural_cfg(m["name"], m["sc"], m["d"], m["L"])
    return cell(label or job, argv, chars=chars, seed=seed, lr=m["lr"], **kw)


register("v4-ref-tune", [ref_cell(j, "10e6", 0) for j in REF_META])


# ---------------------------------------------------------------- selection from records
def best_refs(k: int, l2_only: bool = False) -> list[str]:
    """The k best v4-ref-tune cells by val BPC (finite only); [] if the tune has not run."""
    scored = []
    for job, m in REF_META.items():
        f = OUT / "v4-ref-tune" / f"{job}.json"
        if l2_only and m["L"] != 2 or not f.exists():
            continue
        bpc = json.loads(f.read_text())["val_bpc"]
        if bpc == bpc:
            scored.append((bpc, job))
    return [j for _, j in sorted(scored)[:k]]


def best_d_lr(default: str = "3e-2") -> str:
    """D's lr: lowest mean val BPC over seeds 0-1 at 30M chars across v3-final (lr 3e-2) and
    v3-lr; only lrs with both seeds count. Falls back to 3e-2 if v3-lr has not run."""
    by_lr: dict[float, dict[int, float]] = {}
    for sub, pat in (("v3-final", "D_30M_s*.json"), ("v3-lr", "D_lr*_30M_s*.json")):
        for f in (OUT / sub).glob(pat):
            r = json.loads(f.read_text())
            if r["val_bpc"] == r["val_bpc"]:
                by_lr.setdefault(float(r["args"]["lr"]), {})[int(r["args"]["seed"])] = (
                    r["val_bpc"]
                )
    full = {lr: v for lr, v in by_lr.items() if {0, 1} <= set(v)}
    if len(full) < 2:
        return default
    return f"{min(full, key=lambda lr: full[lr][0] + full[lr][1]):g}"


# ---------------------------------------------------------------- v4-final10
def _final10():
    refs = best_refs(2)
    if not refs:
        return []
    lr = best_d_lr()
    jobs = []
    for s in range(10):
        jobs.append(
            cell(f"D-lr{lr}_30M_s{s}", D, d=72, sc="8", chars="30e6", seed=s, lr=lr)
        )
        for i, job in enumerate(refs, 1):
            jobs.append(ref_cell(job, "30e6", s, label=f"ref{i}-{job}_30M_s{s}"))
        # the reference from the goal-track write-up, for continuity (93K params, lr 3e-2, sc 4)
        jobs.append(
            cell(
                f"ref0-v3_30M_s{s}",
                REF + ["--heads", "8"],
                d=72,
                sc="4",
                chars="30e6",
                seed=s,
            )
        )
    return jobs


register("v4-final10", _final10())


# ---------------------------------------------------------------- v4-scale
def _scale():
    refs = best_refs(1, l2_only=True)
    if not refs:
        return []
    m = REF_META[refs[0]]
    d_lr = best_d_lr()
    jobs = []
    for d in (40, 56, 104):
        for s in (0, 1, 2):
            for lr in (d_lr, half(d_lr)):
                jobs.append(
                    cell(
                        f"D-d{d}-lr{lr}_s{s}",
                        D,
                        d=d,
                        sc="8",
                        chars="30e6",
                        seed=s,
                        lr=lr,
                    )
                )
            for lr in (m["lr"], half(m["lr"])):
                argv, kw = neural_cfg(m["name"], m["sc"], d, 2)
                jobs.append(
                    cell(
                        f"ref-{m['name']}-sc{m['sc']}-d{d}-lr{lr}_s{s}",
                        argv,
                        chars="30e6",
                        seed=s,
                        lr=lr,
                        **kw,
                    )
                )
    return jobs


register("v4-scale", _scale())


# ---------------------------------------------------------------- v4-taps
TAPS = {
    "base": [],
    "taps2": ["--ante-taps", "2"],
    "taps3": ["--ante-taps", "3"],
    "biemb": ["--ngram-init", "1", "--ngram-mode", "emb"],
    "biplace": ["--ngram-init", "1", "--ngram-mode", "placement"],
    "bifull": ["--ngram-init", "1", "--ngram-mode", "full"],
    "trifull": ["--ante-taps", "2", "--ngram-init", "2", "--ngram-mode", "full"],
}
_CURVE = ["--eval-chars", "300000", "--eval-every", "100", "--eval-at-start"]
register(
    "v4-taps",
    [
        cell(
            f"{v}_s{s}",
            D + extra + _CURVE,
            d=72,
            sc="8",
            chars="10e6",
            seed=s,
            lr="3e-2",
        )
        for v, extra in TAPS.items()
        for s in (0, 1, 2)
    ],
)
