"""Rule-view analysis: where the rule reading fails, what rules do, and whether rule usage tracks coherence.

    .venv/bin/python -m flm.rule_story [--ckpt outputs/v18-3b/....pt] [--out outputs/v18-3b/rule_story.json]

Q1/Q3: every rated text of the coherence sets with a key is replayed through its own checkpoint. Per text: m80 (mean number of rules
needed to cover 80% of the rule score per position and layer), usage entropy, rule-set self-similarity (mean largest Jaccard overlap of
a position's top-8 rule set with any earlier position's) and the 8-rule reading's top-1 agreement with the model, by story position and
by repetition (a token trigram already seen). Correlated with the mean rater total and plot, with a shuffled-score control.
Q2: for the 20 most-used rules per layer of the 3B model, label by the pieces the rule's effect favours and check the label is stable
across two disjoint halves of the validation windows.
Predictions P1-P4 are registered in RESULTS.md before this was run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

import numpy as np
import torch

from flm.coherence_auto import SETS, corr
from flm.rule_dist import decompose
from flm.rule_reader import reading
from flm.rules import load_model, make_codec

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
M = 8
EOT_TEXT = "\n<|eot|>\n"


def topset(score: torch.Tensor, m: int) -> list[frozenset]:
    """score (T, R) -> the top-m rule index set at every position."""
    return [frozenset(r) for r in score.topk(m, dim=-1).indices.tolist()]


def self_similarity(sets: list[frozenset]) -> float:
    """Mean over positions of the largest Jaccard overlap with any earlier position's set (0 when there is none)."""
    best = []
    for t in range(1, len(sets)):
        best.append(
            max(len(sets[t] & sets[s]) / len(sets[t] | sets[s]) for s in range(t))
        )
    return float(np.mean(best)) if best else 0.0


def m80(score: torch.Tensor) -> float:
    """Mean number of rules (largest first) needed to reach 80% of each position's total rule score."""
    srt = score.sort(dim=-1, descending=True).values
    cum = srt.cumsum(-1) / srt.sum(-1, keepdim=True).clamp_min(1e-12)
    return float((cum < 0.8).sum(-1).float().add(1).mean())


def entropy(score: torch.Tensor) -> float:
    p = score.sum(0)
    p = p / p.sum().clamp_min(1e-12)
    return float(-(p * p.clamp_min(1e-12).log()).sum() / math.log(2))


@torch.no_grad()
def analyse(model, ids: torch.Tensor) -> dict | None:
    """Rule statistics for one token sequence (1, T)."""
    try:
        D = decompose(model, ids[:, :-1])
    except AssertionError:
        return None
    tgt = ids[0, 1:]
    sets, scores = [], []
    for li, blk in enumerate(model.blocks):
        g = D["gates"][li][0]
        score = g.abs() * (blk.ffn.conseq * model.norm.weight).norm(dim=-1)
        scores.append(score)
        sets.append(topset(score, M))
    both = [
        frozenset(b + 10_000 * li for li, ls in enumerate(sets) for b in ls[t])
        for t in range(len(sets[0]))
    ]
    top1_full = D["logits"][0].argmax(-1)
    top1_read = reading(model, D, M, True)[0].argmax(-1)
    agree = (top1_full == top1_read).float()
    ids_l = ids[0].tolist()
    rep = torch.zeros(len(tgt))
    seen = set()
    for t in range(2, len(ids_l)):
        tri = tuple(ids_l[t - 2 : t + 1])
        if tri in seen and t - 1 < len(rep):
            rep[t - 1] = 1.0
        seen.add(tri)
    T = len(tgt)
    early = torch.arange(T) < 16
    return {
        "m80": float(np.mean([m80(s) for s in scores])),
        "entropy": float(np.mean([entropy(s) for s in scores])),
        "similarity": self_similarity(both),
        "agree": float(agree.mean()),
        "agree_early": float(agree[early].mean()) if early.any() else float("nan"),
        "agree_late": float(agree[~early].mean()) if (~early).any() else float("nan"),
        "agree_rep": float(agree[rep > 0].mean()) if (rep > 0).any() else float("nan"),
        "agree_norep": float(agree[rep == 0].mean()),
        "rep_share": float(rep.mean()),
        "T": T,
    }


def checkpoint_for(name: str) -> Path | None:
    hits = sorted((ROOT / "outputs").glob(f"*/{name}.pt"))
    return hits[0] if hits else None


def q3(args) -> dict:
    cache: dict[str, tuple] = {}
    rows = []
    for s, files in SETS.items():
        stem = "coherence" + ("" if s == "1" else s)
        blind = {
            r["id"]: r["text"]
            for r in json.loads((REPORT / f"{stem}_blind.json").read_text())
        }
        key = json.loads((REPORT / f"{stem}_key.json").read_text())
        scores = [json.loads((REPORT / f).read_text()) for f in files]
        for i, text in blind.items():
            name = key[i]["model"]
            if name not in cache:
                ck = checkpoint_for(name)
                if ck is None:
                    cache[name] = None
                else:
                    model, tok = load_model(ck)
                    cache[name] = (model, make_codec(tok)) if tok else None
            if cache[name] is None:
                continue
            model, codec = cache[name]
            ids = torch.from_numpy(
                codec.bpe.encode(text.replace(EOT_TEXT, "\x1f")).astype(np.int64)
            )[None][:, : model.cfg.max_len]
            r = analyse(model, ids)
            if r is None:
                continue
            r.update(
                set=int(s),
                model=name,
                total=st.mean(sum(sc[i].values()) for sc in scores),
                plot=st.mean(sc[i]["P"] for sc in scores),
            )
            rows.append(r)
    out = {"n": len(rows), "models": sorted({r["model"] for r in rows})}
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rows))
    out["corr"] = {}
    tot = [r["total"] for r in rows]
    nulls = {
        k: [corr([r[k] for r in rows], list(rng.permutation(tot))) for _ in range(500)]
        for k in ("m80", "entropy", "similarity", "agree", "rep_share")
    }
    for k in ("m80", "entropy", "similarity", "agree", "rep_share"):
        x = [r[k] for r in rows]
        out["corr"][k] = {
            "total": round(corr(x, [r["total"] for r in rows]), 3),
            "plot": round(corr(x, [r["plot"] for r in rows]), 3),
            "shuffled_total": round(corr(x, [rows[j]["total"] for j in perm]), 3),
            "null_sd": round(float(np.std(nulls[k])), 3),
        }
    for key in ("agree_early", "agree_late", "agree_rep", "agree_norep"):
        v = [r[key] for r in rows if r[key] == r[key]]
        out[key] = round(float(np.mean(v)), 4)
    out["rows"] = rows
    return out


@torch.no_grad()
def q2(ckpt: Path, windows: int = 128, length: int = 256, top_rules: int = 20) -> dict:
    model, tok = load_model(ckpt)
    codec = make_codec(tok)
    idx = codec.windows(windows, length + 1)
    D = decompose(model, idx[:, :-1])
    tgt = idx[:, 1:]
    V = model.cfg.vocab_size
    half = windows // 2
    res = {}
    for li, blk in enumerate(model.blocks):
        g, pi, s = D["gates"][li], D["pis"][li], D["s"][..., 0]
        score = g.abs() * (blk.ffn.conseq * model.norm.weight).norm(dim=-1)
        use = torch.bincount(score.argmax(-1).reshape(-1), minlength=g.shape[-1])
        rules = []
        for r in use.topk(top_rules).indices.tolist():
            eff = (
                g[..., r] * s * pi[r][tgt]
            )  # (B,T) signed effect on the chosen next piece
            tops = []
            for a, b in ((0, half), (half, windows)):
                tot = torch.zeros(V)
                tot.index_add_(
                    0, tgt[a:b].reshape(-1), eff[a:b].clamp_min(0).reshape(-1)
                )
                tops.append(set(tot.topk(10).indices.tolist()))
            jac = len(tops[0] & tops[1]) / len(tops[0] | tops[1])
            sign = 1.0 if g[..., r].mean() >= 0 else -1.0
            rules.append(
                {
                    "rule": r,
                    "wins": int(use[r]),
                    "raises": [
                        codec.bpe.decode([t])
                        for t in (pi[r] * sign).topk(6).indices.tolist()
                    ],
                    "favours": [
                        codec.bpe.decode([t]) for t in sorted(tops[0] & tops[1])
                    ][:8],
                    "jaccard": round(jac, 3),
                    "stable": jac >= 0.3,
                }
            )
        res[f"layer{li}"] = {
            "stable_share": round(sum(x["stable"] for x in rules) / len(rules), 3),
            "rules": rules,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default=str(ROOT / "outputs/v18-3b/D-d48-bpe1024-ctx512-r32_3000M_s0.pt"),
    )
    ap.add_argument("--out", default=str(ROOT / "outputs/v18-3b/rule_story.json"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    r3 = q3(a)
    print(f"{r3['n']} texts, models: {len(r3['models'])}")
    print("metric       r(total)  r(plot)  r(shuffled total)  null sd (500 shuffles)")
    for k, v in r3["corr"].items():
        print(
            f"{k:12s} {v['total']:+.2f}     {v['plot']:+.2f}    {v['shuffled_total']:+.2f}"
        )
    print(
        "reading top-1 agreement: first 16 positions %.3f, later %.3f; repeating %.3f, not repeating %.3f"
        % (r3["agree_early"], r3["agree_late"], r3["agree_rep"], r3["agree_norep"])
    )
    r2 = q2(Path(a.ckpt))
    for k, v in r2.items():
        print(f"{k}: stable share {v['stable_share']}")
    Path(a.out).write_text(
        json.dumps(
            {"q3": {k: v for k, v in r3.items() if k != "rows"}, "q2": r2}, indent=1
        )
    )
    (Path(a.out).with_suffix(".rows.json")).write_text(json.dumps(r3["rows"]))


if __name__ == "__main__":
    main()
