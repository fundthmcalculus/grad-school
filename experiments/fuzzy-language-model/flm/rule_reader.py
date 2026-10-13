"""A calibrated rule reader: how many rules must be read per position before the reading is faithful?

    .venv/bin/python -m flm.rule_reader CKPT.pt [--windows 128] [--out reader.json]

``flm.rule_dist`` found that a rule's stated consequent, read alone for the dominant rule, contains the true next token
only 15-19% of the time. This asks the follow-up: take the exact direct-path decomposition of the logits, keep only the
m rules per position and layer with the largest |gate| x ||profile||, and measure how well the resulting reading predicts.
Two readings: ``rules only`` (just those rules' terms) and ``rules + context`` (those rules plus the embedding and mixer
terms, which are not rules). Metrics, all on the same windows: hit rate of the true next token in the top-10 of the
reading, overlap of the reading's top-10 with the model's own top-10, and the BPC of the reconstructed distribution (direct
path only; layer-0 rules' downstream effect is not included, as in rule_dist).

Registered before running (CPU session, 2026-10-12), on the 3B-character 96K checkpoint: (P1) with m = 8 rules + context the hit
rate is at least 0.70 (the model's own top-10 is about 0.81); (P2) with rules only and m = 8 it is below 0.50; (P3) it takes
m >= 32 rules per position and layer, plus context, to come within 0.03 of the model's own hit rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flm.rule_dist import bpc_of, decompose
from flm.rules import load_model, make_codec

TOPN = 10


@torch.no_grad()
def reading(model, D, m: int, context: bool):
    """Logits from the m highest-scoring rules per position and layer (+ embedding and mixer terms if ``context``)."""
    out = D["emb"] + sum(D["mix"]) if context else 0.0
    for li, blk in enumerate(model.blocks):
        ffn = blk.ffn
        g, pi = D["gates"][li], D["pis"][li]  # (B,T,R), (R,V)
        score = g.abs() * (ffn.conseq * model.norm.weight).norm(dim=-1)
        keep = score.topk(min(m, g.shape[-1]), dim=-1).indices
        mask = torch.zeros_like(g).scatter_(-1, keep, 1.0)
        out = out + ((g * mask) * D["s"]) @ pi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    model, tok = load_model(a.ckpt)
    codec = make_codec(tok)
    lens = np.asarray(codec.bpe.lens)
    idx = codec.windows(a.windows, a.len + 1)
    tgt = idx[:, 1:]
    D = decompose(model, idx[:, :-1])
    R = model.blocks[0].ffn.R
    full_top = D["logits"].topk(TOPN, dim=-1).indices
    full_hit = (full_top == tgt[..., None]).any(-1).float().mean().item()
    res = {
        "ckpt": a.ckpt,
        "rules": R,
        "model_hit_top10": round(full_hit, 4),
        "windows": a.windows,
        "rows": [],
    }
    print(
        f"model's own top-{TOPN} contains the true token {full_hit:.3f}; R = {R}; full BPC {bpc_of(D['logits'], tgt, lens):.4f}"
    )
    print("context  m  hit@10  overlap@10  BPC(direct)")
    for context in (False, True):
        for m in (1, 2, 4, 8, 16, 32, R):
            lg = reading(model, D, m, context)
            top = lg.topk(TOPN, dim=-1).indices
            hit = (top == tgt[..., None]).any(-1).float().mean().item()
            ov = np.mean(
                [
                    len(set(t.tolist()) & set(f.tolist())) / TOPN
                    for t, f in zip(
                        top.reshape(-1, TOPN)[::7], full_top.reshape(-1, TOPN)[::7]
                    )
                ]
            )
            bpc = bpc_of(lg, tgt, lens)
            res["rows"].append(
                dict(
                    context=context,
                    m=m,
                    hit=round(hit, 4),
                    overlap=round(float(ov), 4),
                    bpc=round(bpc, 4),
                )
            )
            print(f"{str(context):7s} {m:3d}  {hit:.3f}   {ov:.3f}      {bpc:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
