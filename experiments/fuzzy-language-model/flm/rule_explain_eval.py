"""Where does the rule reading work? Hit rate of the 8-rule + context reading by token class and by model confidence.

    .venv/bin/python -m flm.rule_explain_eval CKPT.pt [--windows 128] [--m 8] [--out eval.json]

For every validation position: does the true next token appear in the top-10 of (a) the model, (b) the reading that keeps only
the m strongest rules per layer plus the embedding and mixer terms (flm.rule_reader)? Grouped by the class of the true token
(punctuation or newline, capitalised piece, one of the 50 most frequent pieces, other) and by the model's own probability for
it (below 0.05, 0.05-0.5, above 0.5). Registered before running (CPU session, 2026-10-12): the reading's hit rate is closest
to the model's on punctuation and frequent pieces and furthest on capitalised pieces (names) (65%); it is far below the model
in the low-probability bin (70%).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from flm.rule_dist import decompose
from flm.rule_reader import TOPN, reading
from flm.rules import load_model, make_codec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    model, tok = load_model(a.ckpt)
    codec = make_codec(tok)
    idx = codec.windows(a.windows, a.len + 1)
    tgt = idx[:, 1:]
    with torch.no_grad():
        D = decompose(model, idx[:, :-1])
        full_top = D["logits"].topk(TOPN, dim=-1).indices
        read_top = reading(model, D, a.m, True).topk(TOPN, dim=-1).indices
        prob = torch.softmax(D["logits"], -1).gather(-1, tgt[..., None])[..., 0]
    V = model.cfg.vocab_size
    freq = torch.bincount(idx.reshape(-1), minlength=V)
    frequent = set(freq.topk(50).indices.tolist())
    cls = {}
    for t in range(V):
        s = codec.bpe.decode([t])
        if not any(ch.isalnum() for ch in s):
            cls[t] = "punctuation / newline"
        elif s.strip()[:1].isupper():
            cls[t] = "capitalised piece"
        elif t in frequent:
            cls[t] = "frequent piece"
        else:
            cls[t] = "other piece"
    mh = (full_top == tgt[..., None]).any(-1)
    rh = (read_top == tgt[..., None]).any(-1)
    rows = defaultdict(lambda: [0, 0, 0])
    for t, m_, r_, p in zip(
        tgt.reshape(-1).tolist(),
        mh.reshape(-1).tolist(),
        rh.reshape(-1).tolist(),
        prob.reshape(-1).tolist(),
    ):
        for key in (
            cls[t],
            "p<0.05" if p < 0.05 else ("p 0.05-0.5" if p < 0.5 else "p>0.5"),
        ):
            rows[key][0] += 1
            rows[key][1] += m_
            rows[key][2] += r_
    res = {"ckpt": a.ckpt, "m": a.m, "groups": {}}
    print(
        f"{'group':24s} {'n':>7s} {'model hit@10':>13s} {'reading hit@10':>15s} {'gap':>7s}"
    )
    for k, (n, mm, rr) in sorted(rows.items()):
        res["groups"][k] = {
            "n": n,
            "model": round(mm / n, 3),
            "reading": round(rr / n, 3),
        }
        print(f"{k:24s} {n:7d} {mm / n:13.3f} {rr / n:15.3f} {(mm - rr) / n:7.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
