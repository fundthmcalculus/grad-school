"""Blinded set for decoding strategies at fixed weights: python -m flm.coherence_decode out ckpt [--seeds 4,5]

Strategies: A T0.5/top-k10, B T0.7/top-p0.9, C = A + repetition penalty (last 64 tokens, 1.3),
D = A + 4-gram block. Each: 4 prompts x seeds, cut to 600 chars. Writes out_blind.json / out_key.json.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from flm.bpe import BPE
from flm.models import ModelConfig, TinyLM

PROMPTS = [
    "Once upon a time",
    "Lily and Ben went to the park.",
    "The little dog was sad because",
    "One day, a girl named Mia found",
]
CHARS = 600


def sample_next(logits, history, strat, g):
    logits = logits.clone()
    if strat in ("C",):
        for t in set(history[-64:]):
            logits[t] = logits[t] / 1.3 if logits[t] > 0 else logits[t] * 1.3
    if strat == "D" and len(history) >= 3:
        tail = tuple(history[-3:])
        for i in range(len(history) - 3):
            if tuple(history[i : i + 3]) == tail:
                logits[history[i + 3]] = -float("inf")
    if strat == "B":
        logits = logits / 0.7
        p = torch.softmax(logits, -1)
        sp, si = torch.sort(p, descending=True)
        cut = (torch.cumsum(sp, 0) - sp) > 0.9
        sp[cut] = 0
        sp = sp / sp.sum()
        return int(si[torch.multinomial(sp, 1, generator=g)])
    logits = logits / 0.5
    v, _ = torch.topk(logits, 10)
    logits[logits < v[-1]] = -float("inf")
    return int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))


@torch.no_grad()
def gen(model, bp, prompt, seed, strat):
    g = torch.Generator().manual_seed(seed)
    hist = [int(t) for t in bp.encode(prompt)]
    text = bp.decode(hist)
    while len(text) < CHARS + 40:
        ctx = torch.tensor(hist[-model.cfg.max_len :])[None]
        hist.append(sample_next(model(ctx)[0, -1], hist, strat, g))
        text = bp.decode(hist)
    return text[:CHARS]


def main():
    out, ck_path = sys.argv[1], sys.argv[2]
    seeds = (4, 5)
    torch.set_num_threads(4)
    ck = torch.load(ck_path, map_location="cpu")
    m = TinyLM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["state"])
    m.eval()
    bp = BPE(int(ck["tok"].removeprefix("bpe")))
    items = []
    for strat in "ABCD":
        for p in PROMPTS:
            for s in seeds:
                items.append(
                    {
                        "strategy": strat,
                        "prompt": p,
                        "seed": s,
                        "text": gen(m, bp, p, s, strat),
                    }
                )
        print("done", strat, flush=True)
    random.Random(11).shuffle(items)
    for i, it in enumerate(items):
        it["id"] = f"S{i + 1:03d}"
    Path(out + "_blind.json").write_text(
        json.dumps([{"id": it["id"], "text": it["text"]} for it in items], indent=1)
    )
    Path(out + "_key.json").write_text(
        json.dumps(
            {
                it["id"]: {k: it[k] for k in ("strategy", "prompt", "seed")}
                for it in items
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
