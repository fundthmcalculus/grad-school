"""Validation bits per character of one checkpoint on several corpora, and a greedy blind coherence set.

    BIJ=ci FUZZ_FFN_R=196 FUZZ_FFN_R0=137 python -m flm.eval_cross OUT.json NAME=CKPT [NAME=CKPT ...]

Corpora (validation streams of the same 98-symbol alphabet): TinyStories, the Pile sample, Cosmopedia v2.  2,000 random windows of 256 characters each, the same windows for every checkpoint.  Set BIJ and the rule counts for the checkpoints' shape (one shape per call).
"""

import json
import math
import sys

import numpy as np
import torch

from flm.data import decode, encode
from flm.explain_bijective import CTX, PROMPTS, build

DATA = "C:/personal/grad-school/data"
CORPORA = dict(
    tinystories=f"{DATA}/tinystories/valid.bin",
    pile=f"{DATA}/pile_chars/tinystories/valid.bin",
    cosmopedia=f"{DATA}/cosmopedia/tinystories/valid.bin",
)


def windows(path, n=2000, seed=0):
    v = np.memmap(path, dtype=np.uint8, mode="r")
    starts = np.random.default_rng(seed).integers(0, len(v) - CTX - 1, size=n)
    return np.stack([v[s : s + CTX + 1] for s in starts]).astype(np.int64)


@torch.no_grad()
def bpc(model, w, dev):
    tot = 0.0
    for i in range(0, len(w), 100):
        b = torch.as_tensor(w[i : i + 100], device=dev)
        lg = model(b[:, :-1]).float()
        tot += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1), reduction="sum").item()
    return tot / (w.shape[0] * CTX) / math.log(2)


@torch.no_grad()
def greedy(model, dev, n=200):
    out = {}
    for p in PROMPTS + ["Photosynthesis is the process by which", "Once upon a time, in a small village,"]:
        ids = torch.as_tensor(encode(p)[None], device=dev).long()
        for _ in range(n):
            ids = torch.cat([ids, model(ids[:, -CTX:])[:, -1].argmax(-1, keepdim=True)], 1)
        out[p] = decode(ids[0].cpu().numpy())[len(p) :]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wins = {k: windows(p) for k, p in CORPORA.items()}
    res = {}
    for a in sys.argv[2:]:
        name, path = a.split("=", 1)
        model = build(path, dev)
        res[name] = dict(bpc={k: bpc(model, w, dev) for k, w in wins.items()}, greedy=greedy(model, dev))
        print(name, {k: round(v, 4) for k, v in res[name]["bpc"].items()}, flush=True)
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
