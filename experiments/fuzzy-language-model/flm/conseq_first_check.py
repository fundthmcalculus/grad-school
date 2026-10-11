"""Init-only comparison of rule initialisations on the sub-100K BPE1024 D model (forward passes only; no training).

random: the default init; data: antecedent-first (existing ``--rule-init data``); cf: consequent-first with
centroid consequents; cf-ls: consequent-first with ridge least-squares consequents.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import flm.train as T
from flm.bpe import BPE
from flm.conseq_first import init_conseq_first
from flm.grids_v3 import D
from flm.models import TinyLM, init_rules_from_data


def main(seed=0, d=48, windows=64, ctx=256, eval_chars=300_000):
    torch.set_num_threads(2)
    base = D + ["--ctx", str(ctx), "--rules", "32", "--shortconv", "8", "--out", "x"]
    a = T.get_args(base + ["--d", str(d), "--layers", "2", "--seed", str(seed)])
    T.TOK = tok = BPE(1024)
    a.vocab_size = tok.tok.get_vocab_size()
    train, val = tok.load("train"), tok.load("valid")
    rng = np.random.default_rng(100 + seed)
    starts = rng.integers(0, len(train) - ctx - 2, windows)
    idx = torch.from_numpy(
        np.stack([train[i : i + ctx + 1] for i in starts]).astype(np.int64)
    )
    out = {}
    for name in ("random", "data", "cf", "cf-ls"):
        torch.manual_seed(seed)
        m = TinyLM(T.make_config(a))
        g = torch.Generator().manual_seed(seed)
        info = {}
        if name == "data":
            info = init_rules_from_data(m, idx[:, :-1], generator=g)
        elif name in ("cf", "cf-ls"):
            info = init_conseq_first(
                m, idx, g, mode="centroid" if name == "cf" else "ls"
            )
        bpc = T.evaluate(m, val, ctx, eval_chars)
        out[name] = bpc
        print(f"{name:7s} init val BPC {bpc:.4f}  {info}", flush=True)
    return out


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:2]))
