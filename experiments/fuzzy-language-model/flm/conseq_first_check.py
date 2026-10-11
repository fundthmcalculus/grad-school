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
    from flm.assoc import bigram_counts, equal_mass_groups, ppmi_svd

    V = a.vocab_size
    stream = np.asarray(train[:60_000_000])
    C = bigram_counts(stream, V)
    pred, succ = ppmi_svd(C, d)
    freq = C.sum(0)
    stats = {
        "succ": torch.from_numpy(succ).float(),
        "freq_groups": {72: equal_mass_groups(freq, 72)},
    }
    out = {}
    variants = [("random", None, None, "random"), ("data", None, None, "random")]
    for emb in ("random", "succ"):
        for grp in ("emb", "freq", "assoc"):
            for mode in ("centroid", "ls"):
                variants.append((f"{grp}/{mode}/emb={emb}", grp, mode, emb))
    for name, grp, mode, emb in variants:
        torch.manual_seed(seed)
        m = TinyLM(T.make_config(a))
        if emb == "succ":
            w = torch.from_numpy(succ).float()
            m.emb.weight.data.copy_(w * (0.1 / w.pow(2).mean().sqrt()))
        g = torch.Generator().manual_seed(seed)
        info = {}
        if name == "data":
            info = init_rules_from_data(m, idx[:, :-1], generator=g)
        elif grp is not None:
            info = init_conseq_first(
                m, idx, g, mode=mode, groups=grp, token_stats=stats
            )
        bpc = T.evaluate(m, val, ctx, eval_chars)
        out[name] = bpc
        print(f"{name:28s} init val BPC {bpc:.4f}", flush=True)
    return out


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:2]))
