"""``python -m flm.run_conseq <variant> <train args>``: flm.train with the TRIBBLE-order (consequent-first) rule init.

The variant replaces ``flm.train.init_rules_from_data`` (so ``--rule-init data`` is the switch; train.py is untouched):
  cf        answers = k-means of the model's own embedding, centroid consequents
  cfls      same groups, ridge least-squares consequents
  assocls   answers = k-means of PPMI-SVD successor vectors (association), least-squares consequents
  succcf    embedding SET to the PPMI-SVD successor vectors, then cf (centroid consequents)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import flm.train as T
from flm.assoc import bigram_counts, equal_mass_groups, ppmi_svd
from flm.conseq_first import init_conseq_first

VARIANTS = {
    "cf": dict(groups="emb", mode="centroid", emb=False),
    "cfg1": dict(groups="emb", mode="centroid", emb=False, gains=(0.25, 1.0)),
    "cfwide": dict(groups="emb", mode="centroid", emb=False, widths="global"),
    "cfw300": dict(
        groups="emb", mode="centroid", emb=False
    ),  # same init; the driver sets --warmup 300
    "hyb4": dict(groups="emb", mode="centroid", emb=False, keep=4),
    "hyb12": dict(groups="emb", mode="centroid", emb=False, keep=12),
    "hyb24": dict(groups="emb", mode="centroid", emb=False, keep=24),
    "hyb4g1": dict(groups="emb", mode="centroid", emb=False, keep=4, gains=(0.25, 1.0)),
    "hyb12g1": dict(
        groups="emb", mode="centroid", emb=False, keep=12, gains=(0.25, 1.0)
    ),
    "hyb18g1": dict(
        groups="emb", mode="centroid", emb=False, keep=18, gains=(0.25, 1.0)
    ),
    "cov25": dict(
        groups="emb", mode="centroid", emb=False, keep=18, gains=(0.25, 1.0), cover=True
    ),
    "cov0": dict(
        groups="emb", mode="centroid", emb=False, keep=0, gains=(0.25, 1.0), cover=True
    ),
    "cfls": dict(groups="emb", mode="ls", emb=False),
    "assocls": dict(groups="assoc", mode="ls", emb=False),
    "succcf": dict(groups="emb", mode="centroid", emb=True),
}


def _stats(k: int, V: int):
    cache = Path("outputs/v22-tribble") / f"ppmi_k{k}_v{V}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["pred"], z["succ"], z["freq"]
    from flm.bpe import BPE

    train = np.asarray(BPE(V).load("train")[:60_000_000])
    C = bigram_counts(train, V)
    pred, succ = ppmi_svd(C, k)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pred=pred, succ=succ, freq=C.sum(0))
    return pred, succ, C.sum(0)


def main():
    variant, argv = sys.argv[1], sys.argv[2:]
    cfg = VARIANTS[variant]

    def patched(model, idx, generator=None):
        d = model.cfg.d_model
        V = model.cfg.vocab_size
        _, succ, freq = _stats(d, V)
        if cfg["emb"]:
            w = torch.from_numpy(succ).float()
            model.emb.weight.data.copy_(w * (0.1 / w.pow(2).mean().sqrt()))
        stats = {
            "succ": torch.from_numpy(succ).float(),
            "freq_groups": {
                model.blocks[0].ffn.R: equal_mass_groups(freq, model.blocks[0].ffn.R)
            },
        }
        info = init_conseq_first(
            model,
            idx,
            generator,
            mode=cfg["mode"],
            groups=cfg["groups"],
            token_stats=stats,
            **({"gains": cfg["gains"]} if "gains" in cfg else {}),
            widths=cfg.get("widths", "group"),
            keep=cfg.get("keep"),
            cover=cfg.get("cover", False),
        )
        print("conseq-first init:", info, flush=True)
        return {"variant": variant, **{k: v for k, v in info.items()}}

    T.init_rules_from_data = patched
    T.main(argv)


if __name__ == "__main__":
    main()
