"""``python -m flm.run_tiered <train args>``: flm.train with a frequency-tiered tied embedding (`v30-tiered`).

The TIER_NH most frequent tokens keep a full d_model-wide embedding; the rest live in TIER_E dims and share one up-projection
to d_model.  The tied output follows the same split: head logits are h . E_head, tail logits are (h W_up) . E_tail (rank TIER_E).
Token frequency comes from the cached bigram counts (outputs/v22-tribble/ppmi_k48_v1024.npz).  train.py / models.py are untouched.
"""

import os

import numpy as np
import torch
import torch.nn as nn

import flm.train as T
from flm.models import TinyLM

NH = int(os.environ.get("TIER_NH", "256"))
E = int(os.environ.get("TIER_E", "32"))
_init = TinyLM.__init__


def _new_init(self, cfg, *a, **k):
    _init(self, cfg, *a, **k)
    V, d = cfg.vocab_size, cfg.d_model
    freq = np.load("outputs/v22-tribble/ppmi_k48_v1024.npz")["freq"]
    order = np.argsort(-freq, kind="stable")
    head, tail = order[:NH], order[NH:]
    pos = np.zeros(V, dtype=np.int64)
    pos[head], pos[tail] = np.arange(NH), np.arange(V - NH)
    is_head = np.zeros(V, dtype=bool)
    is_head[head] = True
    del self.emb
    self.up = None
    self.emb_h = nn.Embedding(NH, d)
    self.emb_t = nn.Embedding(V - NH, E)
    self.tier_up = nn.Linear(E, d, bias=False)
    nn.init.normal_(self.emb_h.weight, std=0.1)
    nn.init.normal_(self.emb_t.weight, std=0.1)
    self.register_buffer("t_pos", torch.from_numpy(pos), persistent=False)
    self.register_buffer("t_head", torch.from_numpy(is_head), persistent=False)
    self.register_buffer("t_head_ids", torch.from_numpy(head), persistent=False)
    self.register_buffer("t_tail_ids", torch.from_numpy(tail), persistent=False)


def _embed(self, idx):
    p = self.t_pos[idx]
    xh = self.emb_h(p.clamp(max=NH - 1))
    xt = self.tier_up(self.emb_t(p.clamp(max=self.emb_t.num_embeddings - 1)))
    return torch.where(self.t_head[idx][..., None], xh, xt)


def _logits(self, h):
    h = self.norm(h)
    lh = h @ self.emb_h.weight.T
    lt = (h @ self.tier_up.weight) @ self.emb_t.weight.T
    out = h.new_empty(*h.shape[:-1], self.cfg.vocab_size)
    out[..., self.t_head_ids] = lh
    out[..., self.t_tail_ids] = lt
    return out


def _n_params(self, exclude_embedding=False):
    n = sum(p.numel() for p in self.parameters())
    if exclude_embedding:
        n -= sum(
            p.numel()
            for p in (self.emb_h.weight, self.emb_t.weight, self.tier_up.weight)
        )
    return n


if __name__ == "__main__":
    import sys

    TinyLM.__init__ = _new_init
    TinyLM.embed = _embed
    TinyLM.logits = _logits
    TinyLM.n_params = _n_params
    T.main(sys.argv[1:])
