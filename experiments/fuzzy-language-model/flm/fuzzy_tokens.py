"""Fuzzy token tables for the frequency-tiered tied embedding (PLAN_FUZZYLOCAL.md, stage 5).

A token is a fuzzy set over K prototype vectors: its row of the table holds membership LOGITS, its embedding is the membership-weighted average
of the prototypes (fuzzification), and because the output matrix is tied, its output score is the same weighted average of the prototype scores
(defuzzification): score_v = sum_k mu_vk s_k with s_k = (prototype k) . (final hidden state).  Parameter count is unchanged for the tail tokens
(the K prototypes are the columns of the existing up-projection, the membership logits are the existing tail embeddings) and +d*d for the head
tokens (``all``: one extra prototype matrix; the run trims the FFN to stay under the cap).

``mode``: "tail" (only the E-wide tail tokens) or "all" (head tokens too: d memberships over d prototypes).
"""

import torch
import torch.nn as nn

import flm.run_tiered as R

MODE = "tail"


def fuzzify_init(self, mode: str):
    d = self.cfg.d_model
    nn.init.normal_(self.emb_t.weight, std=1.0)
    if mode == "all":
        nn.init.normal_(self.emb_h.weight, std=1.0)
        self.head_proto = nn.Parameter(torch.randn(d, d) / d**0.5)


def _embed(self, idx):
    p = self.t_pos[idx]
    mt = torch.softmax(self.emb_t(p.clamp(max=self.emb_t.num_embeddings - 1)), dim=-1)
    xt = self.tier_up(mt)
    mh = self.emb_h(p.clamp(max=R.NH - 1))
    xh = (
        (torch.softmax(mh, dim=-1) @ self.head_proto.T)
        if hasattr(self, "head_proto")
        else mh
    )
    return torch.where(self.t_head[idx][..., None], xh, xt)


def _logits(self, h):
    h = self.norm(h)
    mt = torch.softmax(
        self.emb_t.weight, dim=-1
    )  # (V-NH, E) memberships of the tail tokens
    lt = (h @ self.tier_up.weight) @ mt.T
    if hasattr(self, "head_proto"):
        lh = (h @ self.head_proto) @ torch.softmax(self.emb_h.weight, dim=-1).T
    else:
        lh = h @ self.emb_h.weight.T
    out = h.new_empty(*h.shape[:-1], self.cfg.vocab_size)
    out[..., self.t_head_ids] = lh
    out[..., self.t_tail_ids] = lt
    return out
