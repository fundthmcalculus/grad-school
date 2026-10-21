"""Fuzzy token tables for the frequency-tiered tied embedding (PLAN_FUZZYLOCAL.md, stage 5; forms and prior: stage 6, `v54-embed`).

A token is a fuzzy set over K prototype vectors: its row of the table holds membership LOGITS, its embedding is the membership-weighted average
of the prototypes (fuzzification), and because the output matrix is tied, its output score is the same weighted average of the prototype scores
(defuzzification): score_v = sum_k mu_vk s_k with s_k = (prototype k) . (final hidden state).  Parameter count is unchanged for the tail tokens
(the K prototypes are the columns of the existing up-projection, the membership logits are the existing tail embeddings) and +d*d for the head
tokens (``all``: one extra prototype matrix; the run trims the FFN to stay under the cap).

``mode``: "tail" (only the E-wide tail tokens), "all" (head tokens too: d memberships over d prototypes) or "crisp" (the tiered recipe unchanged;
only the optional prior below is applied, as a control).

Stage 6 options (environment; the defaults reproduce stage 5 exactly):
  FUZZ_TOK_FORM = softmax | tanh   membership function: a partition of unity, or signed memberships in (-1, 1) (a free, not convex, combination)
  FUZZ_TOK_RES  = r                head tokens also carry a crisp rank-r residual row on top of the fuzzy part (all mode)
  FUZZ_TOK_PRIOR = 1               add alpha * log unigram probability to every logit (alpha learned, initial 1; one parameter, no table)
"""

import os

import numpy as np
import torch
import torch.nn as nn

import flm.run_tiered as R

MODE = "tail"
FORM = os.environ.get("FUZZ_TOK_FORM", "softmax")
RES = int(os.environ.get("FUZZ_TOK_RES", "0"))
PRIOR = os.environ.get("FUZZ_TOK_PRIOR", "") == "1"


def _mu(logits):
    return torch.softmax(logits, dim=-1) if FORM == "softmax" else torch.tanh(logits)


def fuzzify_init(self, mode: str):
    d = self.cfg.d_model
    self.tok_mode = mode
    if mode != "crisp":
        nn.init.normal_(self.emb_t.weight, std=1.0)
    if mode == "all":
        nn.init.normal_(self.emb_h.weight, std=1.0)
        self.head_proto = nn.Parameter(torch.randn(d, d) / d**0.5)
        if RES:
            self.head_res = nn.Embedding(R.NH, RES)
            self.head_res_up = nn.Parameter(torch.randn(RES, d) * 0.1)
            nn.init.normal_(self.head_res.weight, std=0.1)
    if PRIOR:
        freq = np.load(os.environ["TIER_FREQ"])["freq"].astype(np.float64)
        logp = np.log((freq + 1.0) / (freq + 1.0).sum())
        self.register_buffer(
            "tok_logp", torch.tensor(logp, dtype=torch.float32), persistent=False
        )
        self.prior_alpha = nn.Parameter(torch.ones(()))


def _embed(self, idx):
    if self.tok_mode == "crisp":
        return R._embed(self, idx)
    p = self.t_pos[idx]
    mt = _mu(self.emb_t(p.clamp(max=self.emb_t.num_embeddings - 1)))
    xt = self.tier_up(mt)
    mh = self.emb_h(p.clamp(max=R.NH - 1))
    if hasattr(self, "head_proto"):
        xh = _mu(mh) @ self.head_proto.T
        if hasattr(self, "head_res"):
            xh = xh + self.head_res(p.clamp(max=R.NH - 1)) @ self.head_res_up
    else:
        xh = mh
    return torch.where(self.t_head[idx][..., None], xh, xt)


def _logits(self, h):
    if self.tok_mode == "crisp":
        out = R._logits(self, h)
    else:
        h = self.norm(h)
        mt = _mu(self.emb_t.weight)  # (V-NH, E) memberships of the tail tokens
        lt = (h @ self.tier_up.weight) @ mt.T
        if hasattr(self, "head_proto"):
            lh = (h @ self.head_proto) @ _mu(self.emb_h.weight).T
            if hasattr(self, "head_res"):
                lh = lh + (h @ self.head_res_up.T) @ self.head_res.weight.T
        else:
            lh = h @ self.emb_h.weight.T
        out = h.new_empty(*h.shape[:-1], self.cfg.vocab_size)
        out[..., self.t_head_ids] = lh
        out[..., self.t_tail_ids] = lt
    if PRIOR:
        out = out + self.prior_alpha * self.tok_logp
    return out


def n_params(self, exclude_embedding=False):
    return R._n_params(self, exclude_embedding)
