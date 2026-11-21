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
Stage 6b (`v56-embed2`, from the literature review `literature/04-fuzzy-embeddings.md`):
  FUZZ_TOK_BIAS = head             a learned per-token output bias for the head tokens, initialised at the log unigram probability (+NH parameters)
  FUZZ_TOK_GAIN = 1                a learned output gain and a per-head-token amplitude on the head logits (1 + NH parameters), lifting the bound that a convex combination puts on the scores
  FUZZ_TOK_UNTIE = 1              the head's input and output prototype matrices are separate (+d*d parameters)
  FUZZ_TOK_MOS = M                 a mixture of M softmaxes (low-rank per-component state shift, rank FUZZ_TOK_MOS_RANK, plus a gate)
  FUZZ_TOK_PQ = G                  product-quantised head: G groups of d/G prototypes, a membership vector per group, prototype matrix G*(d/G)*(d/G)
  FUZZ_TOK_HEADRANK = K            the crisp head tokens live in K dims and share one up-projection (like the tail): NH*K + K*d parameters instead of NH*d (`PLAN_SHAPE.md` E7: a smaller shared embedding)
  FUZZ_TOK_MID = "NH2,E1"          a middle tier (`PLAN_SHAPE.md` E5): the NH2 tokens after the head get their own fuzzy sets over E1 prototypes (own up-projection); the tail keeps width TIER_E and shrinks by NH2 rows
  FUZZ_TOK_TIER = 1                a soft tier gate: P(token) = P(tier | h) * P(token | tier), the tier probability a sigmoid of a linear function of h (d + 1 parameters)
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
BIAS = os.environ.get("FUZZ_TOK_BIAS", "")
GAIN = os.environ.get("FUZZ_TOK_GAIN", "") == "1"
TIER = os.environ.get("FUZZ_TOK_TIER", "") == "1"
UNTIE = os.environ.get("FUZZ_TOK_UNTIE", "") == "1"
MOS = int(os.environ.get("FUZZ_TOK_MOS", "0"))
MOS_RANK = int(os.environ.get("FUZZ_TOK_MOS_RANK", "8"))
PQ = int(os.environ.get("FUZZ_TOK_PQ", "0"))
HEADRANK = int(os.environ.get("FUZZ_TOK_HEADRANK", "0"))
MID = (
    tuple(int(x) for x in os.environ["FUZZ_TOK_MID"].split(","))
    if os.environ.get("FUZZ_TOK_MID")
    else ()
)


def _mu(logits):
    return torch.softmax(logits, dim=-1) if FORM == "softmax" else torch.tanh(logits)


def fuzzify_init(self, mode: str):
    d = self.cfg.d_model
    self.tok_mode = mode
    if mode != "crisp":
        nn.init.normal_(self.emb_t.weight, std=1.0)
    if mode == "all":
        nn.init.normal_(self.emb_h.weight, std=1.0)
        if PQ:
            g, k = PQ, d // PQ
            self.head_kind = "pq"
            self.head_pq = nn.Parameter(torch.randn(g, k, d // g) / k**0.5)
        else:
            self.head_kind = "proto"
            self.head_proto = nn.Parameter(torch.randn(d, d) / d**0.5)
            if UNTIE:
                self.head_proto_out = nn.Parameter(torch.randn(d, d) / d**0.5)
        if RES:
            self.head_res = nn.Embedding(R.NH, RES)
            self.head_res_up = nn.Parameter(torch.randn(RES, d) * 0.1)
            nn.init.normal_(self.head_res.weight, std=0.1)
    if (
        MID and mode == "tail"
    ):  # three tiers: head (crisp, width d), middle (fuzzy sets over E1 prototypes), tail (over E prototypes)
        nm, e1 = MID
        mid_ids = self.t_tail_ids[:nm]
        tail_ids = self.t_tail_ids[nm:]
        pos = self.t_pos.clone()
        pos[mid_ids] = torch.arange(nm)
        pos[tail_ids] = torch.arange(len(tail_ids))
        self.t_pos = pos
        self.t_tail_ids = tail_ids
        self.register_buffer("t_mid_ids", mid_ids, persistent=False)
        is_mid = torch.zeros(self.cfg.vocab_size, dtype=torch.bool)
        is_mid[mid_ids] = True
        self.register_buffer("t_mid", is_mid, persistent=False)
        self.emb_t = nn.Embedding(len(tail_ids), self.emb_t.embedding_dim)
        nn.init.normal_(self.emb_t.weight, std=1.0)
        self.emb_m = nn.Embedding(nm, e1)
        self.mid_up = nn.Linear(e1, d, bias=False)
        nn.init.normal_(self.emb_m.weight, std=1.0)
    if (
        HEADRANK and mode != "all"
    ):  # crisp head tokens through a rank-K bottleneck with their own up-projection
        self.emb_h = nn.Embedding(R.NH, HEADRANK)
        self.head_up = nn.Linear(HEADRANK, d, bias=False)
        nn.init.normal_(self.emb_h.weight, std=0.1)
        nn.init.normal_(self.head_up.weight, std=HEADRANK**-0.5)
    if PRIOR or BIAS or TIER:
        freq = np.load(os.environ["TIER_FREQ"])["freq"].astype(np.float64)
        logp = np.log((freq + 1.0) / (freq + 1.0).sum())
        lp = torch.tensor(logp, dtype=torch.float32)
        self.register_buffer("tok_logp", lp, persistent=False)
        if PRIOR:
            self.prior_alpha = nn.Parameter(torch.ones(()))
        if BIAS:
            self.tok_bias = nn.Parameter(lp[self.t_head_ids].clone())
        if TIER:
            share = float(np.exp(logp)[self.t_head_ids.cpu().numpy()].sum())
            self.tier_w = nn.Parameter(torch.zeros(d))
            self.tier_b = nn.Parameter(
                torch.tensor(np.log(share / (1 - share)), dtype=torch.float32)
            )
    if GAIN:
        self.out_gain = nn.Parameter(torch.ones(()))
        self.head_amp = nn.Parameter(torch.ones(R.NH))
    if MOS:
        self.mos_down = nn.Parameter(torch.randn(MOS, d, MOS_RANK) * 0.1)
        self.mos_up = nn.Parameter(torch.randn(MOS, MOS_RANK, d) * 0.1)
        self.mos_pi = nn.Parameter(torch.zeros(d, MOS))


def _embed(self, idx):
    if self.tok_mode == "crisp":
        return R._embed(self, idx)
    p = self.t_pos[idx]
    mt = _mu(self.emb_t(p.clamp(max=self.emb_t.num_embeddings - 1)))
    xt = self.tier_up(mt)
    mh = self.emb_h(p.clamp(max=R.NH - 1))
    kind = getattr(self, "head_kind", None)
    if kind == "pq":
        g, k, dg = self.head_pq.shape
        mu = _mu(mh.reshape(*mh.shape[:-1], g, k))
        xh = torch.einsum("...gk,gkc->...gc", mu, self.head_pq).reshape(
            *mh.shape[:-1], g * dg
        )
    elif kind == "proto":
        xh = _mu(mh) @ self.head_proto.T
        if hasattr(self, "head_res"):
            xh = xh + self.head_res(p.clamp(max=R.NH - 1)) @ self.head_res_up
    else:
        xh = self.head_up(mh) if hasattr(self, "head_up") else mh
    if hasattr(self, "emb_m"):  # the middle tier
        xm = self.mid_up(_mu(self.emb_m(p.clamp(max=self.emb_m.num_embeddings - 1))))
        xt = torch.where(self.t_mid[idx][..., None], xm, xt)
    if kind == "pq" and hasattr(self, "head_res"):
        xh = xh + self.head_res(p.clamp(max=R.NH - 1)) @ self.head_res_up
    return torch.where(self.t_head[idx][..., None], xh, xt)


def _logits(self, h):
    if not MOS:
        return _core(self, h)
    logpi = torch.log_softmax(h @ self.mos_pi, dim=-1)
    comps = []
    for j in range(MOS):
        hj = h + (h @ self.mos_down[j]) @ self.mos_up[j]
        comps.append(torch.log_softmax(_core(self, hj), dim=-1) + logpi[..., j : j + 1])
    return torch.logsumexp(torch.stack(comps, 0), dim=0)


def _core(self, h):
    hn = None  # the normalised state, when this call computed it
    if self.tok_mode == "crisp":
        out = R._logits(self, h)
    else:
        h = hn = self.norm(h)
        mt = _mu(self.emb_t.weight)  # (V-NH, E) memberships of the tail tokens
        lt = (h @ self.tier_up.weight) @ mt.T
        kind = getattr(self, "head_kind", None)
        if kind == "pq":
            g, k, dg = self.head_pq.shape
            sc = torch.einsum(
                "...gc,gkc->...gk", h.reshape(*h.shape[:-1], g, dg), self.head_pq
            )
            lh = torch.einsum(
                "...gk,vgk->...v", sc, _mu(self.emb_h.weight.reshape(-1, g, k))
            )
            if hasattr(self, "head_res"):
                lh = lh + (h @ self.head_res_up.T) @ self.head_res.weight.T
        elif kind == "proto":
            out_proto = self.head_proto_out if UNTIE else self.head_proto
            lh = (h @ out_proto) @ _mu(self.emb_h.weight).T
            if hasattr(self, "head_res"):
                lh = lh + (h @ self.head_res_up.T) @ self.head_res.weight.T
        elif hasattr(self, "head_up"):
            lh = (h @ self.head_up.weight) @ self.emb_h.weight.T
        else:
            lh = h @ self.emb_h.weight.T
        out = h.new_empty(*h.shape[:-1], self.cfg.vocab_size)
        out[..., self.t_head_ids] = lh.to(
            out.dtype
        )  # autocast: lh is bf16/fp16, the buffer fp32
        out[..., self.t_tail_ids] = lt.to(out.dtype)
        if hasattr(self, "emb_m"):
            out[..., self.t_mid_ids] = (
                (h @ self.mid_up.weight) @ _mu(self.emb_m.weight).T
            ).to(out.dtype)
    if PRIOR:
        out = out + self.prior_alpha * self.tok_logp
    if GAIN or BIAS or TIER:
        hi, ti = self.t_head_ids, self.t_tail_ids
        lh, lt = out[..., hi], out[..., ti]
        if GAIN:
            lh = self.out_gain * self.head_amp * lh
        if BIAS:
            lh = lh + self.tok_bias
        if TIER:
            g = (self.norm(h) if hn is None else hn) @ self.tier_w + self.tier_b
            lh = (
                torch.log_softmax(lh, -1) + torch.nn.functional.logsigmoid(g)[..., None]
            )
            lt = (
                torch.log_softmax(lt, -1)
                + torch.nn.functional.logsigmoid(-g)[..., None]
            )
        out = out.index_copy(-1, hi, lh).index_copy(-1, ti, lt)
    return out


def n_params(self, exclude_embedding=False):
    return R._n_params(self, exclude_embedding)
