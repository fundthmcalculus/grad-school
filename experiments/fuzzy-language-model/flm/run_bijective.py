"""``python -m flm.run_bijective <train args>``: flm.train with an invertible order-1 TSK meaning layer between the token codebook and the logic layer (`PLAN_BIJECTIVE.md`).

Arms (env ``BIJ``):
  tied   the usual tied head: output scores are dot products with the embedding table (the reference).
  ident  a codebook C (V x d), distance likelihood  logit_v = -|h - C_v|^2 / (2 tau^2)  (the bijection replaced by the identity: the control that isolates what F adds).
  z      embedding z_v = F(C_v); likelihood in meaning space,  logit_v = -|h - F(C_v)|^2 / (2 tau^2).
  c      embedding z_v = F(C_v); the logic layer's output is routed back through the inverse, u = F^-1(h), likelihood in codebook space,  logit_v = -|u - C_v|^2 / (2 tau^2).
F is ``flm.bijective_tsk.FuzzyBijection(d, BIJ_R rules, BIJ_L couplings, s_max BIJ_SMAX)``.  The blocks are the all-fuzzy ones of ``run_allfuzzy`` (fuzzy norm, fuzzy gates, rule-gated projections,
raw-input first-order TSK feed-forward rules).  train.py / models.py are untouched.  The result file's training log carries ``tau`` and, for the bijective arms, the float32 round-trip recovery rate (see ``recovery``).
"""

import os
import sys

import torch
import torch.nn as nn

import flm.train as T
from flm.bijective_tsk import FuzzyBijection
from flm.fuzzy_parts import install
from flm.models import TinyLM

BIJ = os.environ.get("BIJ", "tied")
R = int(os.environ.get("BIJ_R", "4"))
L = int(os.environ.get("BIJ_L", "2"))
SMAX = float(os.environ.get("BIJ_SMAX", "0.75"))
FFN_R = int(os.environ.get("FUZZ_FFN_R", "12"))
FFN_R0 = int(os.environ.get("FUZZ_FFN_R0", "8"))
GATE_R = int(os.environ.get("FUZZ_GATE_R", "4"))
PROJ_RANKS = tuple(int(x) for x in os.environ.get("FUZZ_PROJ_RANKS", "8,6").split(","))
PROJ_R = int(os.environ.get("FUZZ_PROJ_R", "4"))
_init = TinyLM.__init__


def _new_init(self, cfg, *a, **k):
    _init(self, cfg, *a, **k)
    install(
        self,
        "fuzzy",
        True,
        GATE_R,
        "raw",
        FFN_R,
        "tskl",
        2,
        FFN_R0,
        False,
        PROJ_RANKS,
        PROJ_R,
        "u",
        False,
    )
    if BIJ == "tied":
        return
    V, d = cfg.vocab_size, cfg.d_model
    self.codebook = nn.Parameter(torch.randn(V, d) * 0.5)
    self.log_tau = nn.Parameter(torch.tensor(0.0))
    del self.emb
    self.up = None
    if BIJ in ("z", "c"):
        self.F = FuzzyBijection(d, R, L, SMAX)


def _embed(self, idx):
    if BIJ == "tied":
        return self.emb(idx)
    c = self.codebook[idx]
    return self.F(c)[0] if BIJ in ("z", "c") else c


def _logits(self, h):
    if BIJ == "tied":
        return self.norm(h) @ self.emb.weight.T
    h = self.norm(h)
    if BIJ == "c":
        h, target = self.F.inverse(h), self.codebook
    elif BIJ == "z":
        target = self.F(self.codebook)[0]
    else:
        target = self.codebook
    d2 = (h * h).sum(-1, keepdim=True) - 2.0 * h @ target.T + (target * target).sum(-1)
    return -0.5 * d2 * torch.exp(-2.0 * self.log_tau)


@torch.no_grad()
def recovery(self):
    """Fraction of codewords recovered at the nearest-codeword decode after the float32 round trip F^-1(F(C)) (1.0 means the bijection holds in float32)."""
    if BIJ not in ("z", "c"):
        return None
    c = self.codebook.detach().float()
    back = self.F.float().inverse(self.F.float()(c)[0])
    return (torch.cdist(back, c).argmin(-1) == torch.arange(c.shape[0], device=c.device)).float().mean().item()


def _n_params(self, exclude_embedding=False):
    n = sum(p.numel() for p in self.parameters())
    if exclude_embedding:
        n -= (self.emb.weight if BIJ == "tied" else self.codebook).numel()
    return n


_evaluate = T.evaluate


def _evaluate_logged(model, *a, **k):
    out = _evaluate(model, *a, **k)
    if BIJ != "tied":
        rec = model.recovery()
        print(f"  tau {torch.exp(model.log_tau).item():.4f}" + (f"  float32 round-trip recovery {rec:.4f}" if rec is not None else ""), flush=True)
    return out


if __name__ == "__main__":
    T.evaluate = _evaluate_logged
    TinyLM.__init__ = _new_init
    TinyLM.embed = _embed
    TinyLM.logits = _logits
    TinyLM.n_params = _n_params
    TinyLM.recovery = recovery
    T.main(sys.argv[1:])
