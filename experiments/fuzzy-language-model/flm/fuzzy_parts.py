"""Fuzzy inference replacements for the non-fuzzy parts of an FLM block (PLAN_FUZZYLOCAL.md, stages 2 and 4).

FuzzyNorm  replaces RMSNorm.  A first-order TSK system with ONE input variable, z = log rms(x), R Gaussian membership functions
           (fuzzification), the normalised rule firings (aggregation) and a log-gain output  g = sum_r w_r (b_r + k_r (z - c_r))
           (defuzzification); the layer returns  x * exp(g) * weight.  Initialised with b_r = -c_r and k_r = -1 the output is exactly
           -z for any memberships, i.e. exactly RMSNorm at step 0; training can then bend the gain curve (saturate, boost, ...).
           ``mode="scale"`` keeps only the per-dimension weight (the ablation: no normalisation at all).
FuzzyGate  replaces the sigmoid gates' ``nn.Linear(d, H)``: a zero-order TSK system.  R Gaussian rules over the raw layer input (full-d
           antecedents, product t-norm in the HTSK 1/d exponent scaling, no learned linear projection), consequent logits in R^H per rule,
           output = sum_r w_r c_r (a defuzzified weighted average).  The consequents start at the old bias, so at step 0 every rule says
           the same thing; the rules differentiate as the centres move.
``install(model, norm, gate)`` swaps them into a built TinyLM; flm/run_allfuzzy.py does that inside training.
"""

import math

import torch
import torch.nn as nn


class FuzzyNorm(nn.Module):
    def __init__(
        self, d: int, R: int = 5, mode: str = "fuzzy", eps: float = 1.1920929e-07
    ):
        super().__init__()
        self.mode, self.eps = mode, eps
        self.weight = nn.Parameter(torch.ones(d))
        c = torch.linspace(-1.5, 1.5, R)
        self.centers = nn.Parameter(c)
        self.log_width = nn.Parameter(torch.full((R,), math.log(0.8)))
        self.offset = nn.Parameter(-c.clone())  # consequent intercepts b_r
        self.slope = nn.Parameter(-torch.ones(R))  # consequent slopes k_r
        if mode == "scale":
            for p in (self.centers, self.log_width, self.offset, self.slope):
                p.requires_grad_(False)

    def forward(self, x):
        if self.mode == "scale":
            return x * self.weight
        z = 0.5 * torch.log(x.pow(2).mean(-1, keepdim=True) + self.eps)  # (..., 1)
        logit = -0.5 * ((z - self.centers) / torch.exp(self.log_width)) ** 2
        w = torch.softmax(logit, dim=-1)
        g = (w * (self.offset + self.slope * (z - self.centers))).sum(-1, keepdim=True)
        return x * torch.exp(g) * self.weight


class FuzzyGate(nn.Module):
    def __init__(self, d: int, H: int, R: int = 8, bias: torch.Tensor | None = None):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(R, d) * 0.5)
        self.log_width = nn.Parameter(torch.zeros(R, d))
        base = torch.zeros(H) if bias is None else bias.detach().clone()
        self.conseq = nn.Parameter(base.expand(R, H).clone() + 0.01 * torch.randn(R, H))
        self.scale = 1.0 / d

    def forward(self, x):
        a = torch.exp(-2.0 * self.log_width)  # (R, d)
        ca = self.centers * a
        quad = (x * x) @ a.T - 2.0 * (x @ ca.T) + (self.centers * ca).sum(-1)
        w = torch.softmax(-0.5 * self.scale * quad, dim=-1)
        return w @ self.conseq


def install(model, norm: str = "", gate: bool = False, gate_rules: int = 6):
    """norm: "" (keep RMSNorm), "fuzzy" or "scale"; gate: replace both sigmoid gates of every fuzzy-delta mixer."""
    d = model.cfg.d_model
    if norm:
        for blk in model.blocks:
            blk.n1 = FuzzyNorm(d, mode=norm)
            if blk.n2 is not None:
                blk.n2 = FuzzyNorm(d, mode=norm)
        model.norm = FuzzyNorm(d, mode=norm)
    if gate:
        for blk in model.blocks:
            m = blk.mix
            m.gate = FuzzyGate(d, m.H, R=gate_rules, bias=m.gate.bias)
            m.beta = FuzzyGate(d, m.H, R=gate_rules, bias=m.beta.bias)
    return model
