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
FuzzyFFNRaw replaces the TSK feed-forward by rules on the raw input (no antecedent projection; optionally no slope).
``install(model, norm, gate, ...)`` swaps them into a built TinyLM; flm/run_allfuzzy.py does that inside training.
"""

import math

import torch
import torch.nn as nn


class _DataInit:
    """First-batch, data-driven placement of Gaussian membership functions (classical fuzzification: rule centres are sampled inputs, widths are the
    per-dimension data spread).  Enabled by ``do_init``; runs once, on the first forward pass, with no gradient and no extra parameters.
    """

    do_init = False
    _done = False

    def _maybe_init(self, u):
        if not self.do_init or self._done:
            return
        with torch.no_grad():
            flat = u.detach().reshape(-1, u.shape[-1])
            idx = torch.randperm(flat.shape[0], device=flat.device)[
                : self.centers.shape[0]
            ]
            self.centers.copy_(flat[idx])
            std = flat.std(0).clamp_min(1e-3)
            self.log_width.copy_(torch.log(std).expand_as(self.log_width))
        self._done = True


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


class FuzzyGate(_DataInit, nn.Module):
    def __init__(self, d: int, H: int, R: int = 8, bias: torch.Tensor | None = None):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(R, d) * 0.5)
        self.log_width = nn.Parameter(torch.zeros(R, d))
        base = torch.zeros(H) if bias is None else bias.detach().clone()
        self.conseq = nn.Parameter(base.expand(R, H).clone() + 0.01 * torch.randn(R, H))
        self.scale = 1.0 / d

    def forward(self, x):
        self._maybe_init(x)
        a = torch.exp(-2.0 * self.log_width)  # (R, d)
        ca = self.centers * a
        quad = (x * x) @ a.T - 2.0 * (x @ ca.T) + (self.centers * ca).sum(-1)
        w = torch.softmax(-0.5 * self.scale * quad, dim=-1)
        return w @ self.conseq


class TSKMap(_DataInit, nn.Module):
    """A zero-order TSK system as a map R^d_in -> R^d_out (a drop-in for ``nn.Linear(d_in, d_out, bias=False)``): R Gaussian rules over the raw
    input (product t-norm, 1/d exponent), normalised firings (a fuzzy partition), consequent vectors c_r in R^d_out, output = sum_r w_r c_r.
    Piecewise-constant interpolation between R prototypes: far less expressive than a linear map at equal parameters, which is the point of
    measuring it."""

    def __init__(self, d_in: int, d_out: int, R: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(R, d_in))
        self.log_width = nn.Parameter(torch.zeros(R, d_in))
        self.conseq = nn.Parameter(torch.randn(R, d_out) / math.sqrt(R))
        self.scale = 1.0 / d_in

    def forward(self, x):
        self._maybe_init(x)
        a = torch.exp(-2.0 * self.log_width)
        ca = self.centers * a
        quad = (x * x) @ a.T - 2.0 * (x @ ca.T) + (self.centers * ca).sum(-1)
        return torch.softmax(-0.5 * self.scale * quad, dim=-1) @ self.conseq


class TSKLinear(_DataInit, nn.Module):
    """A first-order TSK map with a shared consequent basis (a drop-in for ``nn.Linear(d_in, d_out, bias=False)``): u = B x (k numbers), R Gaussian
    rules over u give normalised firings w_r, each rule rescales the k basis directions by s_r, and  y = A (g * u)  with  g = sum_r w_r s_r.
    That is  y = sum_r w_r (A diag(s_r) B) x : a fuzzy partition of locally linear maps that share their factors.  With s_r = 1 it is exactly the
    rank-k linear map A B, which is where it starts."""

    def __init__(self, d_in: int, d_out: int, k: int, R: int):
        super().__init__()
        self.B = nn.Linear(d_in, k, bias=False)
        self.A = nn.Linear(k, d_out, bias=False)
        self.centers = nn.Parameter(torch.randn(R, k) * 0.5)
        self.log_width = nn.Parameter(torch.zeros(R, k))
        self.s = nn.Parameter(torch.ones(R, k))
        self.scale = 1.0 / k

    def forward(self, x):
        u = self.B(x)
        self._maybe_init(u)
        a = torch.exp(-2.0 * self.log_width)
        ca = self.centers * a
        quad = (u * u) @ a.T - 2.0 * (u @ ca.T) + (self.centers * ca).sum(-1)
        g = torch.softmax(-0.5 * self.scale * quad, dim=-1) @ self.s
        return self.A(g * u)


class FuzzyFFNRaw(_DataInit, nn.Module):
    """TSK feed-forward whose antecedents are Gaussian membership functions on the RAW input: every rule has a centre and a width for each of
    the d input dimensions and fires as the product t-norm over them (exponent scaled by 1/d, as everywhere), with no learned linear projection
    in front.  ``order=1``: first-order consequent  g_r (v_r . x + b_r) c_r  (the slope is still linear); ``order=0``: a zero-order system
    y = sum_r f_r c_r, with no slope projection at all.  Raw firing strengths (no softmax), as in the model's FFN.
    """

    def __init__(self, d: int, R: int, order: int = 1, taps: int = 1):
        super().__init__()
        self.R, self.order, self.taps, self.d = R, order, taps, d
        self.centers = nn.Parameter(torch.randn(R, taps * d))
        self.log_width = nn.Parameter(torch.zeros(R, taps * d))
        self.conseq = nn.Parameter(torch.randn(R, d) / math.sqrt(R))
        if order == 1:
            self.slope = nn.Linear(d, R, bias=True)
        self.scale = 1.0 / (taps * d)

    def antecedent_input(self, x):
        """taps > 1: the antecedent reads the raw inputs of the last ``taps`` positions side by side (zero before the start): a rule is a
        conjunction over memberships on the current token AND the previous ones, with no learned linear projection (the fuzzy local layer).
        """
        if self.taps == 1:
            return x
        T = x.shape[-2]
        xp = nn.functional.pad(x, (0, 0, self.taps - 1, 0))
        return torch.cat(
            [
                xp[..., self.taps - 1 - j : self.taps - 1 - j + T, :]
                for j in range(self.taps)
            ],
            dim=-1,
        )

    def forward(self, x):
        u = self.antecedent_input(x)
        self._maybe_init(u)
        a = torch.exp(-2.0 * self.log_width)
        ca = self.centers * a
        quad = (u * u) @ a.T - 2.0 * (u @ ca.T) + (self.centers * ca).sum(-1)
        w = torch.exp(-0.5 * self.scale * quad)
        if self.order == 1:
            w = w * self.slope(x)
        return w @ self.conseq


def install(
    model,
    norm: str = "",
    gate: bool = False,
    gate_rules: int = 6,
    ffn: str = "",
    ffn_rules: int = 52,
    proj: str = "",
    ffn_taps: int = 1,
    ffn_rules0: int = 0,
    data_init: bool = False,
    proj_ranks: tuple = (32, 20),
):
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
    if ffn:  # "raw" (first-order) or "raw0" (zero-order): replace the whole TSK FFN
        for li, blk in enumerate(model.blocks):
            first = (
                li == 0 and ffn_taps > 1
            )  # layer 0 reads the last ffn_taps tokens (the fuzzy local layer, replacing the convolution)
            blk.ffn = FuzzyFFNRaw(
                d,
                (ffn_rules0 or ffn_rules) if first else ffn_rules,
                order=0 if ffn == "raw0" else 1,
                taps=ffn_taps if first else 1,
            )
    if (
        proj
    ):  # the mixer's linear q/k/v and output projections: "tsk0" (zero-order rule maps) or "tskl" (rule-gated low-rank maps)
        for blk in model.blocks:
            m = blk.mix
            if proj == "tsk0":
                m.qkv, m.out = TSKMap(d, 3 * d, 28), TSKMap(d, d, 16)
            else:
                m.qkv, m.out = TSKLinear(d, 3 * d, proj_ranks[0], 8), TSKLinear(
                    d, d, proj_ranks[1], 8
                )
    if (
        data_init
    ):  # place every raw-input / gate / projection rule base from the first batch
        for mod in model.modules():
            if isinstance(mod, _DataInit):
                mod.do_init = True
    return model
