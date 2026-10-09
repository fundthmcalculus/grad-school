"""Tiny causal language models with interchangeable sequence mixers and channel mixers.

Sequence mixers (``mixer=``):

* ``softmax`` -- standard causal softmax attention with RoPE. Quadratic in context.
* ``gauss``   -- the same attention written as a zero-order TSK fuzzy system: every
  past token s is a rule ``IF q is about k_s THEN y = v_s`` with a Gaussian membership
  of learned per-dimension width. Softmax attention differs from it only by the
  rule weight ``exp(||k||^2 / 2 sigma^2)`` (see ``theory.md``). Quadratic.
* ``linear``  -- Katharopoulos et al. (2020) linear attention, feature map elu+1,
  with optional per-head exponential decay. Linear in context (recurrent form).
* ``fuzzy``   -- the recurrent fuzzy mixer (FRLM): a FIXED base of R Gaussian rules per
  head. A token is written into each rule's consequent memory in proportion to its
  membership; a query reads the rules it fires. Exactly linear attention with the
  rule-membership vector as the feature map. Linear in context.
* ``delta``   -- DeltaNet (delta-rule fast weights, L2-normalized SiLU features).
* ``fuzzydelta`` -- FRLM, delta form: the fuzzy rules' consequents are fit in context by
  normalized LMS -- a TSK system that trains its own consequents as it reads.
* ``gru``     -- a whole-model GRU baseline (no residual stack).

Channel mixers (``ffn=``): ``mlp`` (GELU MLP) or ``tsk`` (zero-order TSK system with
normalized Gaussian rules over a learned antecedent projection).

Every recurrent mixer has two code paths -- a parallel (training) form and a step
(inference) form -- and ``test_models.py`` pins them equal.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    mixer: str = "softmax"  # softmax | gauss | linear | fuzzy | gru
    ffn: str = "mlp"  # mlp | tsk | none
    ffn_mult: float = 2.0  # MLP hidden = ffn_mult * d ; TSK rules = ffn_mult * d
    tsk_dim: int = 0  # TSK antecedent dimension (0 -> d_model)
    n_rules: int = 16  # rules per head for the fuzzy mixer
    decay: str = "fixed"  # none | fixed | data  (recurrent mixers only)
    shortconv: int = 0  # causal depthwise conv width on mixer input (0 = off)
    exp_norm: str = (
        "sum"  # Gaussian exponent over antecedent dims: sum (classic TSK) | mean (HTSK) | sqrt
    )
    max_len: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- utils


def rope_cache(head_dim: int, max_len: int, base: float = 10000.0):
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(half, dtype=torch.float32) / half))
    t = torch.arange(max_len, dtype=torch.float32)
    ang = torch.outer(t, inv)  # (T, half)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, Dh). Rotates (x[..., :half], x[..., half:]) pairs."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    T = x.shape[-2]
    c, s = cos[:T], sin[:T]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def decay_init(n_heads: int) -> torch.Tensor:
    """RetNet-style multi-scale decays gamma_h = 1 - 2^(-5-h*?) spread over heads,
    returned as logit(gamma) so that sigmoid() recovers it."""
    if n_heads == 1:
        gammas = torch.tensor([1 - 2.0**-5])
    else:
        # half-lives from ~4 tokens up to ~128 tokens
        exps = torch.linspace(2.0, 7.0, n_heads)
        gammas = 1 - 2.0 ** (-exps)
    return torch.log(gammas / (1 - gammas))


class ShortConv(nn.Module):
    """Causal depthwise convolution (token shift generalized)."""

    def __init__(self, d: int, width: int):
        super().__init__()
        self.width = width
        self.conv = nn.Conv1d(d, d, width, groups=d, padding=width - 1, bias=False)
        with torch.no_grad():  # start near identity
            self.conv.weight.zero_()
            self.conv.weight[:, 0, -1] = 1.0

    def forward(self, x):  # (B, T, d)
        T = x.shape[1]
        return self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)


# ----------------------------------------------------------------------------- mixers


class SoftmaxAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        assert d % H == 0
        self.H, self.dh = H, d // H
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        cos, sin = rope_cache(self.dh, cfg.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def qkv_heads(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        return q, k, v  # (B, H, T, dh)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv_heads(x)
        q, k = apply_rope(q, self.cos, self.sin), apply_rope(k, self.cos, self.sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class GaussianTSKAttention(SoftmaxAttention):
    """Zero-order TSK attention with a dynamic rule base (one rule per past token).

    firing_s(q) = exp( -sum_j (q_j - k_sj)^2 / (2 sigma_j^2) )
    y = sum_s firing_s v_s / sum_s firing_s

    Expanding the square, the per-query term ||q/sigma||^2 cancels in the normalization,
    so the logits are (q/sigma).(k/sigma) - ||k/sigma||^2 / 2. sigma is tied within each
    RoPE rotation pair, so scaling commutes with the rotation and the identity is exact.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        # log sigma per head per rotation pair; init so that 1/sigma^2 = 1/sqrt(dh)
        init = 0.25 * math.log(self.dh)
        self.log_sigma = nn.Parameter(torch.full((self.H, self.dh // 2), init))

    def inv_sigma(self):
        s = torch.exp(-self.log_sigma)
        return torch.cat([s, s], dim=-1)[None, :, None, :]  # (1, H, 1, dh)

    def logits_parts(self, x):
        q, k, v = self.qkv_heads(x)
        q, k = apply_rope(q, self.cos, self.sin), apply_rope(k, self.cos, self.sin)
        inv = self.inv_sigma()
        return q * inv, k * inv, v

    def forward(self, x):
        B, T, d = x.shape
        qs, ks, v = self.logits_parts(x)
        # augment so that sdpa computes qs.ks - ||ks||^2/2 exactly
        ones = torch.ones_like(qs[..., :1])
        qa = torch.cat([qs, ones], dim=-1)
        ka = torch.cat([ks, -0.5 * (ks * ks).sum(-1, keepdim=True)], dim=-1)
        y = F.scaled_dot_product_attention(qa, ka, v, is_causal=True, scale=1.0)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class _RecurrentMixerBase(nn.Module):
    """Shared projections, feature maps and decay for the recurrent (linear-cost) mixers.

    Decay modes (``cfg.decay``):
      none  -- alpha_t = 1
      fixed -- alpha_t = gamma_h, a learned per-head constant (RetNet-style init)
      data  -- alpha_t = sigmoid(w_h . x_t + b_h), input-dependent forgetting (GLA-style)
    Gamma[t, s] = prod_{i=s+1..t} alpha_i is formed from cumulative log-decays, masked
    *before* exponentiation so it never overflows.
    """

    eps = 1e-6

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        assert d % H == 0
        self.H, self.dh = H, d // H
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.decay = cfg.decay
        assert self.decay in ("none", "fixed", "data"), self.decay
        if self.decay == "fixed":
            self.decay_logit = nn.Parameter(decay_init(H))
        elif self.decay == "data":
            self.gate = nn.Linear(d, H)
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.copy_(decay_init(H))

    def phi(self, x):  # (B, H, T, dh) -> (B, H, T, F), non-negative
        raise NotImplementedError

    def features(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        return self.phi(q), self.phi(k), v

    def log_alpha(self, x):  # -> (B, H, T) or None
        B, T, _ = x.shape
        if self.decay == "none":
            return None
        if self.decay == "fixed":
            return F.logsigmoid(self.decay_logit)[None, :, None].expand(B, self.H, T)
        return F.logsigmoid(self.gate(x)).transpose(1, 2)

    @staticmethod
    def gamma_matrix(la, T, device, strict=False):
        """(B,H,T,T) with Gamma[t,s] = prod_{i=s+1..t} alpha_i for s<=t (s<t if strict)."""
        idx = torch.arange(T, device=device)
        mask = (
            (idx[:, None] > idx[None, :]) if strict else (idx[:, None] >= idx[None, :])
        )
        if la is None:
            return mask.to(torch.get_default_dtype())[None, None]
        L = la.cumsum(-1)
        diff = L[..., :, None] - L[..., None, :]
        return torch.exp(diff.masked_fill(~mask, -float("inf")))


class _KernelLinearMixer(_RecurrentMixerBase):
    """Normalized linear attention with feature map phi:

        S_t = alpha_t S_{t-1} + phi(k_t) v_t^T      z_t = alpha_t z_{t-1} + phi(k_t)
        y_t = phi(q_t)^T S_t / (phi(q_t)^T z_t)

    Parallel form: A = (phi(Q) phi(K)^T) * Gamma; y = A V / A 1.
    """

    def forward(self, x):
        B, T, d = x.shape
        fq, fk, v = self.features(x)
        A = (fq @ fk.transpose(-1, -2)) * self.gamma_matrix(
            self.log_alpha(x), T, x.device
        )
        y = (A @ v) / (A.sum(-1, keepdim=True) + self.eps)
        return self.out(y.transpose(1, 2).reshape(B, T, d))

    @torch.no_grad()
    def step_forward(self, x):
        """Recurrent (inference) form, one token at a time."""
        B, T, d = x.shape
        fq, fk, v = self.features(x)
        la = self.log_alpha(x)
        S = x.new_zeros(B, self.H, fq.shape[-1], self.dh)
        z = x.new_zeros(B, self.H, fq.shape[-1])
        ys = []
        for t in range(T):
            a = torch.exp(la[:, :, t])[..., None] if la is not None else 1.0
            S = (
                S * (a[..., None] if la is not None else 1.0)
                + fk[:, :, t, :, None] * v[:, :, t, None, :]
            )
            z = z * a + fk[:, :, t]
            num = (fq[:, :, t, :, None] * S).sum(-2)
            den = (fq[:, :, t] * z).sum(-1, keepdim=True) + self.eps
            ys.append(num / den)
        y = torch.stack(ys, dim=2)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


def exp_scale(exp_norm: str, dim: int) -> float:
    """Scale on the Gaussian exponent sum_j (u_j - c_j)^2 / 2 s_j^2 over `dim` antecedent dims.

    sum  : 1        -- classic product-t-norm TSK; the rule softmax saturates as dim grows
    mean : 1/dim    -- HTSK (Cui, Wu & Xu 2021, arXiv:2102.04271 eq. 17): firing = (prod mu)^(1/dim)
    sqrt : 1/sqrt(dim) -- between the two; the Transformer's 1/sqrt(d_k), which that paper cites
    """
    return {"sum": 1.0, "mean": 1.0 / dim, "sqrt": dim**-0.5}[exp_norm]


class _FuzzyRules(nn.Module):
    """R Gaussian rules per head; phi(u) = normalized memberships (a fuzzy partition)."""

    def init_rules(self, H, R, dh, exp_norm="sum"):
        self.R = R
        self.exp_scale = exp_scale(exp_norm, dh)
        self.centers = nn.Parameter(torch.randn(H, R, dh) * 0.5)
        self.log_width = nn.Parameter(torch.zeros(H, R, dh))

    def log_membership(self, x):  # (B,H,T,dh) -> (B,H,T,R)
        # sum_j (x_j - c_rj)^2 a_rj with a = 1/s^2, expanded into matmuls:
        #   (x^2) a_r - 2 x (c_r a_r) + sum_j c_rj^2 a_rj
        a = torch.exp(-2.0 * self.log_width)  # (H,R,dh)
        ca = self.centers * a
        quad = (x * x) @ a.transpose(-1, -2) - 2.0 * (x @ ca.transpose(-1, -2))
        quad = quad + (self.centers * ca).sum(-1)[:, None, :]
        return -0.5 * self.exp_scale * quad

    def phi(self, x):
        return torch.softmax(self.log_membership(x), dim=-1)


class LinearAttention(_KernelLinearMixer):
    def phi(self, x):
        return F.elu(x) + 1.0


class FuzzyRecurrentMixer(_FuzzyRules, _KernelLinearMixer):
    """FRLM mixer (accumulating form). R Gaussian rules per head over key/query space.

    Membership of a vector u in rule r:  mu_r(u) = exp(-sum_j (u_j - c_rj)^2 / (2 s_rj^2)).
    The SAME antecedents fuzzify the keys (write) and the queries (read), so the implied
    kernel K(q,k) = sum_r mu_r(q) mu_r(k) is symmetric PSD. Memberships are normalized
    across rules (a fuzzy partition, as in FCM); the read-side normalization cancels in
    the output ratio, so only the write side's partition of unity carries meaning.

    Read as a TSK system: rule r's consequent at time t is the decayed fuzzy mean of the
    values written into it, ybar_r = S_r / z_r, and the output is
    y = sum_r w_r ybar_r with w_r = mu_r(q) z_r / sum_r' mu_r'(q) z_r'.
    """

    def __init__(self, cfg: ModelConfig):
        _KernelLinearMixer.__init__(self, cfg)
        self.init_rules(self.H, cfg.n_rules, self.dh, cfg.exp_norm)


class _DeltaMixer(_RecurrentMixerBase):
    """Delta-rule fast weights (Schlag et al. 2021; Yang et al. 2024), optional decay:

        S_t = alpha_t S_{t-1} + beta_t (v_t - alpha_t S_{t-1} k_t) k_t^T,   o_t = S_t q_t
    with k = phi(k_raw), q = phi(q_raw). Writing S_t = sum_{s<=t} Gamma[t,s] u_s k_s^T gives
        u_t = beta_t (v_t - sum_{s<t} Gamma[t,s] (k_s . k_t) u_s)
    i.e. the unit-lower-triangular system (I + diag(beta) (tril(K K^T, -1) * Gamma)) U = diag(beta) V,
    and o = (tril(Q K^T) * Gamma) U. One triangular solve per (batch, head).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.beta = nn.Linear(cfg.d_model, self.H)

    def forward(self, x):
        B, T, d = x.shape
        fq, fk, v = self.features(x)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)[..., None]  # (B,H,T,1)
        la = self.log_alpha(x)
        G = self.gamma_matrix(la, T, x.device)
        Gs = self.gamma_matrix(la, T, x.device, strict=True)
        KK = (fk @ fk.transpose(-1, -2)) * Gs
        eye = torch.eye(T, device=x.device, dtype=x.dtype)
        Lmat = eye + beta * KK
        U = torch.linalg.solve_triangular(
            Lmat, beta * v, upper=False, unitriangular=True
        )
        y = ((fq @ fk.transpose(-1, -2)) * G) @ U
        return self.out(y.transpose(1, 2).reshape(B, T, d))

    @torch.no_grad()
    def step_forward(self, x):
        B, T, d = x.shape
        fq, fk, v = self.features(x)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)  # (B,H,T)
        la = self.log_alpha(x)
        S = x.new_zeros(B, self.H, self.dh, fk.shape[-1])  # maps feature -> value
        ys = []
        for t in range(T):
            if la is not None:
                S = S * torch.exp(la[:, :, t])[..., None, None]
            k = fk[:, :, t]
            pred = (S @ k[..., None])[..., 0]
            S = (
                S
                + beta[:, :, t, None, None]
                * (v[:, :, t] - pred)[..., None]
                * k[..., None, :]
            )
            ys.append((S @ fq[:, :, t, :, None])[..., 0])
        y = torch.stack(ys, dim=2)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class DeltaNetMixer(_DeltaMixer):
    """DeltaNet feature map: L2-normalized SiLU (Yang et al. 2024)."""

    def phi(self, x):
        return F.normalize(F.silu(x), dim=-1)


class FuzzyDeltaMixer(_FuzzyRules, _DeltaMixer):
    """FRLM mixer (delta form): a zero-order TSK system that fits its own consequents
    in context. With phi = normalized rule firing (sum_r phi_r = 1),

        theta_r <- alpha theta_r + beta phi_r(k_t) (v_t - sum_r' phi_r'(k_t) alpha theta_r')
        y_t = sum_r phi_r(q_t) theta_r

    which is normalized-LMS training of the TSK consequents theta_r (the columns of S) on
    the (key -> value) pairs seen so far. Stable for beta in (0,1) because ||phi||_2 <= 1.
    """

    def __init__(self, cfg: ModelConfig):
        _DeltaMixer.__init__(self, cfg)
        self.init_rules(self.H, cfg.n_rules, self.dh, cfg.exp_norm)


class GRUModelCore(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.rnn = nn.GRU(
            cfg.d_model, cfg.d_model, num_layers=cfg.n_layers, batch_first=True
        )

    def forward(self, x):
        return self.rnn(x)[0]


# ----------------------------------------------------------------------------- channel mixers


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = max(1, int(round(cfg.ffn_mult * cfg.d_model)))
        self.fc = nn.Linear(cfg.d_model, h, bias=False)
        self.proj = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class TSKFFN(nn.Module):
    """Zero-order TSK channel mixer.

    u = P x (antecedent space, dim da);  w_r(u) = softmax_r(-||(u - c_r)/s_r||^2 / 2)
    y = sum_r w_r(u) a_r      (a_r in R^d: rule consequent)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        da = cfg.tsk_dim or d
        R = max(1, int(round(cfg.ffn_mult * d)))
        self.R, self.da = R, da
        self.exp_scale = exp_scale(cfg.exp_norm, da)
        self.ante = nn.Linear(d, da, bias=False)
        self.centers = nn.Parameter(torch.randn(R, da))
        self.log_width = nn.Parameter(torch.zeros(R, da))
        self.conseq = nn.Parameter(torch.zeros(R, d))
        nn.init.normal_(self.conseq, std=0.02)

    def firing(self, x):
        u = self.ante(x)
        a = torch.exp(-2.0 * self.log_width)  # (R,da)
        ca = self.centers * a
        quad = (u * u) @ a.T - 2.0 * (u @ ca.T) + (self.centers * ca).sum(-1)
        return torch.softmax(-0.5 * self.exp_scale * quad, dim=-1)

    def forward(self, x):
        return self.firing(x) @ self.conseq


# ----------------------------------------------------------------------------- model

MIXERS = {
    "softmax": SoftmaxAttention,
    "gauss": GaussianTSKAttention,
    "linear": LinearAttention,
    "fuzzy": FuzzyRecurrentMixer,
    "delta": DeltaNetMixer,
    "fuzzydelta": FuzzyDeltaMixer,
}
FFNS = {"mlp": MLP, "tsk": TSKFFN}


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d_model)
        self.conv = ShortConv(cfg.d_model, cfg.shortconv) if cfg.shortconv > 1 else None
        self.mix = MIXERS[cfg.mixer](cfg)
        self.n2 = nn.RMSNorm(cfg.d_model) if cfg.ffn != "none" else None
        self.ffn = FFNS[cfg.ffn](cfg) if cfg.ffn != "none" else None

    def mixer_in(self, x):
        h = self.n1(x)
        return self.conv(h) if self.conv is not None else h

    def forward(self, x, step=False):
        h = self.mixer_in(x)
        x = x + (self.mix.step_forward(h) if step else self.mix(h))
        if self.ffn is not None:
            x = x + self.ffn(self.n2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.emb.weight, std=0.1)
        if cfg.mixer == "gru":
            self.core = GRUModelCore(cfg)
            self.blocks = nn.ModuleList()
        else:
            self.core = None
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        # tied unembedding

    def forward(self, idx, step=False):
        x = self.emb(idx)
        if self.core is not None:
            x = self.core(x)
        for b in self.blocks:
            x = b(x, step=step)
        return self.norm(x) @ self.emb.weight.T

    def n_params(self, exclude_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.emb.weight.numel()
        return n

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=None, generator=None):
        """Simple full-context resampling generator (context window = max_len)."""
        for _ in range(n_new):
            ctx = idx[:, -self.cfg.max_len :]
            logits = self(ctx)[:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            p = torch.softmax(logits, -1)
            nxt = torch.multinomial(p, 1, generator=generator)
            idx = torch.cat([idx, nxt], 1)
        return idx


# ----------------------------------------------------------------------------- data-driven rule init


@torch.no_grad()
def init_rules_from_data(
    model: TinyLM, idx: torch.Tensor, generator: torch.Generator | None = None
) -> dict:
    """Place every TSK rule on the data before training.

    For each fuzzy component, run ``idx`` through the (untrained) model, collect the
    vectors its antecedents see (keys for the fuzzy mixers, the projected input u for
    the TSK FFN), set the rule centers to R distinct randomly chosen data vectors, and
    set every rule's per-dimension width to the data's per-dimension std. Layers are
    initialized in order, so each one sees the already-initialized layers below it.
    Returns the number of rules initialized per component.
    """
    done = {}
    for li, blk in enumerate(model.blocks):
        # rerun from the bottom each time so lower layers' new rules are reflected
        x = model.emb(idx)
        for b in model.blocks[:li]:
            x = b(x)
        mix = blk.mix
        if isinstance(mix, _FuzzyRules):
            h = blk.mixer_in(x)
            B, T, _ = h.shape
            k = mix.qkv(h).view(B, T, 3, mix.H, mix.dh)[:, :, 1]  # (B,T,H,dh) raw keys
            k = k.reshape(-1, mix.H, mix.dh)
            for hh in range(mix.H):
                pick = torch.randperm(k.shape[0], generator=generator)[: mix.R]
                mix.centers[hh] = k[pick, hh]
                mix.log_width[hh] = torch.log(k[:, hh].std(0).clamp(min=1e-3)).expand(
                    mix.R, -1
                )
            done[f"L{li}.mix"] = mix.R * mix.H
        x = x + mix(blk.mixer_in(x))
        if isinstance(blk.ffn, TSKFFN):
            ffn = blk.ffn
            u = ffn.ante(blk.n2(x)).reshape(-1, ffn.da)
            pick = torch.randperm(u.shape[0], generator=generator)[: ffn.R]
            ffn.centers.copy_(u[pick])
            ffn.log_width.copy_(torch.log(u.std(0).clamp(min=1e-3)).expand(ffn.R, -1))
            done[f"L{li}.ffn"] = ffn.R
    return done
