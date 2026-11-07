"""Strict sliding-window mixers: can the N-diagonal idea make a fuzzy mixer O(T*w)?

Three ways to let a query see only its last ``w`` tokens (``WIN`` env, default 32); each is a subclass of the
dense mixer it replaces, so the same parameters load into both and the exactness checks in
``flm/test_windowed.py`` are one-line comparisons.

  wgauss   WindowGaussianAttention  explicit window attention. Each query gathers its last w keys and values
                                    (no solve). Receptive field exactly w. Memory O(T w d), compute O(T w d).
  bsolve   WindowFuzzyDeltaSolve    the delta system with every coupling beyond lag w-1 set to zero, solved as
                                    a banded unit-lower-triangular system by forward substitution (the N-diagonal
                                    solve). Memory O(T w d), compute O(T w^2 d). BUT the inverse of a banded
                                    triangular matrix is dense, so u_t depends on the whole past through the
                                    chain u_{t-1}, u_{t-2}, ...: the receptive field is NOT w. Measured in the tests.
  wdelta   WindowFuzzyDeltaRestart  strict window: each query re-solves the delta system on its own last w tokens
                                    only (a batch of T small w x w solves). Receptive field exactly w. Memory
                                    O(T w^2) per head: it beats dense O(T^2) only when w < sqrt(T).

Dense references: ``GaussianTSKAttention`` (wgauss at w >= T), ``FuzzyDeltaMixer`` (bsolve and wdelta at w >= T).
Not implemented: a recurrent ``step_forward`` for the windowed mixers (training and full-context evaluation only).
"""

from __future__ import annotations

import os

import torch

import flm.models as M


def window() -> int:
    return int(os.environ.get("WIN", "32"))


def lag_index(T: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, w) key positions for lags 0..w-1 (lag 0 = the query itself) and a mask of positions that exist (>= 0)."""
    t = torch.arange(T, device=device)[:, None]
    s = t - torch.arange(w, device=device)[None, :]
    return s.clamp(min=0), s >= 0


def _lag_terms(fq, fk, la, idx, ok):
    """Per (t, lag) quantities: Gamma[t, t-l] (decay, 1 without decay), q_t . k_{t-l}, k_{t-l} . k_t."""
    kw = fk[:, :, idx]  # (B,H,T,w,F)
    QK = (fq[:, :, :, None, :] * kw).sum(-1)  # (B,H,T,w)
    KK = (fk[:, :, :, None, :] * kw).sum(-1)  # (B,H,T,w)
    if la is None:
        G = ok.to(fk.dtype).expand_as(QK)
    else:
        L = la.cumsum(-1)  # (B,H,T); Gamma[t,s] = exp(L_t - L_s) <= 1
        G = torch.exp(L[..., None] - L[:, :, idx]).masked_fill(~ok, 0.0)
    return G, QK, KK


def window_attention(qs, ks, v, w: int) -> torch.Tensor:
    """Softmax over each query's last w keys of the Gaussian-TSK logits q.k - |k|^2/2 (the dense form, restricted)."""
    T = qs.shape[2]
    idx, ok = lag_index(T, min(w, T), qs.device)
    kw, vw = ks[:, :, idx], v[:, :, idx]  # (B,H,T,w,dh)
    logit = (qs[:, :, :, None] * kw).sum(-1) - 0.5 * (kw * kw).sum(-1)
    p = torch.softmax(logit.masked_fill(~ok, float("-inf")), dim=-1)
    return (p[..., None] * vw).sum(-2)


def banded_solve(fq, fk, v, beta, la, w: int) -> torch.Tensor:
    """u_t = beta_t (v_t - sum_{1<=l<w} Gamma[t,t-l] (k_{t-l}.k_t) u_{t-l});  y_t = sum_{l<w} Gamma[t,t-l] (q_t.k_{t-l}) u_{t-l}.

    The unit-lower-triangular band is solved by forward substitution: one step per token, each touching only the
    w-1 previous u's. Sequential in t (a Python loop), so this is for the complexity and exactness tests; it is not
    a fast kernel.
    """
    B, H, T, _ = fk.shape
    idx, ok = lag_index(T, w, fk.device)
    G, QK, KK = _lag_terms(fq, fk, la, idx, ok)
    strict = ok & (torch.arange(w, device=fk.device)[None, :] > 0)
    A = beta * G * KK * strict  # (B,H,T,w): coupling of u_t to u_{t-l}, with beta_t on the row
    bv = beta * v
    U: list[torch.Tensor] = []
    for t in range(T):
        acc = bv[:, :, t]
        m = min(t, w - 1)
        if m > 0:
            prev = torch.stack(U[t - m : t][::-1], dim=2)  # lags 1..m, (B,H,m,dh)
            acc = acc - (A[:, :, t, 1 : m + 1][..., None] * prev).sum(2)
        U.append(acc)
    Uw = torch.stack(U, dim=2)[:, :, idx]  # (B,H,T,w,dh)
    return ((G * QK * ok)[..., None] * Uw).sum(-2)


def restart_solve(fq, fk, v, beta, la, w: int) -> torch.Tensor:
    """Strict window: for each t, solve the delta system on positions t-w+1..t alone (rows before the window do not
    exist), then read the last row's query against those local u's. At w >= T this equals the dense recurrence."""
    B, H, T, _ = fk.shape
    off = torch.arange(w, device=fk.device) - (w - 1)
    P = torch.arange(T, device=fk.device)[:, None] + off[None, :]  # (T,w) window positions, row r = position t-(w-1)+r
    ok = P >= 0
    Pc = P.clamp(min=0)
    kP, vP = fk[:, :, Pc], v[:, :, Pc]  # (B,H,T,w,F), (B,H,T,w,dh)
    bP = beta[:, :, Pc] * ok[..., None]  # invalid rows get beta 0: u = 0 there
    KKl = kP @ kP.transpose(-1, -2)  # (B,H,T,w,w)
    rows = torch.arange(w, device=fk.device)
    lower = (rows[:, None] > rows[None, :]) & ok[:, None, :]  # strict lower, valid columns (T,w,w) via broadcast below
    if la is None:
        Gl = torch.ones_like(KKl)
    else:
        LP = la.cumsum(-1)[:, :, Pc]  # (B,H,T,w)
        Gl = torch.exp((LP[..., :, None] - LP[..., None, :]).masked_fill(~lower, -float("inf")))
    Mloc = torch.eye(w, dtype=fk.dtype, device=fk.device) + (bP * Gl * KKl).masked_fill(~lower, 0.0)
    Uloc = torch.linalg.solve_triangular(Mloc, bP * vP, upper=False, unitriangular=True)  # (B,H,T,w,dh)
    # query t against the window: Gamma[t, P_r] and q_t . k_{P_r}
    QK = (fq[:, :, :, None, :] * kP).sum(-1)  # (B,H,T,w)
    if la is None:
        Gq = ok.to(fk.dtype).expand_as(QK)
    else:
        L = la.cumsum(-1)
        Gq = torch.exp(L[..., None] - la.cumsum(-1)[:, :, Pc]).masked_fill(~ok, 0.0)
    return ((Gq * QK)[..., None] * Uloc).sum(-2)


class WindowGaussianAttention(M.GaussianTSKAttention):
    """wgauss: exact window attention, receptive field w."""

    def __init__(self, cfg: M.ModelConfig):
        super().__init__(cfg)
        self.win = window()

    def forward(self, x):
        B, T, d = x.shape
        qs, ks, v = self.logits_parts(x)
        y = window_attention(qs, ks, v, self.win)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class _WindowFuzzyDelta(M._FuzzyRules, M._DeltaMixer):
    """Fuzzy delta mixer (phi = normalized rule memberships) with a window core. Same parameters as FuzzyDeltaMixer."""

    _core = staticmethod(banded_solve)

    def __init__(self, cfg: M.ModelConfig):
        M._DeltaMixer.__init__(self, cfg)
        self.init_rules(self.H, cfg.n_rules, self.dh, cfg.exp_norm, cfg.width_share)
        self.win = window()

    def forward(self, x):
        B, T, d = x.shape
        fq, fk, v = self.features(x)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)[..., None]  # (B,H,T,1)
        la = self.log_alpha(x)
        y = self._core(fq, fk, v, beta, la, min(self.win, T))
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class WindowFuzzyDeltaSolve(_WindowFuzzyDelta):
    """bsolve: banded solve (the N-diagonal idea). Receptive field is NOT w: see the module docstring."""

    _core = staticmethod(banded_solve)


class WindowFuzzyDeltaRestart(_WindowFuzzyDelta):
    """wdelta: strict window, local re-solve per position."""

    _core = staticmethod(restart_solve)


M.MIXERS.update(
    {
        "wgauss": WindowGaussianAttention,
        "bsolve": WindowFuzzyDeltaSolve,
        "wdelta": WindowFuzzyDeltaRestart,
    }
)
