"""Exactness and receptive-field checks for ``flm.windowed`` (run: ``pytest flm/test_windowed.py -q``).

Float64 throughout. Three kinds of check:
  1. full window (w >= T) reproduces the dense mixer, module-level, with the same parameters;
  2. each windowed core matches an independent brute-force reference at w < T (a dense solve on the masked band;
     a dense recurrence re-run on each window);
  3. receptive field: perturb one input position and measure which outputs move. wgauss and wdelta must be exactly
     zero beyond lag w-1; bsolve (the banded solve) is expected to move beyond it. That expectation is the test.
"""

import torch

import flm.windowed as W
from flm import models as M

torch.set_default_dtype(torch.float64)


def cfg(decay="data", T=12):
    return M.ModelConfig(
        vocab_size=98, d_model=16, n_heads=2, n_rules=4, decay=decay, max_len=T, mixer="fuzzydelta"
    )


def dense_delta_core(fq, fk, v, beta, la):
    """Dense recurrence form of _DeltaMixer.forward on a whole sequence (the reference)."""
    n = fk.shape[2]
    G = M._RecurrentMixerBase.gamma_matrix(la, n, fk.device)
    Gs = M._RecurrentMixerBase.gamma_matrix(la, n, fk.device, strict=True)
    KK = (fk @ fk.transpose(-1, -2)) * Gs
    L = torch.eye(n, dtype=fk.dtype)[None, None] + beta * KK
    U = torch.linalg.solve_triangular(L, beta * v, upper=False, unitriangular=True)
    return ((fq @ fk.transpose(-1, -2)) * G) @ U


def random_core_inputs(B=2, H=2, T=12, F=5, dh=3, decay=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    fq = torch.rand(B, H, T, F, generator=g)
    fk = torch.rand(B, H, T, F, generator=g)
    v = torch.randn(B, H, T, dh, generator=g)
    beta = torch.sigmoid(torch.randn(B, H, T, 1, generator=g))
    la = torch.nn.functional.logsigmoid(torch.randn(B, H, T, generator=g)) if decay else None
    return fq, fk, v, beta, la


# ---------------------------------------------------------------- 1. full window == dense, same parameters


def test_wgauss_full_window_equals_dense():
    torch.manual_seed(0)
    c = cfg()
    dense = M.GaussianTSKAttention(c).double()
    win = W.WindowGaussianAttention(c).double()
    win.load_state_dict(dense.state_dict())
    win.win = 12
    x = torch.randn(2, 12, 16)
    assert torch.allclose(win(x), dense(x), atol=1e-10)


def test_fuzzy_delta_full_window_equals_dense():
    for decay in ("none", "data"):
        torch.manual_seed(1)
        c = cfg(decay)
        dense = M.FuzzyDeltaMixer(c).double()
        for cls in (W.WindowFuzzyDeltaSolve, W.WindowFuzzyDeltaRestart):
            win = cls(c).double()
            win.load_state_dict(dense.state_dict())
            win.win = 12
            x = torch.randn(2, 12, 16)
            assert torch.allclose(win(x), dense(x), atol=1e-9), (decay, cls.__name__)


# ---------------------------------------------------------------- 2. cores vs independent brute force at w < T


def test_banded_solve_matches_dense_masked_solve():
    for decay in (False, True):
        fq, fk, v, beta, la = random_core_inputs(decay=decay)
        T, w = fq.shape[2], 4
        got = W.banded_solve(fq, fk, v, beta, la, w)
        idx = torch.arange(T)
        lag = idx[:, None] - idx[None, :]
        band = (lag >= 0) & (lag < w)
        strict = (lag >= 1) & (lag < w)
        G = M._RecurrentMixerBase.gamma_matrix(la, T, fk.device) if decay else band.double()[None, None]
        Gs = M._RecurrentMixerBase.gamma_matrix(la, T, fk.device, strict=True) if decay else (strict.double())[None, None]
        G, Gs = G * band, Gs * strict
        KK = (fk @ fk.transpose(-1, -2)) * Gs
        Lm = torch.eye(T)[None, None] + beta * KK
        U = torch.linalg.solve_triangular(Lm, beta * v, upper=False, unitriangular=True)
        want = ((fq @ fk.transpose(-1, -2)) * G) @ U
        assert torch.allclose(got, want, atol=1e-10), decay


def test_restart_matches_dense_recurrence_on_each_window():
    for decay in (False, True):
        fq, fk, v, beta, la = random_core_inputs(decay=decay)
        T, w = fq.shape[2], 4
        got = W.restart_solve(fq, fk, v, beta, la, w)
        rows = []
        for t in range(T):
            s0 = max(0, t - w + 1)
            sl = slice(s0, t + 1)
            la_s = la[:, :, sl] if decay else None
            y = dense_delta_core(fq[:, :, sl], fk[:, :, sl], v[:, :, sl], beta[:, :, sl], la_s)
            rows.append(y[:, :, -1])
        want = torch.stack(rows, dim=2)
        assert torch.allclose(got, want, atol=1e-10), decay


def test_restart_full_window_equals_dense_core():
    fq, fk, v, beta, la = random_core_inputs()
    T = fq.shape[2]
    assert torch.allclose(W.restart_solve(fq, fk, v, beta, la, T), dense_delta_core(fq, fk, v, beta, la), atol=1e-10)


# ---------------------------------------------------------------- 3. receptive field


def _receptive_field(mod, T=24, w=4, seed=2):
    """|change in y_t| when only x_s moves, for t = T-1 and every lag k = t - s. Returns a list over k."""
    mod.win = w
    torch.manual_seed(seed)
    x = torch.randn(1, T, 16)
    t = T - 1
    base = mod(x)[0, t]
    out = []
    for k in range(T):
        s = t - k
        x2 = x.clone()
        x2[0, s] += torch.randn(16) * 0.5
        out.append((mod(x2)[0, t] - base).abs().max().item())
    return out


def test_receptive_field_wgauss_and_wdelta_are_exactly_w():
    w = 4
    torch.manual_seed(3)
    c = cfg(T=24)
    for cls in (W.WindowGaussianAttention, W.WindowFuzzyDeltaRestart):
        prof = _receptive_field(cls(c).double(), w=w)
        assert all(p == 0.0 for p in prof[w:]), (cls.__name__, prof[w:])
        assert any(p > 1e-6 for p in prof[:w]), cls.__name__


def test_receptive_field_banded_solve_leaks_past_the_window():
    """The N-diagonal solve: the band is w, but the inverse is dense, so the output still moves past lag w-1."""
    w = 4
    torch.manual_seed(3)
    prof = _receptive_field(W.WindowFuzzyDeltaSolve(cfg(T=24)).double(), w=w)
    beyond = max(prof[w:])
    print("bsolve receptive-field profile (lag: |dy|):", [f"{k}:{p:.2e}" for k, p in enumerate(prof) if p > 0][:24])
    assert beyond > 1e-6, "banded solve should leak past the window; if this fails the premise is wrong"
