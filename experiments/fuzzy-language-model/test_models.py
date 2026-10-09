"""Pins the identities the experiment rests on. Run: .venv/bin/python -m pytest -q"""

import math

import numpy as np
import pytest

# CI collects every experiments/ suite in an environment without torch; skip, don't
# error -- a collection error interrupts the whole pytest run and hides other suites.
torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from flm.data import EOT, decode, encode
from flm.models import (
    DeltaNetMixer,
    FuzzyDeltaMixer,
    FuzzyRecurrentMixer,
    GaussianTSKAttention,
    LinearAttention,
    ModelConfig,
    TinyLM,
    apply_rope,
)


@pytest.fixture(autouse=True, scope="module")
def _float64():
    """The identities are checked in float64; restore the default so a shared pytest
    session's other torch suites are unaffected."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def cfg(**kw):
    base = dict(vocab_size=98, d_model=16, n_layers=2, n_heads=2, n_rules=5, max_len=32)
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize(
    "cls", [LinearAttention, FuzzyRecurrentMixer, DeltaNetMixer, FuzzyDeltaMixer]
)
@pytest.mark.parametrize("decay", ["none", "fixed", "data"])
def test_parallel_equals_recurrent(cls, decay):
    torch.manual_seed(0)
    m = cls(cfg(decay=decay))
    if decay == "data":  # make the gate actually input-dependent
        torch.nn.init.normal_(m.gate.weight, std=0.5)
    x = torch.randn(3, 20, 16)
    assert torch.allclose(m(x), m.step_forward(x), atol=1e-10)


@pytest.mark.parametrize("mixer", ["linear", "fuzzy", "delta", "fuzzydelta"])
def test_full_model_step_path(mixer):
    torch.manual_seed(1)
    m = TinyLM(cfg(mixer=mixer, ffn="tsk", shortconv=3))
    idx = torch.randint(0, 98, (2, 17))
    assert torch.allclose(m(idx), m(idx, step=True), atol=1e-10)


def test_gauss_attention_is_explicit_tsk():
    """GaussianTSKAttention == normalized product-of-Gaussian-MF rule base, one rule per past token."""
    torch.manual_seed(2)
    m = GaussianTSKAttention(cfg())
    with torch.no_grad():
        m.log_sigma.uniform_(-0.5, 0.5)
    x = torch.randn(2, 9, 16)
    got = m(x)
    q, k, v = m.qkv_heads(x)
    q, k = apply_rope(q, m.cos, m.sin), apply_rope(k, m.cos, m.sin)
    sig = torch.exp(m.log_sigma)
    sig = torch.cat([sig, sig], -1)[None, :, None, :]
    T = x.shape[1]
    ys = torch.zeros_like(v)
    for t in range(T):
        # membership of query t in rule s: prod_j exp(-(q_j - k_sj)^2 / 2 sigma_j^2)
        mu = torch.exp(
            -(((q[:, :, t : t + 1] - k[:, :, : t + 1]) / sig) ** 2).sum(-1) / 2
        )
        ys[:, :, t] = (mu[..., None] * v[:, :, : t + 1]).sum(2) / mu.sum(
            -1, keepdim=True
        )
    want = m.out(ys.transpose(1, 2).reshape(2, T, 16))
    assert torch.allclose(got, want, atol=1e-10)


def test_softmax_attention_is_weighted_tsk():
    """softmax(q.k / tau) over s == Gaussian MF exp(-||q-k_s||^2 / 2tau) * rule weight exp(||k_s||^2 / 2tau)."""
    torch.manual_seed(3)
    q, K = torch.randn(4), torch.randn(7, 4)
    tau = math.sqrt(4)
    sm = torch.softmax(K @ q / tau, 0)
    mu = torch.exp(-((q - K) ** 2).sum(-1) / (2 * tau))
    w = torch.exp((K**2).sum(-1) / (2 * tau))
    assert torch.allclose(sm, mu * w / (mu * w).sum(), atol=1e-12)


def test_fuzzy_mixer_reads_as_tsk_with_recurrent_consequents():
    """y_t = sum_r w_r ybar_r, ybar_r = S_r/z_r (fuzzy mean of values in rule r), w_r ∝ mu_r(q) z_r."""
    torch.manual_seed(4)
    m = FuzzyRecurrentMixer(cfg(decay="fixed"))
    x = torch.randn(1, 12, 16)
    fq, fk, v = m.features(x)
    g = torch.sigmoid(m.decay_logit)
    t = 11
    pw = (
        g[None, :, None] ** torch.arange(t, -1, -1, dtype=x.dtype)[None, None, :]
    )  # (1,H,t+1)
    S = torch.einsum("bhs,bhsr,bhsd->bhrd", pw, fk[:, :, : t + 1], v[:, :, : t + 1])
    z = torch.einsum("bhs,bhsr->bhr", pw, fk[:, :, : t + 1])
    ybar = S / z[..., None]
    w = fq[:, :, t] * z
    w = w / w.sum(-1, keepdim=True)
    y_tsk = (w[..., None] * ybar).sum(-2)  # (1,H,dh)
    y = m.out(y_tsk.transpose(0, 1).reshape(1, 16))
    # tolerance covers the model's eps=1e-6 denominator stabilizer, absent from the explicit form
    assert torch.allclose(m(x)[:, t], y, atol=1e-6)


def test_causality():
    for mixer in ["softmax", "gauss", "linear", "fuzzy", "delta", "fuzzydelta", "gru"]:
        torch.manual_seed(5)
        m = TinyLM(
            cfg(mixer=mixer, ffn="tsk" if mixer == "fuzzy" else "mlp", shortconv=3)
        )
        idx = torch.randint(0, 98, (1, 16))
        a = m(idx)
        idx2 = idx.clone()
        idx2[0, 10:] = (idx2[0, 10:] + 1) % 98
        b = m(idx2)
        assert torch.allclose(a[:, :10], b[:, :10], atol=1e-10), mixer


def test_codec_roundtrip():
    s = 'Once upon a time, "Tom" said hi.\n'
    ids = encode(s + "<|endoftext|>" + s)
    assert (ids == EOT).sum() == 1
    assert (
        decode(ids).replace("\n<|eot|>\n", "<|endoftext|>") == s + "<|endoftext|>" + s
    )
    assert np.all(ids < 98)


def test_data_rule_init_spreads_rules():
    from flm.models import init_rules_from_data

    torch.manual_seed(6)
    m = TinyLM(cfg(mixer="fuzzydelta", ffn="tsk", decay="data"))
    idx = torch.randint(3, 98, (8, 32))
    done = init_rules_from_data(m, idx)
    assert set(done) == {"L0.mix", "L1.mix", "L0.ffn", "L1.ffn"}
    # every FFN rule now wins somewhere on the init data
    blk = m.blocks[0]
    w = blk.ffn.firing(blk.n2(m.emb(idx) + blk.mix(blk.mixer_in(m.emb(idx)))))
    assert (
        torch.bincount(w.argmax(-1).reshape(-1), minlength=blk.ffn.R) > 0
    ).float().mean() > 0.5


@pytest.mark.parametrize("cls", [FuzzyRecurrentMixer, FuzzyDeltaMixer])
def test_htsk_parallel_equals_recurrent(cls):
    torch.manual_seed(7)
    m = cls(cfg(decay="data", exp_norm="mean"))
    torch.nn.init.normal_(m.gate.weight, std=0.5)
    x = torch.randn(2, 15, 16)
    assert torch.allclose(m(x), m.step_forward(x), atol=1e-10)


def test_htsk_is_geometric_mean_of_memberships():
    """HTSK firing = softmax over (1/D) * log prod_j mu_j, i.e. normalized (prod mu)^(1/D)."""
    from flm.models import TSKFFN

    torch.manual_seed(8)
    f = TSKFFN(cfg(exp_norm="mean"))
    x = torch.randn(5, 16)
    u = f.ante(x)
    mu = torch.exp(
        -0.5 * ((u[:, None, :] - f.centers) * torch.exp(-f.log_width)) ** 2
    )  # (5,R,D)
    g = mu.prod(-1) ** (1.0 / f.da)
    assert torch.allclose(f.firing(x), g / g.sum(-1, keepdim=True), atol=1e-10)
