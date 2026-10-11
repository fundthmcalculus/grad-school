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


def test_ffn_exp_norm_override_is_ffn_only():
    m = TinyLM(cfg(mixer="fuzzy", ffn="tsk", exp_norm="mean", ffn_exp_norm="sum"))
    assert m.blocks[0].mix.exp_scale == 1.0 / 8  # dh = 16 / 2 heads
    assert m.blocks[0].ffn.exp_scale == 1.0


@pytest.mark.parametrize("share", ["dim", "rule"])
@pytest.mark.parametrize("mixer", ["fuzzy", "fuzzydelta"])
def test_width_share_parallel_equals_recurrent(share, mixer):
    torch.manual_seed(9)
    m = TinyLM(
        cfg(mixer=mixer, ffn="tsk", decay="data", width_share=share, exp_norm="mean")
    )
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 3 and p.shape[1] == 1 or p.shape[-1] == 1:
                p.uniform_(-0.3, 0.3)  # make shared widths non-trivial
    idx = torch.randint(0, 98, (2, 13))
    assert torch.allclose(m(idx), m(idx, step=True), atol=1e-10)


def test_width_share_param_counts():
    full = TinyLM(cfg(mixer="fuzzydelta", ffn="tsk", decay="data")).n_params()
    dim = TinyLM(
        cfg(mixer="fuzzydelta", ffn="tsk", decay="data", width_share="dim")
    ).n_params()
    rule = TinyLM(
        cfg(mixer="fuzzydelta", ffn="tsk", decay="data", width_share="rule")
    ).n_params()
    assert full > rule > dim


def test_shared_depth_reuses_blocks():
    from flm.models import init_rules_from_data

    torch.manual_seed(10)
    m = TinyLM(cfg(mixer="fuzzydelta", ffn="tsk", decay="data", n_layers=4, n_unique=1))
    one = TinyLM(cfg(mixer="fuzzydelta", ffn="tsk", decay="data", n_layers=1))
    assert len(m.blocks) == 1 and m.n_params() == one.n_params()
    idx = torch.randint(0, 98, (2, 11))
    assert torch.allclose(m(idx), m(idx, step=True), atol=1e-10)
    # applying the shared block 4x is not the same function as applying it once
    one.load_state_dict(m.state_dict())
    assert not torch.allclose(m(idx), one(idx))
    # runs with shared blocks (needs >= R distinct tokens to place R rules)
    init_rules_from_data(m, torch.randint(0, 98, (4, 16)))


def test_tap_linear_is_a_causal_multitap_map():
    from flm.models import TapLinear

    torch.manual_seed(11)
    m = TapLinear(6, 5, taps=3)
    x = torch.randn(2, 9, 6)
    want = torch.zeros(2, 9, 5)
    for t in range(9):
        for j in range(3):
            if t - j >= 0:
                want[:, t] += x[:, t - j] @ m.weight[j].T
    assert torch.allclose(m(x), want, atol=1e-12)
    x2 = x.clone()
    x2[:, 5:] += 1.0
    assert torch.allclose(m(x)[:, :5], m(x2)[:, :5], atol=1e-12)  # causal


@pytest.mark.parametrize("taps", [2, 3])
def test_ante_taps_keeps_parallel_equals_recurrent_and_causality(taps):
    torch.manual_seed(12)
    m = TinyLM(
        cfg(
            mixer="fuzzydelta", ffn="tsk", decay="data", ante_taps=taps, exp_norm="mean"
        )
    )
    idx = torch.randint(0, 98, (2, 14))
    assert torch.allclose(m(idx), m(idx, step=True), atol=1e-10)
    idx2 = idx.clone()
    idx2[0, 9:] = (idx2[0, 9:] + 1) % 98
    assert torch.allclose(m(idx)[:, :9], m(idx2)[:, :9], atol=1e-10)
    # only block 0 gets the extra taps
    assert m.blocks[0].ffn.ante.weight.shape[0] == taps
    assert isinstance(m.blocks[1].ffn.ante, torch.nn.Linear)


def _markov_chain(n, seed=0, n_states=30, rank=4, scale=1.0):
    """A first-order chain over 30 symbols whose log-transition matrix has rank <= 4, so a
    16-dim readout can represent it and 32 rules can cover every context."""
    rng = np.random.default_rng(seed)
    logits = (
        scale * rng.normal(size=(n_states, rank)) @ rng.normal(size=(rank, n_states))
    )
    T = np.exp(logits - logits.max(-1, keepdims=True))
    cum = (T / T.sum(-1, keepdims=True)).cumsum(-1)
    out = np.zeros(n, dtype=np.uint8)
    u = rng.random(n)
    for i in range(1, n):
        out[i] = min(int(np.searchsorted(cum[out[i - 1]], u[i])), n_states - 1)
    return out


def test_ngram_counts_match_bruteforce_and_conditionals_normalise():
    from flm.ngram import conditionals, count_tables

    a = np.random.default_rng(1).integers(0, 98, 5000).astype(np.uint8)
    tabs = count_tables(n_chars=len(a) - 2, max_k=2, chunk=1000, train=a)
    ref2 = np.zeros((98, 98, 98), dtype=np.int64)
    for t in range(len(a) - 2):
        ref2[a[t], a[t + 1], a[t + 2]] += 1
    assert np.array_equal(tabs[2], ref2)
    probs = conditionals(tabs)
    for p in probs:
        assert np.allclose(p.sum(-1), 1.0)


def test_ngram_init_makes_a_fuzzy_bigram_model():
    """On a synthetic low-rank Markov chain the init must start the model far below a random
    init and near the exact bigram model, with the rules (not the embedding alone) carrying it.
    """
    from flm.ngram import (
        _bpc,
        conditionals,
        count_tables,
        init_rules_from_ngrams,
        window_bpc,
    )

    chain = _markov_chain(150_000)
    tabs = count_tables(n_chars=len(chain) - 3, max_k=2, chunk=50_000, train=chain)
    probs = conditionals(tabs)
    rng = np.random.default_rng(2)

    def win(n):
        s = rng.integers(0, len(chain) - 258, n)
        return torch.from_numpy(
            np.stack([chain[i : i + 257] for i in s]).astype(np.int64)
        )

    stats, calib, held = win(64), win(32), win(32)
    kw = dict(mixer="fuzzydelta", ffn="tsk", decay="data", exp_norm="mean", max_len=256)
    torch.manual_seed(13)
    base = TinyLM(cfg(**kw))
    random_bpc = _bpc(base, held)

    centers_before = base.blocks[0].ffn.centers.detach().clone()
    info = init_rules_from_ngrams(base, stats, calib, 1, tabs, probs, mode="emb")
    assert (
        torch.equal(base.blocks[0].ffn.centers, centers_before)
        and info["mode"] == "emb"
    )

    torch.manual_seed(13)
    m = TinyLM(cfg(**kw))
    info = init_rules_from_ngrams(m, stats, calib, 1, tabs, probs, mode="full")
    full = _bpc(m, held)
    exact = window_bpc(probs, 1, held.numpy())
    assert full < random_bpc - 3.0, (random_bpc, full, info)
    assert full < exact + 0.6, (exact, full)
    with torch.no_grad():  # removing the rules must hurt: they carry the readout
        m.blocks[0].ffn.conseq.zero_()
    assert _bpc(m, held) > full + 1.0
    E = m.emb.weight.detach()  # the embedding is an orthonormal, aligned basis
    assert torch.allclose(E.T @ E, torch.eye(E.shape[1]), atol=1e-4)


def test_ngram_init_on_first_order_raw_firing_ffn_with_surplus_rules():
    """The goal-track feed-forward (first-order, raw firing, tied widths) has R = 108 rules, more
    than the 98 bigram contexts: surplus rules must stay inert, and the init must still start
    the model far below a random init."""
    from flm.ngram import _bpc, conditionals, count_tables, init_rules_from_ngrams

    chain = _markov_chain(150_000)
    tabs = count_tables(n_chars=len(chain) - 3, max_k=2, chunk=50_000, train=chain)
    probs = conditionals(tabs)
    rng = np.random.default_rng(3)

    def win(n):
        s = rng.integers(0, len(chain) - 258, n)
        return torch.from_numpy(
            np.stack([chain[i : i + 257] for i in s]).astype(np.int64)
        )

    stats, calib, held = win(64), win(32), win(32)
    kw = dict(
        mixer="fuzzydelta",
        ffn="tsk",
        decay="data",
        exp_norm="mean",
        max_len=256,
        d_model=48,
        ffn_order=1,
        ffn_norm="none",
        tsk_dim=16,
        ffn_mult=2.25,
        ffn_width="rule",
    )
    torch.manual_seed(14)
    m = TinyLM(cfg(**kw))
    assert m.blocks[0].ffn.R == 108
    random_bpc = _bpc(m, held)
    info = init_rules_from_ngrams(m, stats, calib, 1, tabs, probs, mode="full")
    assert info["n_context_rules"] == 98
    assert _bpc(m, held) < random_bpc - 2.0, (random_bpc, _bpc(m, held), info)
    ffn = m.blocks[0].ffn
    assert torch.all(ffn.conseq[98:107] == 0)  # surplus rules are inert
    assert torch.all(ffn.slope.weight == 0) and torch.all(
        ffn.slope.bias == 1
    )  # zero-order at init


def test_matched_neural_references_have_the_fuzzy_models_parameter_count():
    import json
    from pathlib import Path

    mm = json.loads((Path(__file__).parent / "flm" / "matched_mults.json").read_text())
    for key, p in mm["params"].items():
        _, _, d, L = key.split("|")
        # the 3-layer d=56 reference is matched to D at d=72 (97.5K), not to D at d=56
        target = mm["target"]["d72|L2" if L == "L3" else f"{d}|L2"]
        assert abs(p - target) / target < 0.005, (key, p, target)


def test_v4_grids_select_winners_from_records(tmp_path, monkeypatch):
    """final10 and scale are empty until v4-ref-tune exists, then read its winners; D's lr is
    the lowest-mean v3 cell with both seeds."""
    import json

    import flm.grids_v4 as g

    monkeypatch.setattr(g, "OUT", tmp_path)
    assert g._final10() == [] and g._scale() == []
    assert g.best_d_lr() == "3e-2"  # no records: the documented default

    (tmp_path / "v4-ref-tune").mkdir()
    bpc = {
        "delta-h8-sc8-lr3e-2": 1.30,
        "gla-h4-sc4-lr4e-2": 1.31,
        "delta-h8-L3-sc4-lr3e-2": 1.29,
    }
    for job, v in bpc.items():
        (tmp_path / "v4-ref-tune" / f"{job}.json").write_text(
            json.dumps({"val_bpc": v})
        )
    (tmp_path / "v4-ref-tune" / "gru-sc4-lr2e-2.json").write_text(
        json.dumps({"val_bpc": float("nan")})
    )
    assert g.best_refs(2) == ["delta-h8-L3-sc4-lr3e-2", "delta-h8-sc8-lr3e-2"]
    assert g.best_refs(1, l2_only=True) == ["delta-h8-sc8-lr3e-2"]

    for sub, name, lr, seed, v in [
        ("v3-final", "D_30M_s0", 0.03, 0, 1.28),
        ("v3-final", "D_30M_s1", 0.03, 1, 1.28),
        ("v3-lr", "D_lr2e-2_30M_s0", 0.02, 0, 1.25),
        ("v3-lr", "D_lr2e-2_30M_s1", 0.02, 1, 1.25),
        ("v3-lr", "D_lr6e-2_30M_s0", 0.06, 0, 1.10),  # only one seed: must not win
    ]:
        (tmp_path / sub).mkdir(exist_ok=True)
        (tmp_path / sub / f"{name}.json").write_text(
            json.dumps({"val_bpc": v, "args": {"lr": lr, "seed": seed}})
        )
    assert g.best_d_lr() == "0.02"

    jobs = g._final10()
    assert (
        len(jobs) == 40 and len({n for n, _ in jobs}) == 40
    )  # 10 seeds x (D, 2 refs, old ref)
    assert len(g._scale()) == 36  # 3 widths x 3 seeds x (D, ref) x 2 lrs
    assert all("--device" in argv for _, argv in jobs)


def test_compare_verdict_and_screen_report_names():
    import numpy as np

    from flm.compare import verdict
    from flm.screen_report import NAME

    ref = np.array([1.27, 1.28, 1.26, 1.27, 1.28])
    assert verdict(ref, ref + 0.05)[2] == "ref better, real"
    assert verdict(ref, ref - 0.05)[2] == "arm better, real"
    assert verdict(ref, ref + 0.0005)[2] == "within noise"
    assert verdict(ref, np.array([1.3]))[2].startswith("no verdict")
    assert (
        NAME.match("A-base_d16_s3")["variant"] == "A-base"
        and NAME.match("base_s2")["seed"] == "2"
    )
    assert NAME.match("taps2_s0")["variant"] == "taps2"


def test_ngram_init_refuses_subword_streams():
    from flm.train import main

    with pytest.raises(SystemExit, match="character stream"):
        main(["--tok", "bpe512", "--ngram-init", "1", "--out", "unused.json"])


def test_v6_subword_controls_match_d_and_select_from_records(tmp_path, monkeypatch):
    import json

    import flm.grids_v6 as g
    from flm.grids_v3 import D
    from flm.match_mults import n_params
    from flm.ref_space import NEURAL
    from flm.train import get_args

    d_argv = D + ["--d", "48", "--layers", "2", "--shortconv", "8"]
    target = n_params(d_argv, 1024)
    assert target == 93_344  # the GPU session's recorded count for BPE1024 D d48
    for (name, sc), mult in g.MULT.items():
        argv = NEURAL[name] + [
            "--d",
            "48",
            "--layers",
            "2",
            "--shortconv",
            str(sc),
            "--ffn-mult",
            str(mult),
        ]
        assert abs(n_params(argv, 1024) - target) / target < 0.002, (name, sc)

    monkeypatch.setattr(g, "OUT", tmp_path)
    assert g._final10() == [] and g.best_d_lr() == "3e-2"
    (tmp_path / "v6-ref-tune").mkdir()
    for job, v in {"delta-h8-sc8-lr3e-2": 1.20, "gla-h8-sc4-lr4e-2": 1.21}.items():
        (tmp_path / "v6-ref-tune" / f"{job}.json").write_text(
            json.dumps({"val_bpc": v})
        )
    (tmp_path / "v6-d-lr").mkdir()
    for lr, v in ((0.02, 1.3), (0.04, 1.25)):
        for s in (0, 1):
            (tmp_path / "v6-d-lr" / f"D-lr{lr:g}_10M_s{s}.json").write_text(
                json.dumps({"val_bpc": v, "args": {"lr": lr, "seed": s}})
            )
    assert g.best_d_lr() == "0.04"
    jobs = g._final10()
    assert len(jobs) == 30 and len({n for n, _ in jobs}) == 30  # 10 seeds x (D, 2 refs)
    for _, argv in jobs:
        get_args(argv + ["--out", "unused.json"])  # every flag parses
    assert len(g.REF_JOBS) == 24


class _FakeCodec:
    kind = "char"

    def piece(self, ids):
        return "".join(chr(97 + int(i) % 26) for i in ids)


def _rule_model(**kw):
    base = dict(
        mixer="fuzzydelta",
        ffn="tsk",
        decay="data",
        exp_norm="mean",
        n_heads=2,
        ffn_mult=2.0,
    )
    torch.manual_seed(15)
    return TinyLM(cfg(**{**base, **kw}))


@pytest.mark.parametrize("first_order", [False, True])
def test_rulebook_reads_both_ffn_kinds_without_changing_the_model(first_order):
    from flm.rules import rulebook

    kw = dict(ffn_order=1, ffn_norm="none", tsk_dim=8) if first_order else {}
    m = _rule_model(**kw)
    windows = torch.randint(0, 98, (12, 24))
    before = m(windows)
    text = rulebook(m, "", windows, _FakeCodec(), top=3, max_rules=4)
    assert torch.equal(before, m(windows))  # capture hooks left the model as it was
    assert (
        "first-order, raw firing" if first_order else "zero-order, normalised firing"
    ) in text
    assert "sequence mixer, head 1" in text and "THEN raise" in text
    assert ("gain where it wins" in text) == first_order
    assert text.count("- **rule") >= 4


def test_bpe_codec_renders_pieces_visibly():
    from types import SimpleNamespace

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from flm.rules import BPECodec

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["\x1f"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(
        ["once upon a time there was a little dog\n"] * 50, trainer=tr
    )
    codec = BPECodec.__new__(BPECodec)
    codec.bpe, codec.eot_ch = SimpleNamespace(tok=tok), "\x1f"
    ids = tok.encode("once upon a time\n").ids
    shown = codec.piece(ids)
    assert "·" in shown and "↵" in shown and " " not in shown and "\n" not in shown
    assert codec.piece([tok.token_to_id("\x1f")]) == "⏎⏎"


@pytest.mark.parametrize("first_order", [False, True])
def test_ffn_rule_rows_structure(first_order):
    from flm.rules import ffn_rule_rows

    kw = dict(ffn_order=1, ffn_norm="none", tsk_dim=8) if first_order else {}
    m = _rule_model(**kw)
    windows = torch.randint(0, 98, (12, 24))
    rows = ffn_rule_rows(m, windows, _FakeCodec(), top=3, max_rules=5)
    assert len(rows) == 2 * 5  # two layers, five rules each
    r = rows[0]
    assert {"layer", "rule", "wins", "firing", "fires_on", "examples", "raises"} <= set(
        r
    )
    assert (r["gain"] is not None) == first_order and (
        r["raises_neg"] is not None
    ) == first_order
    assert rows[0]["wins"] >= rows[1]["wins"]  # most used first within a layer
    assert abs(sum(x["wins"] for x in rows if x["layer"] == 0)) <= 1.0 + 1e-9


def test_context_probe_helpers():
    from flm.context_probe import POS_EDGES, STORY_EDGES, bucket, filler

    assert [bucket(t, POS_EDGES) for t in (0, 255, 256, 1023, 1024, 5000)] == [
        0,
        0,
        1,
        2,
        3,
        3,
    ]
    assert (
        bucket(0, STORY_EDGES) == 0
        and bucket(10**6, STORY_EDGES) == len(STORY_EDGES) - 2
    )
    assert filler(0) == "" and filler(2).count(".") == 2 and filler(7).count(".") == 7
