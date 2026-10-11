import torch

import flm.rflm_variants  # noqa: F401
from flm.models import FuzzyDeltaMixer, ModelConfig, TinyLM


def _cfg(mixer, decay="data"):
    return ModelConfig(
        vocab_size=40,
        d_model=16,
        n_layers=2,
        n_heads=2,
        mixer=mixer,
        ffn="tsk",
        ffn_order=1,
        ffn_norm="none",
        n_rules=8,
        decay=decay,
        max_len=32,
    )


def test_propagation_starts_equal_to_d():
    torch.manual_seed(0)
    x = torch.randint(0, 40, (2, 24))
    ref = TinyLM(_cfg("fuzzydelta"))
    for kind in ("fuzzydelta_cpread", "fuzzydelta_cpboth"):
        m = TinyLM(_cfg(kind))
        m.load_state_dict(
            {k: v for k, v in ref.state_dict().items()}, strict=False
        )  # prop is the only extra key
        assert torch.allclose(m(x), ref(x), atol=1e-6)


def test_propagation_is_used_and_step_matches():
    torch.manual_seed(0)
    for kind in ("fuzzydelta_cpread", "fuzzydelta_cpboth", "fuzzydelta_static"):
        m = TinyLM(_cfg(kind)).eval()
        for b in m.blocks:
            if hasattr(b.mix, "prop"):
                b.mix.prop.data.normal_(0, 0.2)
        x = torch.randint(0, 40, (2, 20))
        with torch.no_grad():
            assert torch.allclose(m(x), m(x, step=True), atol=1e-4), kind


def test_static_beta_ignores_input_and_has_fewer_params():
    d = TinyLM(_cfg("fuzzydelta"))
    s = TinyLM(_cfg("fuzzydelta_static"))
    assert s.n_params() < d.n_params()
    mix = s.blocks[0].mix
    a = mix.beta(torch.randn(2, 5, 16))
    b = mix.beta(torch.randn(2, 5, 16))
    assert torch.equal(a, b)
