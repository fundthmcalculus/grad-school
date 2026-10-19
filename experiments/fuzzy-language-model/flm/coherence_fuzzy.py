"""``GRAD_SCHOOL_DATA=<repo>/data/tinystories COH_SEEDS=.. python -m flm.coherence_fuzzy out ckpt...``: flm.coherence_set for checkpoints that may contain fuzzy
replacements (flm/fuzzy_parts.py) and tiered embeddings of any shape.  Each checkpoint's architecture is read from its own state dict:
  blocks.*.n1.centers      -> FuzzyNorm            blocks.*.mix.gate.centers  -> FuzzyGate (rules = rows)
  blocks.*.ffn.slope / no ffn.ante -> FuzzyFFNRaw (order 1 / order 0)
  blocks.*.mix.qkv.centers -> TSKMap               blocks.*.mix.qkv.B.weight  -> TSKLinear (k = rows of B)
plus the tiered-embedding detection of flm.coherence_vocab.  Stock models load as before.
"""

import os

import torch

import flm.coherence_set as C
import flm.run_tiered as R
from flm.fuzzy_parts import FuzzyFFNRaw, FuzzyGate, FuzzyNorm, TSKLinear, TSKMap
from flm.models import TinyLM


def install_from_state(model: TinyLM, st: dict) -> None:
    d = model.cfg.d_model
    for li, blk in enumerate(model.blocks):
        p = f"blocks.{li}."
        if p + "n1.centers" in st:
            R_ = st[p + "n1.centers"].shape[
                0
            ]  # (the weights-only ablation is not a coherence candidate: always the fuzzy mode)
            blk.n1 = FuzzyNorm(d, R=R_)
            if blk.n2 is not None:
                blk.n2 = FuzzyNorm(d, R=R_)
        if p + "mix.gate.centers" in st:
            m = blk.mix
            rules = st[p + "mix.gate.centers"].shape[0]
            m.gate, m.beta = FuzzyGate(d, m.H, R=rules), FuzzyGate(d, m.H, R=rules)
        if p + "ffn.ante.weight" not in st and p + "ffn.log_width" in st:
            rules = st[p + "ffn.centers"].shape[0]
            blk.ffn = FuzzyFFNRaw(
                d, rules, order=1 if p + "ffn.slope.weight" in st else 0
            )
        if p + "mix.qkv.B.weight" in st:  # TSKLinear also has centers: test it first
            blk.mix.qkv = TSKLinear(
                d,
                3 * d,
                st[p + "mix.qkv.B.weight"].shape[0],
                st[p + "mix.qkv.s"].shape[0],
            )
            blk.mix.out = TSKLinear(
                d, d, st[p + "mix.out.B.weight"].shape[0], st[p + "mix.out.s"].shape[0]
            )
        elif p + "mix.qkv.conseq" in st:
            blk.mix.qkv = TSKMap(d, 3 * d, st[p + "mix.qkv.centers"].shape[0])
            blk.mix.out = TSKMap(d, d, st[p + "mix.out.centers"].shape[0])
    if "norm.centers" in st:
        model.norm = FuzzyNorm(d, R=st["norm.centers"].shape[0])


_stock = dict(init=TinyLM.__init__, embed=TinyLM.embed, logits=TinyLM.logits)
_state = {}


def _init(self, cfg, *a, **k):
    tiered = _state.get("tiered", False)
    (R._new_init if tiered else _stock["init"])(self, cfg, *a, **k)
    if _state.get("state") is not None:
        install_from_state(self, _state["state"])


if __name__ == "__main__":
    real_load = torch.load

    def load(path, *a, **k):
        ck = real_load(path, *a, **k)
        st = ck["state"]
        tiered = any(key.startswith("emb_h") for key in st)
        if tiered:
            R.NH, R.E = st["emb_h.weight"].shape[0], st["emb_t.weight"].shape[1]
            v = ck["config"]["vocab_size"]
            os.environ["TIER_FREQ"] = (
                "outputs/v22-tribble/ppmi_k48_v1024.npz"
                if v == 1024
                else f"outputs/freq_bpe{v}.npz"
            )
        _state.update(tiered=tiered, state=st)
        TinyLM.__init__ = _init
        TinyLM.embed = R._embed if tiered else _stock["embed"]
        TinyLM.logits = R._logits if tiered else _stock["logits"]
        return ck

    torch.load = load
    C.torch.load = load
    C.main()
