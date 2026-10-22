"""``GRAD_SCHOOL_DATA=<repo>/data/tinystories COH_SEEDS=.. python -m flm.coherence_fuzzy out ckpt...``: flm.coherence_set for checkpoints that may contain fuzzy
replacements (flm/fuzzy_parts.py) and tiered embeddings of any shape.  Each checkpoint's architecture is read from its own state dict:
  blocks.*.n1.centers      -> FuzzyNorm            blocks.*.mix.gate.centers  -> FuzzyGate (rules = rows)
  blocks.*.ffn.slope / no ffn.ante -> FuzzyFFNRaw (order 1 / order 0)
  blocks.*.mix.qkv.centers -> TSKMap               blocks.*.mix.qkv.B.weight  -> TSKLinear (k = rows of B)
plus the tiered-embedding detection of flm.coherence_vocab and the fuzzy token tables of flm/fuzzy_tokens.py (``head_proto`` / ``head_pq`` in the state: all tokens; 'ftail' / 'e2e_tail' / 'e2e_prior' / 'lr6_wu' (the e2e_tail warm-up runs) in the checkpoint file name: tail tokens only).
Stage 6-8 head options are read from the state (``prior_alpha``, ``tok_bias``, ``tier_w``, ``out_gain``, ``head_proto_out``, ``mos_pi``, ``head_pq``, ``head_res``); the one thing the state cannot say is the membership function, so a checkpoint whose file name contains 'tanh' or '_tp' (``e2e_all_tanh``, ``e2e_all_tp``, ``fall_tanh*``) is loaded with signed (tanh) memberships.  Stock models load as before.
A checkpoint trained on another corpus needs ``COH_TIER_FREQ`` (its token-frequency file: the head set is the most frequent tokens) and ``GRAD_SCHOOL_DATA`` (its tokenizer); without them the head tokens are the wrong ones and the text is gibberish (coherence set 22).
"""

import os

import torch

import flm.coherence_set as C
import flm.fuzzy_tokens as FT
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
            taps = st[p + "ffn.centers"].shape[1] // d
            blk.ffn = FuzzyFFNRaw(
                d, rules, order=1 if p + "ffn.slope.weight" in st else 0, taps=taps
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
    if _state.get("emb_mode"):
        FT.fuzzify_init(self, _state["emb_mode"])
    if _state.get("state") is not None:
        install_from_state(self, _state["state"])


def configure(path, ck):
    """Set up the model classes so that torch rebuilds this checkpoint's architecture (every replacement, token table and head option read from its state dict and file name)."""
    st = ck["state"]
    tiered = any(key.startswith("emb_h") for key in st)
    if tiered:
        R.NH, R.E = st["emb_h.weight"].shape[0], st["emb_t.weight"].shape[1]
        v = ck["config"]["vocab_size"]
        os.environ["TIER_FREQ"] = os.environ.get("COH_TIER_FREQ") or (
            "outputs/v22-tribble/ppmi_k48_v1024.npz"
            if v == 1024
            else f"outputs/freq_bpe{v}.npz"
        )  # COH_TIER_FREQ: the frequency file the checkpoint was trained with (a different corpus has a different head set)
    stem = os.path.basename(str(path))
    FT.FORM = "tanh" if ("tanh" in stem or "_tp" in stem) else "softmax"
    FT.PRIOR = "prior_alpha" in st
    FT.BIAS = "head" if "tok_bias" in st else ""
    FT.TIER = "tier_w" in st
    FT.GAIN = "out_gain" in st
    FT.UNTIE = "head_proto_out" in st
    FT.PQ = st["head_pq"].shape[0] if "head_pq" in st else 0
    FT.MOS = st["mos_pi"].shape[1] if "mos_pi" in st else 0
    FT.MOS_RANK = st["mos_down"].shape[2] if "mos_down" in st else FT.MOS_RANK
    FT.RES = st["head_res.weight"].shape[1] if "head_res.weight" in st else 0
    tail_names = ("ftail", "e2e_tail", "e2e_prior", "lr6_wu", "fz_lr")
    # every arm of the shape screens (outputs/v69-shape, v70-shape300) is an e2e_tail model (fuzzy blocks, crisp head tokens, fuzzy tail); the hybrid ones (v71-hybshape, v72-hybshape300) are stock
    shape_run = (
        "-shape" in str(path).replace(chr(92), "/") or ck.get("emb_mode") == "tail"
    )  # post-trained checkpoints (flm.post_ul) carry the hint
    emb_mode = (
        "all"
        if "head_proto" in st or "head_pq" in st
        else (
            "tail"
            if tiered and (shape_run or any(n in stem for n in tail_names))
            else ""
        )
    )
    if not emb_mode and (FT.PRIOR or FT.TIER):
        emb_mode = "crisp"
    _state.update(tiered=tiered, state=st, emb_mode=emb_mode)
    TinyLM.__init__ = _init
    TinyLM.embed = FT._embed if emb_mode else (R._embed if tiered else _stock["embed"])
    TinyLM.logits = (
        FT._logits if emb_mode else (R._logits if tiered else _stock["logits"])
    )


if __name__ == "__main__":
    real_load = torch.load

    def load(path, *a, **k):
        ck = real_load(path, *a, **k)
        configure(path, ck)
        return ck

    torch.load = load
    C.torch.load = load
    C.main()
