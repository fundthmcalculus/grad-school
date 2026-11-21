"""Identity verification of the invertible fuzzy embedding (`PLAN_BIJECTIVE.md`): text in, the same text back, with the intermediate layers doing nothing.

    BIJ=c FUZZ_FFN_R=32 FUZZ_FFN_R0=21 python -m flm.verify_invertible <checkpoint.pt> ["text to round-trip"]

Loads a trained 10K-parameter model of the ``c`` arm (``flm/run_bijective.py``), then for each text:
  ids -> codebook rows C_v -> z = F(C_v)                      (the embedding layer: a trained, exactly invertible order-1 TSK coupling stack)
  h = z                                                       (every block is the identity: nothing between the embedding and the de-embedding)
  u = F^-1(h) ; token = nearest codeword to u                 (the same F run backwards, then the highest-likelihood token)
  text' = decode(token ids)
and reports whether text' == text, in float32 and float64.  A second route sends h through the model's own output stage (final fuzzy norm, F^-1, distance likelihood) to show what the norm does to the identity.
Characters outside the 98-symbol vocabulary map to one UNK symbol before the embedding and come back as that symbol: the layer is exact on the symbols, the vocabulary is the limit.
"""

import os
import sys

import numpy as np
import torch

os.environ.setdefault("BIJ", "c")
import flm.run_bijective as RB  # noqa: E402
from flm.data import VOCAB_SIZE, decode, encode, load  # noqa: E402
from flm.models import ModelConfig, TinyLM  # noqa: E402


def build(path):
    TinyLM.__init__ = RB._new_init
    TinyLM.embed = RB._embed
    TinyLM.logits = RB._logits
    TinyLM.n_params = RB._n_params
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyLM(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state"])
    return model.eval()


@torch.no_grad()
def roundtrip(model, ids, dtype):
    m = model.to(dtype)
    ids = torch.as_tensor(ids, dtype=torch.long)
    z = m.F(m.codebook[ids])[0]  # embedding layer
    h = z  # the blocks: identity
    u = m.F.inverse(h)  # de-embedding: the same layer backwards
    back = torch.cdist(u, m.codebook).argmin(-1)  # highest-likelihood token
    via_model = m.logits(m.embed(ids)).argmax(
        -1
    )  # route 2: the model's own output stage (fuzzy norm in front of F^-1)
    return (
        back.numpy(),
        via_model.numpy(),
        (u - m.codebook[ids]).abs().max().item(),
        (z - m.codebook[ids]).norm(dim=-1).mean().item(),
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    model = build(sys.argv[1])
    n = sum(p.numel() for p in model.parameters())
    print(
        f"model: arm c, {n:,} parameters, vocabulary {VOCAB_SIZE}; F = {len(model.F.steps)} order-1 TSK couplings"
    )
    texts = {
        "pangram": "The quick brown fox jumps over the lazy dog. 0123456789 -- \"Hello, world!\" (Yes?) Tom's mom said: 'ok'.",
        "story (held-out TinyStories)": decode(np.asarray(load("valid")[1000:3000])),
        "all symbols, shuffled x20": decode(
            np.random.default_rng(0).permutation(np.tile(np.arange(VOCAB_SIZE - 1), 20))
        ),
        "out of vocabulary (é, 日本, emoji)": "café 日本 \U0001f600 ok",
    }
    if len(sys.argv) > 2:
        texts = {"your text": sys.argv[2]} | texts
    for name, text in texts.items():
        ids = encode(text)
        want = decode(ids)  # what the vocabulary can represent of the text
        for dtype in (torch.float64, torch.float32):
            back, via, err, disp = roundtrip(model, ids, dtype)
            ok = decode(back) == want
            print(
                f"{name:38s} {str(dtype)[6:]:8s} {len(ids):6d} symbols  identity route: {'EXACT' if ok else 'MISMATCH'} ({int((back != ids).sum())} wrong)  max |F^-1(F(c)) - c| {err:.1e}  | via output stage (norm): {(via == ids).mean() * 100:.1f}% symbols"
            )
        if name == "your text" or name.startswith("pangram"):
            print(f"   in : {text!r}\n   out: {decode(back)!r}")
    print(
        f"embedding displacement |F(c) - c| averages {disp:.2f} (the meaning space is not the codebook space: F moves every codeword)"
    )


if __name__ == "__main__":
    main()
