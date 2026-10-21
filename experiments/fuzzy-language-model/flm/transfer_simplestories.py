"""C0 transfer check (`PLAN_NEXTROUND.md`): the TinyStories checkpoints (BPE3072 of TinyStories) scored in bits per character on the first 1M characters of the SimpleStories validation text,
re-tokenised with the TinyStories tokenizer.  ``python -m flm.transfer_simplestories ckpt ...`` (run from the experiment directory with GRAD_SCHOOL_DATA unset or pointing at the TinyStories data).
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import flm.coherence_fuzzy as CF
import flm.fuzzy_tokens as FT
import flm.run_tiered as R
import flm.train as T
from flm.bpe import BPE, EOT_CH, stream_text
from flm.models import ModelConfig, TinyLM

REPO = Path(__file__).resolve().parents[3]
NATIVE = (
    os.environ.get("SS_NATIVE") == "1"
)  # SimpleStories-trained checkpoints on their own validation stream (the rebuild check)
os.environ["TIER_FREQ"] = (
    "outputs/freq_simplestories_bpe3072.npz" if NATIVE else "outputs/freq_bpe3072.npz"
)


def simple_valid_tokens(bp: BPE, chars=1_000_000):
    ids = np.fromfile(REPO / "data/simplestories/tinystories/valid.bin", dtype=np.uint8)
    text = stream_text(ids[: chars + 2000])
    lines = [l + "\n" for l in text.split("\n")]
    enc = bp.tok.encode_batch(lines)
    return np.fromiter((t for e in enc for t in e.ids), dtype=np.uint16)


if __name__ == "__main__":
    bp = BPE(3072)
    T.TOK = bp
    val = np.asarray(bp.load("valid")) if NATIVE else simple_valid_tokens(bp)
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        st = ck["state"]
        R.NH, R.E = st["emb_h.weight"].shape[0], st["emb_t.weight"].shape[1]
        stem = Path(path).name
        FT.FORM = "tanh" if "tanh" in stem or "_tp" in stem else "softmax"
        tail = any(n in stem for n in ("e2e_tail", "lr6_wu", "fz_lr"))
        CF._state.update(
            tiered=True,
            state=st,
            emb_mode="all" if "head_proto" in st else ("tail" if tail else ""),
        )
        TinyLM.__init__ = CF._init
        TinyLM.embed = FT._embed if CF._state["emb_mode"] else R._embed
        TinyLM.logits = FT._logits if CF._state["emb_mode"] else R._logits
        m = TinyLM(ModelConfig(**ck["config"]))
        m.load_state_dict(st)
        b = T.evaluate(m, val, 512, 1_000_000)
        print(
            f"{path}: {b:.4f} bits per character on SimpleStories text ",
            flush=True,
        )
