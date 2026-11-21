"""Score an all-fuzzy checkpoint on the mixed corpus's three validation sets: ``GRAD_SCHOOL_DATA=<data>/mix_ts_ss python -m flm.eval_mix <checkpoint.pt> [--vocab 6144] [--chars 1000000]``.

Prints bits per character on ``valid_ts`` (TinyStories), ``valid_ss`` (SimpleStories) and their concatenation ``valid`` (what `flm.train` reports during a run); the same non-overlapping windows and denominator as `flm.train.evaluate`.  The checkpoint must have been trained with the tokenizer of this corpus (``flm.coherence_fuzzy.configure`` rebuilds the fuzzy token tables from it).
"""

import argparse
import os

import numpy as np
import torch

import flm.coherence_fuzzy as CF
import flm.train as T
from flm.bpe import BPE, paths
from flm.models import ModelConfig, TinyLM

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--vocab", type=int, default=6144)
    ap.add_argument("--chars", type=int, default=1_000_000)
    ap.add_argument(
        "--freq",
        default="outputs/freq_mix_ts_ss_bpe{V}.npz",
        help="token-frequency file the checkpoint was trained with ({V} = the vocabulary)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ck.setdefault(
        "emb_mode", "tail"
    )  # a checkpoint outside outputs/*-shape/ would otherwise be rebuilt with the stock tiered embedding
    os.environ["COH_TIER_FREQ"] = a.freq.format(V=a.vocab)
    CF.configure(a.ckpt, ck)
    m = TinyLM(ModelConfig(**ck["config"]))
    m.load_state_dict(ck["state"])
    dev = torch.device(a.device)
    m.to(dev).eval()
    T.TOK = BPE(a.vocab)
    d = paths(a.vocab)["valid"].parent
    for split in ("valid_ts", "valid_ss", "valid"):
        val = np.fromfile(d / f"{split}_bpe{a.vocab}.bin", dtype=np.uint16)
        print(split, round(T.evaluate(m, val, 512, a.chars, device=dev), 5))
