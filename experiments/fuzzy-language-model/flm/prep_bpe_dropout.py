"""Stochastic segmentation of the training stream (BPE dropout, Provilkov et al. 2020): ``python -m flm.prep_bpe_dropout <vocab> <chars> <p> <suffix>``.

The tokenizer's merges are skipped with probability p while encoding, so a word has several segmentations whose frequencies are graded rather
than one crisp segmentation: a fuzzification of the tokenizer at training time.  Writes data/tinystories/train_bpe<vocab><suffix>.bin (read with
FLM_TRAIN_SUFFIX=<suffix>); the validation stream stays deterministic, so bits per character are comparable with every other run.
"""

import sys

from tokenizers import Tokenizer

from flm.bpe import encode_stream, paths
from flm.data import load

if __name__ == "__main__":
    v, chars, p, suffix = (
        int(sys.argv[1]),
        int(float(sys.argv[2])),
        float(sys.argv[3]),
        sys.argv[4],
    )
    pp = paths(v)
    tok = Tokenizer.from_file(str(pp["tok"]))
    tok.model.dropout = p
    out = pp["train"].with_name(f"train_bpe{v}{suffix}.bin")
    ntok = encode_stream(tok, load("train"), chars, out)
    print(f"{out.name}: {ntok:,} tokens for {chars:,} characters (dropout {p})")
