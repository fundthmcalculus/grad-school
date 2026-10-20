"""Encode a longer training stream with an existing BPE tokenizer: ``python -m flm.prep_bpe_long <vocab> <chars> <suffix>``.

Writes data/tinystories/train_bpe<vocab><suffix>.bin (read through FLM_TRAIN_SUFFIX=<suffix>), the first ``chars`` characters of the
stream encoded with data/tinystories/bpe<vocab>.json.  The BPE1024 `_1b` (about 1.3B characters) and `_all` (the whole 2.19B) streams
were made the same way; ``chars`` of 0 means the whole stream.
"""

import sys

from tokenizers import Tokenizer

from flm.bpe import encode_stream, paths
from flm.data import load

if __name__ == "__main__":
    v, chars, suffix = int(sys.argv[1]), int(float(sys.argv[2])), sys.argv[3]
    p = paths(v)
    ids = load("train")
    n = chars or len(ids)
    out = p["train"].with_name(f"train_bpe{v}{suffix}.bin")
    tok = Tokenizer.from_file(str(p["tok"]))
    ntok = encode_stream(tok, ids, min(n, len(ids)), out)
    print(f"{out.name}: {ntok:,} tokens for {min(n, len(ids)):,} characters")
