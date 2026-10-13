"""50/50 TinyStories + Cosmopedia training stream for the 30M-char mixed-corpus arm (`PLAN_CORPUS.md` C4).

    python -m flm.prep_mixed [chars_per_corpus]

Both corpora are encoded with the *Cosmopedia* BPE1024 (flm.prep_cosmopedia_bpe).  Training stream: TinyStories and Cosmopedia pieces of
~1M characters, cut at document ends (EOT), alternated.  Writes data/mixed/tinystories/{train_bpe1024,valid_bpe1024}.bin (valid = the
Cosmopedia validation stream, so flm.train's own eval scores Cosmopedia), plus ts_valid_bpe1024.bin (the TinyStories validation text in
the same tokenizer) for flm.eval_other.  Run with no GRAD_SCHOOL_DATA; it addresses data/ directly.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

import flm.bpe as B
from flm.data import EOT

DATA = Path(__file__).resolve().parents[3] / "data"
COSMO = DATA / "cosmopedia" / "tinystories"
TS = DATA / "tinystories"
OUT = DATA / "mixed" / "tinystories"
PIECE = 1_000_000


def encode(tok, ids: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (
            t
            for seg in B._chunks(ids, len(ids))
            for e in tok.encode_batch(
                [l + "\n" for l in seg.split("\n")[:-1]]
                + ([seg.split("\n")[-1]] if seg.split("\n")[-1] else [])
            )
            for t in e.ids
        ),
        dtype=np.uint16,
    )


def pieces(tokens: np.ndarray, lens: np.ndarray, n_chars: int):
    """Split a token stream into ~PIECE-character pieces ending at EOT tokens, covering about n_chars."""
    eot = 0  # the special token is the first vocabulary entry
    out, start, acc, total = [], 0, 0, 0
    cl = lens[tokens]
    for i, t in enumerate(tokens):
        acc += cl[i]
        if t == eot and acc >= PIECE:
            out.append(tokens[start : i + 1])
            total += acc
            start, acc = i + 1, 0
            if total >= n_chars:
                break
    return out


if __name__ == "__main__":
    n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 15_000_000
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(COSMO / "bpe1024.json"))
    assert tok.id_to_token(0) == B.EOT_CH
    lens = np.load(COSMO / "lens_bpe1024.npy").astype(np.int64)
    shutil.copy(COSMO / "bpe1024.json", OUT / "bpe1024.json")
    shutil.copy(COSMO / "lens_bpe1024.npy", OUT / "lens_bpe1024.npy")
    shutil.copy(COSMO / "valid_bpe1024.bin", OUT / "valid_bpe1024.bin")
    ts_train = np.fromfile(TS / "train.bin", dtype=np.uint8, count=n + 2_000_000)
    ts_valid = np.fromfile(TS / "valid.bin", dtype=np.uint8, count=5_000_000)
    ts_tok = encode(tok, ts_train)
    encode(tok, ts_valid).tofile(OUT / "ts_valid_bpe1024.bin")
    co = np.fromfile(COSMO / "train_bpe1024.bin", dtype=np.uint16, count=n // 2)
    a = pieces(ts_tok, lens, n)
    b = pieces(co, lens, n)
    k = min(len(a), len(b))
    stream = []
    for i in range(k):
        stream += [a[i], b[i]]
    np.concatenate(stream).astype(np.uint16).tofile(OUT / "train_bpe1024.bin")
    ca = sum(int(lens[x].sum()) for x in a[:k])
    cb = sum(int(lens[x].sum()) for x in b[:k])
    print(f"pieces {k}+{k}: TinyStories {ca:,} chars, Cosmopedia {cb:,} chars")
