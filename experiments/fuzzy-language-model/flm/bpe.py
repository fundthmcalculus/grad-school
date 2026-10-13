"""Subword (BPE) token stream with the same metric as the character stream.

The corpus is the SAME 98-symbol stream ``flm.data`` builds, rendered back to ASCII text
(UNK -> '?', EOT -> a single \x1f control character that is a special token). A byte-level BPE is trained on the
first ``TRAIN_CHARS`` characters and the stream is encoded to ``uint16``. Every token knows its length
in characters (EOT counts as 1, exactly as in the character stream), so

    BPC = total cross-entropy (bits) / total characters of the target tokens

is directly comparable with every character-level number in this experiment: same validation text
(first 1M characters of valid.bin), same denominator. Only the context length in *characters* differs
(256 tokens cover ~3-4x more text).

    python -m flm.bpe 512            # train the tokenizer and encode train/valid
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from flm.data import EOT, NL, UNK, data_dir, load

EOT_CH = "\x1f"
TRAIN_CHARS = (
    600_000_000  # how much of the stream is encoded (enough for 300M-char runs)
)
FIT_CHARS = 120_000_000  # how much of it the tokenizer is trained on


def _inverse() -> np.ndarray:
    inv = np.zeros(98, dtype=np.uint8)
    inv[EOT] = ord(EOT_CH)
    inv[UNK] = ord("?")
    inv[NL] = ord("\n")
    for i in range(3, 98):
        inv[i] = 32 + (i - 3)
    return inv


def stream_text(ids: np.ndarray) -> str:
    return _inverse()[np.asarray(ids)].tobytes().decode("ascii")


def paths(v: int):
    d = data_dir()
    return {
        "tok": d / f"bpe{v}.json",
        # FLM_TRAIN_SUFFIX=_1b reads the longer stream (same tokenizer, identical prefix)
        "train": d / f"train_bpe{v}{os.environ.get('FLM_TRAIN_SUFFIX', '')}.bin",
        "valid": d / f"valid_bpe{v}.bin",
        "lens": d / f"lens_bpe{v}.npy",
    }


def _chunks(ids: np.ndarray, n: int, size: int = 4_000_000):
    """Yield text pieces of ~size chars, cut at newlines, covering ids[:n]."""
    pos = 0
    while pos < n:
        end = min(n, pos + size)
        seg = stream_text(ids[pos:end])
        if end < n:
            cut = seg.rfind("\n")
            if cut > 0:
                seg = seg[: cut + 1]
        yield seg
        pos += len(seg)


def train_tokenizer(v: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=v,
        special_tokens=[EOT_CH],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    ids = load("train")
    tok.train_from_iterator(_chunks(ids, FIT_CHARS), trainer=trainer)
    return tok


def token_lengths(tok) -> np.ndarray:
    n = tok.get_vocab_size()
    lens = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        lens[i] = 1 if tok.id_to_token(i) == EOT_CH else len(tok.decode([i]))
    return lens


def encode_stream(tok, ids: np.ndarray, n: int, out: Path) -> int:
    total = 0
    with open(out, "wb") as fo:
        for seg in _chunks(ids, n):
            lines = seg.split("\n")
            lines = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
            enc = tok.encode_batch(lines)
            arr = np.fromiter((t for e in enc for t in e.ids), dtype=np.uint16)
            arr.tofile(fo)
            total += len(arr)
    return total


def prepare(v: int) -> None:
    from tokenizers import Tokenizer

    p = paths(v)
    tok = train_tokenizer(v)
    tok.save(str(p["tok"]))
    lens = token_lengths(tok)
    np.save(p["lens"], lens)
    ntr = encode_stream(tok, load("train"), TRAIN_CHARS, p["train"])
    nva = encode_stream(tok, load("valid"), len(load("valid")), p["valid"])
    cpt = TRAIN_CHARS / ntr
    print(f"bpe{v}: train {ntr:,} tokens ({cpt:.2f} chars/token), valid {nva:,} tokens")
    # round-trip check on the validation stream
    va = np.fromfile(p["valid"], dtype=np.uint16)
    back = sum(int(lens[t]) for t in va)
    assert back == len(load("valid")), (back, len(load("valid")))
    print("length check ok: token char lengths sum to the stream length")


class BPE:
    def __init__(self, v: int):
        from tokenizers import Tokenizer

        p = paths(v)
        self.v = v
        self.tok = Tokenizer.from_file(str(p["tok"]))
        self.lens = np.load(p["lens"])
        self._p = p
        self.eot = self.tok.token_to_id(EOT_CH)

    def load(self, split: str) -> np.memmap:
        return np.memmap(self._p[split], dtype=np.uint16, mode="r")

    def chars_per_token(self, split: str = "train", n: int = 5_000_000) -> float:
        a = np.asarray(self.load(split)[:n])
        return float(self.lens[a].astype(np.int64).sum() / len(a))

    def encode(self, text: str) -> np.ndarray:
        return np.array(self.tok.encode(text).ids, dtype=np.int64)

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            out.append("\n<|eot|>\n" if i == self.eot else self.tok.decode([i]))
        return "".join(out)


if __name__ == "__main__":
    prepare(int(sys.argv[1]))
