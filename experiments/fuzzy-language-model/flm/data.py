"""TinyStories V2 (GPT-4 split) as a character stream for tiny-model experiments.

Vocabulary (98 symbols, fixed, documented):
    0      EOT -- replaces the "<|endoftext|>" story separator (and its surrounding newlines)
    1      UNK -- any byte outside printable ASCII + newline (0.06% of bytes; mostly the
              UTF-8 bytes of curly quotes / dashes)
    2      "\n"
    3..97  printable ASCII 32..126

The metric everywhere is bits per character (BPC) on this mapped stream. Because UNK
collapses the rare non-ASCII bytes, BPC here is not exactly bits-per-byte of the raw
file; every arm sees the same stream, so it is exact for comparisons *between* arms.

Data root: ``$GRAD_SCHOOL_DATA`` (default ``<repo>/data``) / tinystories /. The raw
files are fetched from Hugging Face ``roneneldan/TinyStories`` at a pinned revision
(see ``data/.gitignore``); ``python -m flm.data`` writes ``train.bin`` / ``valid.bin``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HF_REPO = "roneneldan/TinyStories"
HF_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
RAW = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}

EOT, UNK, NL = 0, 1, 2
VOCAB_SIZE = 98
SEP = b"<|endoftext|>"


def data_dir() -> Path:
    root = os.environ.get("GRAD_SCHOOL_DATA")
    base = Path(root) if root else Path(__file__).resolve().parents[3] / "data"
    return base / "tinystories"


def build_lut() -> np.ndarray:
    lut = np.full(256, UNK, dtype=np.uint8)
    lut[ord("\n")] = NL
    for b in range(32, 127):
        lut[b] = 3 + (b - 32)
    return lut


def decode(ids) -> str:
    out = []
    for i in ids:
        i = int(i)
        if i == EOT:
            out.append("\n<|eot|>\n")
        elif i == UNK:
            out.append("�")
        elif i == NL:
            out.append("\n")
        else:
            out.append(chr(i - 3 + 32))
    return "".join(out)


def encode(text: str) -> np.ndarray:
    raw = text.encode("utf-8").replace(b"\n" + SEP + b"\n", SEP).replace(SEP, b"\x00")
    arr = np.frombuffer(raw, dtype=np.uint8)
    ids = build_lut()[arr]
    ids[arr == 0] = EOT
    return ids


def fetch_raw() -> None:
    from huggingface_hub import hf_hub_download

    for f in RAW.values():
        if not (data_dir() / f).exists():
            hf_hub_download(
                HF_REPO,
                f,
                repo_type="dataset",
                revision=HF_REVISION,
                local_dir=data_dir(),
            )


def prepare(split: str, chunk: int = 64 << 20) -> Path:
    src = data_dir() / RAW[split]
    dst = data_dir() / f"{split}.bin"
    lut = build_lut()
    carry = b""
    n = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            block = fi.read(chunk)
            buf = carry + block
            if block:
                # hold back a tail so a separator split across chunks is still matched
                cut = max(0, len(buf) - (len(SEP) + 2))
                buf, carry = buf[:cut], buf[cut:]
            else:
                carry = b""
            buf = buf.replace(b"\n" + SEP + b"\n", SEP).replace(SEP, b"\x00")
            arr = np.frombuffer(buf, dtype=np.uint8)
            ids = lut[arr]
            ids[arr == 0] = EOT
            ids.tofile(fo)
            n += len(ids)
            if not block:
                break
    print(f"{split}: {n:,} chars -> {dst}")
    return dst


def load(split: str) -> np.memmap:
    return np.memmap(data_dir() / f"{split}.bin", dtype=np.uint8, mode="r")


if __name__ == "__main__":
    fetch_raw()
    for s in ("valid", "train"):
        prepare(s)
