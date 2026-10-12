"""Prepare a ~3B-character general-text stream from Cosmopedia v2 (`PLAN_CORPUS.md`; same 98-symbol stream format as flm.data).

    python -m flm.prep_cosmopedia [target_chars]

Source: HuggingFaceTB/smollm-corpus, subset cosmopedia-v2 (synthetic textbooks, stories, WikiHow and web-style articles written by
Mixtral; licence ODC-By), pinned revision below.  Train shards 0.. are consumed in order until `target_chars` is reached; the held-out
validation stream is the first 5M characters of the LAST shard (document-level disjoint).  Per document: curly quotes and dashes are
mapped to ASCII, documents whose remaining non-ASCII byte fraction exceeds 0.5% are dropped (counted), 3+ newlines collapse to 2, and an
EOT symbol ends the document.  Writes data/cosmopedia/tinystories/{train,valid}.bin -- the directory name is deliberate: with
GRAD_SCHOOL_DATA=<repo>/data/cosmopedia every flm.* loader (flm.data, flm.bpe) reads this corpus unchanged.  Nothing here is committed
(AGENTS.md rule 8); `data/.gitignore` records the recovery command.
"""

import json
import sys
from pathlib import Path

import numpy as np

from flm.data import EOT, build_lut

REPO = "HuggingFaceTB/smollm-corpus"
REVISION = "3ba9d605774198c5868892d7a8deda78031a781f"
N_SHARDS = 104
ROOT = Path(__file__).resolve().parents[3] / "data" / "cosmopedia"
OUT = ROOT / "tinystories"
MAP = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    " ": " ",
}
TR = str.maketrans(MAP)


def clean(t: str):
    t = t.translate(TR)
    b = t.encode("utf-8", "replace")
    non = sum(1 for x in b if x >= 128 or (x < 32 and x != 10))
    if not b or non / len(b) > 0.005:
        return None
    t = t.encode("ascii", "replace").decode("ascii")
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t.strip("\n")


def docs(shard: int):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    f = hf_hub_download(
        REPO,
        f"cosmopedia-v2/train-{shard:05d}-of-{N_SHARDS:05d}.parquet",
        repo_type="dataset",
        revision=REVISION,
        local_dir=ROOT / "raw",
    )
    for batch in pq.ParquetFile(f).iter_batches(batch_size=2048, columns=["text"]):
        for t in batch.column("text").to_pylist():
            yield t


def write(path: Path, shards, target: int, stats: dict):
    lut = build_lut()
    n = 0
    with open(path, "wb") as fo:
        for s in shards:
            for t in docs(s):
                c = clean(t)
                stats["docs"] += 1
                if c is None:
                    stats["dropped"] += 1
                    stats["dropped_chars"] += len(t)
                    continue
                arr = np.frombuffer(c.encode("ascii"), dtype=np.uint8)
                ids = lut[arr]
                ids = np.concatenate([ids, np.array([EOT], dtype=np.uint8)])
                ids.tofile(fo)
                n += len(ids)
                stats["kept_chars"] += len(ids)
                if n >= target:
                    break
            print(path.name, "shard", s, f"{n:,} chars", flush=True)
            if n >= target:
                break
    return n


if __name__ == "__main__":
    target = int(float(sys.argv[1])) if len(sys.argv) > 1 else 3_000_000_000
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {
        "revision": REVISION,
        "docs": 0,
        "dropped": 0,
        "dropped_chars": 0,
        "kept_chars": 0,
    }
    nv = write(OUT / "valid.bin", [N_SHARDS - 1], 5_000_000, stats)
    nt = write(OUT / "train.bin", range(N_SHARDS - 1), target, stats)
    stats.update(valid_chars=nv, train_chars=nt)
    (OUT / "prep_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
