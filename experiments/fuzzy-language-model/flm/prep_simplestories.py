"""Prepare a SimpleStories character stream (`PLAN_NEXTROUND.md` track B; same 98-symbol stream format as flm.data).

    python -m flm.prep_simplestories [target_chars]

Source: SimpleStories/SimpleStories (about 2M short stories in simple language written by gpt-4o-mini, parameterised over genre, theme, style and vocabulary level;
Finke et al., NeurIPS 2025 Datasets and Benchmarks, arXiv:2504.09184; licence MIT), pinned revision below; the seven train shards are consumed in order until `target_chars`
is reached and the held-out validation stream is the test shard (document-level disjoint).  The 431 MB `processed.parquet` is not fetched.  Per document: the same character
clean-up as `flm.prep_cosmopedia` (curly quotes and dashes to ASCII, documents with more than 0.5% remaining non-ASCII bytes dropped and counted, 3+ newlines collapsed, an EOT symbol
ends the document).  Writes data/simplestories/tinystories/{train,valid}.bin -- with GRAD_SCHOOL_DATA=<repo>/data/simplestories every flm.* loader reads this corpus unchanged.
Nothing here is committed (AGENTS.md rule 8); `data/.gitignore` records the recovery command.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from flm.data import EOT, build_lut
from flm.prep_cosmopedia import clean

REPO = "SimpleStories/SimpleStories"
REVISION = "e63b8adc3b1a1bdc7cac5b500d150b71346b0628"
ROOT = Path(__file__).resolve().parents[3] / "data" / "simplestories"
OUT = ROOT / "tinystories"
TRAIN = [f"data/train-{i:05d}-of-00007.parquet" for i in range(7)]
TEST = ["data/test-00000-of-00001.parquet"]


def fetch(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            REPO,
            name,
            repo_type="dataset",
            revision=REVISION,
            local_dir=ROOT / "raw",
        )
    )


def docs(name: str, stats: dict):
    import pyarrow.parquet as pq

    f = fetch(name)
    stats["files"][name] = {
        "bytes": f.stat().st_size,
        "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
    }
    for batch in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["story"]):
        for t in batch.column("story").to_pylist():
            yield t


def write(path: Path, files, target: int, stats: dict):
    lut = build_lut()
    n = 0
    with open(path, "wb") as fo:
        for name in files:
            for t in docs(name, stats):
                c = clean(t)
                stats["docs"] += 1
                if c is None:
                    stats["dropped"] += 1
                    stats["dropped_chars"] += len(t)
                    continue
                arr = np.frombuffer(c.encode("ascii"), dtype=np.uint8)
                ids = np.concatenate([lut[arr], np.array([EOT], dtype=np.uint8)])
                ids.tofile(fo)
                n += len(ids)
                stats["kept_chars"] += len(ids)
                if n >= target:
                    break
            print(path.name, name, f"{n:,} chars", flush=True)
            if n >= target:
                break
    return n


if __name__ == "__main__":
    target = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10_000_000_000
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {
        "revision": REVISION,
        "files": {},
        "docs": 0,
        "dropped": 0,
        "dropped_chars": 0,
        "kept_chars": 0,
    }
    nv = write(OUT / "valid.bin", TEST, 50_000_000, stats)
    nt = write(OUT / "train.bin", TRAIN, target, stats)
    stats.update(valid_chars=nv, train_chars=nt)
    (OUT / "prep_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "files"}))
