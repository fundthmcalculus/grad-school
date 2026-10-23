"""Stream a deterministic sample of the Pile into the token files every `flm` trainer already reads (`ISSUE_STREAMING.md`, `PLAN_PILE.md`).

    python -m flm.prep_pile --name pile --shards 8 --train-chars 30e6 --valid-chars 2e6 --tokenizer pythia
    python -m flm.prep_pile --name pile --source local:<dir> ...        # a directory of .parquet / .jsonl files with a ``text`` field (tests, offline machines)

Source: ``EleutherAI/the_pile_deduplicated`` at a pinned revision, read with ``datasets`` ``streaming=True`` (only the rows needed are fetched; nothing is cloned).  ``--shards N`` evenly spaced files of the 1,650 each give ``train_chars / N`` training characters and ``valid_chars / N`` validation characters.
A document goes to validation or training by a hash of ``<shard>:<row>`` alone (``h % 1000 < VALID_PERMILLE`` is validation), never by position, so the split does not depend on download order or on the other settings.
Text is folded to the 98-symbol alphabet of the other corpora (tab -> space, carriage return dropped, every other character outside printable ASCII and newline -> ``?``) so that bits per character keeps the same denominator everywhere.  This is a deviation from Pythia's own text; Pythia must be scored on the folded text too for a like-for-like number.
Tokenizer: ``pythia`` (Pythia's 50,277-token vocabulary, pinned revision) or ``bpe:<V>`` (a byte-level BPE trained on the first ``--fit-chars`` folded training characters).  Stage 1 writes the folded text per shard (``text/``); stage 2 tokenises it.  Stage 1 skips a shard whose output already exists and matches its recorded sha256, so an interrupted run resumes.
Writes ``<data>/<name>/tinystories/{bpe<V>.json, lens_bpe<V>.npy, train_bpe<V>.bin, valid_bpe<V>.bin}`` (the layout ``GRAD_SCHOOL_DATA=<data>/<name>`` + ``--tok bpe<V>`` reads), ``outputs/freq_<name>_bpe<V>.npz`` (token counts, for the tiered head) and ``manifest.json`` (copied to ``report/streams/<name>_manifest.json``).
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_DATA = Path(__file__).resolve().parents[3] / "data"
PILE = dict(
    repo="EleutherAI/the_pile_deduplicated",
    rev="fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb",
    n_files=1650,
)
PYTHIA = dict(
    repo="EleutherAI/pythia-14m", rev="cf967c0a9a04383db6f7b1108d86b2962634b4ac"
)
VALID_PERMILLE = 10  # 1% of documents by hash
EOT_CH = "\x1f"  # document separator in the folded text and the end-of-text token of a trained BPE
_BAD = re.compile(r"[^\n -~]")


def fold(text: str) -> str:
    """The 98-symbol alphabet: newline and printable ASCII kept, tab -> space, carriage return dropped, anything else -> question mark."""
    return _BAD.sub("?", text.replace("\r", "").replace("\t", " "))


def doc_hash(shard: int, row: int) -> int:
    h = hashlib.blake2b(f"{shard}:{row}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % 1000


def pick_shards(n_files: int, n: int) -> list[int]:
    """n evenly spaced file indices out of n_files (the first and the last when n >= 2)."""
    return sorted({int(round(i)) for i in np.linspace(0, n_files - 1, n)})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ sources
def list_files(source: str) -> list[str]:
    if source.startswith("local:"):
        d = Path(source[6:])
        return sorted(str(p) for p in d.iterdir() if p.suffix in (".parquet", ".jsonl"))
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(PILE["repo"], revision=PILE["rev"])
    files = sorted(
        s.rfilename for s in info.siblings if s.rfilename.startswith("data/train-")
    )
    if len(files) != PILE["n_files"]:
        raise SystemExit(
            f"expected {PILE['n_files']} shards, the mirror lists {len(files)}"
        )
    return files


def iter_docs(source: str, file: str):
    """Yield the ``text`` of every document of one source file, in file order."""
    if source.startswith("local:"):
        if file.endswith(".jsonl"):
            for line in open(file, encoding="utf-8"):
                yield json.loads(line)["text"]
        else:
            import pandas as pd

            yield from pd.read_parquet(file)["text"].tolist()
        return
    from datasets import load_dataset

    ds = load_dataset(
        PILE["repo"],
        revision=PILE["rev"],
        data_files=[file],
        split="train",
        streaming=True,
    )
    for r in ds:
        yield r["text"]


# ------------------------------------------------------------------ stage 1: folded text per shard
def stage1(source, files, idx, train_q, valid_q, text_dir, entry):
    """Write ``s<idx>.train.txt`` / ``s<idx>.valid.txt`` (folded documents, each ended by EOT_CH) until both quotas are met; returns the manifest entry."""
    tr, va = text_dir / f"s{idx:05d}.train.txt", text_dir / f"s{idx:05d}.valid.txt"
    if (
        entry
        and tr.exists()
        and va.exists()
        and entry["sha256"] == [sha256(tr), sha256(va)]
        and entry["quota"] == [train_q, valid_q]
    ):
        return entry
    nt = nv = rows = 0
    with (
        open(tr, "w", encoding="ascii", newline="") as ft,
        open(va, "w", encoding="ascii", newline="") as fv,
    ):
        for row, text in enumerate(iter_docs(source, files[idx])):
            rows = row + 1
            t = fold(text)
            if not t.strip():
                continue
            to_valid = doc_hash(idx, row) < VALID_PERMILLE
            if to_valid and nv < valid_q:
                fv.write(t + EOT_CH)
                nv += len(t) + 1
            elif not to_valid and nt < train_q:
                ft.write(t + EOT_CH)
                nt += len(t) + 1
            if nt >= train_q and nv >= valid_q:
                break
    if nt < train_q or nv < valid_q:
        print(
            f"warning: shard {idx} ended with {nt:,}/{train_q:,} train and {nv:,}/{valid_q:,} valid characters",
            file=sys.stderr,
        )
    return dict(
        shard=idx,
        file=files[idx],
        rows_read=rows,
        quota=[train_q, valid_q],
        train_chars=nt,
        valid_chars=nv,
        sha256=[sha256(tr), sha256(va)],
    )


# ------------------------------------------------------------------ stage 2: tokenise
def get_tokenizer(kind: str, text_dir: Path, shards, fit_chars: int, out: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if kind == "pythia":
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(PYTHIA["repo"], "tokenizer.json", revision=PYTHIA["rev"])
        shutil.copy(p, out)
        tok = Tokenizer.from_file(str(out))
        return tok, tok.token_to_id("<|endoftext|>")
    v = int(kind.split(":")[1])
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=v,
        special_tokens=[EOT_CH],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def texts():  # the first fit_chars of the folded training text, shard by shard in order
        n = 0
        for idx in shards:
            body = (text_dir / f"s{idx:05d}.train.txt").read_text(encoding="ascii")
            for doc in body.split(EOT_CH):
                if doc:
                    yield doc
                    n += len(doc)
                    if n >= fit_chars:
                        return

    tok.train_from_iterator(texts(), trainer=trainer)
    tok.save(str(out))
    return tok, tok.token_to_id(EOT_CH)


def encode_split(tok, eot, text_dir, shards, split, out_bin):
    """Token ids of all documents (each followed by the end-of-text id); returns (tokens, characters counting one per document end)."""
    n_tok = n_chr = 0
    with open(out_bin, "wb") as fo:
        for idx in shards:
            body = (text_dir / f"s{idx:05d}.{split}.txt").read_text(encoding="ascii")
            docs = [d for d in body.split(EOT_CH) if d]
            for i in range(0, len(docs), 512):
                chunk = docs[i : i + 512]
                enc = tok.encode_batch(chunk, add_special_tokens=False)
                arr = np.fromiter(
                    (t for e in enc for t in (*e.ids, eot)), dtype=np.uint16
                )
                arr.tofile(fo)
                n_tok += len(arr)
                n_chr += sum(len(d) + 1 for d in chunk)
    return n_tok, n_chr


def token_lengths(tok, eot) -> np.ndarray:
    n = tok.get_vocab_size()
    lens = np.zeros(
        n, dtype=np.uint16
    )  # a token can span more than 255 characters (runs of dashes in the Pile)
    for i in range(n):
        lens[i] = 1 if i == eot else len(tok.decode([i]))
    return lens


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pile")
    ap.add_argument("--source", default="hf", help="hf, or local:<dir>")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--train-chars", type=float, default=30e6)
    ap.add_argument("--valid-chars", type=float, default=2e6)
    ap.add_argument("--tokenizer", default="pythia", help="pythia | bpe:<V>")
    ap.add_argument("--fit-chars", type=float, default=20e6)
    ap.add_argument("--data-root", default=str(REPO_DATA))
    ap.add_argument("--freq-dir", default="outputs")
    ap.add_argument("--report-dir", default="report/streams")
    a = ap.parse_args(argv)
    root = Path(a.data_root) / a.name
    text_dir, tok_dir = root / "text", root / "tinystories"
    text_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    mpath = root / "manifest.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    files = list_files(a.source)
    shards = pick_shards(len(files), a.shards)
    tq, vq = int(a.train_chars / len(shards)), int(a.valid_chars / len(shards))
    prev = {e["shard"]: e for e in old.get("shards", [])}
    entries = []
    for i, idx in enumerate(shards):
        entries.append(stage1(a.source, files, idx, tq, vq, text_dir, prev.get(idx)))
        print(
            f"stage 1: shard {idx} ({i + 1}/{len(shards)}) {entries[-1]['train_chars']:,} train chars",
            flush=True,
        )
    tmp = tok_dir / "_tokenizer.json"
    tok, eot = get_tokenizer(a.tokenizer, text_dir, shards, int(a.fit_chars), tmp)
    V = tok.get_vocab_size()
    tok_path = tok_dir / f"bpe{V}.json"
    shutil.move(str(tmp), tok_path)
    lens = token_lengths(tok, eot)
    np.save(tok_dir / f"lens_bpe{V}.npy", lens)
    ntr, ctr = encode_split(
        tok, eot, text_dir, shards, "train", tok_dir / f"train_bpe{V}.bin"
    )
    nva, cva = encode_split(
        tok, eot, text_dir, shards, "valid", tok_dir / f"valid_bpe{V}.bin"
    )
    for split, n_chr in (("valid", cva), ("train", ctr)):
        ids = np.fromfile(tok_dir / f"{split}_bpe{V}.bin", dtype=np.uint16)
        back = int(lens[ids].astype(np.int64).sum())
        if (
            back != n_chr
        ):  # silence is not success: the token lengths must add up to the text
            raise SystemExit(
                f"{split}: token lengths sum to {back:,} but the text has {n_chr:,} characters"
            )
    ids = np.fromfile(tok_dir / f"train_bpe{V}.bin", dtype=np.uint16)
    freq = np.bincount(ids, minlength=V).astype(np.float64)
    Path(a.freq_dir).mkdir(exist_ok=True)
    np.savez(Path(a.freq_dir) / f"freq_{a.name}_bpe{V}.npz", freq=freq)
    import tokenizers

    manifest = dict(
        name=a.name,
        source=a.source if a.source != "hf" else PILE,
        shards=entries,
        valid_permille=VALID_PERMILLE,
        fold="tab to space, carriage return dropped, other non-printable-ASCII to a question mark",
        tokenizer=dict(
            kind=a.tokenizer,
            vocab=V,
            sha256=sha256(tok_path),
            tokenizers_version=tokenizers.__version__,
            **(PYTHIA if a.tokenizer == "pythia" else dict(fit_chars=int(a.fit_chars))),
        ),
        train=dict(
            tokens=ntr,
            chars=ctr,
            chars_per_token=ctr / ntr,
            sha256=sha256(tok_dir / f"train_bpe{V}.bin"),
        ),
        valid=dict(
            tokens=nva,
            chars=cva,
            chars_per_token=cva / nva,
            sha256=sha256(tok_dir / f"valid_bpe{V}.bin"),
        ),
        numpy=np.__version__,
    )
    mpath.write_text(json.dumps(manifest, indent=1))
    out = Path(a.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(mpath, out / f"{a.name}_manifest.json")
    print(
        f"V{V}: train {ntr:,} tokens ({ctr / ntr:.2f} chars/token), valid {nva:,} tokens; manifest {mpath}"
    )
    return manifest


if __name__ == "__main__":
    main()
