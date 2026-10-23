"""Tests of the Pile streaming prep (`flm/prep_pile.py`) on a synthetic local source: no network, no GPU.  ``pytest experiments/fuzzy-language-model/test_stream.py``"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tokenizers")

from flm import prep_pile as P  # noqa: E402


def make_source(d: Path, n_files=3, n_docs=120):
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    words = [
        "the",
        "cat",
        "sat",
        "on",
        "a",
        "mat",
        "and",
        "then",
        "ran",
        "home",
        "café",
        "世界",
    ]
    for f in range(n_files):
        with open(d / f"part{f}.jsonl", "w", encoding="utf-8") as fo:
            for _ in range(n_docs):
                text = " ".join(rng.choice(words, size=int(rng.integers(5, 60))))
                fo.write(json.dumps({"text": text + "\r\n\tend"}) + "\n")


def run(tmp, name, **kw):
    args = [
        "--name", name, "--source", f"local:{tmp / 'src'}", "--shards", "3",
        "--train-chars", "30000", "--valid-chars", "6000", "--tokenizer", "bpe:300",
        "--fit-chars", "20000", "--data-root", str(tmp / "data"),
        "--freq-dir", str(tmp / "out"), "--report-dir", str(tmp / "rep"),
    ]  # fmt: skip
    return P.main(args)


def test_fold_alphabet():
    t = P.fold("a\tb\r\ncé世")
    assert t == "a b\nc??"
    assert all(c == "\n" or 32 <= ord(c) <= 126 for c in t)


def test_split_is_by_hash_and_disjoint():
    ids = [(s, r) for s in range(3) for r in range(2000)]
    valid = [P.doc_hash(s, r) < P.VALID_PERMILLE for s, r in ids]
    assert 0.005 < np.mean(valid) < 0.02  # about 1% of the documents
    assert [P.doc_hash(s, r) for s, r in ids[:5]] == [
        P.doc_hash(s, r) for s, r in ids[:5]
    ]


def test_pick_shards_spread():
    s = P.pick_shards(1650, 8)
    assert s[0] == 0 and s[-1] == 1649 and len(s) == 8 and s == sorted(s)


def test_deterministic_and_resumable(tmp_path):
    make_source(tmp_path / "src")
    m1 = run(tmp_path, "t")
    tok = tmp_path / "data/t/tinystories"
    V = m1["tokenizer"]["vocab"]
    sha = P.sha256(tok / f"train_bpe{V}.bin")
    # a second run over the same inputs reproduces the same bytes, and reuses the folded text
    stamp = (tmp_path / "data/t/text/s00000.train.txt").stat().st_mtime_ns
    m2 = run(tmp_path, "t")
    assert P.sha256(tok / f"train_bpe{V}.bin") == sha == m2["train"]["sha256"]
    assert (tmp_path / "data/t/text/s00000.train.txt").stat().st_mtime_ns == stamp
    # a fresh directory gives the same tokens (the stream is a function of the manifest, not of the machine)
    m3 = run(tmp_path, "t2")
    assert (
        m3["train"]["sha256"] == sha and m3["valid"]["sha256"] == m1["valid"]["sha256"]
    )


def test_layout_matches_the_loader(tmp_path, monkeypatch):
    make_source(tmp_path / "src")
    m = run(tmp_path, "t")
    V = m["tokenizer"]["vocab"]
    monkeypatch.setenv("GRAD_SCHOOL_DATA", str(tmp_path / "data/t"))
    from flm.bpe import BPE

    bp = BPE(V)  # the loader every trainer uses
    tr, va = bp.load("train"), bp.load("valid")
    assert len(tr) == m["train"]["tokens"] and len(va) == m["valid"]["tokens"]
    assert int(bp.lens[np.asarray(va)].astype(np.int64).sum()) == m["valid"]["chars"]
    assert np.load(tmp_path / f"out/freq_t_bpe{V}.npz")["freq"].sum() == len(tr)
    assert int(np.asarray(tr).max()) < V


def test_valid_documents_follow_the_row_rule(tmp_path):
    make_source(tmp_path / "src")
    run(tmp_path, "t")
    files = P.list_files(f"local:{tmp_path / 'src'}")
    for idx, f in enumerate(files):
        rows = [P.fold(t) for t in P.iter_docs(f"local:{tmp_path / 'src'}", f)]
        want_valid = [
            t for r, t in enumerate(rows) if P.doc_hash(idx, r) < P.VALID_PERMILLE
        ]
        va = (tmp_path / f"data/t/text/s{idx:05d}.valid.txt").read_text()
        got = [d for d in va.split(P.EOT_CH) if d]
        # every validation document came from a validation row, and in file order
        assert all(g in want_valid for g in got)
        assert got == [w for w in want_valid if w in got]


def test_token_lengths_can_exceed_255():
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.BPE(vocab={"-": 0}, merges=[]))
    tok.add_tokens(["-" * 300])
    lens = P.token_lengths(tok, eot=-1)
    assert lens.dtype == np.uint16 and int(lens.max()) == 300
