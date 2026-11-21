"""SuperBPE at the SAME vocabulary size as BPE6144 (`PLAN_FUTURE.md` idea 2): ``python -m flm.prep_superbpe_eq [n_merges, default 512]``.

The existing BPE6144 is cut back to its first ``6144 - n`` merges (a BPE trained on the same text is a prefix-extension of itself, so this is exactly the BPE of that smaller vocabulary); ``n`` superword merges, learned on the stage-1 token stream with no word-boundary restriction (`flm.prep_superbpe.learn_merges`), take the remaining ids.  Vocabulary size, and so the embedding cost, is unchanged; characters per token goes up because frequent phrases ("once upon a time") become single tokens.
Writes the corpus ``data/sb6144/tinystories/{bpe6144.json, lens_bpe6144.npy, train_bpe6144.bin, valid_bpe6144.bin}`` (select it with ``SHAPE_CORPUS=sb6144``; the character streams are read from ``data/tinystories``, nothing there is touched), ``outputs/freq_sb6144_bpe6144.npz`` and ``report/streams/sb6144_manifest.json``.
Checks printed and asserted: the truncated base reproduces BPE6144's token ids on a validation slice for ids below the cut (a manipulation check that the base is the prefix), the validation token lengths add up to the validation characters, and the same 1,000,000 validation characters are scored as for every other tokenizer.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from flm.bpe import encode_stream, token_lengths
from flm.data import data_dir, load
from flm.prep_superbpe import build_tokenizer, learn_merges

V = 6144
LEARN_TOKENS = 60_000_000  # stage-1 tokens the superword merges are learned on
TRAIN_CHARS = 340_000_000  # characters of the training stream encoded (enough for 300M-character runs)
ROOT = Path(__file__).resolve().parents[3] / "data" / "sb6144" / "tinystories"


def truncate_base(src_json: Path, cut: int, out_json: Path):
    """The first ``cut`` ids of a trained byte-level BPE (id 0 the separator, 1-256 the bytes, then merges in order)."""
    j = json.loads(src_json.read_text(encoding="utf-8"))
    vocab = {t: i for t, i in j["model"]["vocab"].items() if i < cut}
    j["model"]["vocab"] = vocab
    j["model"]["merges"] = j["model"]["merges"][: cut - 257]
    out_json.write_text(json.dumps(j), encoding="utf-8")
    return Tokenizer.from_file(str(out_json))


def main(n_merges: int):
    ROOT.mkdir(parents=True, exist_ok=True)
    src = data_dir() / f"bpe{V}.json"
    cut = V - n_merges
    base_json = ROOT / f"_base_bpe{cut}.json"
    base = truncate_base(src, cut, base_json)
    full = Tokenizer.from_file(str(src))
    ids_tr, ids_va = load("train"), load("valid")
    # manipulation check: on text without any token of id >= cut the truncated base and the full tokenizer agree
    from flm.bpe import stream_text

    sample = stream_text(ids_va[:200_000]).split("\n")[:300]
    agree = sum(
        1
        for s in sample
        if max(full.encode(s).ids, default=0) < cut
        and base.encode(s).ids == full.encode(s).ids
    )
    only_low = sum(1 for s in sample if max(full.encode(s).ids, default=0) < cut)
    assert agree == only_low and only_low > 50, (agree, only_low)
    print(
        f"base BPE{cut} reproduces BPE{V} on {agree}/{only_low} lines that use no id >= {cut}",
        flush=True,
    )
    t0 = time.time()
    stage1 = ROOT / f"_stage1_bpe{cut}.bin"
    n1 = encode_stream(base, ids_tr, int(LEARN_TOKENS * 4.2), stage1)
    stream = np.fromfile(stage1, dtype=np.uint16, count=LEARN_TOKENS)
    print(
        f"stage-1 stream: {len(stream):,} tokens ({time.time() - t0:.0f}s)", flush=True
    )
    merges, n_after = learn_merges(stream, cut, n_merges)
    print(
        f"stage-1 tokens {len(stream):,} -> {n_after:,} ({len(stream) / n_after:.2f}x fewer)",
        flush=True,
    )
    tok = build_tokenizer(base_json, merges, cut)
    assert tok.get_vocab_size() == V, tok.get_vocab_size()
    tok.save(str(ROOT / f"bpe{V}.json"))
    lens = token_lengths(tok)
    np.save(ROOT / f"lens_bpe{V}.npy", lens)
    ntr = encode_stream(tok, ids_tr, TRAIN_CHARS, ROOT / f"train_bpe{V}.bin")
    nva = encode_stream(tok, ids_va, len(ids_va), ROOT / f"valid_bpe{V}.bin")
    va = np.fromfile(ROOT / f"valid_bpe{V}.bin", dtype=np.uint16)
    assert int(lens[va].astype(np.int64).sum()) == len(ids_va), "length check failed"
    tr = np.fromfile(ROOT / f"train_bpe{V}.bin", dtype=np.uint16)
    cpt = float(TRAIN_CHARS / ntr)
    print(
        f"sb{V}: train {ntr:,} tokens ({cpt:.3f} chars/token), valid {nva:,} tokens ({len(ids_va) / nva:.3f}); length check ok",
        flush=True,
    )
    Path("outputs").mkdir(exist_ok=True)
    np.savez(
        f"outputs/freq_sb{V}_bpe{V}.npz",
        freq=np.bincount(tr[:100_000_000], minlength=V).astype(np.float64),
    )
    # the longest new tokens, for the record
    inv = {i: t for t, i in tok.get_vocab().items()}
    new = [(i, inv[i]) for i in range(cut, V)]
    show = [tok.decode([i]) for i, _ in new[:12]] + [
        tok.decode([i]) for i, _ in new[-4:]
    ]
    manifest = dict(
        vocab=V,
        base_vocab=cut,
        superword_merges=n_merges,
        learn_tokens=len(stream),
        train_chars=TRAIN_CHARS,
        train_tokens=int(ntr),
        chars_per_token_train=cpt,
        chars_per_token_valid=len(ids_va) / nva,
        first_and_last_new_tokens=show,
        tokenizer_sha256=hashlib.sha256(
            (ROOT / f"bpe{V}.json").read_bytes()
        ).hexdigest(),
    )
    out = Path("report/streams")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sb6144_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("new tokens:", show, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 512)
