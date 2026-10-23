"""A mixed TinyStories + SimpleStories corpus on one shared tokenizer (`PLAN_ROUND3.md` stage 5): ``python -m flm.prep_mix --chars 330e6 --vocab 6144``.

Both sources are the 98-symbol character streams the other prep scripts already wrote (``data/tinystories/train.bin``, ``data/simplestories/tinystories/train.bin``).  The first ``--chars / 2`` characters of each, cut at a document boundary, are interleaved document by document (always the next document of the source that has contributed fewer characters), so the mix is 50/50 by characters.
Validation: the first ``--valid-chars`` characters of each source's validation stream, cut at a document boundary, as ``valid_ts`` and ``valid_ss`` and, interleaved by document, ``valid`` (the file ``flm.train`` evaluates during a run; ``flm.eval_mix`` scores a checkpoint on all three).
A byte-level BPE of ``--vocab`` tokens is trained on the first ``--fit-chars`` of the mix; everything is encoded with it.  Output is the layout ``GRAD_SCHOOL_DATA=<data>/mix_ts_ss`` + ``--tok bpe<V>`` reads (``SHAPE_CORPUS=mix_ts_ss`` for ``flm.run_shape``), plus ``outputs/freq_mix_ts_ss_bpe<V>.npz`` and ``report/streams/mix_ts_ss_manifest.json``.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO_DATA = Path(__file__).resolve().parents[3] / "data"
EOT = 0  # end-of-text symbol of the 98-symbol stream


def docs_of(ids: np.ndarray, n: int) -> list[np.ndarray]:
    """Documents (each ending with its EOT) of the first ``n`` symbols, the last partial document dropped."""
    seg = np.asarray(ids[:n])
    ends = np.flatnonzero(seg == EOT) + 1
    starts = np.concatenate([[0], ends[:-1]])
    return [seg[s:e] for s, e in zip(starts, ends)]


def interleave(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    """Alternate documents so the two sources stay within one document of equal character counts."""
    out, na, nb, i, j = [], 0, 0, 0, 0
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and na <= nb):
            out.append(a[i])
            na += len(a[i])
            i += 1
        else:
            out.append(b[j])
            nb += len(b[j])
            j += 1
    return np.concatenate(out) if out else np.zeros(0, np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chars",
        type=float,
        default=330e6,
        help="training characters in the mix (half from each source)",
    )
    ap.add_argument("--valid-chars", type=float, default=500_000)
    ap.add_argument("--vocab", type=int, default=6144)
    ap.add_argument("--fit-chars", type=float, default=60e6)
    ap.add_argument("--data-root", default=str(REPO_DATA))
    ap.add_argument("--freq-dir", default="outputs")
    ap.add_argument("--report-dir", default="report/streams")
    ap.add_argument("--name", default="mix_ts_ss")
    a = ap.parse_args(argv)
    data = Path(a.data_root)
    srcs = {
        "ts": data / "tinystories",
        "ss": data / "simplestories" / "tinystories",
    }
    out = data / a.name / "tinystories"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["GRAD_SCHOOL_DATA"] = str(data / a.name)
    import flm.bpe as B  # imported after the environment is set: it resolves paths lazily, but be explicit

    half = int(a.chars / 2)
    tr = {
        k: np.memmap(v / "train.bin", dtype=np.uint8, mode="r") for k, v in srcs.items()
    }
    va = {
        k: np.memmap(v / "valid.bin", dtype=np.uint8, mode="r") for k, v in srcs.items()
    }
    mix = interleave(docs_of(tr["ts"], half), docs_of(tr["ss"], half))
    mix.tofile(out / "train.bin")
    d_ts, d_ss = docs_of(va["ts"], int(a.valid_chars)), docs_of(
        va["ss"], int(a.valid_chars)
    )
    v_ts, v_ss = np.concatenate(d_ts), np.concatenate(d_ss)
    v_ts.tofile(out / "valid_ts.bin")
    v_ss.tofile(out / "valid_ss.bin")
    interleave(d_ts, d_ss).tofile(
        out / "valid.bin"
    )  # balanced, so the first N characters are 50/50 whatever N is
    print(
        f"mix: {len(mix):,} training characters; valid ts {len(v_ts):,}, ss {len(v_ss):,}",
        flush=True,
    )

    B.FIT_CHARS = int(a.fit_chars)
    B.TRAIN_CHARS = len(mix)
    V = a.vocab
    B.prepare(
        V
    )  # trains the tokenizer on the mix, encodes train and valid, checks the token lengths against the stream
    from tokenizers import Tokenizer

    p = B.paths(V)
    tok = Tokenizer.from_file(str(p["tok"]))
    lens = np.load(p["lens"])
    for k, ids in (("ts", v_ts), ("ss", v_ss)):
        n = B.encode_stream(tok, ids, len(ids), out / f"valid_{k}_bpe{V}.bin")
        enc = np.fromfile(out / f"valid_{k}_bpe{V}.bin", dtype=np.uint16)
        assert int(lens[enc].astype(np.int64).sum()) == len(
            ids
        ), k  # same denominator as the character stream
    ids = np.fromfile(p["train"], dtype=np.uint16)
    Path(a.freq_dir).mkdir(exist_ok=True)
    np.savez(
        Path(a.freq_dir) / f"freq_{a.name}_bpe{V}.npz",
        freq=np.bincount(ids, minlength=V).astype(np.float64),
    )
    manifest = dict(
        name=a.name,
        sources={k: str(v) for k, v in srcs.items()},
        train_chars=int(len(mix)),
        valid_chars=dict(ts=int(len(v_ts)), ss=int(len(v_ss))),
        vocab=V,
        train_tokens=int(len(ids)),
        chars_per_token=float(len(mix) / len(ids)),
        sha256=hashlib.sha256(open(p["train"], "rb").read()).hexdigest(),
    )
    Path(a.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.report_dir) / f"{a.name}_manifest.json").write_text(
        json.dumps(manifest, indent=1)
    )
    return manifest


if __name__ == "__main__":
    main()
