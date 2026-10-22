"""Train a BPE3072 tokenizer on the SimpleStories character stream, encode 700M characters of it (enough for the 30M and 300M runs of `PLAN_NEXTROUND.md` track B) and write the token-frequency file the tiered head needs.

    GRAD_SCHOOL_DATA=<repo>/data/simplestories python -m flm.prep_simplestories_bpe [vocabulary, default 3072]

Writes data/simplestories/tinystories/{bpe3072.json, train_bpe3072.bin, valid_bpe3072.bin, lens_bpe3072.npy} and outputs/freq_simplestories_bpe3072.npz (token counts over the encoded training stream).
"""

import sys

import numpy as np

import flm.bpe as B

B.TRAIN_CHARS = 700_000_000

if __name__ == "__main__":
    V = int(sys.argv[1]) if len(sys.argv) > 1 else 3072
    B.prepare(V)
    p = B.paths(V)
    tr = np.fromfile(p["train"], dtype=np.uint16)
    freq = np.bincount(tr, minlength=V).astype(np.float64)
    np.savez(f"outputs/freq_simplestories_bpe{V}.npz", freq=freq)
    cov = np.sort(freq)[::-1].cumsum() / freq.sum()
    print(
        f"V{V}: {len(tr):,} training tokens ({B.TRAIN_CHARS / len(tr):.2f} chars/token); top-384 cover {cov[383]:.3f}, top-640 {cov[639]:.3f}; frequency file written"
    )
