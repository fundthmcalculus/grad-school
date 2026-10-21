"""Train a BPE3072 tokenizer on the SimpleStories character stream, encode 700M characters of it (enough for the 30M and 300M runs of `PLAN_NEXTROUND.md` track B) and write the token-frequency file the tiered head needs.

    GRAD_SCHOOL_DATA=<repo>/data/simplestories python -m flm.prep_simplestories_bpe

Writes data/simplestories/tinystories/{bpe3072.json, train_bpe3072.bin, valid_bpe3072.bin, lens_bpe3072.npy} and outputs/freq_simplestories_bpe3072.npz (token counts over the encoded training stream).
"""

import numpy as np

import flm.bpe as B

B.TRAIN_CHARS = 700_000_000

if __name__ == "__main__":
    B.prepare(3072)
    p = B.paths(3072)
    tr = np.fromfile(p["train"], dtype=np.uint16)
    freq = np.bincount(tr, minlength=3072).astype(np.float64)
    np.savez("outputs/freq_simplestories_bpe3072.npz", freq=freq)
    cov = np.sort(freq)[::-1].cumsum() / freq.sum()
    print(
        f"{len(tr):,} training tokens; top-384 tokens cover {cov[383]:.3f} of them; frequency file written"
    )
