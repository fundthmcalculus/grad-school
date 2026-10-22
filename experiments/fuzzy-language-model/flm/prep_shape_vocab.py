"""Tokenizers for the vocabulary arms of `PLAN_SHAPE.md` (E3): ``python -m flm.prep_shape_vocab 2560 6144 8192``.

For each vocabulary: train the byte-level BPE on the TinyStories stream and encode the train and validation streams (`flm.bpe.prepare`: 600M characters), then write the token-frequency file the tiered head needs
(outputs/freq_bpe<V>.npz, token counts over the encoded training stream) and print characters per token and the share of training tokens the most frequent 384 / 256 / 192 tokens cover.
"""

import sys

import numpy as np

import flm.bpe as B

if __name__ == "__main__":
    for v in (int(x) for x in sys.argv[1:]):
        B.prepare(v)
        p = B.paths(v)
        tr = np.fromfile(p["train"], dtype=np.uint16)
        freq = np.bincount(tr, minlength=v).astype(np.float64)
        np.savez(f"outputs/freq_bpe{v}.npz", freq=freq)
        cov = np.sort(freq)[::-1].cumsum() / freq.sum()
        print(
            f"V{v}: {len(tr):,} training tokens ({B.TRAIN_CHARS / len(tr):.2f} chars/token); top 384 cover {cov[383]:.3f}, top 256 {cov[255]:.3f}, top 192 {cov[191]:.3f}",
            flush=True,
        )
