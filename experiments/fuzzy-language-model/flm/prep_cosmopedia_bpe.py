"""Train a BPE1024 tokenizer on the Cosmopedia character stream and encode all of it (`PLAN_CORPUS.md` arm T2).

    GRAD_SCHOOL_DATA=<repo>/data/cosmopedia python -m flm.prep_cosmopedia_bpe

flm.bpe's TRAIN_CHARS (600M, enough for the TinyStories 300M runs) is raised to the full 3B so one stream serves every run length.
"""

import flm.bpe as B

B.TRAIN_CHARS = 3_000_000_000

if __name__ == "__main__":
    B.prepare(1024)
