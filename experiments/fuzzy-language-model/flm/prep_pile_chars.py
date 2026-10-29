"""Character-level corpus from the folded Pile sample: ``python -m flm.prep_pile_chars <pile_dir> <out_root>``.

Reads ``<pile_dir>/text/s*.{train,valid}.txt`` (written by ``flm.prep_pile``, already folded to 98 symbols, documents ended by the unit separator) and writes ``<out_root>/tinystories/{train,valid}.bin`` in the layout every trainer reads (point ``GRAD_SCHOOL_DATA`` at ``<out_root>``): 0 end of text, 1 unknown, 2 newline, 3-97 printable ASCII.
"""

import sys
from pathlib import Path

import numpy as np

from flm.data import EOT, build_lut


def main(pile_dir, out_root):
    lut = build_lut()
    dst = Path(out_root) / "tinystories"
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        n = 0
        with open(dst / f"{split}.bin", "wb") as fo:
            for p in sorted((Path(pile_dir) / "text").glob(f"s*.{split}.txt")):
                arr = np.frombuffer(p.read_bytes(), dtype=np.uint8)
                ids = lut[arr]
                ids[arr == 0x1F] = EOT
                ids.tofile(fo)
                n += len(ids)
        print(f"{split}: {n:,} characters")


if __name__ == "__main__":
    main(*sys.argv[1:3])
