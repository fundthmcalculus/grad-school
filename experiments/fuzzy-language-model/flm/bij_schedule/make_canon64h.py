"""Writes canon64H: the 64-wide constructed dictionary with letters as a 5-bit code instead of one flag each.  python -m flm.bij_schedule.make_canon64h"""

import numpy as np

from flm.canonical_dictionary import build

C = build(letter_code=True)
assert len({tuple(r) for r in C}) == 98
np.savez("outputs/embspace/canon64H.npz", C=C)
print("written", C.shape)
