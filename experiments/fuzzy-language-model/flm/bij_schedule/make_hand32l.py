"""Writes hand32L (letter code weight 1.0): python -m flm.bij_schedule.make_hand32l"""

import numpy as np

from flm.canonical_dictionary32 import W, build

C = build(dict(W, letter=1.0))
assert len({tuple(r) for r in C}) == 98
np.savez("outputs/embspace/hand32L.npz", C=C)
s = (C - C.mean(0)) / C.std() * 0.5
d = np.linalg.norm(s[:, None] - s[None], axis=-1)
print(
    "min distance",
    d[np.triu_indices(98, 1)].min().round(3),
    "letters min",
    d[36:62, 36:62][np.triu_indices(26, 1)].min().round(3),
)
