"""Writes the dictionary variants for the separation and width tests: python -m flm.bij_schedule.make_variants"""

import numpy as np

from flm.canonical_dictionary import W, build, distortion, project

OUT = "outputs/embspace/"
base = build()
V = {
    "canon64": base,
    "dlin64": build(digit_step=0.25),
    "dlin64b": build(digit_step=0.5),
    "kind64": build(W=dict(W, letter=2.0)),
    "soft64": build(W=dict(W, cls=0.25, flag=0.25, code=0.25, digit=0.25)),
}
for k in (56, 32, 16, 8):
    V[f"canonpca{k}"] = project(base, k)
for n, C in V.items():
    assert len({tuple(np.round(r, 5)) for r in C}) == 98, n
    np.savez(OUT + n + ".npz", C=C)
    extra = f" distortion {distortion(base, C):.3f}" if "pca" in n else ""
    d = np.linalg.norm(C[:, None] - C[None], axis=-1)
    dg = [d[ord(str(a)) - 29, ord(str(b)) - 29] for a, b in ((0, 1), (0, 9))]
    print(n, C.shape, "digit 0-1, 0-9:", np.round(dg, 2), extra)
