"""Builds the dictionary-drift page: constructed dictionary against the trained codebooks (300M, seed 0).  python -m flm.drift_page [out.html]"""

import json
import sys

import numpy as np
import torch

from flm.data import decode

SEED = 0
CKPT = "outputs/canon300/{arm}_s%d/ci_lr0.01_300M.pt" % SEED


def codebook(arm):
    sd = torch.load(CKPT.format(arm=arm), map_location="cpu", weights_only=False)
    return sd.get("state", sd.get("model", sd))["codebook"].numpy()


def label(i):
    return {0: "EOT", 1: "UNK", 2: "⏎", 3: "␣"}.get(i) or decode([i])


def group(i):
    ch = decode([i])
    if i < 4:
        return 0
    if ch.isalpha():
        return 1
    if ch.isdigit():
        return 2
    return 3


def dist(M):
    return np.linalg.norm(M[:, None] - M[None], axis=-1)


def align(c0, t):
    """Scale t to the size of c0 and rotate it onto c0 (orthogonal Procrustes)."""
    a, b = c0 - c0.mean(0), t - t.mean(0)
    b = b * (np.linalg.norm(a) / np.linalg.norm(b))
    u, _, vt = np.linalg.svd(b.T @ a)
    return b @ (u @ vt)


def arm_data(c0, t):
    iu = np.triu_indices(98, 1)
    d0, d1 = dist(c0), dist(t)
    prof = [
        float(np.corrcoef(np.delete(d0[i], i), np.delete(d1[i], i))[0, 1])
        for i in range(98)
    ]
    nn = lambda d, i: [int(j) for j in np.argsort(d[i])[1:6]]
    keep = [len(set(nn(d0, i)) & set(nn(d1, i))) for i in range(98)]
    a = align(c0, t)
    u, sv, vt = np.linalg.svd(c0 - c0.mean(0), full_matrices=False)
    p0 = (c0 - c0.mean(0)) @ vt[:2].T
    p1 = a @ vt[:2].T
    return dict(
        corr=float(np.corrcoef(d0[iu], d1[iu])[0, 1]),
        x=[round(float(v), 3) for v in (d0[iu] / d0[iu].mean())[::3]],
        y=[round(float(v), 3) for v in (d1[iu] / d1[iu].mean())[::3]],
        prof=[round(v, 3) for v in prof],
        keep=keep,
        nn0=[nn(d0, i) for i in range(98)],
        nn1=[nn(d1, i) for i in range(98)],
        p0=p0.round(3).tolist(),
        p1=p1.round(3).tolist(),
        moved=[
            round(float(v), 3)
            for v in np.linalg.norm(a - (c0 - c0.mean(0)), axis=1)
        ],
    )


def main():
    c0 = np.load("outputs/embspace/canon64.npz")["C"]
    data = dict(
        labels=[label(i) for i in range(98)],
        groups=[group(i) for i in range(98)],
        before=((c0 - c0.mean(0)) / c0.std() * 0.5).round(3).tolist(),
        after=codebook("canonInit").round(3).tolist(),
        start=arm_data(c0, codebook("canonInit")),
        random=arm_data(c0, codebook("learned")),
    )
    tpl = open("flm/drift_template.html", encoding="utf-8").read()
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs/drift/index.html"
    open(out, "w", encoding="utf-8").write(
        tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    )
    print(data["start"]["corr"], data["random"]["corr"])


if __name__ == "__main__":
    main()
