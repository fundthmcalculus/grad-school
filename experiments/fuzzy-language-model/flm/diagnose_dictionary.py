"""Where does the extra loss of a frozen dictionary fall?  Bits per character by the class of the next character, and by the class of the current one, for several checkpoints.

    BIJ=ci FUZZ_FFN_R=328 FUZZ_FFN_R0=230 python -m flm.diagnose_dictionary OUT.json name=CKPT [name=CKPT ...]

Classes: lower-case letter, upper-case letter, digit, space, newline, mark (any other printable), special (end of text, unknown).  Also the top-1 accuracy per class, the codebook's minimum and mean pairwise distance, and tau.
"""

import json
import math
import sys

import numpy as np
import torch

from flm.data import decode
from flm.explain_bijective import build, windows

CLASSES = ["lower", "upper", "digit", "space", "newline", "mark", "special"]


def cls(i):
    if i < 2:
        return 6
    if i == 2:
        return 4
    ch = decode([i])
    if ch == " ":
        return 3
    if ch.isalpha():
        return 0 if ch.islower() else 1
    if ch.isdigit():
        return 2
    return 5


LUT = np.array([cls(i) for i in range(98)])


@torch.no_grad()
def one(path, dev, x, y):
    model = build(path, dev)
    lg = []
    for i in range(0, x.shape[0], 64):
        lg.append(model(x[i : i + 64]).float())
    lg = torch.cat(lg)
    nll = torch.nn.functional.cross_entropy(
        lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none"
    ).cpu().numpy() / math.log(2)
    acc = (lg.argmax(-1) == y).reshape(-1).cpu().numpy()
    ys, xs = y.reshape(-1).cpu().numpy(), x.reshape(-1).cpu().numpy()
    out = dict(bpc=float(nll.mean()), tau=float(torch.exp(model.log_tau)))
    cb = model.codebook.detach().cpu().numpy()
    d = np.linalg.norm(cb[:, None] - cb[None], axis=-1)[np.triu_indices(98, 1)]
    out.update(code_min=float(d.min()), code_mean=float(d.mean()))
    for name, key in (("next", LUT[ys]), ("current", LUT[xs])):
        out[name] = {
            CLASSES[c]: dict(
                share=float((key == c).mean()),
                bits=float(nll[key == c].mean()),
                acc=float(acc[key == c].mean()),
                contrib=float(nll[key == c].sum() / len(nll)),
            )
            for c in range(7)
            if (key == c).any()
        }
    top = np.argsort(-np.bincount(ys, minlength=98))[:12]
    out["top_chars"] = {
        decode([int(c)]).replace("\n", "\\n"): dict(
            bits=float(nll[ys == c].mean()), acc=float(acc[ys == c].mean())
        )
        for c in top
    }
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = windows(512, dev)
    res = {}
    for a in sys.argv[2:]:
        name, path = a.split("=", 1)
        res[name] = one(path, dev, x, y)
        r = res[name]
        print(
            name,
            "bpc %.4f tau %.3f codebook min %.2f mean %.2f"
            % (r["bpc"], r["tau"], r["code_min"], r["code_mean"]),
        )
        for k in CLASSES:
            v = r["next"].get(k)
            if v:
                print(
                    "   next %-8s share %.3f bits %.3f acc %.3f contributes %.3f"
                    % (k, v["share"], v["bits"], v["acc"], v["contrib"])
                )
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
