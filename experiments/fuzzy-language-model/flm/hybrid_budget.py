"""Parameter accounting for the hybrid recipe at other embedding shapes (`PLAN_SHAPE.md` section 20, the fairness test): the recipe of `flm.run_tiered` (fuzzy delta mixer, first-order TSK feed-forward with ``--ffn-mult`` x d rules per layer, shortconv 8, 8 heads), built on the CPU,
counted, and the rule count R per layer (``--ffn-mult`` = R / d) fitted to sit as close under the cap as possible.  The refit at the old shape (V3072, head 384, tail 12, d48) is the hybrid reference of the fairness test.

    python -m flm.hybrid_budget --table          # fits every arm of ARMS into report/hybrid_arms.json
    from flm.hybrid_budget import count, fit
"""

import json
import sys

import numpy as np

import flm.run_tiered as R
import flm.train as T
from flm.models import TinyLM
from flm.run_tribble import argv

CAP = 99_999
TinyLM.__init__ = R._new_init
TinyLM.embed = R._embed
TinyLM.logits = R._logits
TinyLM.n_params = R._n_params


def count(V=3072, NH=384, E=12, d=48, rules=72, ctx=512):
    a = argv("random", 0)
    a[a.index("--tok") + 1] = "bpe3072"
    a[a.index("--d") + 1] = str(d)
    a[a.index("--ffn-mult") + 1] = str(rules / d)
    a[a.index("--ctx") + 1] = str(ctx)
    args = T.get_args(a)
    args.vocab_size = V
    R.NH, R.E = NH, E
    import os

    np.savez("outputs/_hshape_freq.npz", freq=np.ones(V))
    os.environ["TIER_FREQ"] = "outputs/_hshape_freq.npz"
    return int(TinyLM(T.make_config(args)).n_params())


def fit(cap=CAP, **kw):
    """Largest rule count per layer that keeps the model at or under the cap: (rules, params), None if even 8 rules do not fit."""
    n0 = count(rules=72, **kw)
    slope = count(rules=73, **kw) - n0
    r = max(2, int(72 + (cap - n0) / slope) - 6)
    best = None
    for r in range(r, r + 80):
        n = count(rules=r, **kw)
        if n > cap:
            break
        best = (r, n)
    return best if best and best[0] >= 8 else None


# the hybrid at the old shape, and at the shapes the all-fuzzy search found (and a few neighbours)
ARMS = {
    "hybrid_ref": dict(V=3072, NH=384, E=12, d=48),
    "H1_f7": dict(V=6144, NH=640, E=8, d=32),
    "H2_c6": dict(V=6144, NH=512, E=8, d=40),
    "H3_v6144_h640_d40": dict(V=6144, NH=640, E=8, d=40),
    "H4_f6": dict(V=6144, NH=768, E=8, d=32),
    "H5_v6144_h384_d48": dict(V=6144, NH=384, E=8, d=48),
    "H6_v4096_h384_e10_d48": dict(V=4096, NH=384, E=10, d=48),
    "H7_v6144_h512_d48": dict(V=6144, NH=512, E=8, d=48),
}

if __name__ == "__main__":
    out = {}
    emb = lambda V, NH, E, d: NH * d + (V - NH) * E + E * d
    for name, kw in ARMS.items():
        r = fit(**kw)
        if r is None:
            print(name, "no fit", flush=True)
            continue
        out[name] = dict(
            **kw,
            rules=r[0],
            ffn_mult=round(r[0] / kw["d"], 6),
            params=r[1],
            emb=emb(**kw),
            share=round(emb(**kw) / r[1], 3),
        )
        print(name, out[name], flush=True)
    if "--table" in sys.argv:
        json.dump(out, open("report/hybrid_arms.json", "w"), indent=1)
