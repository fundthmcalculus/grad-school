"""Parameter accounting for the shape experiments (`PLAN_SHAPE.md`): build the all-fuzzy model (`e2e_tail` recipe: fuzzy blocks, crisp head tokens, fuzzy tail table) for a given vocabulary, head count, tail width and block shape on the CPU,
count its parameters, and fit the feed-forward rule count (layer 1 / layer 0 in the ratio of the recipe, 52 / 35) so the total sits as close under the cap as possible.  No training, no tokenizer needed.

    python -m flm.shape_budget                 # the recipe (expect 99,019) and the PLAN_SHAPE grid
    from flm.shape_budget import count, fit
"""

import os
import sys

from flm.run_fuzzy_e2e import ARMS as E2E

_envx, _over, _mult = E2E["e2e_tail"]
os.environ.update(_envx)
os.environ.update(TIER_NH="384", TIER_E="12", TIER_FREQ="outputs/freq_bpe3072.npz")

import numpy as np  # noqa: E402

import flm.fuzzy_tokens as FT  # noqa: E402
import flm.run_allfuzzy as RA  # noqa: E402
import flm.run_tiered as R  # noqa: E402
import flm.train as T  # noqa: E402
from flm.models import TinyLM  # noqa: E402
from flm.run_tribble import argv  # noqa: E402

CAP = 99_999
TinyLM.__init__ = RA._init
TinyLM.embed = FT._embed
TinyLM.logits = FT._logits
TinyLM.n_params = R._n_params


def count(
    V=3072,
    NH=384,
    E=12,
    d=48,
    layers=2,
    heads=None,
    rules=52,
    rules0=35,
    mult=None,
    ctx=512,
    extra=(),
    hr=0,
    mid=(),
):
    """Parameters of the all-fuzzy model with these shapes (`extra`: more train-args, e.g. ('--tsk-dim', '12'))."""
    a = argv("random", 0)
    a[a.index("--tok") + 1] = "bpe3072"
    a[a.index("--ffn-mult") + 1] = mult or _mult
    a[a.index("--d") + 1] = str(d)
    a[a.index("--layers") + 1] = str(layers)
    a[a.index("--ctx") + 1] = str(ctx)
    for k, v in _over.items():
        a[a.index(k) + 1] = v
    if heads is not None:  # default: the recipe's own head count (8)
        if "--heads" in a:
            a[a.index("--heads") + 1] = str(heads)
        else:
            a += ["--heads", str(heads)]
    a += list(extra)
    args = T.get_args(a)
    args.vocab_size = V
    R.NH, R.E = NH, E
    FT.HEADRANK = hr
    FT.MID = tuple(mid)
    RA._ffn_r, RA._r0 = rules, rules0
    freq = np.ones(V)
    np.savez("outputs/_shape_freq.npz", freq=freq)
    os.environ["TIER_FREQ"] = "outputs/_shape_freq.npz"
    m = TinyLM(T.make_config(args))
    return int(m.n_params())  # the project's count (the one in every result file)


def fit(cap=CAP, **kw):
    """Largest layer-1 rule count (layer 0 at 35/52 of it) that keeps the model at or under the cap; returns (rules, rules0, params)."""
    n0 = count(rules=52, rules0=35, **kw)
    slope = count(rules=53, rules0=round(53 * 35 / 52), **kw) - n0
    r = max(2, int(52 + (cap - n0) / slope) - 6)
    best = None
    for r in range(r, r + 60):
        r0 = max(1, round(r * 35 / 52))
        n = count(rules=r, rules0=r0, **kw)
        if n > cap:
            break
        best = (r, r0, n)
    return best


# arms of PLAN_SHAPE.md (stage 1); V / head (NH) / tail width (E) / block width d; the rule counts are fitted by `python -m flm.shape_budget --table`
ARMS = {
    "ref": dict(V=3072, NH=384, E=12, d=48),
    "E1a_h192": dict(V=3072, NH=192, E=12, d=48),
    "E1b_h288": dict(V=3072, NH=288, E=12, d=48),
    "E1c_h512": dict(V=3072, NH=512, E=12, d=48),
    "E1d_h640": dict(V=3072, NH=640, E=12, d=48),
    "E2a_e8": dict(V=3072, NH=384, E=8, d=48),
    "E2b_h192e16": dict(V=3072, NH=192, E=16, d=48),
    "E2c_h256e16": dict(V=3072, NH=256, E=16, d=48),
    "E3a_v2560": dict(V=2560, NH=384, E=12, d=48),
    "E3b_v4096e8": dict(V=4096, NH=384, E=8, d=48),
    "E3c_v4096e10": dict(V=4096, NH=384, E=10, d=48),
    "E3d_v6144h256": dict(V=6144, NH=256, E=6, d=48),
    "E3e_v6144h192": dict(V=6144, NH=192, E=6, d=48),
    "E3f_v8192": dict(V=8192, NH=256, E=6, d=48),
    "E4a_d32": dict(V=3072, NH=512, E=12, d=32),
    "E4b_d40": dict(V=3072, NH=448, E=12, d=40),
    "E4c_d56": dict(V=3072, NH=320, E=12, d=56),
    "E4d_d64": dict(V=3072, NH=192, E=10, d=64),
    # E7: a reversible shared embedding (`PLAN_SHAPE.md`): orthonormal prototypes (ortho), a cycle loss (cycle), a rank-limited head (hr)
    "E7a_ortho1": dict(V=3072, NH=384, E=12, d=48, ortho=1.0),
    "E7b_ortho01": dict(V=3072, NH=384, E=12, d=48, ortho=0.1),
    "E7c_cycle005": dict(V=3072, NH=384, E=12, d=48, cycle=0.05),
    "E7d_cycle02": dict(V=3072, NH=384, E=12, d=48, cycle=0.2),
    "E7e_hr32": dict(V=3072, NH=384, E=12, d=48, hr=32),
    "E7f_hr24": dict(V=3072, NH=384, E=12, d=48, hr=24),
    "E7g_hr16": dict(V=3072, NH=384, E=12, d=48, hr=16),
    "E7h_hr24ortho": dict(V=3072, NH=384, E=12, d=48, hr=24, ortho=1.0),
    "E7i_ortho_scaled": dict(V=3072, NH=384, E=12, d=48, ortho=1.0, ortho_scaled=1),
    # E8: combinations of the stage 1 signals (a larger vocabulary, a larger head, d 40)
    "C1_v4096_h512": dict(V=4096, NH=512, E=10, d=48),
    "C2_v4096_h384_d40": dict(V=4096, NH=384, E=10, d=40),
    "C3_v4096_h512_d40": dict(V=4096, NH=512, E=10, d=40),
    "C4_v6144_h384": dict(V=6144, NH=384, E=8, d=48),
    "C5_v4096_h640_e8": dict(V=4096, NH=640, E=8, d=48),
    "C6_v6144_h512_d40": dict(V=6144, NH=512, E=8, d=40),
    # E9: a second round of combinations around the best arm (a large vocabulary with a large head), fitted
    "D4_v6144_h384_d40": dict(V=6144, NH=384, E=8, d=40),
    "D6_v6144_h448_e6": dict(V=6144, NH=448, E=6, d=48),
    "D7_v6144_h384_e6": dict(V=6144, NH=384, E=6, d=48),
    "D8_v8192_h384_e6_d40": dict(V=8192, NH=384, E=6, d=40),
    "D11_v6144_h512_e6_d40": dict(V=6144, NH=512, E=6, d=40),
    # E5: three-tier embedding, mid = (tokens in the middle tier, its prototype count)
    "E5a_v4096_m768": dict(V=4096, NH=256, E=6, d=48, mid=[768, 16]),
    "E5b_v4096_m1024": dict(V=4096, NH=192, E=6, d=48, mid=[1024, 16]),
    "E5c_v3072_m768": dict(V=3072, NH=384, E=8, d=48, mid=[768, 16]),
}


if __name__ == "__main__":
    if "--table" in sys.argv:
        import json

        def emb(V, NH, E, d, hr=0, mid=(), **_):
            head = NH * hr + hr * d if hr else NH * d
            nm, e1 = mid if mid else (0, 0)
            return head + (V - NH - nm) * E + E * d + nm * e1 + (e1 * d if nm else 0)

        only = [a for a in sys.argv[1:] if not a.startswith("--")]
        out = (
            json.load(open("report/shape_arms.json"))
            if only and os.path.exists("report/shape_arms.json")
            else {}
        )
        for name, kw in ARMS.items():
            if only and name not in only:
                continue
            fk = {k: kw[k] for k in ("V", "NH", "E", "d", "hr", "mid") if k in kw}
            r = fit(**fk)
            out[name] = dict(
                **kw,
                rules=r[0],
                rules0=r[1],
                params=r[2],
                emb=emb(**kw),
                share=round(emb(**kw) / r[2], 3),
            )
            print(name, out[name], flush=True)
        json.dump(out, open("report/shape_arms.json", "w"), indent=1)
        sys.exit(0)
    print(
        "recipe e2e_tail (V3072, head 384, tail 12, d48, 2 layers, rules 52/35):",
        count(),
    )
