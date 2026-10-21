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
    guess = int(52 + (cap - n0) / slope)
    best = None
    for r in range(max(2, guess - 4), guess + 5):
        r0 = max(1, round(r * 35 / 52))
        n = count(rules=r, rules0=r0, **kw)
        if n <= cap and (best is None or n > best[2]):
            best = (r, r0, n)
    return best


if __name__ == "__main__":
    print(
        "recipe e2e_tail (V3072, head 384, tail 12, d48, 2 layers, rules 52/35):",
        count(),
    )
