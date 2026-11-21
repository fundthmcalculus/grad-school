"""Does a surgical edit generalise to the same situation in a different story? (`PLAN_SURGERY.md`) ``python -m flm.surgery_twins``

For the 100 near-miss events of `flm.surgery_exp` (same random seed, same events), apply the 3-rule minimum-norm edit (``rule3``) and look at held-out positions (the second half of the cached windows, which contain no event and come from other stories) that share the event's *rule signature*: the same three strongest last-block feed-forward rules (as a set).
  helped   : signature positions whose true next piece is the event's desired piece and where the model was wrong before: the share now right
  hurt     : signature positions whose true next piece is a different piece and where the model was right before: the share now wrong
A signature is a read-out of the model's own rules, not a statistic of the text: it is the "same situation" in the model's terms.
"""

import json
from pathlib import Path

import numpy as np

import flm.surgery_exp as SX
from flm.surgery_exp import (
    AllFuzzyLM,
    TOKENIZER,
    WEIGHTS,
    load,
    logits_of,
    plan_dx,
    rule_edit,
)

SX.rng = np.random.default_rng(0)
rng = SX.rng
m = AllFuzzyLM(WEIGHTS, TOKENIZER)
n_win, X, S, inp, tgt, prev = load()
half = (n_win // 2) * 512
by_pool = np.arange(half, len(X))
by = np.sort(
    rng.choice(by_pool, 30_000, replace=False)
)  # same draws as flm.surgery_exp, to land on the same events
cand = rng.choice(np.arange(0, half), 6000, replace=False)
lg = logits_of(m, X[cand])
rank = (lg > lg[np.arange(len(cand)), tgt[cand]][:, None]).sum(1) + 1
events = cand[(rank >= 2) & (rank <= 4)][:100]
sig_pool = np.sort(
    np.argsort(-np.abs(S[by_pool]), axis=1)[:, :3], axis=1
)  # the three strongest rules of every held-out position
pool_lg = None
rows = []
for e in events:
    x, s, t = X[e], S[e], int(tgt[e])
    dx = plan_dx(m, x, t)
    if dx is None:
        continue
    idx, dC = rule_edit(s, dx, 3)
    sig = np.sort(idx)
    same = np.flatnonzero((sig_pool == sig).all(1))
    if len(same) == 0:
        rows.append(dict(n_sig=0))
        continue
    P = by_pool[same]
    before = logits_of(m, X[P]).argmax(1)
    after = logits_of(m, X[P] + S[P][:, idx] @ dC).argmax(1)
    y = tgt[P]
    helped_den = int(((y == t) & (before != y)).sum())
    hurt_den = int(((y != t) & (before == y)).sum())
    rows.append(
        dict(
            n_sig=int(len(P)),
            helped_den=helped_den,
            helped=int(((y == t) & (before != y) & (after == y)).sum()),
            hurt_den=hurt_den,
            hurt=int(((y != t) & (before == y) & (after != y)).sum()),
            to_target=int(((after == t) & (before != t)).sum()),
        )
    )
use = [r for r in rows if r.get("n_sig", 0) > 0]
hd, h = sum(r["helped_den"] for r in use), sum(r["helped"] for r in use)
ud, u = sum(r["hurt_den"] for r in use), sum(r["hurt"] for r in use)
summary = dict(
    events=len(rows),
    events_with_signature_matches=len(use),
    median_matches=float(np.median([r["n_sig"] for r in use])) if use else 0,
    helped=[h, hd],
    helped_share=h / max(hd, 1),
    hurt=[u, ud],
    hurt_share=u / max(ud, 1),
    events_with_helped_den=sum(1 for r in use if r["helped_den"] > 0),
)
Path("report").mkdir(exist_ok=True)
Path("report/surgery_twins.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
