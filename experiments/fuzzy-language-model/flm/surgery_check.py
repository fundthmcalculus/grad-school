"""Reference edits for the browser parity test (`demo/web/test_surgery.mjs`): ``python -m flm.surgery_check`` writes ``demo/web/surgery.check.json``.

Three events (the first near-miss and the first two far-miss events of `flm.surgery_exp`, successes only): the window's piece ids up to the event, the desired piece, and the numpy edit (all-rules minimum-norm, `flm.surgery_exp.plan_dx` + `rule_edit`): the relative size of the change, the margin it reaches, and the change itself for the three strongest rules.
"""

import json
from pathlib import Path

import numpy as np

import flm.surgery_exp as SX
from flm.surgery import TracingLM
from flm.surgery_exp import (
    MARGIN,
    TOKENIZER,
    WEIGHTS,
    load,
    logits_of,
    margin_at,
    plan_dx,
    rule_edit,
)

SX.rng = np.random.default_rng(0)
rng = SX.rng
m = TracingLM(WEIGHTS, TOKENIZER)
n_win, X, S, inp, tgt, prev = load()
half = (n_win // 2) * 512
rng.choice(np.arange(half, len(X)), 30_000, replace=False)
cand = rng.choice(np.arange(0, half), 6000, replace=False)
lg = logits_of(m, X[cand])
rank = (lg > lg[np.arange(len(cand)), tgt[cand]][:, None]).sum(1) + 1
near = cand[(rank >= 2) & (rank <= 4)][:100]
far = cand[(rank >= 11) & (rank <= 100)][:100]
tokens = np.load("outputs/surgery_cache.npz")["seq"]
out = []
for e in [near[0], far[0], far[1], far[2], far[3], far[4]]:
    if len(out) == 3:
        break
    x, s, t = X[e], S[e], int(tgt[e])
    dx = plan_dx(m, x, t)
    if dx is None:
        continue
    idx, dC = rule_edit(s, dx, len(s))
    cq = m.W["blocks.1.ffn.conseq"]
    w, pos = divmod(int(e), 512)
    mg0, _ = margin_at(m, x, t)
    mg1, _ = margin_at(m, x + s @ dC, t)
    top = np.argsort(-np.abs(s))[:3]
    out.append(
        dict(
            prefix=[int(i) for i in tokens[w, : pos + 1]],
            target=t,
            margin_before=mg0,
            margin_after=mg1,
            rel_norm=float(np.linalg.norm(dC) / np.linalg.norm(cq)),
            dx_norm=float(np.linalg.norm(dx)),
            top_rules=[int(r) for r in top],
            top_rule_dC=[[float(v) for v in dC[r]] for r in top],
        )
    )
Path("demo/web/surgery.check.json").write_text(json.dumps(out))
print(len(out), "reference edits written")
