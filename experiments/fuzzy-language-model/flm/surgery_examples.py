"""Readable examples of the surgery, chosen without cherry-picking (`PLAN_SURGERY.md`): ``python -m flm.surgery_examples`` -> ``report/surgery_examples.json``.

Events are the ones of `flm.surgery_exp` (same seed, same order).  Selection rule, fixed in advance: the near-miss events number 0, 25, 50, 75 and the far-miss events number 0, 25, 50, 75 in event order (successes and failures alike); then, from the remaining events, the first two far-miss events on which the all-rules edit FAILED (no step up to 50x |x| reaches the margin) and the two on which it did the MOST collateral damage (largest rise in the bystanders' bits per character).
Per event: the last 15 pieces of context, the piece the model chose, the piece the surgeon wanted, its rank before, the top-3 pieces with probabilities before and after the all-rules (rule_all) edit and after the unembedding edit (probabilities at temperature 1, event position only), the edit size (relative norm of the consequent change over all 61 last-block rules), and ~30 pieces of continuation (temperature 0.5, top-k 10, a fixed seed per event) sampled from the context before and after the edit (the edited model: the last block's consequents changed, applied from this position on; that block feeds only the output scores, so the earlier positions are untouched).
Three events also list the three bystander positions (held-out, never an event) whose top-1 flipped with the largest loss of probability on the true piece.
"""

import json
from pathlib import Path

import numpy as np

import flm.surgery_exp as SX
from flm.surgery import EPS, TracingLM, final_logits
from flm.surgery_exp import (
    MARGIN,
    TOKENIZER,
    WEIGHTS,
    load,
    logits_of,
    margin_at,
    plan_dx,
    rule_edit,
    softmax,
)

SX.rng = np.random.default_rng(0)
rng = SX.rng
m = TracingLM(WEIGHTS, TOKENIZER)
n_win, X, S, inp, tgt, prev = load()
N, half = len(X), (n_win // 2) * 512
by_pool = np.arange(half, N)
by = np.sort(rng.choice(by_pool, 30_000, replace=False))
cand = rng.choice(np.arange(0, half), 6000, replace=False)
lg_c = logits_of(m, X[cand])
rank_c = (lg_c > lg_c[np.arange(len(cand)), tgt[cand]][:, None]).sum(1) + 1
near = cand[(rank_c >= 2) & (rank_c <= 4)][:100]
far = cand[(rank_c >= 11) & (rank_c <= 100)][:100]
res = json.load(open("report/surgery.json"))["results"]
tokens = np.load("outputs/surgery_cache.npz")[
    "seq"
]  # (windows, 513) the validation pieces


def piece(i):
    return m.piece(int(i))


def top3(logits):
    p = softmax(logits.astype(np.float64))
    o = np.argsort(-p)[:3]
    return [[piece(i), round(float(p[i]), 4)] for i in o]


def generate(prefix_ids, dC, seed, n=30):
    """Continue from prefix_ids: temperature 0.5, top-k 10; the last block's consequents shifted by dC (rows of the strengths' rules) when given."""
    g = np.random.default_rng(seed)
    m.reset()
    out = []
    ids = list(prefix_ids)
    xs = None
    for tkn in ids:
        xs = m.step_trace(int(tkn))
    for _ in range(n):
        x, s = xs
        xe = x + (s @ dC if dC is not None else 0.0)
        lg = final_logits(m, xe[None].astype(np.float64))[0] / 0.5
        lg = np.where(lg < np.sort(lg)[-10], -np.inf, lg)
        t = int(g.choice(m.V, p=softmax(lg)))
        out.append(t)
        xs = m.step_trace(t)
    return "".join(piece(i) for i in out)


def context_of(e, k=15):
    w, t = divmod(int(e), 512)
    return "".join(piece(i) for i in tokens[w, max(0, t - k + 1) : t + 1])


def prefix_of(e):
    w, t = divmod(int(e), 512)
    return tokens[w, : t + 1]


# selection
chosen = [("near", i, near[i]) for i in (0, 25, 50, 75)] + [
    ("far", i, far[i]) for i in (0, 25, 50, 75)
]
used = {(s, i) for s, i, _ in chosen}
ra = res["far"]["rule_all"]
fails = [
    i for i, r in enumerate(ra) if not r.get("success") and ("far", i) not in used
][:2]
dam = sorted(
    [i for i, r in enumerate(ra) if r.get("success") and ("far", i) not in used],
    key=lambda i: -ra[i]["dbpc"],
)[:2]
chosen += [("far", i, far[i]) for i in fails] + [("far", i, far[i]) for i in dam]
kinds = (
    ["fixed_rule_near"] * 4
    + ["fixed_rule_far"] * 4
    + ["failed_rule_all"] * len(fails)
    + ["max_collateral"] * len(dam)
)
# base bystander scores (held-out, never an event)
Xb, Sb, tb = X[by], S[by], tgt[by]
base_lg = logits_of(m, Xb)
base_lse = np.logaddexp.reduce(base_lg.astype(np.float64), axis=1)
base_top = base_lg.argmax(1)
p_true_old = np.exp(base_lg[np.arange(len(by)), tb] - base_lse)


def bystander_damage(dC, k=3):
    new_top = np.empty(len(by), np.int64)
    p_true_new = np.empty(len(by))
    for i in range(0, len(by), 4096):
        sl = slice(i, i + 4096)
        lg = logits_of(m, Xb[sl] + Sb[sl] @ dC)
        lse = np.logaddexp.reduce(lg.astype(np.float64), axis=1)
        new_top[sl] = lg.argmax(1)
        p_true_new[sl] = np.exp(lg[np.arange(len(lg)), tb[sl]] - lse)
    flipped = np.flatnonzero(new_top != base_top)
    worst = flipped[np.argsort(-(p_true_old[flipped] - p_true_new[flipped]))[:k]]
    rows = []
    for j in worst:
        g = by[j]
        w, t = divmod(int(g), 512)
        rows.append(
            dict(
                context="".join(piece(i) for i in tokens[w, max(0, t - 9) : t + 1]),
                true=piece(tb[j]),
                old_top1=piece(base_top[j]),
                new_top1=piece(new_top[j]),
                p_true_before=round(float(p_true_old[j]), 4),
                p_true_after=round(float(p_true_new[j]), 4),
            )
        )
    return rows


out = []
n_with_bystanders = 0
for (stratum, i, e), kind in zip(chosen, kinds):
    x, s, t = X[e], S[e], int(tgt[e])
    lg0 = final_logits(m, x[None].astype(np.float64))[0]
    rank = int((lg0 > lg0[t]).sum()) + 1
    mg, _ = margin_at(m, x, t)
    rec = dict(
        kind=kind,
        stratum=stratum,
        event_number=i,
        context=context_of(e),
        chose=piece(lg0.argmax()),
        wanted=piece(t),
        wanted_is_head_piece=bool(m.is_head[t]),
        rank_before=rank,
        top3_before=top3(lg0),
    )
    dx = plan_dx(m, x, t)
    if dx is None:
        rec.update(
            top3_after=None,
            top3_after_unembed=None,
            edit_size=None,
            n_rules=0,
            note="no step up to 50x |x| along the planned direction reaches the margin: the edit failed",
            text_before=generate(prefix_of(e), None, 11 + i),
            text_after=None,
            bystanders=[],
        )
        out.append(rec)
        continue
    idx, dC = rule_edit(s, dx, len(s))
    cq = m.W["blocks.1.ffn.conseq"]
    rec["edit_size"] = round(float(np.linalg.norm(dC) / np.linalg.norm(cq)), 4)
    rec["n_rules"] = int(len(idx))
    rec["dx_over_x"] = round(float(np.linalg.norm(dx) / np.linalg.norm(x)), 4)
    rec["top3_after"] = top3(final_logits(m, (x + s @ dC)[None].astype(np.float64))[0])
    lg_u = lg0.copy()
    lg_u[t] += (
        MARGIN - mg
    )  # the unembedding edit at this position: the wanted piece's score raised to the margin
    rec["top3_after_unembed"] = top3(lg_u)
    seed = 11 + i
    rec["text_before"] = generate(prefix_of(e), None, seed)
    rec["text_after"] = generate(prefix_of(e), dC, seed)
    rec["bystanders"] = []
    if (
        kind in ("fixed_rule_near", "fixed_rule_far", "max_collateral")
        and n_with_bystanders < 3
        and (
            (kind == "fixed_rule_near" and i == 0)
            or (kind == "fixed_rule_far" and i == 0)
            or kind == "max_collateral"
        )
    ):
        rec["bystanders"] = bystander_damage(dC)
        n_with_bystanders += 1
    out.append(rec)
    print(kind, i, "done", flush=True)
Path("report/surgery_examples.json").write_text(
    json.dumps(out, indent=1, ensure_ascii=False)
)
print(
    "wrote report/surgery_examples.json",
    len(out),
    "events;",
    n_with_bystanders,
    "with bystander lists",
)
