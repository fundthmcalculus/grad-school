"""The surgery experiments (`PLAN_SURGERY.md`), on the cache written by ``python -m flm.surgery collect``: ``python -m flm.surgery_exp``.

Events: validation positions (first half of the cached windows) where the model's top-1 piece is not the true next piece; the "surgeon" names the true piece as what should have happened.  Two strata: near misses (the true piece ranked 2nd-4th) and far misses (ranked 11th-100th).
Methods (every edit is sized by a line search to make the desired piece the top-1 by a margin of MARGIN logits at the event position, using the exact forward function):
  rule1 / rule3 / rule_all : edit the consequent vectors of the 1 / 3 / all (61) last-block feed-forward rules with the largest strengths at the event, minimum-norm (Delta c_r proportional to s_r); rule_all is exactly one gradient step on the last block's consequents
  bias   : a bias added to the desired piece's output score, everywhere (the cheapest dense edit)
  unembed: a rank-one change of the desired piece's output embedding along the event's final hidden state (the usual locate-and-edit move on a dense model's head)
Measured on 30,000 bystander positions from the second half of the windows (never an event): top-1 flips, change in bits per character, mean KL(old || new), and the fraction of "twin" errors fixed (second-half positions with the same last two input pieces and the same true piece where the model was also wrong; and the looser same-current-piece twins).
"""

import json
import time
from pathlib import Path

import numpy as np

from flm.surgery import (
    CACHE,
    DATA,
    TOKENIZER,
    WEIGHTS,
    AllFuzzyLM,
    final_logits,
    softmax,
)

MARGIN = 0.5  # the desired piece must beat every other piece by this many logits
OUT = Path("report")
rng = np.random.default_rng(0)


def load():
    z = np.load(CACHE)
    n_win = int(z["n_win"])
    X = z["X"].reshape(-1, z["X"].shape[-1]).astype(np.float64)
    S = z["S"].reshape(-1, z["S"].shape[-1]).astype(np.float64)
    seq = z["seq"]
    inp, tgt = seq[:, :-1].reshape(-1), seq[:, 1:].reshape(-1)
    prev = np.concatenate([np.full((n_win, 1), -1), seq[:, :-2]], 1).reshape(
        -1
    )  # the piece before the current one
    return n_win, X, S, inp, tgt, prev


def logits_of(m, X):
    out = np.empty((len(X), m.V), np.float32)
    for i in range(0, len(X), 4096):
        out[i : i + 4096] = final_logits(m, X[i : i + 4096])
    return out


def margin_at(m, x, t):
    lg = final_logits(m, x[None])[0]
    other = np.delete(lg, t)
    return float(lg[t] - other.max()), lg


def grad_dir(m, x, t, p):
    """d(logit_t - logit_p)/dx by central differences (32 dimensions)."""
    g = np.zeros_like(x)
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = 1e-3 * max(1.0, abs(x[j]))
        a = final_logits(m, (x + e)[None])[0]
        b = final_logits(m, (x - e)[None])[0]
        g[j] = ((a[t] - a[p]) - (b[t] - b[p])) / (2 * e[j])
    return g


def plan_dx(m, x, t):
    """The smallest step along the (t - p) gradient direction that makes t the top-1 by MARGIN; (None) if no step up to 50x |x| does."""
    mg, lg = margin_at(m, x, t)
    if mg >= MARGIN:
        return np.zeros_like(x)
    lg2 = lg.copy()
    lg2[t] = -np.inf
    p = int(lg2.argmax())
    g = grad_dir(m, x, t, p)
    u = g / (np.linalg.norm(g) + 1e-12)
    lo, hi = 0.0, 0.05 * np.linalg.norm(x)
    while margin_at(m, x + hi * u, t)[0] < MARGIN:
        lo, hi = hi, hi * 2
        if hi > 50 * np.linalg.norm(x):
            return None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if margin_at(m, x + mid * u, t)[0] < MARGIN else (lo, mid)
    return hi * u


def rule_edit(s, dx, k):
    """Minimum-norm consequent change of the k strongest rules that moves the residual by dx: Delta c_r = s_r dx / sum s^2.  Returns (rule indices, Delta C (k, d))."""
    idx = np.argsort(-np.abs(s))[:k] if k < len(s) else np.arange(len(s))
    return idx, np.outer(s[idx] / (s[idx] ** 2).sum(), dx)


def bystander_stats(
    m, Xb, base_logits, base_top, base_nll_bits, chars, Xnew=None, dcol=None, t=None
):
    """Compare the bystanders after an edit with before.  Either Xnew (the residuals changed) or dcol (the desired piece's score changed by dcol per position)."""
    if Xnew is not None:
        new = logits_of(m, Xnew)
    else:
        new = base_logits.copy()
        new[:, t] += dcol
    lse = np.logaddexp.reduce(new.astype(np.float64), axis=1)
    top = new.argmax(1)
    return new, top, lse


def summarize(m, new, lse_new, top_new, base_lg, base_lse, base_top, tgt_b, chars_b):
    n = len(new)
    flips = float((top_new != base_top).mean())
    nll_new = (lse_new - new[np.arange(n), tgt_b]) / np.log(2)
    nll_old = (base_lse - base_lg[np.arange(n), tgt_b]) / np.log(2)
    dbpc = float((nll_new - nll_old).sum() / chars_b.sum())
    p_old = np.exp(base_lg - base_lse[:, None])
    kl = float(
        (p_old * ((base_lg - base_lse[:, None]) - (new - lse_new[:, None])))
        .sum(1)
        .mean()
    )
    return dict(flips=flips, dbpc=dbpc, kl=kl)


def main():
    t0 = time.time()
    m = AllFuzzyLM(WEIGHTS, TOKENIZER)
    lens = np.load(DATA / "lens_bpe6144.npy")
    n_win, X, S, inp, tgt, prev = load()
    N = len(X)
    half = (n_win // 2) * 512
    ev_pool = np.arange(0, half)
    by_pool = np.arange(half, N)
    by = np.sort(rng.choice(by_pool, 30_000, replace=False))
    Xb, Sb, tb = X[by], S[by], tgt[by]
    chars_b = lens[tb].astype(np.int64)
    base_lg = logits_of(m, Xb)
    base_lse = np.logaddexp.reduce(base_lg.astype(np.float64), axis=1)
    base_top = base_lg.argmax(1)
    base_bpc = float(
        ((base_lse - base_lg[np.arange(len(by)), tb]) / np.log(2)).sum() / chars_b.sum()
    )
    print(
        f"bystanders {len(by)}: base BPC {base_bpc:.4f}, top-1 accuracy {(base_top == tb).mean():.3f} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    # rank of the true piece at event candidates (first half): sample candidates and score them
    cand = rng.choice(ev_pool, 6000, replace=False)
    lg_c = logits_of(m, X[cand])
    rank = (lg_c > lg_c[np.arange(len(cand)), tgt[cand]][:, None]).sum(1) + 1
    near = cand[(rank >= 2) & (rank <= 4)][:100]
    far = cand[(rank >= 11) & (rank <= 100)][:100]
    print(
        f"events: {len(near)} near-miss, {len(far)} far-miss ({time.time() - t0:.0f}s)",
        flush=True,
    )
    # twin pools in the second half
    key2 = (prev[by_pool].astype(np.int64) * 7000 + inp[by_pool]) * 7000 + tgt[by_pool]
    key1 = inp[by_pool].astype(np.int64) * 7000 + tgt[by_pool]
    lg_pool_top = None
    results = {}
    methods = ["rule1", "rule3", "rule_all", "bias", "unembed"]
    hn_b = None
    for stratum, events in (("near", near), ("far", far)):
        rows = {mth: [] for mth in methods}
        for n_ev, e in enumerate(events):
            x, s, t = X[e], S[e], int(tgt[e])
            dx = plan_dx(m, x, t)
            if dx is None:
                for mth in methods:
                    rows[mth].append(dict(success=False))
                continue
            tw2 = by_pool[key2 == (int(prev[e]) * 7000 + int(inp[e])) * 7000 + t]
            tw1 = by_pool[key1 == int(inp[e]) * 7000 + t]
            tw2 = tw2[:200]
            tw1 = tw1[:400]
            for mth in methods:
                rec = dict(success=True)
                if mth.startswith("rule"):
                    k = {"rule1": 1, "rule3": 3, "rule_all": len(s)}[mth]
                    idx, dC = rule_edit(s, dx, k)
                    C = m.W["blocks.1.ffn.conseq"][idx]
                    rec["rel_norm"] = float(np.linalg.norm(dC) / np.linalg.norm(C))
                    rec["n_rules"] = len(idx)
                    xe = x + s[idx] @ dC
                    ok = margin_at(m, xe, t)[0] >= MARGIN - 1e-6
                    new, top, lse = bystander_stats(
                        m,
                        Xb,
                        base_lg,
                        base_top,
                        None,
                        chars_b,
                        Xnew=Xb + Sb[:, idx] @ dC,
                    )
                    # twins
                    for name, tw in (("tw2", tw2), ("tw1", tw1)):
                        if len(tw):
                            wrong = logits_of(m, X[tw]).argmax(1) != tgt[tw]
                            after = (
                                logits_of(m, X[tw] + S[tw][:, idx] @ dC).argmax(1)
                                == tgt[tw]
                            )
                            rec[name] = (int(wrong.sum()), int((wrong & after).sum()))
                else:
                    mg, lg = margin_at(m, x, t)
                    need = MARGIN - mg
                    if mth == "bias":
                        dcol = np.full(len(by), need)
                    else:  # unembed: E_out[t] += gamma * hn / |hn|^2, so the event's score rises by gamma
                        if hn_b is None:
                            from flm.surgery import EPS

                            def hn(X_):
                                W = m.W
                                z = 0.5 * np.log((X_**2).mean(-1) + EPS)
                                w = softmax(
                                    -0.5
                                    * (
                                        (z[:, None] - W["norm.centers"])
                                        / np.exp(W["norm.log_width"])
                                    )
                                    ** 2
                                )
                                g_ = (
                                    w
                                    * (
                                        W["norm.offset"]
                                        + W["norm.slope"]
                                        * (z[:, None] - W["norm.centers"])
                                    )
                                ).sum(-1)
                                return X_ * np.exp(g_)[:, None] * W["norm.weight"]

                            hn_b = hn
                        h_e = hn_b(x[None])[0]
                        dcol = need * (hn_b(Xb) @ h_e) / (h_e @ h_e)
                    rec["rel_norm"] = float("nan")
                    rec["n_rules"] = 0
                    new, top, lse = bystander_stats(
                        m, Xb, base_lg, base_top, None, chars_b, dcol=dcol, t=t
                    )
                    for name, tw in (("tw2", tw2), ("tw1", tw1)):
                        if len(tw):
                            lgt = logits_of(m, X[tw])
                            wrong = lgt.argmax(1) != tgt[tw]
                            if mth == "bias":
                                lgt[:, t] += need
                            else:
                                lgt[:, t] += need * (hn_b(X[tw]) @ h_e) / (h_e @ h_e)
                            rec[name] = (
                                int(wrong.sum()),
                                int((wrong & (lgt.argmax(1) == tgt[tw])).sum()),
                            )
                rec.update(
                    summarize(
                        m, new, lse, top, base_lg, base_lse, base_top, tb, chars_b
                    )
                )
                rows[mth].append(rec)
            if n_ev % 20 == 0:
                print(
                    f"  {stratum} event {n_ev + 1}/{len(events)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        results[stratum] = rows
    out = dict(
        base_bpc=base_bpc,
        n_bystanders=int(len(by)),
        n_near=len(near),
        n_far=len(far),
        margin=MARGIN,
        results=results,
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "surgery.json").write_text(json.dumps(out, default=float))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
