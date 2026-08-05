"""Screen candidate antecedents by measured predictive power, before fitting anything.

Adding an input to a rule base is expensive twice over: it costs time in the innermost
loop, and it costs generalisation, because every extra input multiplies the parameters the
GA has to fit against a training set that is already too small. So far features have been
chosen by argument and then judged by whether the whole system got better — which conflates
"is this signal informative" with "did the GA manage to exploit it".

This separates the two, by way of a shift in what the rule base is asked.

**The reframing.** `EFFORT` is currently asked "how much effort does this city deserve?"
That has no ground truth: there is no label for the right breadth at a city, which is why
the only way to evaluate an input has been to fit a rule base around it and run the solver.
Asked instead as **"will searching this city pay off, and by how much?"** the target becomes
directly measurable — run the search, record whether the city yielded an improving move and
how large the gain was. Effort allocation is then a monotone response to predicted payoff,
and any candidate feature can be scored on the prediction task alone.

So this module runs a normal Lin-Kernighan local search, and at every city the queue pops it
records the candidate features *before* the search and the outcome *after*. Features are
then ranked by

* **AUC** for predicting "this city yielded an improving move" — the probability that a
  randomly chosen paying city scores above a randomly chosen non-paying one. 0.5 is
  worthless, and the sign says which direction the rule should read.
* **Spearman correlation with the realised gain, over the paying cities only.** Whether a
  city pays and how much it pays are different questions, and a feature can be good at one
  and useless at the other. Restricting to the paying subset is what makes the second
  question answerable: with 89% of the gain column exactly zero, a correlation computed over
  everything is mostly measuring the same thing the AUC does.

A feature that cannot beat 0.5 AUC on this task cannot help a rule base allocate effort, and
no amount of GA budget will make it. That is worth knowing for the cost of one instrumented
run rather than one fitting campaign.

The one caveat, stated because it bounds what this can conclude: the labels come from a
*fixed-parameter* search, so this measures "would effort here have paid off at these
settings", not "what is the optimal effort here". It screens out useless features and ranks
plausible ones; it does not by itself prove a feature will earn its runtime.

Run:  python features_probe.py [--instances 6] [--out feature_screen.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fis
import synth
from core import build_candidates, dist, greedy_edge_tour, make_pos, nn_stats, pred, succ
from lk import N_STATS, improve_city
from tsplib import load

HERE = Path(__file__).resolve().parent

# Candidate features. The first six are what `EFFORT` reads today; the rest are proposals.
# Each is a name, a short description, and the direction a rule base would be expected to
# read it ("+" = higher means search harder).
FEATURES = [
    ("excess", "mean incident edge / nearest-neighbour distance", "+"),
    ("fails", "how often this city has already come up empty", "-"),
    ("turn", "local turn sharpness", "+"),
    ("progress", "fraction of the run's work already spent", "-"),
    ("rank", "nearer neighbours the worse tour edge ignores, / k", "+"),
    ("peak", "nearest-neighbour distance / mean candidate distance", "?"),
    # --- proposals ---
    ("probe", "best depth-1 positive gain available, / first broken edge", "+"),
    ("probe_frac", "fraction of candidates passing the depth-1 gain test", "+"),
    ("nbr_active", "fraction of candidate neighbours still queued", "+"),
    ("pos_spread", "tour-position spread of the candidate neighbours, / n", "+"),
    ("edge_asym", "|d_succ - d_pred| / (d_succ + d_pred)", "+"),
    ("cand_step", "(2nd nearest - nearest) / nearest", "?"),
    ("in_degree", "how many cities hold this one in their candidate list, / k", "?"),
]
NAMES = [f[0] for f in FEATURES]


def _in_degree(cand):
    """How many cities list each city as a candidate. A hub is structurally different from
    a leaf, and this is the cheapest statement of that; computed once per instance."""
    n, k = cand.shape
    deg = np.zeros(n, np.float64)
    np.add.at(deg, cand.ravel(), 1.0)
    return deg / k


def collect(inst, k=32, breadth=8, max_depth=6, deep_breadth=16, or_seg=3):
    """(X, gains) over every city scan of one fixed-parameter local search.

    The driver loop mirrors ``lk_solve``'s queue discipline exactly — FIFO, don't-look bits,
    re-activate the cities whose edges changed — because features like ``nbr_active`` and
    ``progress`` are properties *of that discipline* and would mean something different
    under any other one.
    """
    cand, cand_d = build_candidates(inst.coords, k, inst.ceil)
    nn1, mean_c = nn_stats(cand_d)
    in_deg = _in_degree(cand)
    coords, ceil, n = inst.coords, inst.ceil, inst.n

    tour = greedy_edge_tour(coords, cand, ceil)
    pos = make_pos(tour)
    stats = np.zeros(N_STATS, np.int64)
    rev_i = np.empty(max_depth + 1, np.int64)
    rev_j = np.empty(max_depth + 1, np.int64)
    touched = np.empty(max(8, 4 * (max_depth + 1)), np.int32)
    xc = np.empty(fis.CH_N_IN, np.float64)
    mu = np.empty((fis.CH_N_IN, fis.N_TERMS), np.float64)

    cap = n + 1
    queue = np.empty(cap, np.int32)
    in_queue = np.zeros(n, np.uint8)
    for i in range(n):
        queue[i] = tour[i]
        in_queue[tour[i]] = 1
    qh, qt, qn = 0, n, n
    fails = np.zeros(n, np.float64)
    work_scale = 3.0 * n

    rows = []
    gains = []
    pops = 0
    while qn > 0:
        t1 = int(queue[qh])
        qh = (qh + 1) % cap
        qn -= 1
        in_queue[t1] = 0
        pops += 1

        s1 = succ(tour, pos, n, t1)
        p1 = pred(tour, pos, n, t1)
        d_s = dist(coords, t1, s1, ceil)
        d_p = dist(coords, t1, p1, ceil)
        d_long = max(d_s, d_p)

        f = {}
        f["excess"] = min(max((0.5 * (d_s + d_p) / nn1[t1] - 1.0) * 0.5, 0.0), 1.0)
        f["fails"] = min(fails[t1] / 3.0, 1.0)

        ax, ay = coords[t1, 0] - coords[p1, 0], coords[t1, 1] - coords[p1, 1]
        bx, by = coords[s1, 0] - coords[t1, 0], coords[s1, 1] - coords[t1, 1]
        na, nb = np.hypot(ax, ay), np.hypot(bx, by)
        f["turn"] = 0.5 if na <= 0 or nb <= 0 else 0.5 * (
            1.0 - min(max((ax * bx + ay * by) / (na * nb), -1.0), 1.0)
        )
        f["progress"] = min(pops / work_scale, 1.0)

        r = int(np.searchsorted(cand_d[t1], d_long, side="left"))
        f["rank"] = r / k
        f["peak"] = nn1[t1] / mean_c[t1]

        # --- the probe: what a single level of search can already see from here.
        # Both directions of the broken edge are tried, exactly as the chain would.
        best_g1 = 0.0
        n_pass = 0
        for d_break, t2 in ((d_s, s1), (d_p, p1)):
            for t in range(k):
                g1 = d_break - cand_d[t2, t]
                if g1 <= 1e-9:
                    break  # candidates ascend, so no later one passes either
                n_pass += 1
                if g1 > best_g1:
                    best_g1 = g1
        f["probe"] = min(best_g1 / max(d_long, 1e-9), 1.0)
        f["probe_frac"] = min(n_pass / (2.0 * k), 1.0)

        act = 0
        span = 0.0
        for t in range(k):
            c = cand[t1, t]
            act += int(in_queue[c])
            dp = abs(int(pos[c]) - int(pos[t1]))
            span += min(dp, n - dp)
        f["nbr_active"] = act / k
        f["pos_spread"] = min(span / (k * 0.25 * n), 1.0)

        f["edge_asym"] = abs(d_s - d_p) / max(d_s + d_p, 1e-9)
        f["cand_step"] = min((cand_d[t1, 1] - cand_d[t1, 0]) / max(cand_d[t1, 0], 1e-9), 1.0)
        f["in_degree"] = min(in_deg[t1] / 3.0, 1.0)

        gain, nt = improve_city(
            tour, pos, n, coords, cand, cand_d, ceil, t1,
            breadth, max_depth, deep_breadth, or_seg,
            rev_i, rev_j, touched, stats,
            False, fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS, xc, mu,
        )
        rows.append([f[name] for name in NAMES])
        gains.append(max(float(gain), 0.0))

        if gain > 1e-9:
            fails[t1] = 0.0
            for i in range(nt):
                c = int(touched[i])
                if not in_queue[c] and qn < cap - 1:
                    queue[qt] = c
                    qt = (qt + 1) % cap
                    qn += 1
                    in_queue[c] = 1
        else:
            fails[t1] += 1.0

    return np.array(rows), np.array(gains)


def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney). Ties handled by average ranks."""
    pos = labels > 0
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tied score groups, or a constant feature scores 1.0 not 0.5
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _tied_ranks(a):
    """Ranks with ties averaged. Ordinal ranks are wrong here and not subtly: 89% of the
    gain column is exactly 0, so ranking those zeros arbitrarily manufactures correlation
    out of array order."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    s = a[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _tied_ranks(np.asarray(a, float)), _tied_ranks(np.asarray(b, float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "feature_screen.json"))
    args = ap.parse_args()

    # A spread of families and both real and synthetic instances, so a feature that only
    # works on uniform points cannot look good here.
    targets = [load("pr1002"), load("d1291"), load("rl1889")] + [
        synth.make("clustered", 1400, 21),
        synth.make("grid", 1600, 31),
        synth.make("mixed", 1800, 41),
    ]

    X, y = [], []
    for inst in targets:
        xi, yi = collect(inst)
        print(f"  {inst.name:>18s} n={inst.n:6d}  {len(yi):6d} city scans, "
              f"{100.0 * (yi > 0).mean():5.1f}% paid off", flush=True)
        X.append(xi)
        y.append(yi)
    X = np.vstack(X)
    y = np.concatenate(y)

    print(f"\n{len(y)} city scans, {100.0 * (y > 0).mean():.1f}% yielded an improving move\n")
    paid = y > 0
    print(f"{'feature':>12s} {'AUC':>7s} {'|AUC-.5|':>9s} {'rho|paid':>9s}  expect  note")
    res = []
    for i, (name, desc, direction) in enumerate(FEATURES):
        a = auc(X[:, i], y)
        r = spearman(X[paid, i], y[paid])
        res.append({"feature": name, "auc": a, "rho_paid": r, "desc": desc, "expect": direction})
        new = "" if i < 6 else "  <-- proposal"
        print(f"{name:>12s} {a:7.4f} {abs(a - 0.5):9.4f} {r:9.4f}  {direction:>6s}{new}")

    res.sort(key=lambda d: -abs(d["auc"] - 0.5))
    print("\nranked by |AUC - 0.5|:")
    for d in res:
        print(f"  {d['feature']:>12s} {abs(d['auc'] - 0.5):.4f}   {d['desc']}")
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
