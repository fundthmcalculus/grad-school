"""A random joint sample of the all-fuzzy model's allocation space (`PLAN_ROUND3.md` stage 4): vocabulary, head size, tail width, block width, mixer rules per head and the split of the feed-forward rules between layer 0 and layer 1, each sample fitted to the parameter cap.

``python -m flm.random_shape --n 24 --seed 7 --out report/random_arms.json`` writes arms in the format of ``report/shape_arms.json`` (plus ``mixrules``) that ``SHAPE_ARMS=report/random_arms.json SHAPE_OUT=outputs/v77-random python -m flm.run_shape`` runs.
The fitted feed-forward layer-1 rule count is the largest one under the cap (``shape_budget.CAP``) for the sampled ratio r0/r1; a sample whose fitted layer-1 count falls below ``MIN_RULES`` is rejected and redrawn (a model with almost no rules is not an allocation, it is a broken model).
``python -m flm.random_shape --analyse outputs/v77-random`` fits a ridge regression of the 30M validation bits per character on the sampled settings and prints the coefficients and the ranked samples.
"""

import argparse
import json
import random
from pathlib import Path

CHOICES = dict(
    V=[4096, 6144, 8192],
    NH=list(range(256, 897, 64)),
    E=[4, 6, 8, 10],
    d=[24, 32, 40, 48],
    mixrules=[8, 16, 24, 32, 48],
)
RATIO = (
    0.25,
    1.0,
)  # layer-0 rules as a fraction of layer-1 rules (the recipe: 35 / 52 = 0.67)
MIN_RULES = 8


def fit_rules(cap, ratio, **kw):
    """Largest layer-1 rule count whose model is at or under ``cap`` (parameters are increasing in it): bisection on ``shape_budget.count``."""
    from flm.shape_budget import count

    def n(r):
        return count(rules=r, rules0=max(1, round(r * ratio)), **kw)

    lo, hi = 2, 400
    if n(lo) > cap:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if n(mid) <= cap else (lo, mid)
    return lo, max(1, round(lo * ratio)), n(lo)


def sample(n, seed):
    from flm.shape_budget import CAP

    rng = random.Random(seed)
    arms, seen = {}, set()
    while len(arms) < n:
        a = {k: rng.choice(v) for k, v in CHOICES.items()}
        ratio = round(rng.uniform(*RATIO), 2)
        key = (*a.values(), ratio)
        if key in seen:
            continue
        seen.add(key)
        fit = fit_rules(
            CAP,
            ratio,
            V=a["V"],
            NH=a["NH"],
            E=a["E"],
            d=a["d"],
            extra=("--rules", str(a["mixrules"])),
        )
        if fit is None or fit[0] < MIN_RULES:
            continue
        r, r0, params = fit
        name = f"R{len(arms):02d}_v{a['V']}_h{a['NH']}_e{a['E']}_d{a['d']}_m{a['mixrules']}_q{int(ratio * 100)}"
        arms[name] = dict(a, rules=r, rules0=r0, ratio=ratio, params=params)
    return arms


def analyse(out_dir):
    import numpy as np

    arms = json.load(open("report/random_arms.json"))
    rows = []
    for name, a in arms.items():
        fs = sorted(Path(out_dir).glob(f"{name}_s*.json"))
        if fs:
            v = [json.load(open(f))["val_bpc"] for f in fs]
            rows.append((name, a, float(np.mean(v)), len(v)))
    rows.sort(key=lambda r: r[2])
    for name, a, m, k in rows:
        print(
            f"{m:.4f} (n={k}) {name} params {a['params']} rules {a['rules']}/{a['rules0']}"
        )
    if len(rows) >= 8:
        X = np.array(
            [
                [
                    r[1]["V"] / 1000,
                    r[1]["NH"] / 100,
                    r[1]["E"],
                    r[1]["d"] / 10,
                    r[1]["mixrules"] / 10,
                    r[1]["ratio"],
                    r[1]["rules"] / 10,
                ]
                for r in rows
            ]
        )
        y = np.array([r[2] for r in rows])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        lam = 1.0
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
        names = ["V", "head", "tail", "d", "mixer rules", "ratio", "ffn rules"]
        print("ridge (per standard deviation of each setting; negative = better):")
        for nme, c in sorted(zip(names, w), key=lambda t: t[1]):
            print(f"  {nme:12s} {c:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="report/random_arms.json")
    ap.add_argument("--analyse", default="")
    a = ap.parse_args()
    if a.analyse:
        analyse(a.analyse)
    else:
        arms = sample(a.n, a.seed)
        Path(a.out).write_text(json.dumps(arms, indent=1))
        for k, v in arms.items():
            print(k, v["params"], v["rules"], v["rules0"])
