"""Which regression problems are actually slow for a neural network to converge on?

Part 4 refuted the warm-start hypothesis for a precise reason: a warm start saves
*optimizer steps*, and every problem in the ladder needed almost none. PhiUSIIL
at 235,795 rows reached 2% error in 25 minibatch updates -- 0.03 s -- against a
2 s FIS fit. Scaling the row count did not help, because row count buys per-epoch
cost, not epochs-to-target.

So the search is not for a *bigger* problem. It is for one where gradient descent
genuinely needs many updates. This script measures that directly, and the metric
is **minibatch updates to reach a target quality**, deliberately not seconds and
not epochs: updates are what an initialization can skip, and unlike epochs they
are comparable across datasets of different sizes.

Every candidate is either already in `data/` or generated here, because the
egress policy blocks the usual dataset hosts.

    python experiments/fis-to-neural-net/find_slow_problem.py
    python experiments/fis-to-neural-net/find_slow_problem.py --problems chirp pendulum-n2

Writes `outputs/slow_problems.md`.

A candidate is worth taking to a full warm-start experiment when it needs
*thousands* of updates, not tens -- that is the only regime in which saving the
first few hundred can beat the cost of a FIS fit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "reproduce", "tables"))
sys.path.insert(0, os.path.join(REPO, "AnalyticalDynamics", "chaos"))

#: Every generated artifact goes here. Kept out of the source directory so the
#: scripts and the things they produce never have to be told apart by eye, and
#: so `outputs/.gitignore` can drop derived CSVs without a rule that could ever
#: match a hand-written file.
OUTPUTS = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

import fis2nn  # noqa: E402

#: Fraction of the target's variance the network must explain. 0.9 is a low bar
#: on purpose -- the question is how many updates a network needs to become
#: *useful*, which is the point a warm start would let it skip, not how long it
#: takes to squeeze out the last decimal.
TARGET_R2 = 0.90
MAX_UPDATES = 60_000
LR_GRID = (1e-3, 3e-3, 1e-2)
HIDDEN = 128
BATCH = 128


# ---------------------------------------------------------------------------
# Candidate problems. Each returns (X, y, note).
# ---------------------------------------------------------------------------


def p_concrete():
    """The ladder's reference point: known to converge in tens of updates."""
    import _fuzzy_models as fm

    X, y = fm.load_concrete()
    return X.to_numpy(float), y.to_numpy(float), "reference (Part 1 workhorse)"


def p_chirp(n=4000, k=40.0, seed=0):
    """A swept-frequency 1-D signal -- the canonical spectral-bias probe.

    A ReLU network fits low frequencies quickly and high ones slowly, so the
    number of updates needed is controlled directly by ``k`` rather than by the
    row count. This is the cheapest knob in the file for manufacturing a
    slow problem, and the one whose difficulty is understood analytically.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = np.sin(2 * np.pi * k * x**2)
    return x[:, None], y, f"sin(2*pi*{k:g}*x^2), 1 input"


def p_highfreq_2d(n=8000, k=12.0, seed=0):
    """Two-dimensional oscillation: the same spectral bias with interactions."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    y = np.sin(2 * np.pi * k * X[:, 0]) * np.sin(2 * np.pi * k * X[:, 1])
    return X, y, f"sin({k:g}x)sin({k:g}y), 2 inputs"


def p_illconditioned(n=8000, d=20, cond=1e4, seed=0):
    """A linear target on an input basis with condition number ``cond``.

    Slow for a different reason than oscillation -- the loss surface is a long
    narrow valley rather than a high-frequency one. Adam's per-parameter scaling
    absorbs much of this, which is exactly what makes it worth measuring instead
    of assuming.
    """
    rng = np.random.default_rng(seed)
    scales = np.logspace(0, np.log10(cond), d)
    X = rng.normal(size=(n, d)) * scales
    w = rng.normal(size=d) / scales
    return X, X @ w, f"{d} inputs, condition number {cond:g}"


def p_pendulum(n_links=2, friction=False):
    """The chaos time-step operator: (theta_1(0)..theta_n(0), t) -> theta(t).

    `AnalyticalDynamics/chaos` already builds this, already applies TRIBBLE to
    it, and its own module docstring calls the target "a plain (if violently
    oscillatory) 2-input regression surface" -- which is the profile this search
    is looking for, in the author's own domain rather than a synthetic stand-in.
    """
    import pendulum_data

    ds = pendulum_data.generate(n_links, friction)
    theta = np.asarray(ds.theta_deg, dtype=float)  # (n_ic, n_steps, n_links)
    n_ic, n_steps = theta.shape[0], theta.shape[1]
    t = np.asarray(ds.t, dtype=float)[:n_steps]
    X = np.concatenate(
        [
            np.repeat(np.asarray(ds.ic_deg, dtype=float), n_steps, axis=0),
            np.tile(t, n_ic)[:, None],
        ],
        axis=1,
    )
    y = theta[:, :, 0].reshape(-1)  # first angle, the paper's headline output
    return X, y, f"n={n_links}{' + friction' if friction else ''} time-step operator"


# The first sweep only produced "a few hundred updates" or "never", which is a
# *capacity* wall, not a convergence cost: 128 ReLU units cannot represent 80
# oscillations at any number of updates. The frequency ladder below fills in the
# band between, because the band is the whole point -- a warm start can only
# help on a problem the network will eventually solve.
PROBLEMS = {
    "concrete": p_concrete,
    "illcond": p_illconditioned,
    "chirp-k4": lambda: p_chirp(k=4.0),
    "chirp-k8": lambda: p_chirp(k=8.0),
    "chirp-k16": lambda: p_chirp(k=16.0),
    "chirp-k40": lambda: p_chirp(k=40.0),
    "sine2d-k2": lambda: p_highfreq_2d(k=2.0),
    "sine2d-k4": lambda: p_highfreq_2d(k=4.0),
    "sine2d-k8": lambda: p_highfreq_2d(k=8.0),
    "pendulum-n2": lambda: p_pendulum(2, False),
    "pendulum-n2-fric": lambda: p_pendulum(2, True),
    "pendulum-n3": lambda: p_pendulum(3, False),
}


# ---------------------------------------------------------------------------


def updates_to_target(X, y, lr, seed, max_updates, batch=BATCH, hidden=HIDDEN):
    """Minibatch updates for a He-init MLP to first reach TARGET_R2 on a holdout.

    Uses the same `fis2nn` network and trainer as the rest of the experiment, so
    a number here is directly comparable to Part 4's "25 updates" for PhiUSIIL.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]

    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ym, ys = float(y[tr].mean()), float(y[tr].std()) or 1.0
    ytr, yte = (y[tr] - ym) / ys, (y[te] - ym) / ys

    n_batches = max(1, int(np.ceil(len(Xtr) / batch)))
    epochs = int(np.ceil(max_updates / n_batches))
    net = fis2nn.he_start(rng, Xtr.shape[1], hidden)

    # Record often enough to resolve a fast problem, but not so often that a
    # slow one spends all its time being evaluated.
    every = max(1, n_batches // 4)
    _, hist = fis2nn.train_adam(
        net,
        Xtr,
        ytr,
        X_test=Xte,
        y_test=yte,
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        seed=seed,
        eval_batches=every,
        track_train=False,
    )
    curve = np.asarray(hist.test_rmse, dtype=float)  # RMSE in standardized units
    r2 = 1.0 - curve**2 / (float(np.var(yte)) or 1.0)
    hit = np.flatnonzero(r2 >= TARGET_R2)
    best = float(np.nanmax(r2))
    if not hit.size:
        return None, best
    return float(hist.epochs[hit[0]]) * n_batches, best


def guard_output(path, force):
    """Refuse to overwrite an existing run of record unless asked twice.

    Writing this file cost tens of minutes; a smoke run with the default `--out`
    costs seconds and silently replaces it. That happened once while this
    directory was being reorganized -- a one-seed, five-epoch synth1d run landed
    on top of the ten-seed, 150-epoch `results.json` -- and it is the same
    failure `WORKINGDOC.md` catalogues under REPRO_OUTPUT_DIR. Recovering it
    needed `git checkout`, which only worked because the file happened to be
    staged.
    """
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"{os.path.relpath(path, REPO)} already exists.\n"
            "Pass --force to replace it, or --out <path> to write elsewhere "
            "(which is what a smoke run should do)."
        )
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="*", default=list(PROBLEMS))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-updates", type=int, default=MAX_UPDATES)
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing run of record"
    )
    args = ap.parse_args()

    rows = []
    for name in args.problems:
        try:
            X, y, note = PROBLEMS[name]()
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] unavailable: {type(exc).__name__}: {exc}", flush=True)
            continue
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        t0 = time.perf_counter()
        # Per learning rate: how many seeds arrived, how many updates they took,
        # and the best R2 seen. The problem is credited with its *easiest*
        # setting -- anything else would measure the sweep, not the problem.
        swept = []
        for lr in LR_GRID:
            got, r2 = [], []
            for s in range(args.seeds):
                u, b = updates_to_target(X, y, lr, s, args.max_updates)
                got.append(u)
                r2.append(b)
            arrived = [u for u in got if u is not None]
            swept.append(
                {
                    "lr": lr,
                    "n_arrived": len(arrived),
                    "updates": float(np.mean(arrived)) if arrived else None,
                    "r2": float(np.mean(r2)),
                }
            )
        # Most seeds arriving wins; then fewest updates; then best R2 among
        # learning rates where nothing arrived at all.
        best = max(
            swept,
            key=lambda c: (
                c["n_arrived"],
                -(c["updates"] if c["updates"] is not None else 0.0),
                c["r2"],
            ),
        )
        best_lr = best["lr"]
        best_updates = best["updates"]
        best_r2 = max(c["r2"] for c in swept)
        rows.append(
            {
                "problem": name,
                "note": note,
                "rows": len(X),
                "features": X.shape[1],
                "lr": best_lr,
                "updates_to_r2": best_updates,
                "best_r2": best_r2,
                "sweep": swept,
                "seconds": time.perf_counter() - t0,
            }
        )
        print(
            f"  {name:14s} {len(X):>7,} x {X.shape[1]:<3d} "
            + (
                f"updates to R2>={TARGET_R2}: {best_updates:,.0f}"
                if best_updates is not None
                else f"NEVER reached R2 {TARGET_R2} (best {best_r2:.3f})"
            )
            + f"  [{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )

    rows.sort(
        key=lambda r: (r["updates_to_r2"] is not None, -(r["updates_to_r2"] or 0))
    )
    lines = [
        "# Which problems are actually slow to converge?",
        "",
        f"Minibatch updates for a He-initialized {HIDDEN}-unit ReLU network "
        f"(batch {BATCH}, Adam, best of {list(LR_GRID)}) to first reach "
        f"R2 >= {TARGET_R2} on a 20% holdout, mean of {args.seeds} seeds, capped at "
        f"{args.max_updates:,}.",
        "",
        "Updates, not seconds and not epochs: updates are what an initialization "
        "can skip, and unlike epochs they are comparable across dataset sizes. "
        "**PhiUSIIL needed 25.** A warm start costing a 2 s FIS fit needs a problem "
        "in the thousands before it can repay itself.",
        "",
        "| problem | rows x features | updates to R2>=0.9 | best R2 | what it is |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        u = (
            f"**{r['updates_to_r2']:,.0f}**"
            if r["updates_to_r2"] is not None
            else f"never (best R2 {r['best_r2']:.2f})"
        )
        lines.append(
            f"| `{r['problem']}` | {r['rows']:,} x {r['features']} | {u} | "
            f"{r['best_r2']:.3f} | {r['note']} |"
        )
    lines.append("")

    path = os.path.join(OUTPUTS, "slow_problems.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    guarded = guard_output(os.path.join(OUTPUTS, "slow_problems.json"), args.force)
    with open(guarded, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n" + "\n".join(lines))
    print(f"wrote {os.path.relpath(path, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
