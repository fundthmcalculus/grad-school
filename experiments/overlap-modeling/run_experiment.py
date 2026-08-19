"""Driver: does overlapping the output buckets improve a TRIBBLE regression model?

Six arms per (dataset, n_output_buckets, tsk_order, seed), all sharing one split,
one scaler, one feature-selection pass and one ridge strength, so the only thing
that differs between them is where the overlap is applied and how wide it is:

    baseline        hard buckets, global firing-weighted ridge   [the shipped model]
    soft-ante       overlapped membership-function fits, global solve
    soft-random     same, but the band is drawn at random         [control for soft-ante]
    local-hard      per-rule fit on the hard bucket only          [control for "local"]
    local-overlap   per-rule fit on the overlapped slice          [the literal request]
    full-overlap    overlapped antecedents *and* overlapped per-rule fits
    fusion          global solve + adjacent-consequent agreement penalty

Two of the six are controls, and each answers a specific objection.

`local-hard` is what makes the reading of `local-overlap` possible: going local
is itself a large change, and without that control any difference would be a
mixture of "local instead of global" and "soft instead of hard".

`soft-random` is what makes `soft-ante` readable. Widening a bucket's fitting
slice does three things at once -- it softens the boundary, it puts more rows
into every membership fit, and (because the sweep selects on validation) it hands
that family 14 candidates where the baseline has one. `soft-random` borrows the
same number of rows with the same weights from the same neighbours, drawn
uniformly rather than at the shared edge, so it holds all three of those fixed
and varies only the boundary structure. Any `soft-ante` gain that `soft-random`
also shows is not a gain from softening a boundary.

Protocol
--------
Each seed draws an 80/20 train/test split and then a 75/25 inner split of the
train fold. Every arm is fitted on the inner-train fold and scored on both the
validation fold and the test fold. Nothing is ever selected on test: the headline
table picks the overlap width on **validation** R2 per seed and only then reports
what that choice scored on test, which is the number a user would actually get.
The per-width curves are also reported, as diagnostics, clearly marked as such.

X is unit-scaled and y standardized on the inner-train fold only -- FIS
membership functions want bounded inputs (see the note in
`FuzzySystemsExperiments/concrete.py`), and fitting the scaler on anything wider
than the fitting fold would leak.

Usage
-----
    python experiments/overlap-modeling/run_experiment.py
    python experiments/overlap-modeling/run_experiment.py --datasets concrete --seeds 0,1,2
    python experiments/overlap-modeling/run_experiment.py --quick

Writes ``outputs/results.json`` (every arm, every seed) plus the generated tables
quoted by RESULTS.md.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import os
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "reproduce", "tables"))

# The library fans its per-(feature, label) Gaussian fits out over a thread pool
# sized from the CPU count. This driver already parallelizes at the cell level,
# and the two together oversubscribe a 4-core box badly, so the inner pool is
# pinned to one worker. Set before the import that reads it.
os.environ.setdefault("TRIBBLE_GAUSSIAN_WORKERS", "1")

import _fuzzy_models as fm  # noqa: E402
from overlap import OverlapTribbleRegressor  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("OVERLAP_SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
FRACTIONS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
SHAPES = ("flat", "ramp")
FUSION_REGS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
BUCKETS = (3, 5, 7)
ORDERS = ("2nd", "full-2nd")
L2_REG = 1e-2          # the value `FuzzySystemsExperiments/concrete.py` runs at
PARTITION = "quantile"  # the request is about quantiles; see README for why


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
def load_concrete():
    """Concrete compressive strength: 1,030 x 8 -> MPa. The plan's reference target."""
    X, y = fm.load_concrete()
    return X, y


# `cnt` is exactly `casual + registered`, and the shared loader drops neither.
# `experiments/fis-to-neural-net` found this and deliberately left the shared
# loader alone, because archived proposal tables quote it; the same reasoning
# applies here, so this experiment also carries its own leak-free wrapper.
LEAKY_BIKESHARE_COLUMNS = ["casual", "registered"]


def load_bikeshare():
    """Bike sharing hourly: 17,379 x 12 -> rides. The large, heavily tied target."""
    X, y = fm.load_bikeshare()
    return X.drop(columns=LEAKY_BIKESHARE_COLUMNS, errors="raise"), y


# WEC's target has the same problem in a different costume: `Total_Power` is the
# sum of the 100 `Power*` columns, so a model handed those columns is doing
# addition, not regression. `experiments/fis-to-neural-net/run_experiment.py`'s
# `load_wec` keeps them -- it drops only `Total_Power` -- and its WEC row should
# be read with that in mind. Dropped here.
LEAKY_WEC_PREFIX = "Power"


def load_wec():
    """Wave energy converters (Sydney, 100 buoys): 2,318 x 200 -> total power.

    The wide one: 200 candidate columns for TRIBBLE's feature selection to cut
    down, which is the regime where the bucket partition is doing the most work
    (it is what the differentiation score is computed against).
    """
    df = pd.read_csv(os.path.join(REPO, "data", "WEC_Sydney_100.csv")).dropna()
    y = df["Total_Power"].astype(float)
    y.name = "y_value"
    drop = ["Total_Power"] + [c for c in df.columns if c.startswith(LEAKY_WEC_PREFIX)]
    X = df.drop(columns=drop).select_dtypes(include=[np.number]).astype(float)
    return X, y


def _synth(seed, kind, n=1500):
    """Two synthetic rungs that make the hypothesis falsifiable at the mechanism level.

    The claim under test is that hard bucket edges are an artifact -- that the
    response surface does not actually change at the quantile cuts, so a rule
    fitted only inside one cut is fitted on an arbitrary slice. That claim has a
    contrapositive, and a real dataset cannot separate the two:

    ``smooth``
        One smooth surface, no regimes at all. Every bucket edge is arbitrary by
        construction, so this is the best case the idea can have.
    ``piecewise``
        Three genuine regimes with *different* response functions and a jump
        between them, arranged along an index that is roughly uniform so the
        regime edges land near the target's tertiles. Here the hard edge is
        approximately the right model and blending across it should *cost*
        accuracy.

    If overlap helps on both, it is not doing what the story says. If it helps on
    ``smooth`` and hurts on ``piecewise``, the mechanism is confirmed even where
    the net effect on real data is small.
    """
    rng = np.random.default_rng(1000 + seed)
    X = pd.DataFrame({f"x{i}": rng.uniform(0.0, 1.0, n) for i in range(1, 6)})
    x1, x2, x3 = X.x1.to_numpy(), X.x2.to_numpy(), X.x3.to_numpy()
    noise = 0.05 * rng.normal(size=n)
    if kind == "smooth":
        y = np.sin(3.0 * x1) + x2 ** 2 + 0.7 * x1 * x3 + noise
    else:
        t = 0.6 * x1 + 0.4 * x2                      # ~uniform index -> even regimes
        lo, hi = np.quantile(t, [1 / 3, 2 / 3])
        y = np.where(
            t < lo, 0.5 + 1.0 * t + 0.4 * x3,
            np.where(t < hi, 2.0 - 0.8 * t - 0.6 * x3,
                     3.5 + 2.0 * t + 0.9 * x3)) + noise
    # x4 and x5 are pure noise columns, left in so feature selection has work.
    return X, pd.Series(y, name="y_value")


def load_synth_smooth():
    """One smooth surface: 1,500 x 5 (3 informative). The idea's best case."""
    return _synth(0, "smooth")


def load_synth_piecewise():
    """Three genuine regimes with jumps: 1,500 x 5. The idea's worst case."""
    return _synth(0, "piecewise")


DATASETS = {
    "concrete": load_concrete,
    "bikeshare": load_bikeshare,
    "synth-smooth": load_synth_smooth,
    "synth-piecewise": load_synth_piecewise,
    "wec": load_wec,
}

# WEC is loadable but excluded from the default run, and that exclusion is a
# finding rather than a convenience. With the `Power*` columns dropped -- and they
# have to be dropped, they sum to the target -- `calculate_gaussian_correlation`
# scores every one of the 198 remaining buoy-coordinate columns at exactly
# 0.0000. `take_top_features` therefore keeps all 198, and the resulting model
# predicts the training mean: test R2 = -0.000 for all 31 arms in a probe run, at
# 4-8 s a fit. A rung that scores every arm identically at zero cannot
# discriminate between them, so including it would only add an hour of runtime
# and four columns of zeros. Reproduce with `--datasets wec`.
DEFAULT_DATASETS = ["concrete", "bikeshare", "synth-smooth", "synth-piecewise"]


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------
def arm_configs():
    """(arm, label, kwargs) for every configuration, in report order."""
    out = [("baseline", "baseline", dict(overlap=0.0, consequent_fit="global"))]
    for arm, band in (("soft-ante", "adjacent"), ("soft-random", "random")):
        for shape in SHAPES:
            for f in FRACTIONS:
                out.append((arm, f"{arm}/{shape}/{f:g}", dict(
                    overlap=f, overlap_shape=shape, overlap_band=band,
                    overlap_antecedents=True, overlap_means=True,
                    consequent_fit="global")))
    out.append(("local-hard", "local-hard", dict(overlap=0.0, consequent_fit="local")))
    for shape in SHAPES:
        for f in FRACTIONS:
            out.append(("local-overlap", f"local-overlap/{shape}/{f:g}", dict(
                overlap=f, overlap_shape=shape, overlap_antecedents=False,
                overlap_means=True, consequent_fit="local")))
    for shape in SHAPES:
        for f in FRACTIONS:
            out.append(("full-overlap", f"full-overlap/{shape}/{f:g}", dict(
                overlap=f, overlap_shape=shape, overlap_antecedents=True,
                overlap_means=True, consequent_fit="local")))
    for lam in FUSION_REGS:
        out.append(("fusion", f"fusion/{lam:g}", dict(
            overlap=0.0, consequent_fit="global", fusion_reg=lam)))
    return out


def _r2(y_true, y_pred):
    """R2 with non-finite predictions dropped, and how many were dropped.

    An arm that returns NaN for part of the test fold has not scored well on the
    rest of it -- silently dropping those rows would flatter it. The count is
    carried through to the report so a high R2 on a fraction of the fold cannot
    be read as a high R2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_pred)
    n_dropped = int((~keep).sum())
    if keep.sum() < 2:
        return float("nan"), n_dropped
    yt, yp = y_true[keep], y_pred[keep]
    denom = float(np.sum((yt - yt.mean()) ** 2))
    if denom == 0:
        return float("nan"), n_dropped
    return float(1.0 - np.sum((yt - yp) ** 2) / denom), n_dropped


def split3(X, y, seed):
    """80/20 train/test, then 75/25 inner-train/validation inside the train fold."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(round(0.2 * len(X)))
    test, train = idx[:n_test], idx[n_test:]
    n_val = int(round(0.25 * len(train)))
    val, inner = train[:n_val], train[n_val:]

    def take(ix):
        return X.iloc[ix].reset_index(drop=True), y.iloc[ix].reset_index(drop=True)
    return take(inner), take(val), take(test)


def prepare(X_fit, folds):
    """Unit-scale X and standardize y using ``X_fit``'s statistics only."""
    scaler, _ = fm.fit_scaler(X_fit[0], scaler="unit")
    centre = float(np.mean(X_fit[1]))
    scale = float(np.std(X_fit[1])) or 1.0
    return [(fm.apply_scaler(scaler, Xf), (yf - centre) / scale) for Xf, yf in folds]


def run_cell(dataset, n_buckets, order, seed, configs):
    """Every arm for one (dataset, buckets, order, seed). Returns a list of records."""
    warnings.filterwarnings("ignore")
    X, y = DATASETS[dataset]()
    inner, val, test = split3(X, y, seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare(inner, [inner, val, test])

    records = []
    for arm, label, kwargs in configs:
        t0 = time.perf_counter()
        try:
            # `calculate_gaussian_correlation` prints its ranking table on every
            # call; 31 arms x 180 cells of that is not a log.
            with contextlib.redirect_stdout(io.StringIO()):
                model = OverlapTribbleRegressor(
                    n_output_buckets=n_buckets, output_partition=PARTITION,
                    tsk_order=order, l2_reg=L2_REG, pin_extremes=False,
                    random_state=seed, **kwargs).fit(Xtr, ytr)
                fit_s = time.perf_counter() - t0
                r2_val, drop_val = _r2(yva, model.predict(Xva))
                r2_test, drop_test = _r2(yte, model.predict(Xte))
                area = model.membership_overlap_area()
        except Exception as exc:                       # noqa: BLE001 -- recorded, not hidden
            records.append(dict(
                dataset=dataset, n_buckets=n_buckets, order=order, seed=seed,
                arm=arm, label=label, error=f"{type(exc).__name__}: {exc}"))
            continue
        records.append(dict(
            dataset=dataset, n_buckets=n_buckets, order=order, seed=seed,
            arm=arm, label=label, r2_val=r2_val, r2_test=r2_test,
            dropped_val=drop_val, dropped_test=drop_test, fit_seconds=fit_s,
            n_rules=int(model.n_rules_), n_features=len(model.top_features_),
            overlap_area=area,
            **{k: v for k, v in kwargs.items()}))
    return records


def provenance():
    def git(path, *args):
        try:
            return subprocess.run(["git", "-C", path, *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:                              # noqa: BLE001
            return "unknown"
    return dict(
        repo_commit=git(REPO, "rev-parse", "HEAD"),
        tribble_fis_commit=git(os.path.join(REPO, "tribble-fis"), "rev-parse", "HEAD"),
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        seeds=SEEDS,
        l2_reg=L2_REG,
        output_partition=PARTITION,
    )


def dump_payload(payload: dict, path: str) -> str:
    """Write a run's payload as gzipped compact JSON, returning the path written.

    These files are the run of record and are committed, so their size is a review
    cost, not just a disk cost. Four stages of per-(cell, arm) records is 19 MB of
    plain JSON -- 4.5x the largest artifact any other experiment in this repo commits
    -- and 2.3 MB gzipped, which is below it. Nobody reads these by eye; `analyze*.py`
    is the reader, and `load_payload` handles either form.

    A ``.json`` path is redirected to ``.json.gz``; the returned path is what to
    report to the user.
    """
    if not path.endswith(".gz"):
        path += ".gz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return path


def load_payload(path: str) -> dict:
    """Read a run payload, accepting ``.json.gz`` or plain ``.json``.

    Falls back across the two so an older uncompressed artifact, or one a user
    gunzipped by hand to poke at, still loads.
    """
    candidates = [path]
    if path.endswith(".gz"):
        candidates.append(path[:-3])
    else:
        candidates.insert(0, path + ".gz")
    for candidate in candidates:
        if os.path.exists(candidate):
            opener = gzip.open if candidate.endswith(".gz") else open
            with opener(candidate, "rt", encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(f"no run payload at {' or '.join(candidates)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--buckets", default=",".join(str(b) for b in BUCKETS))
    ap.add_argument("--orders", default=",".join(ORDERS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true",
                    help="concrete only, 3 seeds, one bucket count and order")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "results.json"))
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    buckets = [int(b) for b in args.buckets.split(",")]
    orders = args.orders.split(",")
    if args.quick:
        datasets, seeds, buckets, orders = ["concrete"], [0, 1, 2], [5], ["2nd"]

    configs = arm_configs()
    cells = [(d, b, o, s) for d in datasets for b in buckets for o in orders for s in seeds]
    print(f"{len(cells)} cells x {len(configs)} arms = {len(cells) * len(configs)} fits "
          f"on {args.jobs} workers")

    from joblib import Parallel, delayed
    t0 = time.time()
    batches = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_cell)(d, b, o, s, configs) for d, b, o, s in cells)
    records = [r for batch in batches for r in batch]
    elapsed = time.time() - t0

    payload = dict(provenance=provenance(), wall_clock_seconds=elapsed,
                   fractions=list(FRACTIONS), shapes=list(SHAPES),
                   fusion_regs=list(FUSION_REGS), records=records)
    written = dump_payload(payload, args.out)

    n_err = sum("error" in r for r in records)
    print(f"\n{len(records)} records in {elapsed:.1f}s ({n_err} errors) -> {written}")
    if n_err:
        for r in records:
            if "error" in r:
                print(f"  {r['dataset']}/{r['n_buckets']}/{r['order']}/"
                      f"seed{r['seed']}/{r['label']}: {r['error']}")
                break


if __name__ == "__main__":
    main()
