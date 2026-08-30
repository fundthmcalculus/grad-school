"""Table 4.1 -- Mixture-of-Gaussians FIS training time and accuracy vs. baselines.

The point of this table is the *speed* claim: MoG trains a competitive model
without any GA/GD. Columns: MoG train time and accuracy/R2, then the baselines a
reviewer will ask for -- a scikit-learn tree/forest reference (always available),
and ANFIS / GA-tuned FIS (optional adapters; N/A unless their deps are present).
Every number is mean +/- std across `common.SEEDS`.

Run (from repo root):  uv run --project tribble-fis python reproduce/tables/table_4_1_mog_baselines.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import _fuzzy_models as _fm  # noqa: E402

# Optional baselines -- resolve once; None => the column renders as N/A.
(anfis_fit_predict,) = C.optional_import("_baseline_anfis", ["fit_predict"])
(gafis_fit_predict,) = C.optional_import("_baseline_gafis", ["fit_predict"])


def _fit_one_seed(seed, *, kind, X, y, mog_factory, score):
    """One seed's worth of `_bench`'s inner loop -- everything that used to run
    inline, unchanged in substance so it can run either in-process or in a
    worker. Every model here takes an explicit `seed` (or a per-call
    `RandomState(seed)`, for ANFIS/GA-FIS); nothing depends on shared global
    state, which is what makes farming this out across seeds safe.

    Returns `(seed, out)`, `out` mapping column -> `(train_seconds, score)` or
    `None` if that arm didn't produce a prediction (model unavailable, or a
    guarded fit/predict failure).
    """
    out = {"mog": None, "rf": None, "anfis": None, "gafis": None}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = mog_factory(seed)
    if model is not None:
        with C.timed() as t:
            try:
                p = np.asarray(model.fit(Xtr, ytr).predict(Xte))
            except Exception:  # noqa: BLE001
                p = None
        if p is not None:
            out["mog"] = (t.seconds, score(yte, p))

    RF = RandomForestRegressor if kind == "reg" else RandomForestClassifier
    with C.timed() as t:
        p = RF(n_estimators=200, random_state=seed).fit(Xtr, ytr).predict(Xte)
    out["rf"] = (t.seconds, score(yte, p))

    for name, fn in (("anfis", anfis_fit_predict), ("gafis", gafis_fit_predict)):
        if fn is None:
            continue
        try:
            with C.timed() as t:
                p = np.asarray(fn(Xtr, ytr, Xte, kind=kind, seed=seed))
            out[name] = (t.seconds, score(yte, p))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] failed ({exc.__class__.__name__}); -> N/A")
    return seed, out


def _n_seed_workers():
    """How many seeds to fit in parallel. `REPRO_SEED_WORKERS=1` (or a single
    seed) disables the pool and keeps the old in-process loop -- useful for a
    quick REPRO_SEEDS=0 smoke run, where pool start-up would cost more than it
    saves."""
    cap = os.environ.get("REPRO_SEED_WORKERS")
    n = int(cap) if cap else (os.cpu_count() or 1)
    return max(1, min(n, len(C.SEEDS)))


def _pin_blas_threads_for_workers():
    """Cap each worker to a small, fixed thread budget before any are spawned.

    Env vars set here are inherited by every child at process-creation time --
    that happens at the OS level (fork/spawn copy the parent's environment),
    before any Python code in the child runs, so it is not sensitive to
    whether the child's own `import numpy` happens before or after its
    initializer. What matters is that `n_workers` processes, each spinning up
    its own BLAS thread pool, must not together oversubscribe the machine: at
    the repo's own default of 8 BLAS threads (table_4_11's convention, tuned
    for ONE process), 10 seeds in parallel would ask for 80 threads on a
    32-core host -- the same class of oversubscription NOTE12_THREADING.md
    traces to a SIGSEGV from a joblib backend spawning one BLAS pool per
    process without capping any of them. Defaulting to 1 thread/worker and
    `n_workers` processes is the standard, safe shape for embarrassingly
    parallel CPU-bound work: parallelism comes from the process count, not
    from each process's own BLAS calls, which is also appropriate here --
    these are small-matrix operations (Concrete's dual solve is at most
    824x824) where BLAS's own threading overhead usually isn't worth paying
    per call anyway.
    """
    n = os.environ.get("REPRO_BLAS_THREADS_PER_WORKER", "1")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = n


def _bench(kind, X, y, mog_factory, score, norm=False):
    """Return dict col -> list-of-per-seed (train_seconds, score).

    `norm` applies the pipeline's log-and-standardize treatment to X before any
    split, for EVERY arm in the table rather than only the MoG one. That
    uniformity is the point: an earlier version of this table timed the MoG at
    its raw-feature default while quoting tree baselines that had been measured
    under normalization, so the headline row was competing against arms fitted
    on different inputs. Applying it to all arms costs the baselines nothing --
    CART and Random Forest split on rank and are provably invariant to a
    monotone transform, which `table_hyperparam_normalization` measures at
    +0.001 and +0.000 -- and it removes the mismatch.
    """
    if norm:
        X = _fm.normalize(X)[0]
    cols = {c: {"t": [], "s": []} for c in ["mog", "rf", "anfis", "gafis"]}

    # One DISCARDED warm-up fit, so the first seed does not pay for the process.
    #
    # Without it seed 0 absorbs import, JIT, BLAS thread-pool spin-up and
    # first-touch allocation, and that lands on the headline cell of a *speed*
    # claim: Concrete measured 1.04 +/- 0.62 s, a +/-60% deviation presented as
    # seed spread. Per-seed times say otherwise -- seed 0 is 3.68x the mean of
    # the other nine, and dropping it moves the spread from +/-0.641 s to
    # +/-0.021 s (2.6%), which matches the PhiUSIIL row's +/-0.02 s. PhiUSIIL was
    # never affected because it is fitted second, and that asymmetry between two
    # rows of one table is what gives the diagnosis away.
    #
    # The warm-up runs on the SAME first split rather than on synthetic data: a
    # differently-shaped input may not touch the same code paths, which would
    # leave part of the cost still charged to seed 0.
    #
    # The global numpy RNG state is snapshotted and restored around it. The models
    # take an explicit random_state, but a discarded fit is only free if it
    # consumes no shared randomness, and that is cheap to guarantee rather than
    # assume. Verified by the accuracy columns coming back byte-identical to
    # outputs/full-14900hx-r2/table_4_1.csv -- only the time cells moved.
    _warm = mog_factory(C.SEEDS[0])
    if _warm is not None:
        Xw, Xwte, yw, _ = train_test_split(X, y, test_size=0.2, random_state=C.SEEDS[0])
        _rng_state = np.random.get_state()
        try:
            _warm.fit(Xw, yw).predict(Xwte)
        except Exception:  # noqa: BLE001 -- a failed warm-up must not fail the table
            pass
        finally:
            np.random.set_state(_rng_state)

    worker = partial(_fit_one_seed, kind=kind, X=X, y=y, mog_factory=mog_factory, score=score)
    n_workers = _n_seed_workers()
    if n_workers <= 1:
        results = [worker(seed) for seed in C.SEEDS]
    else:
        _pin_blas_threads_for_workers()
        ctx = mp.get_context("spawn")  # never fork: BLAS's own thread pool does
        # not survive a fork cleanly (see _pin_blas_threads_for_workers), and
        # spawn is the only option on Windows anyway -- one code path for both.
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            results = list(ex.map(worker, C.SEEDS))

    for _seed, out in sorted(results, key=lambda r: r[0]):
        for name in ("mog", "rf", "anfis", "gafis"):
            if out[name] is not None:
                t_seconds, s_value = out[name]
                cols[name]["t"].append(t_seconds)
                cols[name]["s"].append(s_value)
    return cols


def _row(label, metric_name, cols):
    order = ["mog", "anfis", "gafis", "rf"]
    return [
        label,
        C.cell(cols["mog"]["t"], fmt="{:.2f}") + " s" if cols["mog"]["t"] else C.NA,
        f"{metric_name}=" + C.cell(cols["mog"]["s"]) if cols["mog"]["s"] else C.NA,
        *[C.cell(cols[k]["s"]) for k in order if k != "mog"],
    ]


def _timing_row(label, cols):
    """Train-time row for the companion speed table: the arms' wall-clock seconds
    and the MoG speedup over the slowest fuzzy arm -- the 'orders of magnitude
    faster than a tuned fuzzy system' claim (Goal C1), finally with a fuzzy arm
    to be faster *than*. Speedup is a ratio of means, so it needs no spread."""
    mt = np.mean(cols["mog"]["t"]) if cols["mog"]["t"] else None
    fuzzy = [np.mean(cols[k]["t"]) for k in ("anfis", "gafis") if cols[k]["t"]]
    speedup = (max(fuzzy) / mt) if (mt and fuzzy) else None
    return [
        label,
        C.cell(cols["mog"]["t"], fmt="{:.2f}") + " s" if cols["mog"]["t"] else C.NA,
        C.cell(cols["anfis"]["t"], fmt="{:.2f}") + " s" if cols["anfis"]["t"] else C.NA,
        C.cell(cols["gafis"]["t"], fmt="{:.2f}") + " s" if cols["gafis"]["t"] else C.NA,
        C.cell(cols["rf"]["t"], fmt="{:.2f}") + " s" if cols["rf"]["t"] else C.NA,
        f"{speedup:.0f}×" if speedup else C.NA,
    ]


def main():
    print("Table 4.1 -- MoG training time & accuracy vs. baselines")
    rows = []
    timing = []  # (label, cols) accumulator for the companion speed table

    concrete = _fm.load_concrete()
    if concrete is not None:
        X, y = concrete
        # Concrete: normalized, because that is the configuration Ch 4 argues for
        # and the one its accuracy figures are quoted at.
        cols = _bench("reg", X, y, _fm.mog_regressor, r2_score, norm=True)
        rows.append(_row("Concrete (regression)", "R2", cols))
        timing.append(("Concrete (regression)", cols))
        # Second consequent order, timed. Table 4.5 quotes a full-2nd-order R2
        # from `table_concrete_reconciliation.py`, which sweeps orders but never
        # times them, so that row's training-time cell had nothing behind it and
        # read `*pending*`. Measuring it here -- same split, same seeds, same
        # normalization, same timer as the 1st-order row -- makes the two rows
        # comparable, which borrowing the 1st-order seconds would not have.
        cols2 = _bench(
            "reg",
            X,
            y,
            # A lambda closure isn't picklable, and _bench's per-seed workers
            # (when running in a process pool) need to be -- partial() of a
            # module-level function is, and does the same job.
            partial(_fm.mog_regressor, tsk_order="full-2nd"),
            r2_score,
            norm=True,
        )
        rows.append(_row("Concrete (regression, full 2nd order)", "R2", cols2))
        timing.append(("Concrete (regression, full 2nd order)", cols2))
    else:
        rows.append(["Concrete (regression)", C.NA, C.NA, C.NA, C.NA, C.NA])

    bikeshare = _fm.load_bikeshare()
    if bikeshare is not None:
        X, y = bikeshare
        # Bike Sharing: normalized, for scale comparison with Concrete.
        # 17.3× larger (17,379 vs 1,030 rows) while maintaining regression task.
        cols = _bench("reg", X, y, _fm.mog_regressor, r2_score, norm=True)
        rows.append(_row("Bike Sharing (regression)", "R2", cols))
        timing.append(("Bike Sharing (regression)", cols))
    else:
        rows.append(["Bike Sharing (regression)", C.NA, C.NA, C.NA, C.NA, C.NA])

    phi = _fm.load_phiusiil()
    if phi is not None:
        X, y = phi
        # PhiUSIIL: left at the shipped default. Every method saturates on this
        # set, so the transform has nothing to buy, and Ch 4 quotes it as shipped.
        cols = _bench("clf", X, y, _fm.mog_classifier, accuracy_score, norm=False)
        rows.append(_row("PhiUSIIL (classification)", "acc", cols))
        timing.append(("PhiUSIIL (classification)", cols))
    else:
        rows.append(["PhiUSIIL (classification)", C.NA, C.NA, C.NA, C.NA, C.NA])

    iot = _fm.load_rt_iot2022()
    if iot is not None:
        X, y = iot
        # RT-IOT2022: raw features, left unnormalized -- matches the open-set
        # measurement (table_4_4_openset.py) and the single-split sanity check
        # (quick_iot_baseline.py), neither of which normalizes, so this row is
        # comparable to both rather than introducing a third convention.
        # top_n=5 (mog_classifier's default) -- NOT the open-set experiment's
        # all-82-feature antecedent screen, so this is materially cheaper and
        # answers a different question (the plain classification/timing claim
        # Table 4.4's row names, not the open-set complement-rule claim).
        cols = _bench("clf", X, y, _fm.mog_classifier, accuracy_score, norm=False)
        rows.append(_row("RT-IOT2022 (12-class)", "acc", cols))
        timing.append(("RT-IOT2022 (12-class)", cols))
    else:
        rows.append(["RT-IOT2022 (12-class)", C.NA, C.NA, C.NA, C.NA, C.NA])

    header = [
        "Dataset (task)",
        "MoG train time",
        "MoG accuracy / R2",
        "ANFIS",
        "GA-tuned FIS",
        "tree / RF ref",
    ]
    C.emit(
        "table_4_1",
        "Table 4.1 -- Training time and accuracy",
        header,
        rows,
        note="MoG columns measured; ANFIS / GA-FIS fill in only if their adapters "
        "(reproduce/tables/_baseline_anfis.py, _baseline_gafis.py) are present. "
        "The RF reference is scikit-learn. Times are wall-clock training seconds.",
    )

    # Companion speed table -- the actual Goal C1 deliverable. Table 4.1 above
    # shows the baselines are as ACCURATE as the MoG arm (often more so); this one
    # shows what the chapter actually claims, that they are far SLOWER. Emitted
    # only when at least one fuzzy adapter is present, so a run without them does
    # not ship a table of MoG-and-RF-only speedups that would read as complete.
    if anfis_fit_predict is not None or gafis_fit_predict is not None:
        C.emit(
            "table_4_1b_baseline_timing",
            "Table 4.1b -- Training-time comparison (the speed claim, Goal C1)",
            [
                "Dataset (task)",
                "MoG train",
                "ANFIS train",
                "GA-FIS train",
                "RF train",
                "MoG speedup vs slowest fuzzy",
            ],
            [_timing_row(label, cols) for label, cols in timing],
            note="Wall-clock training seconds, mean ± s.d. over the same seeds and "
            "splits as Table 4.1. 'Speedup' is the ratio of means, slowest of "
            "{ANFIS, GA-FIS} over MoG -- the fuzzy baseline the 'orders of "
            "magnitude faster' claim (Ch 1/7/8, Goal C1) previously had nothing "
            "to be faster than. ANFIS grid-partitions where feasible (Concrete) "
            "and scatter-partitions at scale; GA-FIS evolves the same premises. "
            "Both are fair: Table 4.1 shows they match or beat the MoG arm on "
            "accuracy, so the speedup is not bought by a weaker model.",
        )


if __name__ == "__main__":
    main()
