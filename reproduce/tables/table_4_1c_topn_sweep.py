"""Table 4.1c -- MoG quality and cost against features kept, per feature scorer.

Table 4.1 compares the shipped MoG default against ANFIS, GA-FIS and a random
forest. On PhiUSIIL the MoG cell reads 0.440 +/- 0.181 against ~0.999 for the
fuzzy baselines. That cell is `TribbleClassifier(top_n=5)` under the default
Wasserstein scorer -- exactly Table A.2's Wasserstein cell at k = 5 (0.4401) --
so what is measured there is a *feature selection*, not the model family. This
table separates the two: the same estimator, swept over `top_n` and over the
four scorers (`scoring_method`, tribble-fis #239), beside the same baselines.

Three tables and one raw file:

  4.1c-a  comparison     shipped MoG default | ANFIS | GA-FIS | RF, score and
                         train time, plus the best cell the sweep found
  4.1c-b  top_n sweep    score / mean fit seconds for every (dataset, k, scorer)
  4.1c-c  sweep cost     total fit seconds for the WHOLE sweep, and per run
                         (mean, min, max), against ten seeds of each baseline
  table_4_1c_topn_runs.csv   one line per fit -- the evidence a later diff uses

Expectations registered before the first run (a result that contradicts one
needs its own explanation, not a quiet edit of this list):

  E1  PhiUSIIL, wasserstein, k = 5 reproduces Table 4.1's MoG cell: 0.440 +/- 0.181.
  E2  Scorer choice moves the PhiUSIIL score by tens of points at small k and by
      about a point at k >= 20 (Table A.2, 10 seeds).
  E3  Concrete and Bike Sharing move far less with the scorer than PhiUSIIL does.
  E4  Fit time grows with k and is dominated by scoring at small k: every fit
      re-scores all features, so the sweep pays for scoring once per (k, seed).

Caveats that travel with the numbers:
  * "Best cell" is chosen on the same test splits it is scored on across
    (scorers x k) cells, so it is OPTIMISTIC -- read it as a ceiling, not a
    result. Nothing here selects a configuration on held-out data.
  * ANFIS is not deterministic run to run (PROVENANCE_MAP 2026-11-20); its cells
    are a sample, not a constant.
  * `top_n` is the REQUEST. The estimator's 0.85 redundancy filter can keep
    fewer; the cost table counts the runs where it did.
  * RT-IOT2022 is not in this table: 92k rows x 82 features x 11 labels makes a
    48-cell sweep cost hours (Table 4.4 spends 2h09m on one pass), and the
    dataset is currently absent. It reads as excluded, not as a result.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_4_1c_topn_sweep.py

Knobs:
    REPRO_SEEDS="0,1"      seeds (smoke only; never cite)
    REPRO_SCORERS="wasserstein,bhattacharyya,composite,higher_order"
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import _fuzzy_models as _fm  # noqa: E402
import table_4_1_mog_baselines as T  # noqa: E402

SCORERS = [
    s.strip()
    for s in os.environ.get(
        "REPRO_SCORERS", "wasserstein,bhattacharyya,composite,higher_order"
    ).split(",")
]
K_BASE = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 47]

# (label, loader, task kind, normalize-before-split) -- the same four settings
# Table 4.1 uses: Concrete and Bike Sharing are normalized for every arm,
# PhiUSIIL is left as shipped.
DATASETS = [
    ("Concrete (regression)", _fm.load_concrete, "reg", True),
    ("Bike Sharing (regression)", _fm.load_bikeshare, "reg", True),
    ("PhiUSIIL (classification)", _fm.load_phiusiil, "clf", False),
]


def _grid(m):
    """The requested top_n values for an m-feature dataset; ends at 'all of them'."""
    return [k for k in K_BASE if k < m] + [m]


def _model(kind, scorer, k, seed):
    if kind == "reg":
        from tribblefis.gaussian_regressor import TribbleRegressor

        return TribbleRegressor(
            n_output_buckets=3,
            tsk_order="1st",
            top_n=k,
            scoring_method=scorer,
            random_state=seed,
        )
    from tribblefis.gaussian_classifier import TribbleClassifier

    return TribbleClassifier(top_n=k, scoring_method=scorer, random_state=seed)


def _sweep(kind, X, y, score):
    """Fit every (scorer, k, seed). Returns (runs, wall_seconds).

    `runs` is a list of dicts. The whole sweep is wrapped in one wall-clock
    stopwatch -- which includes prediction, scoring-by-the-metric and the
    bookkeeping -- separately from the per-run fit seconds, because the two
    answer different questions: "what does one model cost" and "what does the
    experiment cost".
    """
    grid = _grid(X.shape[1])
    splits = {
        seed: train_test_split(X, y, test_size=0.2, random_state=seed)
        for seed in C.SEEDS
    }

    # One DISCARDED warm-up fit, so seed 0 of the first cell does not pay for
    # import, JIT and thread-pool start-up (same rationale as table_4_1's _bench).
    # RNG state is snapshotted around it so a discarded fit consumes nothing shared.
    Xw, Xwt, yw, _ = splits[C.SEEDS[0]]
    state = np.random.get_state()
    try:
        _model(kind, SCORERS[0], grid[0], C.SEEDS[0]).fit(Xw, yw).predict(Xwt)
    except Exception:  # noqa: BLE001 -- a failed warm-up must not fail the table
        pass
    finally:
        np.random.set_state(state)

    runs = []
    t_wall = time.perf_counter()
    for scorer in SCORERS:
        for k in grid:
            for seed in C.SEEDS:
                Xtr, Xte, ytr, yte = splits[seed]
                rec = dict(scorer=scorer, k=k, seed=seed, fit_s=None, score=None, kept=None)
                try:
                    model = _model(kind, scorer, k, seed)
                    start = time.perf_counter()
                    model.fit(Xtr, ytr)
                    rec["fit_s"] = time.perf_counter() - start
                    rec["score"] = float(score(yte, np.asarray(model.predict(Xte))))
                    rec["kept"] = len(model.top_features_)
                except Exception as exc:  # noqa: BLE001 -- counted, never skipped
                    print(f"  [fail] {scorer} k={k} seed={seed}: {exc.__class__.__name__}: {exc}")
                runs.append(rec)
    return runs, time.perf_counter() - t_wall


def _vals(runs, scorer, k, field):
    return [r[field] for r in runs if r["scorer"] == scorer and r["k"] == k and r[field] is not None]


def main():
    wall0 = time.perf_counter()
    print("Table 4.1c -- top_n sweep: MoG quality and cost vs features kept")
    print(f"  seeds={C.SEEDS}  scorers={SCORERS}")

    comparison, sweep_rows, cost_rows, long_rows = [], [], [], []
    notes = []

    for label, loader, kind, norm in DATASETS:
        data = loader()
        if data is None:
            print(f"  [{label}] dataset unavailable -> N/A (NOT a result)")
            comparison.append([label] + [C.NA] * 9)
            sweep_rows.append([label, C.NA] + [C.NA] * len(SCORERS))
            cost_rows.append([label, C.NA, C.NA, C.NA, C.NA, C.NA, C.NA, C.NA])
            continue
        X, y = data
        if norm:
            X = _fm.normalize(X)[0]
        score = r2_score if kind == "reg" else accuracy_score
        metric = "R2" if kind == "reg" else "acc"
        print(f"\n== {label}: N={len(X)} M={X.shape[1]} metric={metric}")

        # ---- the shipped default and the baselines, via Table 4.1's own harness ----
        factory = _fm.mog_regressor if kind == "reg" else _fm.mog_classifier
        t_base = time.perf_counter()
        cols = T._bench(kind, X, y, factory, score, norm=False)  # X already normalized
        base_wall = time.perf_counter() - t_base

        # ---- the sweep ----
        runs, sweep_wall = _sweep(kind, X, y, score)
        grid = _grid(X.shape[1])
        for r in runs:
            long_rows.append([label, r["scorer"], r["k"], r["seed"], r["fit_s"], r["score"], r["kept"]])

        # ---- 4.1c-b: score / mean fit seconds per (k, scorer) ----
        for k in grid:
            row = [label, str(k)]
            for sc in SCORERS:
                v, t = _vals(runs, sc, k, "score"), _vals(runs, sc, k, "fit_s")
                row.append(f"{C.cell(v)} / {np.mean(t):.2f} s" if v and t else C.NA)
            sweep_rows.append(row)

        # ---- best cell (optimistic) ----
        best = None
        for sc in SCORERS:
            for k in grid:
                v = _vals(runs, sc, k, "score")
                if v and (best is None or np.mean(v) > best[0]):
                    best = (float(np.mean(v)), sc, k, C.cell(v))
        best_txt = f"{best[3]} ({best[1]}, k={best[2]})" if best else C.NA

        def tcell(name):
            t = cols[name]["t"]
            return C.cell(t, fmt="{:.2f}") + " s" if t else C.NA

        def scell(name):
            s = cols[name]["s"]
            return C.cell(s) if s else C.NA

        comparison.append(
            [label, f"{metric}={scell('mog')}", tcell("mog"), scell("anfis"), tcell("anfis"),
             scell("gafis"), tcell("gafis"), scell("rf"), tcell("rf"), best_txt]
        )

        # ---- 4.1c-c: cost of the whole sweep, and of one run ----
        for sc in SCORERS:
            t = [r["fit_s"] for r in runs if r["scorer"] == sc and r["fit_s"] is not None]
            shrunk = sum(1 for r in runs if r["scorer"] == sc and r["kept"] is not None and r["kept"] < r["k"])
            failed = sum(1 for r in runs if r["scorer"] == sc and r["fit_s"] is None)
            cost_rows.append(
                [label, sc, str(len(t) + failed), f"{sum(t):.1f} s",
                 f"{np.mean(t):.2f} s" if t else C.NA,
                 f"{min(t):.2f}-{max(t):.2f} s" if t else C.NA,
                 str(shrunk), str(failed)]
            )
        all_t = [r["fit_s"] for r in runs if r["fit_s"] is not None]
        cost_rows.append(
            [label, "ALL SCORERS (sweep total)", str(len(runs)), f"{sum(all_t):.1f} s",
             f"{np.mean(all_t):.2f} s" if all_t else C.NA,
             f"{min(all_t):.2f}-{max(all_t):.2f} s" if all_t else C.NA,
             str(sum(1 for r in runs if r["kept"] is not None and r["kept"] < r["k"])),
             str(len(runs) - len(all_t))]
        )
        cost_rows.append(
            [label, "sweep wall-clock (fit + predict + scoring)", "", f"{sweep_wall:.1f} s", "", "", "", ""]
        )
        for name, nice in (("mog", "MoG shipped default"), ("anfis", "ANFIS"), ("gafis", "GA-FIS"), ("rf", "RF")):
            t = cols[name]["t"]
            if t:
                cost_rows.append(
                    [label, f"baseline: {nice} x{len(t)} seeds", str(len(t)), f"{sum(t):.1f} s",
                     f"{np.mean(t):.2f} s", f"{min(t):.2f}-{max(t):.2f} s", "", "0"]
                )
        print(f"  baselines (Table 4.1 harness) {base_wall:.0f}s wall | sweep {sweep_wall:.0f}s wall, "
              f"{sum(all_t):.0f}s of fits over {len(runs)} runs")

    total_wall = time.perf_counter() - wall0
    notes.append(f"Whole generator wall-clock: {total_wall:.0f} s.")

    base_note = (
        "Seeds: " + ",".join(map(str, C.SEEDS)) + ". 'Best cell' is the highest mean over "
        "(scorers x k), chosen on the same test splits it is scored on -- optimistic, a ceiling "
        "not a result. ANFIS is nondeterministic run to run. RT-IOT2022 excluded (cost; dataset "
        "absent). " + " ".join(notes)
    )
    C.emit(
        "table_4_1c_comparison",
        "Table 4.1c-a -- MoG default vs ANFIS / GA-FIS / RF, and the best the top_n sweep reaches",
        ["Dataset (task)", "MoG default (shipped)", "MoG train", "ANFIS", "ANFIS train",
         "GA-FIS", "GA-FIS train", "RF", "RF train", "MoG best sweep cell"],
        comparison,
        note=base_note,
    )
    C.emit(
        "table_4_1c_topn_sweep",
        "Table 4.1c-b -- score / mean fit seconds by top_n and feature scorer (MoG)",
        ["Dataset (task)", "top_n (requested)"] + SCORERS,
        sweep_rows,
        note=base_note + " Each cell: mean +/- std of the score over seeds / mean fit seconds "
        "(scoring included -- every fit re-scores all features).",
    )
    C.emit(
        "table_4_1c_sweep_cost",
        "Table 4.1c-c -- cost of the whole top_n sweep and of one run",
        ["Dataset (task)", "Arm", "Runs", "Total fit time", "Mean / run", "Min-max / run",
         "Runs below requested top_n", "Failed"],
        cost_rows,
        note=base_note + " 'Total fit time' sums per-run fit seconds; the wall-clock row adds "
        "prediction and scoring. 'Runs below requested top_n' counts fits where the 0.85 "
        "redundancy filter kept fewer features than asked for.",
    )
    raw = os.path.join(C.OUTPUT_DIR, "table_4_1c_topn_runs.csv")
    with open(raw, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "scorer", "top_n", "seed", "fit_seconds", "score", "features_kept"])
        w.writerows(long_rows)
    print(f"  wrote {raw}  ({len(long_rows)} runs)")
    print(f"\nTotal wall-clock: {total_wall:.0f} s")


if __name__ == "__main__":
    main()
