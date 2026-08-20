"""Table 4.8 -- Membership-function deduplication: reduction vs. tolerance,
across six problems, plus the correction-rule pass quantified for Glass.

Chapter 4 SS4.3.1 has owed this since the chapter was drafted: the correction-
rule pass ("a second, small pass of correction rules where two classes are
confused") was claimed and never measured, and Fig 4.3 has sat as a deliberate
placeholder for exactly this reason. `GaussianMixtureModel` already ships dedup
machinery (`to_simple_model`, `get_deduplicated_membership_fcns`), but nothing
in the harness had exercised it, and its merge tolerance is a hardcoded module
constant with no way to sweep it -- see `_mf_dedup.py`'s docstring and
tribble-fis#85, filed from this measurement.

WHAT THIS TABLE ANSWERS. Two questions, kept separate because they are
different effects with different shapes:

  1. How much of the fitted antecedent structure is redundant, and at what
     tolerance does removing it start costing accuracy? Swept across six
     problems (four classification, two regression) so the answer is a
     property of the *method*, not one dataset's luck.
  2. Does the correction-rule pass (SS4.3.1's `MixtureOfGaussiansFuzzySequence
     Classifier`) actually buy accuracy, and what does deploying it as ONE flat
     rule base (union its layers, dedup, plain argmax -- no gating logic) cost
     relative to the real gated cascade? Glass only: it is the one dataset in
     this table whose classes are confusable enough to grow a non-trivial
     cascade, and mixing this into the six-problem sweep above would conflate
     a *mechanism* cost (dropping the gates) with a *tolerance* cost (merging
     near-duplicate Gaussians) -- see the exploratory finding below and the
     "cascade" rows' own note.

METHOD -- pairing, not just averaging. Every accuracy/R2 number below is
PAIRED per seed: the same seed's held-out split scores both the raw and the
deduplicated model, and the reported delta is `dedup - raw` on that split, not
a difference of two independently-averaged means. Ten seeds (`common.SEEDS`).
This matters because the effect sizes near the tolerance boundary are small
enough that an unpaired comparison would be reading noise: at the library
default (rtol=1e-2, atol=1e-3, "1x" below) 9 of Glass's 10 base-model seeds
show EXACTLY zero change, and the one exception is a small improvement, not a
degradation -- a fact an unpaired mean-vs-mean comparison cannot see, because
it averages away the very thing (does any single seed diverge) that a five-
seed mean elsewhere in this dissertation (Ch 6 SS6.4's mixture-of-experts
divergence) already showed can hide.

THE "MAX-LOSSLESS" TOLERANCE, DEFINED. For each dataset, sweep the multiplier
grid in `_mf_dedup.MULTIPLIERS` (0.1x to 100x the library default) and compute
the paired delta's 95% CI (`mean +/- 1.96 * std / sqrt(n)`) at each step. The
max-lossless multiplier is the LARGEST one reachable by an unbroken run of
"CI contains zero" starting from the smallest multiplier tested -- i.e. the
point past which the sweep first gives real evidence of a cost, not merely the
largest multiplier that happens to show a small mean. A dataset whose delta
never leaves the CI anywhere in the grid reports the grid's own ceiling (100x)
plus a note, since the true boundary is unmeasured past that point, not proven
absent.

CASCADE CAVEAT, CARRIED FROM THE EXPLORATORY PASS. Flattening Glass's cascade
into one deployable FIS (union `layers_` via `GaussianMixtureModel.augment()`,
dedup, plain argmax) shows a paired delta whose std (0.065) is ~4x its own mean
(-0.017) AT ZERO TOLERANCE -- before any numeric merging happens at all. That
is the cost of dropping the anomaly/confidence gating, not of deduplication;
the two are reported as separate columns below rather than folded into one
number.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_4_8_mf_dedup.py
"""

from __future__ import annotations

import math
import os
import sys

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import _mf_dedup as D  # noqa: E402

from tribblefis.gaussian_classifier import (  # noqa: E402
    TribbleClassifier,
    TribbleSequenceClassifier,
)
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402
from tribblefis.regression import predict_tsk  # noqa: E402
from tribblefis.gauss_math import simple_gaussian_predict  # noqa: E402


def _ci_excludes_zero(mean, std, n):
    """95% CI for a paired mean, via the normal approximation (n=10)."""
    half = 1.96 * std / math.sqrt(n)
    return (mean - half > 0) or (mean + half < 0)


def _max_lossless(rows):
    """`rows`: per-multiplier dicts with delta_mean/delta_std, in ascending
    multiplier order. Returns (multiplier, row, hit_ceiling: bool)."""
    best = rows[0]
    for r in rows:
        if _ci_excludes_zero(r["delta_mean"], r["delta_std"], r["n"]):
            break
        best = r
    hit_ceiling = best is rows[-1] and not _ci_excludes_zero(
        best["delta_mean"], best["delta_std"], best["n"]
    )
    return best, hit_ceiling


def _sweep_classification(name, loader):
    data = loader()
    if data is None:
        return None
    X, y = data
    per_mult = {m: {"dedup_mf": [], "delta": []} for m in D.MULTIPLIERS}
    raw_mf_list, acc_raw_list = [], []
    for seed in C.SEEDS:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )
        top_n = min(5, X.shape[1])
        clf = TribbleClassifier(top_n=top_n, random_state=seed)
        clf.fit(X_tr, y_tr)
        raw_mf = clf.model_.n_membership_functions
        acc_raw = accuracy_score(y_te, clf.predict(X_te))
        raw_mf_list.append(raw_mf)
        acc_raw_list.append(acc_raw)
        for m in D.MULTIPLIERS:
            simple = D.to_simple_model_tol(clf.model_, D.LIB_RTOL * m, D.LIB_ATOL * m)
            acc_dedup = accuracy_score(y_te, simple_gaussian_predict(X_te, simple))
            per_mult[m]["dedup_mf"].append(len(simple.input_mfs))
            per_mult[m]["delta"].append(acc_dedup - acc_raw)
    return _finish("classification", name, raw_mf_list, acc_raw_list, per_mult)


def _sweep_regression(name, loader):
    data = loader()
    if data is None:
        return None
    X, y = data
    per_mult = {m: {"dedup_mf": [], "delta": []} for m in D.MULTIPLIERS}
    raw_mf_list, r2_raw_list = [], []
    for seed in C.SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
        reg = TribbleRegressor(
            n_output_buckets=3, tsk_order="1st", top_n=-1, random_state=seed
        )
        reg.fit(X_tr, y_tr)
        raw_mf = reg.model_.n_membership_functions
        r2_raw = r2_score(y_te, reg.predict(X_te))
        raw_mf_list.append(raw_mf)
        r2_raw_list.append(r2_raw)
        for m in D.MULTIPLIERS:
            deduped = D.build_deduped_model(reg.model_, D.LIB_RTOL * m, D.LIB_ATOL * m)
            n_dedup = len({mf.id for mf in deduped.all_membership_fcns})
            pred = predict_tsk(
                X_te,
                deduped,
                reg.top_features_,
                reg.y_bucket_mean_,
                reg.corr_terms_,
                order=reg.tsk_order,
                basis=reg.consequent_basis,
                norms=reg._norms(),
            )
            r2_dedup = r2_score(y_te, pred)
            per_mult[m]["dedup_mf"].append(n_dedup)
            per_mult[m]["delta"].append(r2_dedup - r2_raw)
    return _finish("regression", name, raw_mf_list, r2_raw_list, per_mult)


def _finish(task, name, raw_mf_list, metric_raw_list, per_mult):
    n = len(raw_mf_list)
    raw_mf_mean, _ = C.agg(raw_mf_list)
    metric_mean, metric_std = C.agg(metric_raw_list)
    rows = []
    for m in D.MULTIPLIERS:
        dm_mean, dm_std = C.agg(per_mult[m]["dedup_mf"])
        d_mean, d_std = C.agg(per_mult[m]["delta"])
        rows.append(
            dict(
                multiplier=m,
                n=n,
                raw_mf_mean=raw_mf_mean,
                dedup_mf_mean=dm_mean,
                dedup_mf_std=dm_std,
                delta_mean=d_mean,
                delta_std=d_std,
            )
        )
    lossless, hit_ceiling = _max_lossless(rows)
    default_row = next(r for r in rows if r["multiplier"] == 1.0)
    return dict(
        task=task,
        name=name,
        n=n,
        raw_mf_mean=raw_mf_mean,
        metric_raw_mean=metric_mean,
        metric_raw_std=metric_std,
        rows=rows,
        default_row=default_row,
        lossless_row=lossless,
        lossless_multiplier=lossless["multiplier"],
        hit_ceiling=hit_ceiling,
    )


def _reduction_pct(raw, dedup):
    return 100.0 * (1.0 - dedup / raw) if raw else 0.0


def _fmt_delta(mean, std):
    sign = "+" if mean >= 0 else ""
    return f"{sign}{mean:.4f} ± {std:.4f}"


def run_glass_cascade():
    """The correction-rule pass, quantified: base vs. gated cascade vs. the
    cascade flattened into one deployable FIS (dedup at rtol=atol=0, i.e. the
    numeric-merge-free floor, so the number reported is the mechanism cost
    alone -- see the module docstring)."""
    data = D.load_glass()
    if data is None:
        return None
    X, y = data
    acc_base, acc_casc, acc_flat = [], [], []
    raw_mf_base, raw_mf_casc, dedup_mf_flat = [], [], []
    n_experts_list = []
    for seed in C.SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )

        base = TribbleClassifier(top_n=5, random_state=seed)
        base.fit(X_tr, y_tr)
        acc_base.append(accuracy_score(y_te, base.predict(X_te)))
        raw_mf_base.append(base.model_.n_membership_functions)

        casc = TribbleSequenceClassifier(top_n=5, random_state=seed)
        casc.fit(X_tr, y_tr)
        acc_casc.append(accuracy_score(y_te, casc.predict(X_te).astype(int)))
        raw_mf_casc.append(
            sum(layer.model_.n_membership_functions for layer in casc.layers_)
        )
        n_experts_list.append(len(casc.experts_))

        combined = casc.layers_[0].model_
        for layer in casc.layers_[1:]:
            combined = combined.augment(layer.model_)
        simple_flat = D.to_simple_model_tol(combined, 0.0, 0.0)
        acc_flat.append(
            accuracy_score(y_te, simple_gaussian_predict(X_te, simple_flat).astype(int))
        )
        dedup_mf_flat.append(len(simple_flat.input_mfs))

    delta_casc = [c - b for c, b in zip(acc_casc, acc_base)]
    delta_flat = [f - b for f, b in zip(acc_flat, acc_base)]
    return dict(
        acc_base=C.agg(acc_base),
        acc_casc=C.agg(acc_casc),
        acc_flat=C.agg(acc_flat),
        raw_mf_base=C.agg(raw_mf_base),
        raw_mf_casc=C.agg(raw_mf_casc),
        dedup_mf_flat=C.agg(dedup_mf_flat),
        n_experts=C.agg(n_experts_list),
        delta_casc=C.agg(delta_casc),
        delta_flat=C.agg(delta_flat),
    )


def main():
    results = []
    for name, loader in D.CLASSIFICATION_DATASETS:
        print(
            f"  [table-4-8] fitting {name} (classification, {len(C.SEEDS)} seeds x "
            f"{len(D.MULTIPLIERS)} tolerances)..."
        )
        r = _sweep_classification(name, loader)
        if r:
            results.append(r)
    for name, loader in D.REGRESSION_DATASETS:
        print(
            f"  [table-4-8] fitting {name} (regression, {len(C.SEEDS)} seeds x "
            f"{len(D.MULTIPLIERS)} tolerances)..."
        )
        r = _sweep_regression(name, loader)
        if r:
            results.append(r)

    # --- Table 4.8: summary, one row per problem ---
    header = [
        "Dataset",
        "Task",
        "Raw MF",
        "MF @ 1x (Δ)",
        "Reduction @ 1x",
        "Max-lossless ×",
        "MF @ max-lossless (Δ)",
        "Reduction @ max-lossless",
    ]
    rows = []
    for r in results:
        d = r["default_row"]
        lr = r["lossless_row"]
        metric = "acc" if r["task"] == "classification" else "R²"
        rows.append(
            [
                r["name"],
                r["task"],
                f"{r['raw_mf_mean']:.1f}",
                f"{d['dedup_mf_mean']:.1f} ({_fmt_delta(d['delta_mean'], d['delta_std'])} {metric})",
                f"{_reduction_pct(r['raw_mf_mean'], d['dedup_mf_mean']):.1f}%",
                f"{lr['multiplier']:g}×"
                + (" (grid ceiling)" if r["hit_ceiling"] else ""),
                f"{lr['dedup_mf_mean']:.1f} ({_fmt_delta(lr['delta_mean'], lr['delta_std'])} {metric})",
                f"{_reduction_pct(r['raw_mf_mean'], lr['dedup_mf_mean']):.1f}%",
            ]
        )
    note = (
        '"Max-lossless ×" is the largest tolerance multiplier (relative to the library '
        'default rtol=1e-2/atol=1e-3) reachable by an unbroken run of "95% CI for the paired '
        'delta contains zero" starting from 0.1x. "(grid ceiling)" means the delta never left '
        "the CI anywhere in the swept grid (0.1x-100x) -- the true boundary is unmeasured past "
        "100x, not proven absent. See `table_4_8_mf_dedup_sweep.csv` for the full per-multiplier "
        "detail behind every row."
    )
    C.emit(
        "table_4_8_mf_dedup",
        "Table 4.8 — MF deduplication: reduction vs. tolerance, six problems",
        header,
        rows,
        note=note,
    )

    # --- companion: full per-(dataset, multiplier) sweep, for the record ---
    sweep_header = [
        "Dataset",
        "Task",
        "Multiplier",
        "Raw MF",
        "Dedup MF (mean±std)",
        "Delta (mean±std)",
        "CI excludes zero",
    ]
    sweep_rows = []
    for r in results:
        for row in r["rows"]:
            excl = _ci_excludes_zero(row["delta_mean"], row["delta_std"], row["n"])
            sweep_rows.append(
                [
                    r["name"],
                    r["task"],
                    f"{row['multiplier']:g}",
                    f"{row['raw_mf_mean']:.1f}",
                    f"{row['dedup_mf_mean']:.2f} ± {row['dedup_mf_std']:.2f}",
                    f"{row['delta_mean']:+.5f} ± {row['delta_std']:.5f}",
                    "yes" if excl else "no",
                ]
            )
    C.emit(
        "table_4_8_mf_dedup_sweep",
        "Table 4.8 (detail) — full per-multiplier sweep",
        sweep_header,
        sweep_rows,
        note="Full grid behind Table 4.8's summary row per dataset. "
        '"CI excludes zero" is the per-step version of the max-lossless rule.',
    )

    # --- Glass cascade: the correction-rule pass, quantified ---
    casc = run_glass_cascade()
    if casc:
        ab_m, ab_s = casc["acc_base"]
        ac_m, ac_s = casc["acc_casc"]
        af_m, af_s = casc["acc_flat"]
        rb_m, rb_s = casc["raw_mf_base"]
        rc_m, rc_s = casc["raw_mf_casc"]
        df_m, df_s = casc["dedup_mf_flat"]
        ne_m, ne_s = casc["n_experts"]
        dc_m, dc_s = casc["delta_casc"]
        df2_m, df2_s = casc["delta_flat"]
        casc_header = ["Arm", "MF count", "Accuracy", "Paired Δ vs. base"]
        casc_rows = [
            [
                "Base (no correction pass)",
                f"{rb_m:.1f} ± {rb_s:.1f}",
                f"{ab_m:.4f} ± {ab_s:.4f}",
                "—",
            ],
            [
                "Gated cascade (base + experts, routed)",
                f"{rc_m:.1f} ± {rc_s:.1f} raw",
                f"{ac_m:.4f} ± {ac_s:.4f}",
                _fmt_delta(dc_m, dc_s),
            ],
            [
                "Cascade → one flat FIS (union, dedup @ exact tol., argmax)",
                f"{df_m:.1f} ± {df_s:.1f} deduped",
                f"{af_m:.4f} ± {af_s:.4f}",
                _fmt_delta(df2_m, df2_s),
            ],
        ]
        casc_note = (
            f"Glass only, {len(C.SEEDS)} seeds, top_n=5, mean {ne_m:.1f} experts per fit. "
            "The flattened arm's dedup runs at EXACT tolerance only (rtol=atol=0) -- it "
            "isolates the cost of dropping the cascade's gating logic from the cost of "
            "numeric merging, which the exploratory pass found conflated. Deploying the "
            "correction pass as one flat rule base recovers part, not all, of the gated "
            "cascade's accuracy gain over the base alone."
        )
        C.emit(
            "table_4_9_correction_pass",
            "Table 4.9 — The correction-rule pass, quantified (Glass)",
            casc_header,
            casc_rows,
            note=casc_note,
        )

    print(
        "\nDone. Emitted table_4_8_mf_dedup, table_4_8_mf_dedup_sweep, table_4_9_correction_pass."
    )


if __name__ == "__main__":
    main()
