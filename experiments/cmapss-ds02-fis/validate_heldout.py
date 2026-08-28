"""Held-out validation of the two committed 'wins': raw_memory's 4 output
buckets and the whole_cycle/raw_memory blend.

Both were originally chosen by minimising the *test* metric, which is model
selection on the reporting set -- not a real generalisation result. This carves
a validation set out of the TRAINING engines (grouped by engine, stratified
across datasets so no engine leaks and every file is represented), selects each
hyperparameter on validation, then reports on the untouched official test
engines. A gain is only real if the validation-selected setting also wins on
test.

Run from the repo root (needs data/nasa-cmapps2/):
    python experiments/cmapss-ds02-fis/validate_heldout.py
"""

import contextlib
import io
import os

import numpy as np  # noqa: E402

from _ds02_harness import bootstrap, rmse  # noqa: E402

bootstrap("FuzzySystemsExperiments", os.path.dirname(__file__))
import cmapss_all_datasets as cad  # noqa: E402
from tribble_predictive_health import TribblePredictiveHealth  # noqa: E402
from tribble_predictive_health.metrics import nasa_score  # noqa: E402


def split_val(train, frac_every=3):
    """~1/frac_every of each dataset's engines held out as validation, taken by
    sorted order (deterministic, no RNG). Returns (fit_tbl, val_tbl)."""
    val_engines = []
    for _, sub in train.groupby("dataset"):
        engs = sorted(sub["engine"].unique())
        val_engines += engs[::frac_every]
    mask = train["engine"].isin(val_engines)
    return train[~mask].copy(), train[mask].copy(), len(val_engines)


def fit_rm(cfg, table, cols):
    eng = TribblePredictiveHealth(condition_correction=False, unit_col="engine", **cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        eng.fit_featurized(table, cols)
    return eng


def per_cycle(eng, table):
    s = eng.predict_samples_featurized(table)
    return (
        s.groupby(["engine", "cycle"])
        .agg(pred=("predicted_rul", "mean"), true=("rul", "mean"))
        .reset_index()
    )


def pe_last(df, col):
    last = df.sort_values("cycle").groupby("engine").last()
    return rmse(last["true"], last[col]), nasa_score(last["true"], last[col])


def main(h5_dir):
    print(f"Loading + pooling {h5_dir} ...")
    pooled, processed, skipped = cad.gather(h5_dir)
    tr_rm, te_rm, cols_rm = pooled["raw_memory"]
    tr_wc, te_wc, cols_wc = pooled["whole_cycle"]

    fit_rm_t, val_rm_t, n_val = split_val(tr_rm)
    fit_wc_t, val_wc_t, _ = split_val(tr_wc)
    print(
        f"  train engines {tr_rm['engine'].nunique()}  -> "
        f"fit {fit_rm_t['engine'].nunique()} / val {val_rm_t['engine'].nunique()}"
        f" ; test engines {te_rm['engine'].nunique()}"
    )

    # ---- Validation A: n_output_buckets for raw_memory --------------------
    print("\n=== A. raw_memory n_output_buckets ===")
    print("  nb | VAL per-sample | TEST per-sample | TEST per-engine")
    base_cfg = {
        k: v for k, v in cad.CONFIGS["raw_memory"].items() if k != "n_output_buckets"
    }
    A = []
    for nb in (2, 3, 4, 6, 8):
        cfg = {**base_cfg, "n_output_buckets": nb}
        val_ps = fit_rm(cfg, fit_rm_t, cols_rm).score_featurized(val_rm_t)[
            "per_sample_rmse"
        ]
        m = fit_rm(cfg, tr_rm, cols_rm).score_featurized(te_rm)
        print(
            f"  {nb:2d} | {val_ps:13.2f} | {m['per_sample_rmse']:14.2f} | {m['per_engine_rmse']:14.2f}"
        )
        A.append((nb, val_ps, m["per_sample_rmse"], m["per_engine_rmse"]))
    nb_val = min(A, key=lambda r: r[1])[0]
    nb2 = next(r for r in A if r[0] == 2)
    nb_sel = next(r for r in A if r[0] == nb_val)
    print(
        f"  -> validation picks nb={nb_val}. TEST per-sample: nb=2 {nb2[2]:.2f} vs "
        f"nb={nb_val} {nb_sel[2]:.2f}  ({'IMPROVEMENT' if nb_sel[2] < nb2[2] else 'NO IMPROVEMENT'})"
    )

    # ---- Validation B: blend alpha ---------------------------------------
    print("\n=== B. blend alpha (raw_memory at committed config) ===")
    rm_fit, wc_fit = fit_rm(cad.CONFIGS["raw_memory"], fit_rm_t, cols_rm), fit_rm(
        cad.CONFIGS["whole_cycle"], fit_wc_t, cols_wc
    )
    rm_full, wc_full = fit_rm(cad.CONFIGS["raw_memory"], tr_rm, cols_rm), fit_rm(
        cad.CONFIGS["whole_cycle"], tr_wc, cols_wc
    )

    def joined(rm_eng, wc_eng, rm_tbl, wc_tbl):
        rm = per_cycle(rm_eng, rm_tbl).rename(columns={"pred": "pred_rm"})
        wc = per_cycle(wc_eng, wc_tbl).rename(columns={"pred": "pred_wc"})
        j = wc.merge(rm[["engine", "cycle", "pred_rm"]], on=["engine", "cycle"])
        return j

    jv = joined(rm_fit, wc_fit, val_rm_t, val_wc_t)
    jt = joined(rm_full, wc_full, te_rm, te_wc)

    print(
        "  alpha | VAL per-engine | TEST per-engine | TEST NASA   (alpha=1 -> whole_cycle)"
    )
    B = []
    for a in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0):
        jv["b"] = a * jv["pred_wc"] + (1 - a) * jv["pred_rm"]
        jt["b"] = a * jt["pred_wc"] + (1 - a) * jt["pred_rm"]
        v, _ = pe_last(jv, "b")
        t, tn = pe_last(jt, "b")
        print(f"  {a:5.1f} | {v:13.2f} | {t:14.2f} | {tn:9,.0f}")
        B.append((a, v, t, tn))
    a_val = min(B, key=lambda r: r[1])[0]
    a1 = next(r for r in B if r[0] == 1.0)
    a_sel = next(r for r in B if r[0] == a_val)
    print(
        f"  -> validation picks alpha={a_val}. TEST per-engine: whole_cycle(a=1) "
        f"{a1[2]:.2f} vs blend {a_sel[2]:.2f}  "
        f"({'IMPROVEMENT' if a_sel[2] < a1[2] else 'NO IMPROVEMENT'})"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", default="data/nasa-cmapps2")
    main(ap.parse_args().h5_dir)
