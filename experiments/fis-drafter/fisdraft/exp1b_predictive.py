"""Experiment 1b -- the predictive ceiling on the deployable feature set.

The FIS is being asked to predict shape parameters from cheap features. Before
building it, establish what a strong, unconstrained model gets on the same
features. If gradient boosting only reaches R^2 = 0.15, the FIS has nothing to
be interpretable *about*, and the finding is about the features, not the model.

Two design points that the numbers depend on entirely:

**Split by prompt, never by step.** Steps within one generation are strongly
dependent, so a random row split puts near-duplicate rows on both sides and
inflates every score. All splits here are on `prompt_id`.

**Report the three tiers separately.** Tier A+B is deployable. Tier C requires
the forward pass the drafter exists to avoid, so C is not a candidate model --
it is the ceiling that says how much of this is knowable at all.

Baselines are included at the same status as the models, because in
`experiments/fuzzy-lm-anomaly.md` a single scalar statistic beat every learned
detector and the study nearly missed it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

TIER_A = [
    "prev1_entropy", "prev2_entropy", "prev3_entropy",
    "prev1_varentropy", "prev2_varentropy", "prev3_varentropy",
    "prev1_top1_prob", "prev2_top1_prob", "prev3_top1_prob",
    "prev1_log_margin_12", "prev2_log_margin_12", "prev3_log_margin_12",
    "ent_ema_short", "ent_ema_long", "ent_cummax",
]
TIER_B = ["step", "prompt_len", "tok_len", "tok_is_space", "tok_is_punct",
          "tok_is_alpha", "tok_logfreq"]

TARGETS = ["entropy", "top1_prob", "log_margin_12", "nucleus_90"]


def add_tier_b(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Surface features of the token being conditioned on. No forward pass."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    ids = df.input_token.to_numpy()
    uniq = np.unique(ids)
    strs = {int(i): tok.convert_ids_to_tokens(int(i)) or "" for i in uniq}

    # Unigram frequency estimated from this corpus. A drafter could hold a
    # precomputed table, so this stays tier B.
    counts = pd.Series(ids).value_counts()
    freq_map = (counts / counts.sum()).to_dict()

    s = pd.Series([strs[int(i)] for i in ids], index=df.index)
    df = df.copy()
    df["tok_len"] = s.str.len().astype(float)
    # SmolLM2 uses the GPT-2 byte-level convention: 'Ġ' marks a leading space.
    df["tok_is_space"] = s.str.startswith("Ġ").astype(float)
    df["tok_is_punct"] = s.str.strip("Ġ").str.match(r"^[^\w\s]+$").fillna(False).astype(float)
    df["tok_is_alpha"] = s.str.strip("Ġ").str.match(r"^[A-Za-z]+$").fillna(False).astype(float)
    df["tok_logfreq"] = np.log([freq_map.get(int(i), 1e-9) for i in ids])
    return df


def tier_c_features(rundir: Path, df: pd.DataFrame, n_pca: int = 32):
    """Hidden state (PCA-reduced) and per-layer norms. Not deployable."""
    hid = np.load(rundir / "hidden_last.npy")[df.row.to_numpy()]
    lns = np.load(rundir / "layer_norms.npy")[df.row.to_numpy()]
    return hid, lns, n_pca


def evaluate(Xtr, ytr, Xte, yte, seed=0) -> dict:
    out = {}
    d = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    out["constant"] = r2_score(yte, d.predict(Xte))

    # Single best feature, chosen on train only. This is the arm that beat every
    # learned detector in the anomaly study.
    best, best_r2 = None, -np.inf
    for j in range(Xtr.shape[1]):
        m = HistGradientBoostingRegressor(
            max_iter=100, random_state=seed
        ).fit(Xtr[:, [j]], ytr)
        r = r2_score(yte, m.predict(Xte[:, [j]]))
        if r > best_r2:
            best, best_r2 = j, r
    out["single_best_feature"] = best_r2
    out["single_best_feature_idx"] = int(best)

    out["ridge"] = r2_score(yte, RidgeCV().fit(Xtr, ytr).predict(Xte))
    out["gbm"] = r2_score(
        yte,
        HistGradientBoostingRegressor(max_iter=400, random_state=seed)
        .fit(Xtr, ytr)
        .predict(Xte),
    )
    return out


def run(rundir: Path, seed: int = 0) -> dict:
    meta = json.loads((rundir / "meta.json").read_text())
    df = pd.read_parquet(rundir / "steps.parquet")
    df = add_tier_b(df, meta["config"]["model_id"])

    # Drop step 0: it has no tier-A history by construction, and imputing it
    # would let the imputation value itself become a predictive feature.
    df = df[df.step > 0].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    pids = df.prompt_id.unique()
    rng.shuffle(pids)
    cut = int(0.7 * len(pids))
    tr_p, te_p = set(pids[:cut]), set(pids[cut:])
    tr = df[df.prompt_id.isin(tr_p)]
    te = df[df.prompt_id.isin(te_p)]

    hid, lns, n_pca = tier_c_features(rundir, df)
    pca = PCA(n_components=n_pca, random_state=seed).fit(hid[df.index.isin(tr.index)])
    hid_p = pca.transform(hid)

    tr_m = df.index.isin(tr.index)
    te_m = df.index.isin(te.index)

    banks = {
        "A": df[TIER_A].to_numpy(dtype=np.float64),
        "B": df[TIER_B].to_numpy(dtype=np.float64),
        "C": np.hstack([hid_p, lns]),
    }
    combos = {
        "A": ["A"],
        "B": ["B"],
        "A+B (deployable)": ["A", "B"],
        "A+B+C (ceiling)": ["A", "B", "C"],
    }

    results: dict = {
        "n_train_rows": int(tr_m.sum()),
        "n_test_rows": int(te_m.sum()),
        "n_train_prompts": len(tr_p),
        "n_test_prompts": len(te_p),
        "tier_c_pca_explained": float(pca.explained_variance_ratio_.sum()),
        "targets": {},
    }

    for tgt in TARGETS:
        y = df[tgt].to_numpy(dtype=np.float64)
        if tgt == "nucleus_90":
            y = np.log(y)  # heavy tailed; R^2 on the raw scale is dominated by outliers
        per_combo = {}
        for name, keys in combos.items():
            X = np.hstack([banks[k] for k in keys])
            names = sum([TIER_A if k == "A" else TIER_B if k == "B" else
                         [f"C{i}" for i in range(banks["C"].shape[1])] for k in keys], [])
            ok = np.isfinite(X).all(1)
            r = evaluate(X[tr_m & ok], y[tr_m & ok], X[te_m & ok], y[te_m & ok], seed)
            r["single_best_feature_name"] = names[r.pop("single_best_feature_idx")]
            r["n_features"] = int(X.shape[1])
            per_combo[name] = r
            print(f"  {tgt:16s} {name:18s} "
                  f"1feat={r['single_best_feature']:+.3f} "
                  f"ridge={r['ridge']:+.3f} gbm={r['gbm']:+.3f} "
                  f"({r['single_best_feature_name']})", flush=True)
        results["targets"][tgt] = per_combo
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    a = ap.parse_args()
    rundir = Path(a.run)
    res = run(rundir)
    (rundir / "exp1b_predictive.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
