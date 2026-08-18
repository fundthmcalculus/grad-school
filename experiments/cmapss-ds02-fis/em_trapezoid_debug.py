import sys, contextlib, io
import numpy as np
sys.path.insert(0, "FuzzySystemsExperiments")
from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss
from tribble_predictive_health.preprocessing import (
    apply_condition_correction, build_memory_features, fit_condition_correction)
H5 = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"
print("featurising ...", flush=True)
dev, cond, sensors = load_ncmapss(H5, "dev"); test, _, _ = load_ncmapss(H5, "test")
mdl = fit_condition_correction(dev, sensors, cond)
dev_c = apply_condition_correction(dev, sensors, cond, mdl)
test_c = apply_condition_correction(test, sensors, cond, mdl)
train_tab, feat_cols = build_memory_features(dev_c, sensors)
test_tab, _ = build_memory_features(test_c, sensors)

def fit(mf, tm=None):
    kw = dict(member_function=mf)
    if tm: kw["trapz_method"] = tm
    e = TribblePredictiveHealth(condition_correction=False, aggregation="raw_memory",
                                tsk_order="full-2nd", top_p=0.95, l2_reg=0.01, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        e.fit_featurized(train_tab, feat_cols)
    return e

engs = {"gaussian": fit("gaussian"), "fast": fit("trap","fast"), "em": fit("trap","em")}
e0 = engs["em"]
Xtr = e0.scaler_.transform(train_tab[e0.feature_cols_].to_numpy(float))  # scaled train (MFs fit on this)
top = list(e0.regressor_.top_features_)
name2col = {n: int(n.split("_")[1]) for n in top}

print(f"\n{len(top)} top features; inspecting first 5\n")
for f in top[:5]:
    col = Xtr[:, name2col[f]]
    q = np.percentile(col, [0, 5, 50, 95, 100])
    print(f"--- {f}  data[min,5,50,95,max]={np.round(q,2)} ---")
    for tag in ("gaussian", "fast", "em"):
        model = engs[tag].regressor_.model_
        lm = model.feature_models[f].label_models
        parts = []
        for label, lmod in lm.items():
            for mf in lmod.memberships:
                mu = mf.evaluate(col)
                cov = float((mu > 1e-6).mean())
                if hasattr(mf, "a"):  # trapezoid
                    parts.append(f"L{label}:[a={mf.a:.2f},b={mf.b:.2f},c={mf.c:.2f},"
                                 f"d={mf.d:.2f}] cov={cov:.0%}")
                else:  # gaussian
                    parts.append(f"L{label}:[mu={mf.mu:.2f},sig={mf.sigma:.2f}] cov={cov:.0%}")
        print(f"  {tag:9s} " + "  ".join(parts))
    print()

print("\n=== product-firing coverage (fraction of rows where a rule can fire) ===")
print(f"    p = {len(top)} antecedent features (full-2nd top_p=0.95)")
for tag, e in engs.items():
    model = e.regressor_.model_
    tf = list(e.regressor_.top_features_)
    # per (rule=label) firing support: a rule fires at row x iff EVERY feature has
    # >=1 component with membership>0 at x (product t-norm -> any zero kills it).
    labels = list(model.feature_models[tf[0]].label_models.keys())
    row_fires_any = np.zeros(Xtr.shape[0], dtype=bool)
    percov = {}
    for label in labels:
        rule_fires = np.ones(Xtr.shape[0], dtype=bool)
        for f in tf:
            col = Xtr[:, int(f.split("_")[1])]
            feat_fire = np.zeros(Xtr.shape[0], dtype=bool)
            for mf in model.feature_models[f].label_models[label].memberships:
                feat_fire |= mf.evaluate(col) > 1e-9
            rule_fires &= feat_fire
        row_fires_any |= rule_fires
    print(f"  {tag:9s} rows with >=1 rule firing: {row_fires_any.mean():6.1%}")

print("\n=== coverage vs conditioning (full-2nd, p=18) ===")
from tribble_predictive_health.metrics import rmse
Xte = e0.scaler_.transform(test_tab[e0.feature_cols_].to_numpy(float))
def firing_cov(model, tf, Xm):
    labels = list(model.feature_models[tf[0]].label_models.keys())
    any_fire = np.zeros(Xm.shape[0], dtype=bool)
    for label in labels:
        rf = np.ones(Xm.shape[0], dtype=bool)
        for f in tf:
            col = Xm[:, int(f.split("_")[1])]
            ff = np.zeros(Xm.shape[0], dtype=bool)
            for mf in model.feature_models[f].label_models[label].memberships:
                ff |= mf.evaluate(col) > 1e-9
            rf &= ff
        any_fire |= rf
    return any_fire.mean()
for tag, e in engs.items():
    tf = list(e.regressor_.top_features_)
    tr = rmse(train_tab["rul"], e.regressor_.predict(Xtr))  # note: uncapped rul as ref
    te = e.score_featurized(test_tab)["per_sample_rmse"]
    cov_te = firing_cov(e.regressor_.model_, tf, Xte)
    ncomp = sum(len(e.regressor_.model_.feature_models[f].label_models[lab].memberships)
                for f in tf for lab in e.regressor_.model_.feature_models[f].label_models)
    print(f"  {tag:9s} train-RMSE={tr:6.2f}  test-RMSE={te:6.2f}  "
          f"test-firing-cov={cov_te:5.1%}  total-MF-components={ncomp}")
