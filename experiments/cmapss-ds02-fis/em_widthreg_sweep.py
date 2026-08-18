"""Does a support-width regularizer fix EM-trapezoid antecedents on DS02?
Sweep trapz_math.WIDTH_REG for member_function='trap', trapz_method='em',
full-2nd. Reference: gaussian 6.48, fast-trap 6.42."""
import sys, contextlib, io, time
import numpy as np, pandas as pd
sys.path.insert(0, "FuzzySystemsExperiments")
import tribblefis.trapz_math as tm
from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss
from tribble_predictive_health.preprocessing import (
    apply_condition_correction, build_memory_features, fit_condition_correction)
from tribble_predictive_health.metrics import rmse
H5 = "NASA-CMAPSS/N-CMAPSS_DS02-006.h5"; OUT = "outputs/hdbscan-ds02"
CSV = f"{OUT}/em_widthreg.csv"
print("featurising ...", flush=True)
dev, cond, sensors = load_ncmapss(H5, "dev"); test, _, _ = load_ncmapss(H5, "test")
mdl = fit_condition_correction(dev, sensors, cond)
train_tab, feat_cols = build_memory_features(apply_condition_correction(dev, sensors, cond, mdl), sensors)
test_tab, _ = build_memory_features(apply_condition_correction(test, sensors, cond, mdl), sensors)

def em_fit(wr):
    tm.WIDTH_REG = wr
    e = TribblePredictiveHealth(condition_correction=False, aggregation="raw_memory",
                                tsk_order="full-2nd", top_p=0.95, l2_reg=0.01,
                                member_function="trap", trapz_method="em")
    with contextlib.redirect_stdout(io.StringIO()):
        e.fit_featurized(train_tab, feat_cols)
    return e

def widths(e):
    ws, ps = [], []
    for f, fm in e.regressor_.model_.feature_models.items():
        for lab, lm in fm.label_models.items():
            for mf in lm.memberships:
                if hasattr(mf, "a"):
                    ws.append(mf.d - mf.a); ps.append(mf.c - mf.b)
    return float(np.mean(ws)), float(np.mean(ps))

open(CSV, "w").write("width_reg,per_sample,train_rmse,mean_support,mean_plateau,fit_s\n")
for wr in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
    t = time.perf_counter(); e = em_fit(wr); dt = time.perf_counter() - t
    Xtr = e.scaler_.transform(train_tab[e.feature_cols_].to_numpy(float))
    tr = rmse(train_tab["rul"], e.regressor_.predict(Xtr))
    ps = e.score_featurized(test_tab)["per_sample_rmse"]
    ms, mp = widths(e)
    open(CSV, "a").write(f"{wr},{ps:.4f},{tr:.4f},{ms:.3f},{mp:.3f},{dt:.1f}\n")
    print(f"  WIDTH_REG={wr:<5} per-sample={ps:6.3f} train={tr:6.3f} "
          f"support={ms:5.2f} plateau={mp:5.2f} ({dt:.0f}s)", flush=True)
tm.WIDTH_REG = 0.0
print("done", flush=True)
