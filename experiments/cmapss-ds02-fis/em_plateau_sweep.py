"""With support width fixed (WIDTH_REG=1.0), sweep PLATEAU_REG to see if keeping a
real trapezoid plateau closes the last gap to fast-trapezoid (6.42) on DS02."""
import sys, contextlib, io, time
import numpy as np
sys.path.insert(0, "FuzzySystemsExperiments")
import tribblefis.trapz_math as tm
from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss
from tribble_predictive_health.preprocessing import (
    apply_condition_correction, build_memory_features, fit_condition_correction)
H5="NASA-CMAPSS/N-CMAPSS_DS02-006.h5"; OUT="outputs/hdbscan-ds02"
CSV=f"{OUT}/em_plateau.csv"
print("featurising ...", flush=True)
dev,cond,sensors=load_ncmapss(H5,"dev"); test,_,_=load_ncmapss(H5,"test")
mdl=fit_condition_correction(dev,sensors,cond)
train_tab,fc=build_memory_features(apply_condition_correction(dev,sensors,cond,mdl),sensors)
test_tab,_=build_memory_features(apply_condition_correction(test,sensors,cond,mdl),sensors)
tm.WIDTH_REG=1.0
open(CSV,"w").write("plateau_reg,per_sample,mean_support,mean_plateau,fit_s\n")
for pr in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0]:
    tm.PLATEAU_REG=pr
    e=TribblePredictiveHealth(condition_correction=False,aggregation="raw_memory",
        tsk_order="full-2nd",top_p=0.95,l2_reg=0.01,member_function="trap",trapz_method="em")
    t=time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()): e.fit_featurized(train_tab,fc)
    dt=time.perf_counter()-t
    ps=e.score_featurized(test_tab)["per_sample_rmse"]
    mfs=[mf for fm in e.regressor_.model_.feature_models.values()
         for lm in fm.label_models.values() for mf in lm.memberships if hasattr(mf,"a")]
    ms=np.mean([mf.d-mf.a for mf in mfs]); mp=np.mean([mf.c-mf.b for mf in mfs])
    open(CSV,"a").write(f"{pr},{ps:.4f},{ms:.3f},{mp:.3f},{dt:.1f}\n")
    print(f"  PLATEAU_REG={pr:<4} per-sample={ps:6.3f} support={ms:5.2f} plateau={mp:5.2f} ({dt:.0f}s)",flush=True)
tm.WIDTH_REG=0.0; tm.PLATEAU_REG=0.0
print("done",flush=True)
