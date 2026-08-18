import sys, contextlib, io, time
import numpy as np
sys.path.insert(0, "FuzzySystemsExperiments")
import tribblefis.trapz_math as tm
from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss
from tribble_predictive_health.preprocessing import (
    apply_condition_correction, build_memory_features, fit_condition_correction)
H5="NASA-CMAPSS/N-CMAPSS_DS02-006.h5"; OUT="outputs/hdbscan-ds02"
dev,cond,sensors=load_ncmapss(H5,"dev"); test,_,_=load_ncmapss(H5,"test")
mdl=fit_condition_correction(dev,sensors,cond)
train_tab,fc=build_memory_features(apply_condition_correction(dev,sensors,cond,mdl),sensors)
test_tab,_=build_memory_features(apply_condition_correction(test,sensors,cond,mdl),sensors)
open(f"{OUT}/em_widthreg.csv","a")  # append to existing
for wr in [1.5, 2.0, 3.0, 5.0]:
    tm.WIDTH_REG=wr
    e=TribblePredictiveHealth(condition_correction=False,aggregation="raw_memory",
        tsk_order="full-2nd",top_p=0.95,l2_reg=0.01,member_function="trap",trapz_method="em")
    t=time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()): e.fit_featurized(train_tab,fc)
    dt=time.perf_counter()-t
    ps=e.score_featurized(test_tab)["per_sample_rmse"]
    ws=np.mean([mf.d-mf.a for fm in e.regressor_.model_.feature_models.values()
                for lm in fm.label_models.values() for mf in lm.memberships if hasattr(mf,"a")])
    open(f"{OUT}/em_widthreg.csv","a").write(f"{wr},{ps:.4f},,{ws:.3f},,{dt:.1f}\n")
    print(f"  WIDTH_REG={wr:<5} per-sample={ps:6.3f} support={ws:5.2f} ({dt:.0f}s)",flush=True)
tm.WIDTH_REG=0.0
