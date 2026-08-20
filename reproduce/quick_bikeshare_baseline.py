#!/usr/bin/env python
"""Quick baseline: Bike Sharing Demand (regression)."""

import os
import sys
import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables"))
import _fuzzy_models as F  # noqa: E402

X, y = F.load_bikeshare()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
# Easy unit scalar
ss_x = StandardScaler()
ss_x.set_output(transform="pandas")
Xtr = ss_x.fit_transform(Xtr)
Xte = ss_x.transform(Xte)
ss_y = StandardScaler()
ytr = pd.Series(
    ss_y.fit_transform(ytr.values.reshape(-1, 1)).ravel(),
    index=ytr.index,
    name=ytr.name,
)
yte = pd.Series(
    ss_y.transform(yte.values.reshape(-1, 1)).ravel(), index=yte.index, name=yte.name
)

mog_model = F.mog_regressor(seed=42)
for name, model in [
    (
        "Random Forest",
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    ),
    ("Tribble (MoG)", mog_model),
]:
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0
    r2 = r2_score(yte, model.predict(Xte))
    print(f"{name}: R2={r2:.4f}  train={train_time:.2f}s")

t0 = time.time()
rm, bucket_mean = F.ruspinize_regressor(mog_model, Xtr, ytr)
ruspini_time = time.time() - t0
r2 = r2_score(yte, F.ruspini_predict_regression(rm, bucket_mean, Xte))
print(
    f"Tribble (Ruspini): R2={r2:.4f}  train={ruspini_time:.2f}s  rules={len(rm.rules)}"
)
F.plot_membership_functions(rm, Xtr, "quick_bikeshare_ruspini_mfs")
