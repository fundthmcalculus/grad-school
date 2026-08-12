#!/usr/bin/env python
"""Quick baseline: RT-IOT2022 open-set intrusion detection (classification)."""

import os
import sys
import os

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables"))
import _fuzzy_models as F  # noqa: E402

print("=" * 70)
print("RT-IOT2022 Open-Set Detection — Single Example Baseline")
print("=" * 70)
print()

# Load data
X, y = F.load_rt_iot2022()
if X is None:
    print("ERROR: Could not load RT-IOT2022")
    sys.exit(1)

print(f"Dataset: {len(X)} rows × {X.shape[1]} features, {len(np.unique(y))} classes")
classes = np.unique(y)
print(f"Classes: {classes}")
print()

# Simple train/test split (80/20, one seed)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
# Easy unit scalar
ss_x = StandardScaler()
ss_x.set_output(transform="pandas")
Xtr = ss_x.fit_transform(Xtr)
Xte = ss_x.transform(Xte)
# ss_y = StandardScaler()
# ytr = ss_y.fit_transform(ytr)
# yte = ss_y.transform(yte)

mog_model = F.mog_classifier(seed=42)
for name, model in [
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    ),
    ("Tribble (MoG)", mog_model),
]:
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0
    acc = accuracy_score(yte, model.predict(Xte))
    print(f"{name}: acc={acc:.4f}  train={train_time:.2f}s")

t0 = time.time()
rm = F.ruspinize_classifier(mog_model, Xtr, ytr)
ruspini_time = time.time() - t0
acc = accuracy_score(yte, rm.predict(Xte))
print(
    f"Tribble (Ruspini): acc={acc:.4f}  train={ruspini_time:.2f}s  rules={len(rm.rules)}"
)
F.plot_membership_functions(rm, Xtr, "quick_iot_ruspini_mfs")
