#!/usr/bin/env python
"""Quick baseline: BETH host telemetry anomaly detection (classification)."""

import os
import sys
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables"))
import _fuzzy_models as F  # noqa: E402

splits = F.load_beth()
Xtr, ytr = splits["train"]
Xte, yte = splits["test"]

mog_model = F.mog_classifier(seed=42)
for name, model in [
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
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
print(f"Tribble (Ruspini): acc={acc:.4f}  train={ruspini_time:.2f}s  rules={len(rm.rules)}")
F.plot_membership_functions(rm, Xtr, "quick_beth_ruspini_mfs")
