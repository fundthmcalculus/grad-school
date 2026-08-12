#!/usr/bin/env python
"""Quick baseline: RT-IOT2022 open-set intrusion detection (classification)."""

import os
import sys
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables"))
import _fuzzy_models as F  # noqa: E402

X, y = F.load_rt_iot2022()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in [
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("Tribble (MoG)", F.mog_classifier(seed=42)),
]:
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0
    acc = accuracy_score(yte, model.predict(Xte))
    print(f"{name}: acc={acc:.4f}  train={train_time:.2f}s")
