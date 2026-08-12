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

for name, model in [
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ("Tribble (MoG)", F.mog_classifier(seed=42)),
]:
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0
    acc = accuracy_score(yte, model.predict(Xte))
    print(f"{name}: acc={acc:.4f}  train={train_time:.2f}s")
