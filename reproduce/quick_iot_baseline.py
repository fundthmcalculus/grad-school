#!/usr/bin/env python
"""Quick single-example baseline on RT-IOT2022 for open-set detection."""

import sys
import os

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "tables"))
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
print(f"Train: {len(Xtr)} samples, Test: {len(Xte)} samples")
print()

# Baseline 1: Random Forest classifier
print("Baseline 1: Random Forest (200 trees)")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
t0 = time.time()
rf.fit(Xtr, ytr)
rf_train_time = time.time() - t0
t0 = time.time()
rf_pred = rf.predict(Xte)
rf_pred_time = time.time() - t0
rf_acc = accuracy_score(yte, rf_pred)
print(f"  Training: {rf_train_time:.2f}s")
print(f"  Prediction: {rf_pred_time:.3f}s")
print(f"  Accuracy: {rf_acc:.4f}")
print()

# Baseline 2: MoG classifier (if available)
print("Baseline 2: MoG Classifier (Tribble)")
try:
    mog = F.mog_classifier(seed=42)
    if mog is None:
        print("  [unavailable]")
    else:
        t0 = time.time()
        mog.fit(Xtr, ytr)
        mog_train_time = time.time() - t0
        t0 = time.time()
        mog_pred = mog.predict(Xte)
        mog_pred_time = time.time() - t0
        mog_acc = accuracy_score(yte, mog_pred)
        print(f"  Training: {mog_train_time:.2f}s")
        print(f"  Prediction: {mog_pred_time:.3f}s")
        print(f"  Accuracy: {mog_acc:.4f}")
except Exception as exc:
    print(f"  [failed: {exc.__class__.__name__}]")
print()

print("=" * 70)
print("Summary")
print("=" * 70)
print(f"Dataset: RT-IOT2022, 123k samples, 12 classes")
print(f"Split: 80/20 (train={len(Xtr)}, test={len(Xte)})")
print()
print(f"Random Forest (200 trees):")
print(f"  Training: {rf_train_time:.2f}s")
print(f"  Prediction: {rf_pred_time:.3f}s (on {len(Xte):,} samples)")
print(f"  Accuracy: {rf_acc:.4f}")
print()
try:
    print(f"MoG Classifier:")
    print(f"  Training: {mog_train_time:.2f}s")
    print(f"  Prediction: {mog_pred_time:.3f}s (on {len(Xte):,} samples)")
    print(f"  Accuracy: {mog_acc:.4f}")
    print()
    speedup = rf_train_time / mog_train_time
    print(f"RF/MoG training speedup: {speedup:.2f}x (RF is {speedup:.2f}x faster)")
except NameError:
    pass
print()
