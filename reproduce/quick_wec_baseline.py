#!/usr/bin/env python
"""Quick single-example baseline on Wave Energy Farm (WEC_Perth_49) dataset."""

import time
import sys
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "tables"))
import _fuzzy_models as F  # noqa: E402

print("=" * 70)
print("Wave Energy Farm (WEC_Perth_49) — Single Example Baseline")
print("=" * 70)
print()

# Load data
data_path = os.path.join(F.DATA_DIR, "WEC_Perth_49.csv")
if not os.path.exists(data_path):
    print(f"ERROR: Dataset not found at {data_path}")
    sys.exit(1)

try:
    df = pd.read_csv(data_path)
    print(f"Loaded: {data_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print()

    # Extract features and target
    if "Total_Power" not in df.columns:
        print("ERROR: 'Total_Power' column not found")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    y = df["Total_Power"].astype(float)
    # Keep only spatial coordinates (X/Y columns), exclude Power columns and qW
    exclude_cols = ["Total_Power", "qW"] + [c for c in df.columns if c.startswith("Power")]
    X = df.drop(columns=exclude_cols).select_dtypes(include=[np.number]).astype(float)

    print(f"Features: {X.shape[1]} numeric columns")
    print(f"Target: 'Total_Power' ({len(y)} samples)")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target mean ± std: {y.mean():.2f} ± {y.std():.2f}")
    print()

except Exception as exc:
    print(f"ERROR: Failed to load/parse dataset ({exc.__class__.__name__}: {exc})")
    sys.exit(1)

# Train/test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train/test split: {len(Xtr)} / {len(Xte)} samples")
print()

# Baseline 1: Random Forest regressor
print("Baseline 1: Random Forest Regressor (200 trees)")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
t0 = time.time()
rf.fit(Xtr, ytr)
rf_train_time = time.time() - t0
t0 = time.time()
rf_pred = rf.predict(Xte)
rf_pred_time = time.time() - t0
rf_r2 = r2_score(yte, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(yte, rf_pred))
print(f"  Training: {rf_train_time:.3f}s")
print(f"  Prediction: {rf_pred_time:.4f}s")
print(f"  R²: {rf_r2:.4f}")
print(f"  RMSE: {rf_rmse:.4f}")
print()

# Baseline 2: MoG regressor
print("Baseline 2: MoG Regressor (Tribble)")
try:
    mog = F.mog_regressor(seed=42)
    if mog is None:
        print("  [unavailable]")
    else:
        t0 = time.time()
        mog.fit(Xtr, ytr)
        mog_train_time = time.time() - t0
        t0 = time.time()
        mog_pred = mog.predict(Xte)
        mog_pred_time = time.time() - t0
        mog_r2 = r2_score(yte, mog_pred)
        mog_rmse = np.sqrt(mean_squared_error(yte, mog_pred))
        print(f"  Training: {mog_train_time:.3f}s")
        print(f"  Prediction: {mog_pred_time:.4f}s")
        print(f"  R²: {mog_r2:.4f}")
        print(f"  RMSE: {mog_rmse:.4f}")
except Exception as exc:
    print(f"  [failed: {exc.__class__.__name__}]")
print()

print("=" * 70)
print("Summary")
print("=" * 70)
print(f"Dataset: WEC_Perth_49, {len(X):,} samples, {X.shape[1]} features")
print(f"Target: Total Power (regression)")
print(f"Split: 80/20 (train={len(Xtr):,}, test={len(Xte):,})")
print()
print(f"Random Forest (200 trees):")
print(f"  Training: {rf_train_time:.3f}s")
print(f"  Prediction: {rf_pred_time:.4f}s")
print(f"  R²: {rf_r2:.4f}")
print(f"  RMSE: {rf_rmse:.4f}")
print()
try:
    print(f"MoG Regressor:")
    print(f"  Training: {mog_train_time:.3f}s")
    print(f"  Prediction: {mog_pred_time:.4f}s")
    print(f"  R²: {mog_r2:.4f}")
    print(f"  RMSE: {mog_rmse:.4f}")
    print()
    speedup = mog_train_time / rf_train_time
    print(f"MoG/RF training ratio: {speedup:.2f}x")
    print(f"R² gap: {abs(rf_r2 - mog_r2):.4f} ({rf_r2 - mog_r2:+.4f})")
except NameError:
    pass
print()
