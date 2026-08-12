#!/usr/bin/env python
"""Quick smoke test: verify all dataset loaders work and find their files."""

import sys
import os

# Add the repo to the path
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = _HERE
sys.path.insert(0, os.path.join(REPO_ROOT, "tables"))

from _fuzzy_models import (
    load_concrete,
    load_phiusiil,
    load_rt_iot2022,
    load_beth,
    load_shuttle,
    load_bikeshare,
)

LOADERS = [
    ("Concrete", load_concrete, {}),
    ("PhiUSIIL", load_phiusiil, {"sample_size": None}),
    ("RT-IOT2022", load_rt_iot2022, {}),
    ("BETH", load_beth, {}),
    ("Shuttle", load_shuttle, {}),
    ("Bike Sharing", load_bikeshare, {}),
]

print("=" * 70)
print("Dataset Loader Smoke Test")
print("=" * 70)
print()

results = {}
for name, loader, kwargs in LOADERS:
    print(f"Testing {name}...")
    result = loader(**kwargs)
    if result is None:
        results[name] = "FAILED (returned None)"
        print(f"  ✗ {name}: loader returned None")
    else:
        X, y = result
        results[name] = f"OK ({len(X)} rows × {X.shape[1]} features)"
        print(f"  ✓ {name}: {len(X)} rows × {X.shape[1]} features, target shape {y.shape}")
    print()

print("=" * 70)
print("Summary")
print("=" * 70)
for name, status in results.items():
    status_icon = "✓" if status.startswith("OK") else "✗"
    print(f"{status_icon} {name:15} {status}")
print()

failed = [name for name, status in results.items() if not status.startswith("OK")]
if failed:
    print(f"⚠ {len(failed)} loader(s) failed. Check file paths and data integrity.")
    sys.exit(1)
else:
    print("✓ All loaders passed!")
    sys.exit(0)
