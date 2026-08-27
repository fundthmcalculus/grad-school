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
        print(f"  [FAILED] {name}: loader returned None")
    elif name == "BETH":
        # BETH returns a dict with train/val/test splits
        if isinstance(result, dict) and all(
            k in result for k in ["train", "val", "test"]
        ):
            total_rows = sum(len(result[s][0]) for s in ["train", "val", "test"])
            features = result["train"][0].shape[1]
            results[name] = f"OK ({total_rows} total rows, {features} features)"
            print(
                f"  [OK] {name}: train={len(result['train'][0])}, "
                f"val={len(result['val'][0])}, test={len(result['test'][0])} "
                f"({total_rows} total, {features} features)"
            )
        else:
            results[name] = "FAILED (invalid format)"
            print(f"  [FAILED] {name}: invalid return format")
    else:
        X, y = result
        results[name] = f"OK ({len(X)} rows, {X.shape[1]} features)"
        print(f"  [OK] {name}: {len(X)} rows, {X.shape[1]} features")
    print()

# --------------------------------------------------------------------------- #
# Leakage / spec guard
# --------------------------------------------------------------------------- #
# `load_rt_iot2022` shipped an unnamed index column as a feature until
# 2026-08-27, and that column was not a harmless row number: RT_IOT2022.csv
# concatenates twelve per-class captures and the counter restarts at zero for
# each, so it encoded the label. `load_bikeshare` had already learned this
# lesson and drops `instant`. This guard is here so the next loader does not
# have to learn it a third time, and so a feature count can never drift from
# `dataset_specs.yaml` unnoticed.
SPEC_KEYS = {
    "Concrete": "concrete",
    "RT-IOT2022": "rt_iot2022",
    "Shuttle": "shuttle",
    "BETH": "beth",
}
# PhiUSIIL is deliberately absent: from a clean checkout its loader falls
# through to a ucimlrepo fetch that returns a DIFFERENT feature set, which the
# loader's own comment flags. Pinning its width here would fail for the wrong
# reason. Bike Sharing is absent because it has no spec row yet.

guard_failures = []
try:
    sys.path.insert(0, REPO_ROOT)
    import dataset_specs

    specs = dataset_specs.load_specs()
except Exception as exc:  # noqa: BLE001
    print(f"[warn] spec cross-check unavailable ({exc.__class__.__name__}: {exc})")
    specs = None

for name, loader, kwargs in LOADERS:
    result = loader(**kwargs)
    if result is None:
        continue
    frames = (
        [result[s][0] for s in ("train", "val", "test")]
        if name == "BETH"
        else [result[0]]
    )
    for X in frames:
        bad = [
            c
            for c in X.columns
            if str(c).startswith("Unnamed") or str(c).lower() in ("instant", "index", "id")
        ]
        if bad:
            guard_failures.append(f"{name}: index-like column(s) kept as features: {bad}")

    if specs and name in SPEC_KEYS:
        spec = specs[SPEC_KEYS[name]]
        expected = spec.get("features")
        # The spec counts features that REACH A MODEL. A loader may legitimately
        # return more than that and leave the last drop to the generator: BETH's
        # loader hands back `sus` and `timestamp`, which table_4_11's
        # `load_splits` drops as leaky (`sus` is BETH's second label). Applying
        # the spec's own drop list here compares like with like.
        drop = set((spec.get("verify") or {}).get("drop_columns") or [])
        actual = len([c for c in frames[0].columns if c not in drop])
        if expected is not None and actual != expected:
            guard_failures.append(
                f"{name}: {actual} modelled features, but dataset_specs.yaml "
                f"says {expected}"
            )

print("=" * 70)
print("Leakage / spec guard")
print("=" * 70)
if guard_failures:
    for f in guard_failures:
        print(f"  [FAIL] {f}")
else:
    print("  [PASS] no index-like features; every checked width matches the spec")
print()

print("=" * 70)
print("Summary")
print("=" * 70)
for name, status in results.items():
    status_icon = "PASS" if status.startswith("OK") else "FAIL"
    print(f"[{status_icon}] {name:15} {status}")
print()

failed = [name for name, status in results.items() if not status.startswith("OK")]
if failed:
    print(
        f"ERROR: {len(failed)} loader(s) failed. Check file paths and data integrity."
    )
    sys.exit(1)
elif guard_failures:
    print(f"ERROR: {len(guard_failures)} leakage/spec guard failure(s).")
    sys.exit(1)
else:
    print("SUCCESS: All loaders passed!")
    sys.exit(0)
