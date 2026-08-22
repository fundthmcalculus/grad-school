#!/usr/bin/env python3
"""Diagnose the classification-accuracy collapse in Table 4.1.

The archived run of record (`reproduce/outputs/goal-8h-2026-08-11-fullsuite/`,
tribble-fis 80e98d7) and a re-run at the current pin (141596e) disagree on the
two CLASSIFICATION rows by margins no seed spread covers:

    row                    archive            current pin
    PhiUSIIL               0.997 +/- 0.001    0.729 +/- 0.023
    RT-IOT2022 (12-class)  0.927 +/- 0.002    0.500 +/- 0.244

while the three REGRESSION rows moved the other way, slightly up
(0.795 -> 0.808, 0.852 -> 0.867, 0.939 -> 0.965), and every training time fell
by 5-7x. A model that fits six times faster and scores twenty-seven points worse
is doing less work, so this prints the size of what got fitted alongside the
score rather than the score alone.

Checklist B13 recorded this bump as "byte-identical across the bump" on the
strength of the three R2 values -- which do match. The accuracy columns were not
part of that check.

    uv run --project tribble-fis python \
        reproduce/experiments/diagnose_table_4_1_accuracy.py [--dataset phiusiil]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "reproduce"))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))


def describe_model(model) -> str:
    """Count what actually got fitted, without assuming the internal layout."""
    inner = getattr(model, "model_", model)
    bits = []
    for attr in ("features_", "selected_features_", "feature_names_", "classes_"):
        val = getattr(inner, attr, None)
        if val is not None:
            try:
                bits.append(f"{attr}={len(val)}")
            except TypeError:
                bits.append(f"{attr}={val!r}")

    # Walk whatever rule/mixture container exists and count Gaussian terms.
    n_terms = 0
    n_rules = 0
    for attr in ("rules_", "rules", "class_models_", "mixtures_", "gaussians_"):
        container = getattr(inner, attr, None)
        if container is None:
            continue
        try:
            n_rules = len(container)
            items = container.values() if isinstance(container, dict) else container
            for rule in items:
                if isinstance(rule, dict):
                    for v in rule.values():
                        n_terms += len(v) if hasattr(v, "__len__") else 1
                elif hasattr(rule, "__len__"):
                    n_terms += len(rule)
        except Exception:  # pragma: no cover - purely diagnostic
            pass
        bits.append(f"{attr}: rules={n_rules} terms={n_terms}")
        break
    return "  ".join(bits) if bits else "(model internals not introspectable)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="phiusiil")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    import _fuzzy_models as fm
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    import tribblefis

    print(f"tribblefis from : {tribblefis.__file__}")
    print(f"dataset         : {args.dataset}")

    loader = getattr(fm, f"load_{args.dataset}", None)
    if loader is None:
        for name in dir(fm):
            if args.dataset in name.lower() and name.startswith("load"):
                loader = getattr(fm, name)
                break
    if loader is None:
        print(f"no loader for {args.dataset}; available: "
              f"{[n for n in dir(fm) if n.startswith('load')]}")
        return 2

    X, y = loader()
    X = np.asarray(X)
    y = np.asarray(y)
    print(f"shape           : {X.shape}, classes={len(np.unique(y))}")
    print()

    accs = []
    for seed in range(args.seeds):
        xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        model = fm.mog_classifier(seed)
        t0 = time.perf_counter()
        model.fit(xtr, ytr)
        fit_s = time.perf_counter() - t0
        pred = model.predict(xte)
        acc = accuracy_score(yte, pred)
        accs.append(acc)
        # A collapsed classifier usually predicts one label everywhere; say so
        # explicitly rather than leaving it to be inferred from the accuracy.
        vals, counts = np.unique(pred, return_counts=True)
        top = counts.max() / counts.sum()
        print(f"  seed {seed}: acc={acc:.4f}  fit={fit_s:6.2f}s  "
              f"distinct_preds={len(vals)}  most_common_frac={top:.3f}")
        print(f"           {describe_model(model)}")

    a = np.asarray(accs)
    print(f"\n  acc = {a.mean():.4f} +/- {a.std():.4f}  over {args.seeds} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
