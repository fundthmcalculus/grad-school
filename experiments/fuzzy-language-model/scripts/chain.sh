#!/bin/bash
# Unattended chain: tune (incl. lr=3e-2 extension) -> edge check -> scaling -> ablate.
cd "$(dirname "$0")/.."
# (previous tune process already gone; tune below resumes)
.venv/bin/python -m flm.sweep tune --workers 4 >> outputs/tune.log 2>&1
.venv/bin/python - <<'PY' || { echo "EDGE CHECK FAILED - not launching scaling"; exit 1; }
import json, sys
from flm.grids import TUNE_LRS, TUNE_CONV
from flm.sweep import ARMS, OUT
bad = []
for arm in ARMS:
    cells = []
    for lr in TUNE_LRS:
        for sc in TUNE_CONV:
            r = json.loads((OUT / "tune" / f"{arm}_lr{lr}_sc{sc}.json").read_text())
            cells.append((r["val_bpc"] if r["val_bpc"] == r["val_bpc"] else 9e9, lr, sc))
    b = min(cells)
    print(f"{arm:12s} best lr={b[1]} sc={b[2]} bpc={b[0]:.4f}")
    if b[1] == TUNE_LRS[-1]:
        bad.append(arm)
if bad:
    print("best lr at grid edge for:", bad); sys.exit(1)
PY
echo "=== scaling start $(date)"
.venv/bin/python -m flm.sweep scaling --workers 4 > outputs/scaling.log 2>&1
echo "=== ablate start $(date)"
.venv/bin/python -m flm.sweep ablate --workers 4 > outputs/ablate.log 2>&1
echo "=== chain done $(date)"
echo "=== headline start $(date)"
.venv/bin/python -m flm.sweep headline --workers 4 > outputs/headline.log 2>&1
echo "=== headline done $(date)"
