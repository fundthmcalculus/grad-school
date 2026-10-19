#!/bin/bash
# FRLM-v2 chain: tune the HTSK arms on the shared grid -> edge check -> scaling2 -> headline2.
cd "$(dirname "$0")/.."
.venv/bin/python -m flm.sweep tune --workers 2 >> outputs/tune.log 2>&1
.venv/bin/python - <<'PY' || { echo "V2 EDGE CHECK FAILED"; exit 1; }
import json, sys
from flm.grids import TUNE_LRS, TUNE_CONV, ARMS_V2
from flm.sweep import OUT
bad = []
for arm in ARMS_V2:
    cells = []
    for lr in TUNE_LRS:
        for sc in TUNE_CONV:
            r = json.loads((OUT / "tune" / f"{arm}_lr{lr}_sc{sc}.json").read_text())
            cells.append((r["val_bpc"] if r["val_bpc"] == r["val_bpc"] else 9e9, lr, sc))
    b = min(cells); print(f"{arm} best lr={b[1]} sc={b[2]} bpc={b[0]:.4f}")
    if b[1] in (TUNE_LRS[0], TUNE_LRS[-1]): bad.append(arm)
if bad: print("edge:", bad); sys.exit(1)
PY
for g in scaling2 headline2; do
  n=$(.venv/bin/python -m flm.sweep $g --list | tail -1 | cut -d' ' -f1)
  [ "$n" -gt 0 ] || { echo "GRID $g EMPTY - stopping"; exit 1; }
  echo "=== $g start ($n jobs) $(date)"
  .venv/bin/python -m flm.sweep $g --workers 2 > outputs/$g.log 2>&1
done
echo "=== chain2 done $(date)"
