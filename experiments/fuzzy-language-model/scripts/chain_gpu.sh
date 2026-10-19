#!/bin/bash
# GPU track: device parity at the tune budget, then the 3-seed qualification grid on CUDA.
# Registered in RESULTS.md ("the GPU track"). Hyperparameters come from the CPU tune sweep.
# On Windows the venv interpreter is .venv/Scripts/python; override with PY=...
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
W=${WORKERS:-6}
for g in device-check scaling-gpu; do
  n=$($PY -m flm.sweep $g --list | tail -1 | cut -d' ' -f1)
  [ "$n" -gt 0 ] || { echo "GRID $g EMPTY - stopping"; exit 1; }
  echo "=== $g start ($n jobs) $(date)"
  $PY -m flm.sweep $g --workers $W > outputs/$g.log 2>&1
done
echo "=== chain_gpu done $(date)"
