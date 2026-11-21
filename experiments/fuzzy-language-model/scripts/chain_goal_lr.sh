#!/bin/bash
# goal-track LR study on CUDA: lrwidth (H10a, remaining cells), then the D / neural LR curves
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
for g in lrwidth v3-lr; do
  echo "=== $g start $(date)"
  $PY -m flm.sweep $g --device cuda --workers 4 > outputs/$g.log 2>&1
done
echo "=== chain_goal_lr done $(date)"
