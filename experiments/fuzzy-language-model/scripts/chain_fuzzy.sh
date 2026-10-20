#!/bin/bash
# Resumable queue for the all-fuzzy stages (PLAN_FUZZYLOCAL.md): every runner skips cells whose result JSON exists, so re-running after a crash continues.
cd "$(dirname "$0")/.."
export PYTHONPATH=.
for step in "flm.run_fuzzy_local_300m" "flm.run_fuzzy_ffn" "flm.run_fuzzy_proj" "flm.run_fuzzy_stage fgate12"; do
  echo "=== $step start $(date)"
  python -m $step
done
echo "=== chain_fuzzy done $(date)"
