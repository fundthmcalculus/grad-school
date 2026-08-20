#!/usr/bin/env bash
# Scale the causal clamp across a model size ladder: capture streaming activations
# then scan where the exogenous-response subspace lives + its effective rank.
set -e
cd /home/sphillips/PycharmProjects/grad-school/experiments/fis-drafter
export HF_HUB_DISABLE_XET=1
PY=.venv/bin/python

run_one () {
  local mid="$1"; local tag="$2"
  echo "=== $mid -> runs/stream_$tag ==="
  if [ ! -f "runs/stream_$tag/win.npy" ]; then
    $PY -m fisdraft.fmri_stream --model-id "$mid" --batch-size 8 --n-base 60 \
        --n-preamble 4 --out "runs/stream_$tag"
  fi
  $PY -m fisdraft.clamp_scan --run "runs/stream_$tag" --n-probe 40
}

run_one "HuggingFaceTB/SmolLM2-360M-Instruct" "smol360"
run_one "Qwen/Qwen2.5-1.5B-Instruct" "qwen1p5b"
run_one "Qwen/Qwen2.5-7B-Instruct" "qwen7b"
echo "ALL DONE"
