#!/bin/bash
# GPU smoke test: the 50K invertible model, 0.5M characters, the same seed on CPU and on CUDA.
# Passes when the CUDA run is finite, recorded as cuda, and its validation BPC is within 0.01 of the CPU run
# (the D1 tolerance in flm/device.py). Needs the CUDA environment: .venv-cuda (torch 2.14.1+cu130) and a working driver.
#   bash flm/gpu_smoke.sh
cd "$(dirname "$0")/.." && export GRAD_SCHOOL_DATA=${GRAD_SCHOOL_DATA:-/home/scott/grad-school/data} PYTHONPATH=.
PY=.venv-cuda/bin/python
[ -x $PY ] || { echo "missing $PY (see README: GPU host setup)"; exit 2; }
$PY -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda available:', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 3)" || { echo "CUDA not available: the NVIDIA driver is not loaded"; exit 3; }
OUT=outputs/gpu_smoke; mkdir -p $OUT
COMMON="--mixer fuzzydelta --decay data --exp-norm mean --ffn-exp-norm mean --heads 2 --ffn tsk --ffn-order 1 --ffn-norm none --tsk-dim 8 --ffn-mult 1.5 --d 48 --layers 2 --rules 6 --shortconv 0 --ctx 256 --batch 32 --lr 0.01 --lr-floor 0.02 --warmup 100 --chars 0.5e6 --eval-chars 200000 --eval-every 0 --precision fp32 --seed 0"
for dev in cpu cuda; do
  env BIJ=ci FUZZ_FFN_R=68 FUZZ_FFN_R0=48 $PY -m flm.run_bijective $COMMON --device $dev --threads 8 --out $OUT/$dev.json > $OUT/$dev.log 2>&1 || { echo "$dev run failed, see $OUT/$dev.log"; exit 1; }
done
$PY - <<'PYEOF'
import json, math, sys
c = json.load(open("outputs/gpu_smoke/cpu.json")); g = json.load(open("outputs/gpu_smoke/cuda.json"))
d = abs(g["val_bpc"] - c["val_bpc"])
print(f"cpu {c['val_bpc']:.5f}  cuda {g['val_bpc']:.5f}  diff {d:.5f}  device_name {g.get('device_name')}  cuda c/s {g['chars_per_sec']}  cpu c/s {c['chars_per_sec']}")
ok = math.isfinite(g["val_bpc"]) and g.get("device") == "cuda" and d < 0.01
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYEOF
