#!/bin/bash
# Same arms as run.sh, run one at a time: the concurrent batch diverged (NaN) in 5 of 6 runs in two attempts, while every run alone was healthy.
cd "$(dirname "$0")/../.." && export GRAD_SCHOOL_DATA=/home/scott/grad-school/data
OUT=outputs/window_t2048/serial
COMMON="--decay data --exp-norm mean --ffn-exp-norm mean --heads 2 --ffn tsk --ffn-order 1 --ffn-norm none --tsk-dim 8 --ffn-mult 1.5 --d 48 --layers 2 --rules 6 --shortconv 0 --ctx 2048 --batch 8 --chars 4e6 --device cpu --threads 16 --eval-chars 500000 --eval-every 0 --warmup 100 --lr 0.01 --lr-floor 0.02 --precision fp32 --ckpt-every 100 --resume auto"
run() {
  local name=$1 mixer=$2 win=$3 seed=$4
  env BIJ=ci FUZZ_FFN_R=68 FUZZ_FFN_R0=48 FIXED_C=outputs/embspace/oh48.npz WIN=$win PYTHONPATH=. \
    .venv/bin/python -m flm.run_window --mixer $mixer $COMMON --seed $seed \
    --out $OUT/${name}_s${seed}.json --ckpt-dir $OUT/ckpt/${name}_s${seed} > $OUT/${name}_s${seed}.log 2>&1
}
for s in 0 1; do
  run T2048-dense fuzzydelta "" $s
  run T2048-w32 wdelta 32 $s
  run T2048-w128 wdelta 128 $s
done
echo ALL_DONE > $OUT/done.txt
