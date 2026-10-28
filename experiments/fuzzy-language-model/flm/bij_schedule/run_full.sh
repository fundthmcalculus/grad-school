#!/bin/bash
# usage: run_full.sh ARM LR CHARS_M OUTDIR [extra train args]   (100K arms, d 56, full TinyStories stream, batch 32, context 256; checkpoints every 2000 steps, --resume auto)
cd /c/personal/grad-school-bij/experiments/fuzzy-language-model; export GRAD_SCHOOL_DATA=${DATA_ROOT:-/c/personal/grad-school/data}
ARM=$1; LR=$2; CH=$3; OUT=$4; shift 4
case $ARM in tied|ident) FR=186; FR0=130;; z|c|ci) FR=156; FR0=109;; esac
mkdir -p $OUT
BIJ=$ARM FUZZ_FFN_R=$FR FUZZ_FFN_R0=$FR0 PYTHONPATH=. /c/personal/grad-school/.venv/Scripts/python -m flm.run_bijective --mixer fuzzydelta --decay data --exp-norm mean --ffn-exp-norm mean --heads 2 --ffn tsk --ffn-order 1 --ffn-norm none --tsk-dim 8 --ffn-mult 1.5 --d 56 --layers 2 --rules 6 --shortconv 0 --ctx 256 --batch 32 --chars ${CH}e6 --device cuda --eval-chars 1000000 --eval-every 2000 --warmup 1000 --lr $LR --lr-floor 0.02 --seed 0 --save-model --ckpt-dir $OUT/ckpt --ckpt-every 2000 --resume auto --out $OUT/${ARM}_lr${LR}_${CH}M.json "$@" >> $OUT/${ARM}_lr${LR}_${CH}M.log 2>&1
tail -n 1 $OUT/${ARM}_lr${LR}_${CH}M.log
