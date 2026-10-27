#!/bin/bash
# usage: run_size.sh SIZE ARM SEED LR CHARS_M OUTDIR   (SIZE in 5k 10k 20k 40k 80k: width and rule counts chosen so every arm has the same total, measured)
cd /c/personal/grad-school-bij/experiments/fuzzy-language-model; export GRAD_SCHOOL_DATA=/c/personal/grad-school/data
SIZE=$1; ARM=$2; SEED=$3; LR=$4; CH=$5; OUT=$6
case $SIZE in
  5k)  D=12; T="16 11"; Z="7 5";;
  10k) D=16; T="42 29"; Z="32 21";;
  20k) D=24; T="70 47"; Z="54 38";;
  40k) D=32; T="120 84"; Z="101 71";;
  70k) D=40; T="181 127"; Z="159 111";;
  80k) D=48; T="171 120"; Z="145 102";;
  100k) D=56; T="186 130"; Z="156 109";;
esac
case $ARM in tied|ident) set -- $T;; z|c|ci) set -- $Z;; esac
mkdir -p $OUT
BIJ=$ARM FUZZ_FFN_R=$1 FUZZ_FFN_R0=$2 PYTHONPATH=. /c/personal/grad-school/.venv/Scripts/python -m flm.run_bijective --mixer fuzzydelta --decay data --exp-norm mean --ffn-exp-norm mean --heads 2 --ffn tsk --ffn-order 1 --ffn-norm none --tsk-dim 8 --ffn-mult 1.5 --d $D --layers 2 --rules 6 --shortconv 0 --ctx 256 --batch 32 --chars ${CH}e6 --device cuda --eval-chars 1000000 --eval-every 200 --warmup 100 --lr $LR --lr-floor 0.1 --seed $SEED --out $OUT/${SIZE}_${ARM}_lr${LR}_s${SEED}.json > $OUT/${SIZE}_${ARM}_lr${LR}_s${SEED}.log 2>&1
tail -n 1 $OUT/${SIZE}_${ARM}_lr${LR}_s${SEED}.log
