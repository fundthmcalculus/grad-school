#!/bin/bash
# Shared settings for the 50K TRIBBLE-head study (PLAN_FROZEN_HEAD.md section 4). CPU, 8 threads per run.
# usage: source flm/tribble_head/common.sh; run_flm ARM OUTDIR CHARS SEED [extra env via env]
export GRAD_SCHOOL_DATA=${GRAD_SCHOOL_DATA:-/home/scott/grad-school/data}
export PYTHONPATH=.
PY=.venv/bin/python
SHAPE_ENV="BIJ=ci FUZZ_FFN_R=68 FUZZ_FFN_R0=48"
ARGS="--mixer fuzzydelta --decay data --exp-norm mean --ffn-exp-norm mean --heads 2 --ffn tsk --ffn-order 1 --ffn-norm none --tsk-dim 8 --ffn-mult 1.5 --d 48 --layers 2 --rules 6 --shortconv 0 --ctx 256 --batch 32 --device cpu --threads 8 --eval-chars 1000000 --warmup 100 --lr-floor 0.02 --precision fp32"
