#!/bin/bash
# Donor models at 50K, TinyStories 30M characters, seed 0: (a) learned codebook, (b) frozen oh48 codebook with F and G trained.
# (b) supplies the frozen head (C fixed, F, G learned) that the TRIBBLE construction targets.
cd "$(dirname "$0")/../.." && source flm/tribble_head/common.sh
OUT=outputs/tribble50k/donor
mkdir -p $OUT
env $SHAPE_ENV $PY -m flm.run_bijective $ARGS --lr 0.01 --chars 30e6 --seed 0 --eval-every 2000 \
  --save-model --ckpt-dir $OUT/learned_ckpt --ckpt-every 2000 --resume auto \
  --out $OUT/learned_s0.json > $OUT/learned_s0.log 2>&1 &
env $SHAPE_ENV FIXED_C=outputs/embspace/oh48.npz $PY -m flm.run_fixed_embedding $ARGS --lr 0.01 --chars 30e6 --seed 0 --eval-every 2000 \
  --save-model --ckpt-dir $OUT/frozen_ckpt --ckpt-every 2000 --resume auto \
  --out $OUT/frozen_s0.json > $OUT/frozen_s0.log 2>&1 &
wait
