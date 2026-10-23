# Inference precision: fp32 against fp16 against int8 (report, 2026-10-23)

Question (owner): do fp32, fp16 and int8 inference match, and is int8 faster? Models: the shipped 100K model (F7), the 14M model trained on TinyStories (300M characters, 0.668 bits per character) and the 14M model trained on the Pile sample (1.97). Same trained fp32 weights in every format; RTX 4080 Laptop GPU, and 3 CPU threads for the CPU numbers (run alone, after the GPU runs). Script `flm/infer_precision.py`, tables `flm/infer_precision_report.py`, raw `report/infer_precision_*.json`, artifact https://claude.ai/artifact/6vqvcQsn1g8j75w9VLjS2j. Not registered beforehand (the question arrived mid-session); my expectation before running was "smaller, not faster", and it held.

## Answer
- **fp16 matches fp32.** Change in bits per character at most +0.00002 on all three models; the same top prediction at 99.8% of positions or more; KL divergence below 2e-5 bits; 19 or 20 of 20 greedy continuations identical. Half the size. On the GPU the forward pass is 1.13x to 1.14x faster for the 14M models and 1.02x for the 100K one.
- **int8 (one scale per matrix row) matches in average quality, not word for word.** Change +0.0007 (100K), +0.0009 (14M TinyStories), +0.0011 (14M Pile) bits per character, about a tenth of a percent; the same top prediction at 96.9% to 97.4% of positions; KL 0.0025 to 0.0037 bits; only 4, 4 and 12 of 20 greedy continuations identical (the mean first difference is 25 to 46 pieces in). A quarter of the size (0.114 MB, 14.2 MB, 14.2 MB).
- **int8 is not faster.** Throughput 0.99x to 1.01x of fp32 on the GPU (the arithmetic is the same: there are no int8 kernels for these layers); holding the weights as int8 and converting before each pass is 0.73x to 0.85x. On the CPU everything computes in fp32, so there is no speed difference to find (0.94x to 1.02x, within run-to-run noise).
- **bf16** is between the two on quality (+0.0002 to +0.0003; top-1 98.2% to 98.6%; greedy identical 4, 9 and 10 of 20) and the same speed as fp16.

## What this means for shipping
fp16 is a free halving (28 MB for a 14M model) with no measurable loss. int8 quarter-sizes the model (14 MB) for about +0.001 bits per character, and is the right choice when download size matters and word-for-word greedy decoding does not (the demo samples at temperature 0.5, so continuations differ run to run anyway). Neither makes inference faster on this hardware beyond the 13% fp16 gain of the 14M models on the GPU.

## Caveats
- The decoding times (one greedy 64-piece continuation, batch 1, no cache; 24 pieces on the CPU) come from a single run each and moved by tens of percent between runs (for example the 100K model's CPU fp16 decode shows 0.62x of fp32, which is noise: its forward throughput is 1.01x); only the throughput numbers are medians over repetitions and should be read as speed.
- Quality is measured on the checkpoint's own validation text, 1M characters, 200 windows for agreement and KL; one checkpoint per model, one seed.
- int8 here is weight-only, per-row, without calibration or clipping; "int8 held, rebuilt each pass" is the honest cost of int8 storage with no kernel. Activation quantisation, int4 and below, and quantisation-aware training were not tested (the earlier probe, `report/ptq_probe.md`, found int4 costs about half a bit per character).
- The CPU has no fast fp16 or int8 path for the fuzzy layers, so the CPU speed rows say "no change", not "no benefit available" with a different engine.

## F7, 99,882 parameters, TinyStories (300M characters)

### Quality against fp32 (GPU arithmetic, 1M validation characters, 200 windows of 512 pieces, 20 greedy continuations of 64 pieces)

| format | stored size | bits/char | change | top-1 agrees | KL (bits) | largest logit change | greedy identical | mean first difference |
|---|---|---|---|---|---|---|---|---|
| fp32 (reference) | 0.400 MB (1.00x) | 0.95577 | +0.00000 | 100.00% | 0.00e+00 | 0.00 | 20/20 | 64.0 |
| fp16 | 0.200 MB (0.50x) | 0.95578 | +0.00001 | 99.78% | 1.70e-05 | 0.14 | 19/20 | 60.8 |
| bf16 | 0.200 MB (0.50x) | 0.95607 | +0.00030 | 98.30% | 1.15e-03 | 1.39 | 4/20 | 28.0 |
| int8, per row | 0.114 MB (0.29x) | 0.95649 | +0.00072 | 97.00% | 3.06e-03 | 3.11 | 4/20 | 24.6 |
| int8 held, rebuilt each pass | 0.114 MB (0.29x) | 0.95649 | +0.00072 | 97.00% | 3.06e-03 | 3.11 | 4/20 | 24.6 |

### GPU speed (cuda; forward-only throughput at 16 x 512, and greedy decoding with batch 1 and no cache)

| format | pieces/s | against fp32 | ms per piece, decoding | against fp32 |
|---|---|---|---|---|
| fp32 (reference) | 286,828 | 1.00x | 8.22 | 1.00x |
| fp16 | 291,387 | 1.02x | 7.50 | 0.91x |
| bf16 | 292,330 | 1.02x | 7.75 | 0.94x |
| int8, per row | 288,362 | 1.01x | 8.79 | 1.07x |
| int8 held, rebuilt each pass | 210,443 | 0.73x | 13.07 | 1.59x |

### CPU speed and quality (3 threads; fp16 and int8 are stored as such and computed in fp32 on the CPU, which has no fast fp16 or int8 path for these layers)

| format | pieces/s (4 x 512) | against fp32 | ms per piece, decoding | against fp32 | bits/char (300K characters) |
|---|---|---|---|---|---|
| fp32 (reference) | 9,283 | 1.00x | 4.82 | 1.00x | 0.96472 |
| fp16 | 9,379 | 1.01x | 2.99 | 0.62x | 0.96472 |
| int8, per row | 9,434 | 1.02x | 3.82 | 0.79x | 0.96558 |
| int8 held, rebuilt each pass | 9,010 | 0.97x | 3.03 | 0.63x | 0.96558 |

## 14M, 14,067,715 parameters, TinyStories (300M characters)

### Quality against fp32 (GPU arithmetic, 1M validation characters, 200 windows of 512 pieces, 20 greedy continuations of 64 pieces)

| format | stored size | bits/char | change | top-1 agrees | KL (bits) | largest logit change | greedy identical | mean first difference |
|---|---|---|---|---|---|---|---|---|
| fp32 (reference) | 56.271 MB (1.00x) | 0.66833 | +0.00000 | 100.00% | 0.00e+00 | 0.00 | 20/20 | 64.0 |
| fp16 | 28.135 MB (0.50x) | 0.66834 | +0.00001 | 99.84% | 1.20e-05 | 0.29 | 20/20 | 64.0 |
| bf16 | 28.135 MB (0.50x) | 0.66857 | +0.00024 | 98.64% | 7.05e-04 | 2.59 | 9/20 | 49.8 |
| int8, per row | 14.157 MB (0.25x) | 0.66927 | +0.00094 | 97.44% | 3.65e-03 | 4.86 | 4/20 | 29.4 |
| int8 held, rebuilt each pass | 14.157 MB (0.25x) | 0.66927 | +0.00094 | 97.44% | 3.65e-03 | 4.86 | 4/20 | 29.4 |

### GPU speed (cuda; forward-only throughput at 16 x 512, and greedy decoding with batch 1 and no cache)

| format | pieces/s | against fp32 | ms per piece, decoding | against fp32 |
|---|---|---|---|---|
| fp32 (reference) | 103,336 | 1.00x | 13.50 | 1.00x |
| fp16 | 116,711 | 1.13x | 17.39 | 1.29x |
| bf16 | 116,683 | 1.13x | 19.78 | 1.47x |
| int8, per row | 101,813 | 0.99x | 14.51 | 1.07x |
| int8 held, rebuilt each pass | 84,907 | 0.82x | 43.68 | 3.24x |

### CPU speed and quality (3 threads; fp16 and int8 are stored as such and computed in fp32 on the CPU, which has no fast fp16 or int8 path for these layers)

| format | pieces/s (4 x 512) | against fp32 | ms per piece, decoding | against fp32 | bits/char (300K characters) |
|---|---|---|---|---|---|
| fp32 (reference) | 3,643 | 1.00x | 11.48 | 1.00x | 0.68026 |
| fp16 | 3,467 | 0.95x | 12.38 | 1.08x | 0.68026 |
| int8, per row | 3,495 | 0.96x | 13.63 | 1.19x | 0.68126 |
| int8 held, rebuilt each pass | 2,432 | 0.67x | 21.11 | 1.84x | 0.68126 |

## 14M, 14,067,715 parameters, Pile sample (300M characters)

### Quality against fp32 (GPU arithmetic, 1M validation characters, 200 windows of 512 pieces, 20 greedy continuations of 64 pieces)

| format | stored size | bits/char | change | top-1 agrees | KL (bits) | largest logit change | greedy identical | mean first difference |
|---|---|---|---|---|---|---|---|---|
| fp32 (reference) | 56.271 MB (1.00x) | 1.96941 | +0.00000 | 100.00% | 0.00e+00 | 0.00 | 20/20 | 64.0 |
| fp16 | 28.135 MB (0.50x) | 1.96943 | +0.00002 | 99.80% | 1.30e-05 | 0.37 | 19/20 | 63.9 |
| bf16 | 28.135 MB (0.50x) | 1.96963 | +0.00022 | 98.21% | 6.79e-04 | 2.23 | 10/20 | 48.0 |
| int8, per row | 14.157 MB (0.25x) | 1.97046 | +0.00105 | 96.88% | 2.53e-03 | 5.18 | 12/20 | 45.8 |
| int8 held, rebuilt each pass | 14.157 MB (0.25x) | 1.97046 | +0.00105 | 96.88% | 2.53e-03 | 5.18 | 12/20 | 45.8 |

### GPU speed (cuda; forward-only throughput at 16 x 512, and greedy decoding with batch 1 and no cache)

| format | pieces/s | against fp32 | ms per piece, decoding | against fp32 |
|---|---|---|---|---|
| fp32 (reference) | 101,596 | 1.00x | 15.49 | 1.00x |
| fp16 | 115,384 | 1.14x | 17.40 | 1.12x |
| bf16 | 114,828 | 1.13x | 16.70 | 1.08x |
| int8, per row | 102,348 | 1.01x | 11.64 | 0.75x |
| int8 held, rebuilt each pass | 86,344 | 0.85x | 35.53 | 2.29x |

### CPU speed and quality (3 threads; fp16 and int8 are stored as such and computed in fp32 on the CPU, which has no fast fp16 or int8 path for these layers)

| format | pieces/s (4 x 512) | against fp32 | ms per piece, decoding | against fp32 | bits/char (300K characters) |
|---|---|---|---|---|---|
| fp32 (reference) | 2,814 | 1.00x | 12.65 | 1.00x | 1.92601 |
| fp16 | 3,004 | 1.07x | 14.47 | 1.14x | 1.926 |
| int8, per row | 2,640 | 0.94x | 14.17 | 1.12x | 1.92744 |
| int8 held, rebuilt each pass | 2,943 | 1.05x | 22.22 | 1.76x | 1.92744 |

