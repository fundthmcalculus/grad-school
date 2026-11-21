# Inference precision: fp32 against fp16 against int8 (tables)

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

