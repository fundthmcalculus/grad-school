# Post-training quantisation of the 100K all-fuzzy model (2026-10-23; CPU only, 3 threads; `flm/ptq_probe.py`, `report/ptq_probe.json`)

Model: F7 (V6144 / head 640 / tail 8 / d32, 99,882 parameters), 300M characters, floor 0, seed 0 (`outputs/v70-shape300/F7_v6144_h640_d32_floor0_s0.pt`, trained in FP32). The CPU FP32 evaluation reproduces the training run's number exactly (0.95577 on the first 1,000,000 validation characters), so the harness is faithful. Method: simplest PTQ, symmetric uniform rounding of the weights (fake quantisation: dequantised back to float), per-row scales unless stated, no calibration, no clipping search, no retraining; activations stay FP32. Size = weights at their bit width + 16 bits per scale (FP32 = 399,528 B).

## All parameters, one bit width (300,000 validation characters, reference 0.96472)
| weights | size | bits per character | change |
|---|---|---|---|
| fp32 | 399,528 B | 0.96472 | -- |
| fp16 | 199,764 B | 0.96472 | +0.0000 |
| bf16 | 199,764 B | 0.96494 | +0.0002 |
| int8, per row | 114,224 B | 0.96558 | **+0.0009** |
| int8, per tensor | 100,030 B | 0.96713 | +0.0024 |
| int6, per row | 89,253 B | 0.97902 | +0.0143 |
| int6, per tensor | 75,059 B | 1.01301 | +0.0483 |
| int5, per row | 76,768 B | 1.04140 | +0.0767 |
| int5, per tensor | 62,574 B | 1.16726 | +0.2025 |
| int4, per row | 64,283 B | 1.45821 | +0.4935 |
| int4, per tensor | 50,089 B | 2.52103 | +1.556 |
| int3, per row | 51,797 B | 2.86651 | +1.902 |
| int2, per row | 39,312 B | 3.89974 | +2.935 |
(per-tensor int3 4.34, int2 16.38.) The full 1M-character check of the headline: all-int8 per row 0.95649 against 0.95577 (**+0.0007**), all-int7 per row 0.95971 (+0.0039).

## One group quantised at a time, per row (change in bits per character against 0.96472)
| group (parameters) | int4 | int3 | int2 |
|---|---|---|---|
| tier_up (256) | +0.008 | +0.027 | +0.18 |
| emb_head (20,480) | +0.027 | +0.167 | +1.35 |
| ffn_cons (6,630) | +0.032 | +0.19 | +1.17 |
| proj (10,752) | +0.035 | +0.164 | +1.18 |
| norm (260) | +0.055 | +0.217 | +3.67 |
| mix_other (8,320) | +0.056 | +0.406 | +2.03 |
| emb_tail (44,032) | +0.059 | +0.289 | **+0.48** |
| ffn_ante (rule centres and widths, 9,152) | +0.064 | +0.326 | +1.85 |
Every group loses 0.01-0.07 at 4 bits on its own, and the losses compound: all groups at int4 cost +0.49. Only the fuzzy tail memberships (emb_tail, 68% of the embedding and 44% of all parameters) survive 2 bits at all (+0.48), the rest do not.

## Mixed widths (full 1M characters, reference 0.95577)
| config | size | bits per character | change |
|---|---|---|---|
| all int8 row | 114,224 B | 0.95649 | +0.0007 |
| embeddings int8, rest fp16 | 147,540 B | 0.95599 | +0.0002 |
| embeddings int6, rest int8 | 98,096 B | 0.95948 | +0.0037 |
| embeddings int6, rest fp16 | 131,412 B | 0.95897 | +0.0032 |
| all int7 row | 101,738 B | 0.95971 | +0.0039 |
| embeddings int5, rest int8 | 90,032 B | 0.97056 | +0.0148 |
| embeddings int4, rest int8 | 81,968 B | 1.04112 | +0.0853 |
| emb_tail int4 only, rest fp32 | 256,424 B | 1.01479 | +0.0590 |
(`emb int4, rest fp16` 1.04027; the `emb_tail int6 only` row did not finish inside the time limit.)

## Findings
- **Robust to 16-bit and 8-bit PTQ, not below.** fp16 is exactly lossless and bf16 costs 0.0002; int8 per row costs 0.0007-0.0009 bits per character (0.07%) for a 3.5x smaller file (114 KB); int7 and "embeddings int6" cost about 0.004 (0.4%); int6 for everything costs 0.014; int5 and below are damaging and int4 is a collapse (+0.49 for all, +0.06 even for the most robust single group).
- **Per-row scales matter at low bits** (int5: +0.077 per row, +0.20 per tensor; int4: +0.49 against +1.56) but cost 2 bytes per row; for the 5,504-row tail table with 8 columns the scale overhead is 2 bits per weight, so the size accounting favours int8 for the tail.
- **No single sensitive part is to blame**; the rule centres and log-widths (`ffn_ante`), mixer centres, and the norm's centres/widths are the worst per parameter at 3 bits and below (they define where rules fire, so small shifts change membership), but even the embeddings lose 0.03 at 4 bits. A 100K model has little redundancy to spend.
- The model was **not trained for quantisation**: quantisation-aware training or per-group clipping/calibration (not tried here) would very likely move int4 from collapse to usable; this probe says only what the simplest PTQ does.
- **Bug found while probing (relevant to `--precision bf16`/`fp16` training and to autocast inference):** `flm/fuzzy_tokens.py` `_core` (line 197, `out[..., self.t_head_ids] = lh`) writes a bf16 tensor into an fp32 buffer, so **any autocast forward pass on the all-fuzzy model fails** (`Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source`). The queued precision screen (`outputs/chain_prec.sh`: tf32 is unaffected, bf16 and fp16 arms) will crash at the first step on this path. Not fixed here (out of scope; the file is in use by running jobs).
