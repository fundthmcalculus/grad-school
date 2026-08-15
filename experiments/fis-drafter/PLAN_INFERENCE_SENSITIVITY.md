# Plan: inference-parameter sensitivity battery

Goal: does varying how the target model is run change the injection monitor's
sensitivity? Written after a first cut, with one subtlety pinned down up front.

## The subtlety that scopes the whole battery

The monitor reads the **prompt forward pass** — the per-layer activation atlas
while the model *reads* the prompt, before it generates anything. Those
activations are a deterministic function of (weights, prompt, precision). So:

* **Sampling parameters — temperature, top_p, top_k, repetition_penalty, beam
  width — have exactly zero effect on the monitor.** They only shape token
  *sampling* during generation; the prompt activations do not depend on them.
  Varying them is a null experiment *unless* the monitor is extended to
  generation-time (streaming) activations.

So the axes that can actually move sensitivity are the ones that change the
prompt activations or the "normal" manifold, not the decoding knobs.

## Axes, in priority order

| axis | why it could matter | status |
|---|---|---|
| **Precision** fp32 / bf16 / fp16 | numerical drift in activations | **done — no effect** (det@1%FP 0.572/0.572/0.571, wl-AUROC 0.867 all three, SmolLM2-360M/deepset) |
| **Quantization** int8 / int4 (bitsandbytes, GPTQ, AWQ) | large activation perturbation, realistic serving | TODO — the real precision question; bf16↔fp16↔fp32 already shown equivalent |
| **System prompt / chat template** | the wrapper shifts every activation | **DONE (Part 16) — highest risk realised.** Mean-pool FPR blows up to ~1.0 under any shift; last-token+pooled-prompts holds FPR on unseen prompts at a detection cost. Recalibrate per system prompt. |
| **Attention backend** sdpa / eager / flash-attn | should be numerically ~equivalent | TODO — quick equivalence check |
| **Prompt context / position** injection embedded in a long benign document vs bare | realistic deployment; does the signal survive dilution? | TODO — construct embedded-injection probes |
| **Batch / padding side** | left vs right padding, batch size | TODO — expected null, worth confirming |
| **Generation-time monitoring** + sampling params | the ONLY place decoding knobs enter; a per-step version of the monitor | TODO — larger build (streaming atlas); Part-4 capture already records per-step hidden states |

## First-cut result

Precision is a non-issue: fp32, bf16, and fp16 produce detection identical to
three decimal places. The PCA-whitening + z-scored trimmed score is invariant to
the ~1e-3 numerical differences between float formats. This means the bf16
numbers reported throughout are not a precision artefact, and half-precision
serving costs nothing here.

## What to build next

1. **Quantization** (int8/int4) — the precision axis that could actually bite, on
   a mid model (Qwen-3B) on deepset + safeguard. Re-capture under bitsandbytes
   4-bit and compare det@1%FP.
2. **System-prompt shift** — the highest-risk axis. Capture benign+injection
   under system prompt A (fit "normal") and system prompt B (test); measure the
   realized FPR under the shift. If it blows up, the monitor needs per-deployment
   recalibration, which is an important operational finding.
3. **Embedded injection** — inject into a long benign document and test whether
   detection survives context dilution.

Each reuses `fmri_capture.py` (add a `--quant`/`--system-prompt` option) and the
existing detectors; the analysis is the same within-length det@1%FP pipeline.
