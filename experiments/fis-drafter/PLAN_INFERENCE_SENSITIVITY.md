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
| **Quantization** int8 / int4 (bitsandbytes) | large activation perturbation | **DONE (Part 17).** Matched degrades gracefully (int8 0.61, int4 0.50 vs bf16 0.66); mismatched (fit bf16 -> deploy int4) catastrophic (FPR 1.0). Calibrate at deployment precision. |
| **System prompt / chat template** | the wrapper shifts every activation | **DONE (Part 16) — highest risk realised.** Mean-pool FPR blows up to ~1.0 under any shift; last-token+pooled-prompts holds FPR on unseen prompts at a detection cost. Recalibrate per system prompt. |
| **Attention backend** sdpa / eager | numerical drift | **DONE (Part 17) — no effect** (0.663 vs 0.656). |
| **Prompt context / position** injection embedded in a long benign document | realistic deployment | **DONE (Part 18).** Mean-pool collapses to chance under dilution; last-token recovers it (wl-AUROC 0.80, few-shot 0.55) when the injection is near the end. |
| **Batch / padding side** | left vs right padding | **DONE (Part 17) — expected null** (masked mean and last-token are padding-invariant by construction). |
| **Generation-time monitoring** + logit shape | the ONLY place decoding knobs enter | **DONE (Part 19) — decisive negative.** Built full per-step streaming capture (`fmri_generate.py`) recording streaming hidden states AND output-logit shape. Logit shape carries weak, non-directional signal (wl-AUROC 0.64/0.60) that adds *exactly nothing* on top of the prompt readout (combined == prompt to 3 dp), as `logits = h·Eᵀ` predicts. Streaming activations never beat the prompt readout; `gen_last` is near-chance. Monitor the prompt, don't decode. |

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
