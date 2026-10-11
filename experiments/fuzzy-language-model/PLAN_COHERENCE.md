# Plan: a fuzzy language model that writes coherent stories and can still be read

CPU session, 2026-10-11, for the GPU session. The user's goal: develop the FLM into a model capable of
coherent text that stays explainable; delegate large compute to the GPU. This is the program; the
context question (PLAN_CONTEXT.md, `v8-*`) is its first stage.

## Success criteria (fixed now, before any stage runs)

* **Coherent:** on the blind two-rater protocol (`flm/coherence_set.py`, 8 prompts per model, T 0.5 /
  top-k 10, 600 characters), a total of at least **12/15 and a plot score of at least 3.5/5 from both
  raters**. Today's best is 8.6-9.9/15 with plot 1.9-2.5 (BPE1024 D, 93K parameters, 300M characters).
* **Explainable:** the model is the same architecture (fuzzy recurrent mixer + first-order TSK FFN) at
  every size, and for the final model (a) `flm.rules` produces a rulebook of its FFN rules, (b) a
  **rule-ablation test** shows the rulebook is causal: zeroing the firing of the 8 most-cited rules
  raises validation bits per character by more than zeroing 8 random rules (this script does not exist yet;
  the CPU session writes it, `flm/rule_ablation.py`), and (c) the rulebook's coverage is reported (share of
  token positions where at least one rule listed in the book fires above 0.5).
  If scaling breaks (b), the model is bigger but not explained, and the claim is dropped.

## Stages

1. **Context (`v8-ctx`, `v8-rules`).** Does more context or state help? See PLAN_CONTEXT.md.
2. **Size (`v9-lr`, then `v9-size`).** Same architecture, 0.25M to 7M parameters, 100M characters, ctx 512.
   `v9-lr` (4 cells, 10M characters) picks the learning rate for `v9-size` (it reads the finished v9-lr
   records when the grid is *loaded*, so run v9-lr to completion first, then start v9-size in a new process).
   Run one cell and read steps/s before the rest. The question: where on this ladder do plot scores pass 3?
3. **Grade.** Add the v9 checkpoints (and the best v8 one) to `flm/coherence_set.py`; two blind raters.
   The CPU session can be the second rater again.
4. **Explain.** `flm.rules` rulebook for the best checkpoint; `flm/rule_ablation.py` (CPU session).
5. **If the ladder stalls:** the next levers, each one change at a time: more characters at the best size
   (`--chars 300e6`), a larger vocabulary (BPE1536 gained little at 30M), and decoding (repetition
   control at sampling time, graded blind). Distillation is **out** (H16 refuted).

## Registered predictions

* **H33:** plot score rises with size and passes 3/5 (both raters) by the 3.3M model. ~40%.
* **H34:** no size below 2M parameters reaches 12/15. ~85%.
* **H35:** rule-ablation of the most-cited rules costs more than random rules at every size (rulebook is causal). ~70%.
* **H36:** the best learning rate falls as the model grows (2e-2 or lower at d 192). ~60%.

## Caveats

One seed per size; coherence is 8 prompts x 2 raters of one model family (a human check of the final model is
owed); BPC is not coherence, so the ladder is judged by the graded text, not by validation loss alone.

**Update (user, 2026-10-11):** keep to story size for now and look at Fibonacci attention (PLAN_FIBONACCI.md). Stage 2 (`v9-*`, the 0.25M-7M size ladder) is **on hold**: do not run it until the user releases it. The success criteria stand.
