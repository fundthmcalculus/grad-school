> **Status (2026-10-11):** superseded: context past story size was dropped. Clean summary: [`REVIEW.md`](REVIEW.md).

# Plan: does the fuzzy recurrent model use more context? (for the GPU session)

Written by the CPU session, 2026-10-11. The user's question: scale out attention and context until the
stories become coherent. The user is less interested in neural comparisons than in what the fuzzy
models can do.

**Two facts that shape the plan.**
1. The headline model D *is* the fuzzy recurrent LM (FRLM): its mixer is `--mixer fuzzydelta`, a
   delta-rule fuzzy recurrent mixer with a fixed state per head (8 heads x 16 rules), and a
   first-order TSK FFN. There is no separate "FRLM vs D" arm to run. "Scaling attention" means
   scaling D's context and its state.
2. Every run so far uses a 256-token context (`--ctx` default). At BPE1024 that is roughly 600-800
   characters, about one story. Stories are packed back to back in the training stream with `<|eot|>`
   between them, so context beyond a story's own length holds a *different* story: it cannot help
   predict this one. Longer context can therefore help only (a) stories longer than 256 tokens and
   (b) the model's handling of the story start. Expect modest gains in bits per character; the more
   informative measures are the probes below.

## Probes (`flm/context_probe.py`, runs on any saved BPE checkpoint, CPU is fine)

1. bits per character by window position (0-255, 256-511, 512-1023, 1024+) over long sequential
   validation windows. D has no positional encoding (RoPE is only in the softmax/gauss mixers), so a
   model trained at ctx 256 can be run at 1024 or 2048;
2. bits by distance from the story start (since the last `<|eot|>`);
3. name recall: a story names a character, k neutral filler sentences follow, then the model is asked to
   continue 'her mom said, "Come here,'. The score is log2 p(name) - log2 p(distractor), minus the same
   with the name statement removed (per pair, so name priors cancel).

Baseline (the existing 100M BPE1024 D checkpoint, ctx 256, 24 windows of 1,024 tokens; recorded in
RESULTS.md, H29): name-recall gain 2.01 bits at k = 0 (24 tokens), 1.26 at k = 3 (54 tokens), 1.07 at
k = 12 (152 tokens), 0.79 at k = 24 (280 tokens, just past the training context). The model recalls a
name and the signal decays smoothly rather than collapsing.

## Grids (`flm/grids_v8.py`, registered in `flm.sweep`)

| grid | cells | what | GPU-h (estimate; run one cell first) |
|---|---|---|---|
| `v8-ctx` | 8 | D, BPE1024, d 48, lr 6e-2, ctx 256 / 512 (story size), constant 8,192 tokens per step, 30M chars, seeds 0-1, checkpoints saved (4 cells) | ~0.5 |
| `v8-rules` | 2 | ctx 512 with 32 rules per head (more recurrent state) | ~0.3 |

The ctx-256 cells are the baseline. Command: `python -m flm.sweep v8-ctx --workers 2` (one cell first).

## Order and decisions

1. Run the probe on the existing 100M checkpoint at `--len 2048` (done on CPU at 1,024; repeat at 2,048).
2. Run `v8-ctx`. Then run the probe on every checkpoint it saves:
   `python -m flm.context_probe outputs/v8-ctx/D-ctx1024_30M_s0.pt --out outputs/v8-ctx/D-ctx1024_30M_s0.probe.json`
3. Decision rule (registered as H30-H32 in RESULTS.md): if name-recall gain at k = 24 and bits at positions
   beyond 256 do not improve with training ctx, longer context is not the limit; the limit is
   state size or model capacity, and `v8-rules` plus a bigger model (BPE1024 d64) are the next cells.
   If they improve, extend to the 100M budget and re-grade coherence.
4. Re-grade coherence on samples from the best ctx model with the blind two-rater protocol
   (`flm/coherence_set.py`); the plot axis is the one to watch (1.9/5 at 300M for ctx 256).

## Caveats

* Bits per character at a longer training context is averaged over more positions, so
  `val_bpc` is not directly comparable across ctx; compare the probe's per-position numbers.
* Two seeds per cell; differences under about 0.01 bits per character are within seed spread.
* The name-recall probe is one template and one filler set; it measures reuse of a stated name, not
  plot coherence.

**Update (user, 2026-10-11):** keep to story size for now. The 1024 and 2048 arms were removed from `v8-ctx`; `v8-rules` runs at ctx 512. The new main line is Fibonacci attention (PLAN_FIBONACCI.md, `v10-fib`).
