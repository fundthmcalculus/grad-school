# Fuzzy language model: review document

Written 2026-10-11 by the CPU session for your review. This is the clean entry point: what works, what was dropped, what the evidence says about
coherence and explainability, and the path from here. Every claim points to a section of [`RESULTS.md`](RESULTS.md), where hypotheses are registered
before runs and outcomes (including the ones that went against us) are recorded. Decisions I took while you were away are in [`DECISIONS.md`](DECISIONS.md).
The progress page (charts) is https://claude.ai/artifact/SEDQDg6ZPvBeHZZth515kP; the local demo is `demo/` (`python demo/server.py`).

## 1. Goal and where we stand

**Goal:** a fuzzy language model that writes coherent stories and is still explainable.

| | criterion (set before the work) | best so far |
|---|---|---|
| Coherence | blind total of at least 12/15 **and** plot at least 3.5/5, from both raters | about 8.4/15 average over models, plot about 2.0/5 (best single set: 9.9/15, not replicated) |
| Explainability | rule book is causal and a short reading reproduces the model's prediction | met for next-token prediction; not for plot (section 5) |

**Status: not met.** The model got much better as a language model (1.62 to 0.98 bits per character at about 96K parameters; 0.83 at 1.32M) and its
rules are readable, but blind coherence has stayed at 6 to 8.5 out of 15 across every change tried, at every size from 93K to 1.32M parameters and
up to 3 billion training characters. The one lever with a visible trend that has not been tested properly is **size combined with data** (section 7).

## 2. What works (keep)

The model ("D"), 96,416 parameters: BPE1024 subword tokens; 2 layers, width 48; each layer is a causal convolution, a **delta-form fuzzy recurrent
mixer** (8 heads, 32 fuzzy memory rules per head) and a **first-order TSK feed-forward** of 72 rules over a 16-dimension antecedent (raw Gaussian
firing, mean-over-dimensions exponent); context 512; learning rate 6e-2; warm-up 100 (300 for runs of about 30M characters).

| result | number | where |
|---|---|---|
| Fuzzy D beats two tuned neural references at equal parameters, same tokenizer | 1.1291 vs 1.1564 and 1.1593 bits per character, 10 seeds, 30M characters, more than 2 s.e. each | `v6-final10` |
| Original character-level fuzzy design to the final model, same budget | about 1.62 to 1.11 at 30M characters (the TSK feed-forward fix and subword tokens did the work) | "Conclusion of the sub-100K track" |
| Context 256 to 512 at matched steps | 1.1324 to 1.1232 (2 seeds) | `v8-ctx`, `v8-ctx-b16` |
| 32 rules per head, ctx 512 | 1.1095 +/- 0.0048 at 30M (5 seeds) | `v13-state`, `v14-rule-seeds` |
| More data, 96K model | 30M 1.11, 300M 1.0015 (2 seeds), 1B 0.982 (2 seeds), 3B 0.9777 (flat from 1B) | `v12`, `v17-1b`, `v18-3b` |
| More parameters at equal data (30M characters) | 93K 1.123, 251K 0.970, 1.32M 0.830 | CPU runs, "Outcome of the 251K / 1.32M CPU run" |
| Warm-up | 300 beats 100 by 0.011 at 30M (5/5 seeds); gain is about zero at 300M, so use it only for short runs | `v23-wu30`, `v23-wu300m` |
| The recurrent fuzzy mixer carries long-range memory | name recall 2.1 bits at distance 0, 0.47 at about 280 tokens; plain attention recalls none at 10M characters | `v10-fib` (name-recall table) |
| Explainability tooling | rulebook (`flm/rules.py`), exact decomposition of the output into rules (`flm/rule_dist.py`), rule ablation, calibrated reader, rule inspector, demo | section 5 |
| Measurement | blind coherence sets with fresh seeds, two raters, key published only after the second rater scores | 9 sets, `report/coherence*` |

## 3. How we got here (one paragraph per phase)

CPU phase (WRITEUP.md): softmax and linear attention tie at small size; fuzzy attention costs nothing; the TSK feed-forward was the whole deficit; HTSK scaling and a
first-order consequent fixed it. GPU phase: subword tokens, then tuning at equal effort (D beats tuned neural references), then every lever listed in section 4.
Scale phase: data to 3 billion characters (flat past 1B), then size on CPU (loss falls a lot, coherence does not move).

## 4. What did not work (dropped; do not revisit without new evidence)

| tried | result | evidence |
|---|---|---|
| n-gram initialisation of the rules | null: within noise of random start | CPU phase, `ngram` stage 2 |
| Distillation from a 280K teacher (0.15 bits better) | no gain beyond extra compute; EMA adds 0.0005 | H16, `v3-post` |
| Fibonacci-lag attention (local window plus Fibonacci lags) | better than dense attention by 0.031, but 0.068 behind D; recalls no names at 10M characters | `v10-fib` |
| D plus one Fibonacci attention layer (both orders) | 0.027 worse than D; recall drops (0.47 to 0.11) | `v11-hybrid` |
| Factorised (low-rank) tied embedding | 0.040 to 0.126 worse | `v16-factemb` |
| Decoding changes (T 0.7 + top-p, repetition penalty, 4-gram block) | none beat the baseline sampler (7.6 vs 6.1 to 7.1 out of 15) | `coherence3` |
| Rules per head beyond 32 (64, 128, 192, 256) | flat within noise (1.1095, 1.1082, 1.1002, 1.1050, 1.1072) | `v14-rule-seeds`, `v14-state2` |
| Larger vocabularies (BPE1536, BPE2048) | gain shrank from 0.027 at 10M to 0.009 at 30M (2 seeds); not adopted | `v7-vocab`, `v7-confirm` |
| Weight-shared depth (4 or 6 applications of 2 blocks) | 0.012 worse, 2 to 3 times the compute | `v15-share` |
| Optimiser knobs other than warm-up (lr floor, beta2, clip, weight decay, lr 4e-2 to 9e-2) | flat | `v19-optA30` |
| Consequent-first / PPMI / coverage rule initialisations (cf, cfls, assocls, succcf, cfwide, hybrids) | no gain over random initialisation | `v22-tribble`, `v26-hybrid`, `cfwide`; `v27-cover` still running |
| Repetition-aware training loss (unlikelihood, repeat-weighted) | repeats fell 20% but blind coherence fell too (7.25 to 5.62 out of 15, one checkpoint) | `v25-rep`, coherence set 9 |
| Automatic coherence proxies (repeats, out-of-vocabulary words, names, repeated sentences) | correlate with raters at only |r| up to 0.24 | `coherence_auto` |
| Longer context past one story (1024, 2048 tokens) | never run (stories pack back to back, so extra context is another story) | `PLAN_CONTEXT.md` |
| More data past 1B characters (96K model) | 0.982 to 0.978 at 3B, inside seed noise | `v18-3b` |
| Character-level models, neural references as products | superseded by the subword fuzzy model; references kept only as controls | `v3-*`, `v6-final10` |
| RFLM variants (fixed recurrence, consequent propagation) | screened at 4M characters only, no clear effect; not carried forward | `v20-rflm` |

## 5. Explainability: what is established

* The model's output is an exact sum of per-rule terms plus an embedding term and mixer terms (reconstruction error 1.7e-5): `flm/rule_dist.py`.
* The cited rules are causal: silencing a layer's 8 most-used rules costs more than silencing 8 random ones, in every model and layer tested (96K, 251K, 1.32M): `flm/rule_ablation.py`.
* A reading of the 8 strongest rules per layer **plus the embedding and mixer terms** reproduces 94% of the model's top-10 hit rate (0.767 vs 0.814) and 70% of its top-10 set; 16 rules give 98%. Rules alone, even all 72, reach only 0.68: `flm/rule_reader.py`. The gap to the model is at most 0.07 for every token class: `flm/rule_explain_eval.py`.
* Rule roles are stable (90 to 95% of the top 20 rules per layer keep the same favoured next pieces across data halves), for example a determiner rule and a names-at-story-start rule: `flm/rule_story.py`.
* Repetition, seen as the same rule sets and token runs recurring in a text, tracks lower rater scores (r about -0.3 over 144 texts). It is one part of incoherence the rules can see.
* **Not established:** why a story lacks a plot. The rules explain how a next piece's score is assembled, not narrative structure. At 1.32M parameters the explanation thins: 8 of 288 rules per layer explain only 0.2 to 3% of the loss.
* **Demo** (`demo/`): generation with per-piece alternatives and rule effects, a rule timeline, a most-used-rules browser and per-rule membership functions (16 Gaussian dimensions whose product is the firing).

## 6. Measuring coherence (read this before believing any coherence number)

Blind sets: a checkpoint generates 8 stories (4 prompts, fresh seeds, T 0.5, top-k 10, 600 characters), shuffled across models; raters score grammar, consistency and plot 1 to 5 each
(maximum 15) before seeing the key. Both raters are Claude instances, so this checks rater noise, not rater bias. Inter-rater correlation 0.6 to 0.86 over sets. The **same checkpoint moves by about 2 points between sets**,
so differences under about 2 points are not resolved; 8 texts per model per set. No human has rated these. Seeds and files are in `report/coherence*` (scores, blind sets, keys).

## 7. Clean path to coherent results

The evidence says the remaining plausible levers are, in order:

1. **Finish the current GPU round** (`v27-cover`, then `v28-emb`: embedding initialisation, scale, weight decay). These are loss-side screens; I expect them to be null like their predecessors and do not expect them to change coherence.
2. **The decisive experiment: 1.32M parameters on 1 billion characters** (`PLAN_BIG1B.md`). Today the 1.32M model has seen about 7 training tokens per parameter against about 3,300 for the 96K model on 1B, so size and data are not separated. It is above the GPU session's 99,999-parameter cap, so it
   needs the GPU session's user to approve it. You asked me to schedule it after the current round; I have queued the request and set a follow-up (section 9).
3. **If scale moves blind coherence** (registered: total 10/15 or more and plot 2.5 or more from two raters): run the size ladder (`flm/grids_v9.py`, 0.25M to 7M, 100M characters) to find where plot passes 3.5, and re-certify explainability at that size with the calibrated reader (the reading needs more rules per layer at larger sizes, and the rule view needs a way to summarise 288+ rules).
4. **If scale does not move it**: the conclusion is that this architecture at this scale does not produce plot, and the remaining options are a different mechanism (an explicit story or entity state read by the rules) or a different training signal. Do that only after a human rater has confirmed the null, because the measure is noisy.
5. **In parallel, cheap and independent**: a human rater for the blind sets (two or three people reading 24 texts is enough to calibrate the Claude raters).

## 8. Decisions needed from you

* **Parameter cap.** The GPU session works under a 99,999-parameter cap (its user's instruction). Items 2 and 3 need that lifted, by its user. I cannot change it from here.
* **Human rater.** Who can read the blind sets (they are in `report/`; I can build a rating page in the demo if useful).
* **Archive pass.** Move the dropped levers' code and plans out of the main tree (list in section 10). I did not do this now because the GPU session's sweeps import code from disk while they run; it is a safe 15-minute job once they finish.

## 9. What I scheduled

A request to the GPU session to run the 1.32M x 1B job after `v28-emb` (it needs its user's approval for the cap, and said it will not run it without), and a recurring check that pings it when the round has finished. Details in `PLAN_BIG1B.md` and `DECISIONS.md` D14.

## 10. Map of this directory

* **Read:** this file; `RESULTS.md` (the record); `DECISIONS.md` (what I did overnight); `README.md` (older overview).
* **Code kept:** `flm/models.py`, `train.py`, `data.py`, `bpe.py`, `sweep.py`, `compare.py`; explainability `rules.py`, `rule_dist.py`, `rule_reader.py`, `rule_ablation.py`, `rule_explain_eval.py`, `rule_story.py`; measurement `coherence_set.py`, `coherence_auto.py`; demo `demo/`.
* **Dropped, to archive after the round ends (code, plans):** `flm/grids_v10.py`, `grids_v11.py` (Fibonacci, hybrids), `context_probe.py` (kept as a tool, grid dropped), `grids_v13.py`, `grids_v14.py` (rule counts), `conseq_first*.py`, `assoc.py`, `run_conseq.py`, `run_cover_sweep.py`, `run_tribble*.py` (initialisations), `run_rep*.py`, `rep_share.py` (repetition loss), `rflm_variants.py`, `run_rflm.py`, `run_variant.py` (RFLM), `ngram.py` (n-gram initialisation), `PLAN_CONTEXT.md`, `PLAN_FIBONACCI.md`, `PLAN_RFLM.md`, `PLAN_BACKWARD.md`, `PLAN_TUNE96K.md`. Their results stay in `RESULTS.md`.
* **Data and checkpoints:** `outputs/` (records and committed checkpoints under 5 MB; the 1.32M checkpoint is 5.3 MB and is not committed), `report/` (coherence sets), `data/tinystories/` (tokenizers; training streams are fetched or rebuilt).
