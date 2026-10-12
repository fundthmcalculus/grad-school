# PLAN_CORPUS: same-size models on a larger general text corpus (registered 2026-10-12; plan only, nothing run)

**Why.** Every sub-100K result so far is TinyStories-only. TinyStories is narrow (a vocabulary of roughly 1,500 words, formulaic plots), so a 96K model saturates it (0.982 BPC at 1B chars, 0.978 at 3B) and the only lever that moved coherence was size. The question is whether a *broader* corpus changes what a fixed 96K-98K budget learns, and how much of the TinyStories result is the narrow distribution. This is a different data lever from "more of the same data" (already flat).

**Held fixed.** Architecture and parameter count (96,416 reference and the 98,280 tiered recipe), the TSK/FLM stack, ctx 512, batch 16, lr 6e-2, warm-up 100, T 0.5 / top-k 10 sampling for coherence sets. Only the corpus and, necessarily, the tokenizer change.

## Corpus candidates (pick one; decide before downloading)
| candidate | why | cost / risk |
|---|---|---|
| FineWeb-Edu sample (HF `HuggingFaceFW/fineweb-edu`, `sample-10BT`, first ~1-3 GB) | clean educational web text, plain register | ODC-By licence; pin a revision; large download |
| Wikipedia (English, one pinned dump, plain text) | factual register, well-formed sentences | markup cleaning needed |
| SlimPajama / C4 sample | broad web | noisiest; likely hardest for 96K |
| TinyStories + a general corpus mixed (e.g. 50/50 by characters) | tests whether breadth costs story quality | needs a mixing recipe |

**Default recommendation:** FineWeb-Edu first (single download, one pinned file set), with the TinyStories-mixed arm second.

## Data handling (AGENTS.md rule 8: nothing over 5 MB committed)
Fetched under `$GRAD_SCHOOL_DATA/<corpus>/` at a pinned revision, with a recovery note in `data/.gitignore`. A new loader `flm/data_general.py` (do not edit `flm/data.py` while sweeps run) exposes the same `load(split)` memmap contract. The validation slice is taken by document, never by byte offset into the training stream. A deterministic ASCII filter keeps the character stream at 98 symbols; report the fraction of text removed per corpus.

## Tokenizer
BPE1024 was trained on TinyStories; on a general corpus it is a different object. Arms: **T1** keep the TinyStories BPE1024 (cheapest, expect poor compression); **T2** retrain BPE1024 on the new corpus; **T3** retrain BPE2048 (embedding cost doubles, so pair it with the tiered embedding, `flm/run_tiered.py`). Compare in **bits per character**, never per token, on identical text.

## Experiments (30M-char screens first, then one confirmation)
- **C0 (minutes):** score the existing TinyStories checkpoints (96K 3B, tiered 300M) on the general validation slice. The transfer gap tells us how much is distribution. Also compute a size-matched n-gram baseline (`flm/ngram.py`) on the same slice: the trivial baseline comes before any claim.
- **C1:** 96K reference trained on the general corpus, arms T1/T2/T3, 30M chars, seeds 0-4 (a screen: ranking, not a claim).
- **C2:** tiered 98,280 recipe on the general corpus with the best tokenizer from C1, 30M chars, seeds 0-4.
- **C3 (confirmation):** the better of C1/C2 at 300M chars, seeds 0-1.
- **C4 (mixed corpus):** 50/50 TinyStories + general, evaluated on **both** validation sets, 30M chars, 3 seeds.
- **Metrics:** BPC on the general slice; BPC on TinyStories valid (breadth cost); a blind coherence set on story prompts and on expository prompts, two raters, scores written before the key.

## Registered predictions
1. C0: TinyStories models score above 2.0 BPC on general text (90%). Below 1.6 would mean the corpora overlap more than expected (10%).
2. C1: the 96K model at 30M scores BPC >= 1.6 on general text (85%), and lands within 0.15 of the size-matched n-gram baseline (50%).
3. T2 beats T1 by >= 0.05 BPC on every seed (80%). T3 beats T2 by >= 0.02 under the tiered embedding (35%).
4. C4: mixed training costs TinyStories BPC < 0.03 against the TinyStories-only reference while beating C1 on general BPC (40%). More likely it trades one for the other.
5. Story-prompt coherence from a general-corpus model scores below the TinyStories-only 96K model (~5/15) (75%). If equal or higher, breadth helps coherence and becomes the headline.
6. Whether breadth plus size helps is **not** tested here; that needs the same runs at ~650K and is not planned until C1-C3 are read.

## Decision rules
- If C1 BPC is within 0.15 of the n-gram baseline, the architecture adds almost nothing over counts at this size on broad text: stop and write that up as the limit finding.
- If T2/T3 move BPC by >= 0.05, the tokenizer is a first-class lever and gets its own ten-seed round.
- No run over 30M chars until C0 and the n-gram baseline are scored. Anything above ~1 hour of GPU per cell (300M+) goes back to the user for an explicit yes.

## Compute
30M-char cells ~3 min each on the RTX 4080 (96K): C1/C2 ~ 5 seeds x 4 arms x 3 min = 1 h; C3 ~ 1 h per seed. The download and ASCII filtering dominate wall-clock for a 1-3 GB corpus. CPU-session work in parallel: corpus fetch and filter, BPE retraining, the n-gram baseline, and `flm/data_general.py` with tests.

## Corpus chosen and built (2026-10-12; user: "Pick a decent corpus that's around 3B characters, similar to the one we have used, but maybe more coherent. Your choice.")
**Cosmopedia v2** (`HuggingFaceTB/smollm-corpus`, subset `cosmopedia-v2`, revision `3ba9d605774198c5868892d7a8deda78031a781f`, ODC-By): synthetic LLM-written text like TinyStories (so also coherent, edited-register prose), but broader: textbooks, stories, WikiHow-style and web-style articles. FineWeb-Edu was the earlier default; Cosmopedia's synthetic origin is closer to the corpus already used and is cleaner. Built by `flm/prep_cosmopedia.py`: train shards 0-2 consumed in order to **3,000,002,970 characters** (818,062 documents seen over all shards read; 15,823 dropped for non-ASCII > 0.5%, 58.9M characters, about 1.9%), the validation stream is the first 5,001,115 characters of shard 103 (document-level disjoint); same 98-symbol format, data under `data/cosmopedia/tinystories/` (gitignored, recovery note in `data/.gitignore`). BPE1024 retrained on the first 120M characters (`flm/prep_cosmopedia_bpe.py`): **2.59 characters/token** (TinyStories ~3.7), 1.157B train tokens, 1.93M validation tokens; the 768 most frequent tokens cover 98.4% of the stream.
**Baselines before any training** (`flm/corpus_baselines.py`, 300M training tokens, BPC on the first 1M validation characters): unigram 3.409, bigram 2.242, **trigram 1.622**. Registered prediction 2 (96K at 30M within 0.15 of the n-gram baseline) therefore reads: 96K BPC at 30M between 1.47 and 1.77 (50%).
Runs: `python -m flm.run_cosmo_sweep` (`v36-cosmo`): ref and tiered arms, seeds 0-4, same recipe (lr 6e-2, warm-up 100) as TinyStories (a transfer of the recipe, not a re-tune: lr might differ), tiered head frequencies from the corpus tokens (`TIER_FREQ`).
Additional predictions: tiered beats ref at 30M on >= 4/5 seeds (55%) (head 768 of 1024 tokens covers 98%, a smaller saving here than on TinyStories); ref BPC at 30M is between 1.3 and 1.8 (70%).

## C1/C2 outcome (`v36-cosmo`, 30M chars, seeds 0-4, BPE1024 retrained on Cosmopedia; paired within seed)
| arm | params | BPC (seeds 0-4) | mean |
|---|---|---|---|
| ref | 96,416 | 1.8716 / 1.8810 / 1.8748 / 1.8698 / 1.8680 | **1.873** |
| tiered t768e16m225 | 98,280 | 1.8619 / 1.8582 / 1.8518 / 1.8577 / 1.8503 | **1.856** |
Paired tiered - ref **-0.017, 5/5 seeds better**. Baselines on the same validation text: unigram 3.409, bigram 2.242, trigram 1.622 (fit on 300M tokens, ten times the 30M-char model's data).
Predictions: ref BPC between 1.3 and 1.8 (70%): **wrong** (1.873); within 0.15 of the n-gram baseline, i.e. 1.47-1.77 (50%): **wrong**: at 30M chars the 96K model is *0.25 above* a trigram table fit on more data, so on broad text the model is still data-starved, not saturated (TinyStories at 30M: 1.11). Tiered beats ref on >= 4/5 seeds (55%): **right**, 5/5. BPC well above TinyStories (>= 1.6, 85%): right.
Reading: Cosmopedia is far harder per character for this size (1.87 vs 1.11 at the same budget); the recipe lever (tiered) transfers. Whether 96K ever beats the trigram baseline on this corpus is the 300M/3B question; the decision rule says to compare against the baseline before reading anything into architecture.

### C3 registered before running (300M chars, seed 0, ref then tiered; `COSMO_CHARS=300e6 COSMO_SEEDS=0 python -m flm.run_cosmo_sweep`, ~30 min per arm)
Predictions: ref BPC between 1.45 and 1.65 (55%), i.e. it passes the 300M-token trigram (1.622) only just if at all (beats it: 45%); tiered beats ref by 0.01-0.03 (60%); no model reaches 1.40 (85%). Coherence is not rated until BPC is read.
