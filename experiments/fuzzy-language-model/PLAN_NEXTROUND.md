# PLAN_NEXTROUND: train the all-fuzzy model more efficiently, then test it on a larger dataset (drafted 2026-10-21; plan only, nothing run)

Owner's goal for the round: "train efficiently and improve model quality. We can switch to another dataset if needed." The 300M budget stays the default maximum; the 1B-character run of 2026-10-21 was a single exception at the owner's request, so any further 1B run needs a fresh yes.

## 1. Where we stand (all measured; sources in `PLAN_FUZZYLOCAL.md`)

| fact | number |
|---|---|
| Best all-fuzzy model (`e2e_tail`, 99,019 parameters, lr 6e-2, warm-up 1000) | 0.98929 bits per character (BPC) at 300M, 0.97486 at 1B (two seeds) |
| Hybrid recipe (98,528 parameters) | 0.95887 at 300M, 0.94526 at 1B |
| Gap to the hybrid | 0.030 at 300M, 0.0296 at 1B, 0.082 at 30M: **constant from 300M up**; both models gain about 0.014 per threefold of data |
| Fully fuzzy embedding (both token tables) | +0.011 BPC over the crisp-head all-fuzzy model at 300M |
| Wall-clock | all-fuzzy step 83.5 ms against 75.2 ms; 746-761 s against 721-744 s per 300M run; the extra is kernel-launch overhead (about 2,700 launches per step against 1,400); with the step CUDA-graphed it is 75.5 against 73.4 ms |
| Learning per character | training-loss gap to the hybrid: 1.34 nats at step 100, 0.35 at 1000, 0.18 at 5000, 0.114 at the end of 300M (warm-up 1000 against 100 is a confound) |
| Levers already closed | projection variants (stage 10: none beat the reference), head repairs (every 30M gain except signed memberships vanished at 300M), more blocks, more data |
| Coherence at 1B (sets 20-21, pooled, 2 raters) | all-fuzzy 7.67 / 7.50 of 15, hybrid 7.88 / 7.97: indistinguishable at this sample size |

Two readings drive the plan. (a) The gap is **not data-limited on TinyStories**: it is a constant cost, so more TinyStories cannot show whether it is a representation cost or an optimiser cost. (b) TinyStories is nearly saturated at 99K parameters (0.98 BPC from 1B to 3B characters for the earlier 96K model), so quality changes of a few hundredths are hard to separate from noise there. A broader dataset is the way to test whether the 3% gap is a TinyStories effect.

## 2. Track A: training efficiency (TinyStories; every number stays comparable with all earlier stages)

**A1. CUDA-graph wrapper** (`flm/run_fuzzy_graph.py`, a runner like `run_allfuzzy.py`; `train.py` and `models.py` stay untouched).
Capture forward, backward, gradient clip and the optimiser step as one graph with static input and a learning rate held as a tensor (`AdamW(capturable=True)`); replay per step; evaluation stays eager.
- Manipulation check before any claim: the graphed and eager runs must follow the same loss curve (max |difference| in the first 200 logged losses below 1e-3) and end within 0.002 BPC at 30M; launch counts per step are printed.
- Cost: about half a day of work, then one 30M cell per arm (75 s eager, about 63 s graphed).
- Expected saving: 15-19% on the fuzzy model, about 4% on the hybrid (measured on a synthetic batch; the real loop also has batch sampling and clipping).

**A2. Rule-parameter learning rates** (the slow-start lever, `flm/run_fuzzy_groups.py`).
Parameter groups by name: rule centres and log-widths (`centers`, `log_width`), consequents (`slope`, `s`, `conseq`), gains and everything else. Arms at 30M, five seeds, warm-up 600, lr 6e-2 base, paired with `lr6_wu600` (1.1207):
| arm | centres and widths | consequents | rest |
|---|---|---|---|
| g_c03 | ×0.3 | ×1 | ×1 |
| g_c3 | ×3 | ×1 | ×1 |
| g_s3 | ×1 | ×3 | ×1 |
| g_all_wd | all ×1, weight decay 0 on rule parameters (they are already excluded for centres) | | |
Cost: 4 arms × 5 seeds × 75 s ≈ 25 minutes. (An earlier message of mine said "about 20 minutes per arm"; that was wrong, it is about 6 minutes per arm.)

**A3. Warm-up shape** (only if A2 shows a lever): linear warm-up 1000 against a shorter warm-up with centres and widths frozen for the first 300 steps, 30M, five seeds, about 12 minutes.

**A4. Decision rule for the track.** A gain counts if it beats `lr6_wu600` by at least 0.005 BPC on at least four of five seeds. Anything that passes goes to one 300M confirmation, two seeds, warm-up 1000, against 0.98929 (about 25 minutes). The head gains showed that 30M gains can vanish at 300M, so no 30M gain is quoted without it.

**A5. Registered predictions (Track A).**
1. A1 reproduces the eager loss curve within 1e-3 over the first 200 logged losses (85%).
2. A1 saves 12-20% wall-clock on the all-fuzzy model (70%) and 2-6% on the hybrid (60%).
3. At least one A2 arm beats `lr6_wu600` by 0.005 on four of five seeds (30%).
4. If one does, it keeps at least half its gain at 300M (40%, from the pattern of stages 6-9).
5. The gap to the hybrid at 300M falls by at least 0.01 from any Track A change (15%).

## 3. Track B: a larger dataset

### 3.1 Dataset options

Verified items cite the source I checked; "from earlier work" items come from this repository's own logs; "unverified" items I have not checked and must be before any download.

| # | dataset | what it is | size | licence | difficulty for 99K | status / cost |
|---|---|---|---|---|---|---|
| 1 | **SimpleStories** (`SimpleStories/SimpleStories`) | about 2M short stories in simple language written by `gpt-4o-mini`, parameterised over genre, theme, style and vocabulary level (Finke et al., NeurIPS 2025 Datasets and Benchmarks, arXiv:2504.09184) | 2M stories (Hugging Face bracket 1M-10M; byte and character totals unverified) | MIT (verified on the dataset card) | **slightly harder than TinyStories**: same family, broader vocabulary and plots | download and prepare; cost: hours of CPU, no new tooling beyond `prep_*` |
| 2 | **TinyStories + SimpleStories mix** | 50/50 by characters, evaluated on both validation sets | by construction | MIT and CDLA | between 1 and TinyStories | `flm/prep_mixed.py` already builds mixes (`data/mixed` exists from earlier work) |
| 3 | **Cosmopedia v2** (already built) | LLM-written textbooks, stories, how-to and web articles | 3.0B characters on disk, pinned revision `3ba9d605` | ODC-By | **much harder**: from earlier work the 96K model scores 1.873 BPC at 30M characters (TinyStories 1.11) and sits 0.25 above a trigram table; 1.2M-parameter models need 1B characters to reach 1.13 | no preparation (the BPE for it exists at 1024; a BPE3072 would be trained) |
| 4 | **TinyStories + Cosmopedia mix** (already built, `data/mixed`) | breadth against story quality | built | mixed | harder | no preparation; earlier `mixed_30M` checkpoints exist |
| 5 | **BabyLM 100M-word strict corpus** | about 100M words of child-directed speech, children's books, Gutenberg, subtitles, Simple English Wikipedia and Switchboard (BabyLM challenge call and 2024-2025 findings papers) | about 100M words (roughly 500M characters) | **unverified** (each source carries its own terms) | hard and heterogeneous (dialogue, archaic prose) | download and filter; mixed registers need a per-source report |
| 6 | **Simple English Wikipedia** | simple-language encyclopedia, plain text | about 15M words as the BabyLM share; whole dump size unverified | CC BY-SA | factual register; small | cleaning of markup; small enough for several epochs |
| 7 | **FineWeb-Edu `sample-10BT`** | educational web pages | 10B tokens, about 28 GB of parquet (Hugging Face listing) | ODC-By | **very hard** for this size; the model would remain data-starved | large download; not a "slightly larger" step |

**Recommendation.** Take **#1 SimpleStories** as the primary dataset: it is the smallest change from what we have (same synthetic-story family, LLM-written, simple register) while adding the diversity TinyStories lacks, so it is the cleanest test of whether the 3% gap is a TinyStories effect. Add **#2 (the mix)** as the breadth-cost arm. Keep **#3/#4** as the far end of the ladder, since they exist already and cost no preparation. Leave #5-#7 for later: each changes register as well as size, which confounds the question.

### 3.2 Protocol (same for any chosen dataset)

1. **Pin and prepare**: record the revision, fetch under `$GRAD_SCHOOL_DATA/<name>/`, add a recovery note in `data/.gitignore`, validation split by document, an ASCII filter with the dropped fraction reported (nothing over 5 MB committed).
2. **Tokenizer**: train BPE3072 on the new corpus (the current recipe, with the tiered 384-head / 12-prototype embedding; head frequencies from the new tokens). Compare in bits per character on identical text.
3. **C0, minutes, before anything trains**: score the existing TinyStories checkpoints (hybrid and all-fuzzy at 300M and 1B) on the new validation slice, and fit unigram/bigram/trigram tables on a comparable amount of text (`flm/corpus_baselines.py`). The trivial baseline comes before any claim.
4. **Reference first**: the hybrid recipe on the new corpus at 30M (five seeds) and 300M (two seeds). All-fuzzy numbers are meaningless without it, since TinyStories-tuned settings may not transfer.
5. **All-fuzzy at 30M** (five seeds, `lr6_wu600`) and **300M** (two seeds, warm-up 1000), recipe transferred and not re-tuned, with one learning-rate arm (3e-2 and 6e-2) per model, both models given the same freedom.
6. **Quality metrics**: BPC; gap to the hybrid at 30M and 300M; blind coherence (two raters, fresh seeds, scores written before the key, at least 16 texts per model so the sampling noise measured in sets 20-21, a standard error of about 0.7 for 8 texts, is smaller than the effect).
7. **Decision rule**: if the gap to the hybrid is below 0.015 BPC on the new corpus at 300M, the 3% gap is a TinyStories effect and the write-up changes; if it is above 0.04, the gap grows with breadth and the representation cost is real; in between, report it as unchanged.

### 3.3 Compute (one RTX 4080, one cell at a time)

| step | cells | time |
|---|---|---|
| preparation (SimpleStories download, filter, BPE3072, frequency file) | CPU | 1-3 hours, parallel to GPU work |
| C0 baselines | 6 scores | minutes |
| hybrid reference 30M | 5 | about 6 min |
| all-fuzzy 30M, two learning rates | 10 | about 13 min |
| 300M confirmations (hybrid, all-fuzzy) | 4 | about 50 min |
| coherence rating (sets of 32-48 texts, two sub-agent raters) | 1 | minutes |
Total about 1.5 GPU hours, within the 300M budget. A 1B run on the new corpus is **not** planned.

### 3.4 Registered predictions (Track B; SimpleStories as the corpus, to be revised if the owner picks another)
1. Hybrid at 30M on SimpleStories is harder than TinyStories (1.1) but easier than Cosmopedia (1.87): between 1.2 and 1.6 BPC (70%).
2. The all-fuzzy gap to the hybrid at 30M is between 0.04 and 0.15 (65%; TinyStories 0.082).
3. The gap at 300M is between 0.02 and 0.05 (55%); below 0.015 (10%); above 0.05 (25%).
4. Blind coherence of the all-fuzzy model is within 1 point of the hybrid, pooled over at least 16 texts (65%).
5. The TinyStories-tuned warm-up (1000) is still within 0.01 BPC of the best of the two tried (55%).

## 4. Order of work and what I need from the owner

1. **Now, CPU**: SimpleStories fetch and preparation (needs a yes on a download of unknown size, probably several GB; I will state the size before downloading), C0 baselines for the existing checkpoints, `run_fuzzy_graph.py`.
2. **GPU, short**: A1 numerics check (minutes), then A2 (25 minutes).
3. **GPU, after preparation**: Track B reference and all-fuzzy runs (about 1 hour).
4. **Write-up** into `PLAN_FUZZYLOCAL.md` with each prediction scored; dashboard and demo updated if a model changes.

Decisions needed: (1) the dataset (recommendation #1, with #2 as the second arm); (2) a yes for the download once its size is known; (3) whether the CUDA-graph wrapper goes in the PR flow or on a separate branch.

## 5. Risks and what would change the plan
- **Hybrid reference may need re-tuning** on a new corpus (the recipe was tuned on TinyStories): the reference gets the same learning-rate freedom as the fuzzy model, or the comparison is biased against the fuzzy model.
- **A larger corpus can make both models data-starved** (the trigram baseline beat the 96K model on Cosmopedia at 30M): in that case the 30M screen says little, and the 300M cell decides.
- **CUDA graphs and the tiered embedding's `index_copy`** may not capture cleanly; A1's fallback is capturing only the block stack and leaving the head eager.
- **Rule-parameter learning rates may interact with the 1000-step warm-up** (centres and widths move early): A3 exists for that.
- **The mid-run coherence noise** (±1.5 points on 8 texts) means no coherence claim on fewer than 16 texts per model.
