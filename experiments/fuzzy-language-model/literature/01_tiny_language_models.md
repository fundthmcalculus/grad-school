# 01 — Tiny causal language models on TinyStories: how small, measured how

Literature review for the fuzzy-language-model experiment (CPU-trained causal LM on TinyStories;
parameter-count study; linear vs softmax attention at tiny scale). Compiled 2026-10-08.
Every number below was read from the cited page on that date unless marked **(unverified)** or
**(our computation)**. Numbers marked "self-reported" come from a model card or README and were not
independently reproduced.

---

## 1. TinyStories (Eldan & Li, 2023)

**Source:** R. Eldan, Y. Li, *TinyStories: How Small Can Language Models Be and Still Speak
Coherent English?* <https://arxiv.org/abs/2305.07759>

**Dataset construction.** The authors assembled a vocabulary of "about 1500 basic words" meant to
mimic a 3–4-year-old's vocabulary, split into nouns, verbs and adjectives. Each generation prompt
draws one random verb, noun and adjective, plus a random subset of story features (dialogue, plot
twist, bad ending, moral value), and GPT-3.5 / GPT-4 writes a 3–5-paragraph story that uses them
(§2). The ~1500-word list constrains *generation prompts*, not the tokenizer — the stories
themselves are open text.

**HF dataset.** `roneneldan/TinyStories` (<https://huggingface.co/datasets/roneneldan/TinyStories>),
license CDLA-Sharing-1.0. The Hugging Face datasets-server reports **2,119,719 train / 21,990
validation stories**, ~1.0 GB of parquet (~2.0 GB in memory)
(<https://datasets-server.huggingface.co/size?dataset=roneneldan/TinyStories>). The repo also ships
`TinyStoriesV2-GPT4-{train,valid}.txt`. The dataset card says V2 is "based on generations by GPT-4
only (the original dataset also has generations by GPT-3.5 which are of lesser quality)." It also
says the paper's models were trained on `TinyStories-train.txt`. The V2 validation file we
downloaded holds 27,630 stories / 22.08 MB of UTF-8 text **(our computation)**.

**Models and tokenizer.** The paper uses the GPT-Neo architecture with "window size 256 and context
length 512" and the "GPT-Neo tokenizer but only keep[s] the top 10K most common tokens" (footnote 2).
Released checkpoints: TinyStories-1M/3M/8M/28M/33M/1Layer-21M/2Layers-33M and Instruct variants.
Sizes studied range from "roughly 1M and 35M parameters" with 1–8 layers, "trained on a single
V100 GPU within at most 30 hours" (§4). The largest model they trained on TinyStories had "roughly
80M parameters" (§3.1). Configs read from the HF `config.json` files:

| HF model | hidden | layers | heads | vocab in released config |
|---|---|---|---|---|
| [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M) | 64 | 8 | 16 | 50,257 |
| [TinyStories-3M](https://huggingface.co/roneneldan/TinyStories-3M) | 128 | 8 | 16 | 50,257 |
| [TinyStories-8M](https://huggingface.co/roneneldan/TinyStories-8M) | 256 | 8 | 16 | 50,257 |
| [TinyStories-28M](https://huggingface.co/roneneldan/TinyStories-28M) | 512 | 8 | 16 | 50,257 |
| [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M) | 768 | 4 | 16 | 50,257 |
| [TinyStories-1Layer-21M](https://huggingface.co/roneneldan/TinyStories-1Layer-21M) | 1024 | 1 | 16 | 50,257 |

**Pitfall — the names are not total parameter counts.** We parsed the released `pytorch_model.bin`
files, ignoring the causal-mask buffers and counting the tied LM head once **(our computation)**:

- TinyStories-1M: **3,745,984 total**, of which 3,216,448 sit in the token embedding (50,257 × 64)
  plus 131,072 in positional embeddings. That leaves **398,464 non-embedding** parameters.
- TinyStories-3M: **8,278,400 total**, 1,583,360 non-embedding.

So "1M" is neither the total count (3.7M) nor the non-embedding count (0.4M). It is roughly what you
get when you count a 10K-row embedding: 10,000 × 64 + 0.4M ≈ 1.04–1.17M. That fits footnote 2, but
the reading is our inference. SimpleStories (Finke et al. 2025, <https://arxiv.org/abs/2504.09184>,
§4) also flags that TinyStories model names "do not include embedding parameters." It recommends
that future work on the "smallest model that outputs grammatical English" count *all* parameters.

**Evaluation ("GPT-Eval").** GPT-4 is shown a story beginning and the model's completion, and asked
to grade it "as if those were stories written by students." It scores grammar, creativity,
consistency with the story's beginning, and (for Instruct) instruction-following and plot (§3).
Findings (§3.1):

- the grammar score plateaus earlier than the others;
- "grammar can be mastered by relatively small models", but consistency and creativity emerge
  only at larger size;
- consistency with the story's beginning "emerges when the hidden size of the model increases from
  64 to 128";
- 1-layer models struggle with instruction following, and 2 layers are partly sufficient;
- depth matters more for context-tracking, while embedding width matters more for factual
  knowledge (§4).

**Headline claim.** TinyStories lets models "below 10 million total parameters", or with "only one
transformer block", produce "fluent and consistent stories with several paragraphs that are diverse
and have almost perfect grammar" (abstract). The 1M/8-layer model does visibly worse: it "fails to
answer any factual prompt correctly, and often generates sentences that do not make sense or do not
follow the grammar" (§4). No validation-loss table is given per model. Loss appears only in
training curves (Fig. 3).

**Caveat on the "readability" story.** Lee & Berg-Kirkpatrick (2025,
<https://arxiv.org/abs/2510.13915>) build matched-structure datasets with different readability.
They find that readability alone "does not predict coherence or learning efficiency in SLMs." Their
stronger predictor is statistical simplicity (n-gram diversity). This bears on how we interpret
any "tiny model is coherent" result.

---

## 2. The ~19K-parameter model announced on Reddit — **FOUND: MacroStories**

| Field | Value | Source |
|---|---|---|
| Name | **MacroStories** | HF card |
| Link | <https://huggingface.co/raincandy-u/MacroStories> | — |
| Reddit post | <https://www.reddit.com/r/LocalLLaMA/comments/1wzp2ja/trained_a_20k_lm_probably_smallest_that_can_still/> (r/LocalLLaMA; the URL slug reads "Trained a 20K LM, probably smallest that can still …"). **We could not open the post itself:** Reddit returned 403 to our tools, so its full text is **(unverified)** | URL from the AGI Hunt summary below |
| Secondary coverage | AGI Hunt, "This 20K-parameter, 81KB language model can still write coherent 300-word stories", credited to "x_Raincandy_x · reddit · 2026-10-07" (<https://agihunt.info/en/p/1a1151173541939fc1b7f53ac97>) | — |
| Author | HF user `raincandy-u` (Reddit handle given as `x_Raincandy_x` by AGI Hunt — **(unverified)** on Reddit itself) | HF / AGI Hunt |
| Date | HF repo created 2026-10-07T05:11Z; Reddit post 2026-10-07 per AGI Hunt | HF API |
| Parameters | **19,969** total; FP32 weights = 81,364 bytes | HF card |
| Parameter breakdown | tied embedding/output 12,096 (= 378 × 32, **60.6 %** of the total); attention 3,072; SwiGLU 4,608; RMSNorm 160; inactive early-exit gate 33 | HF card |
| Architecture | ByteDance **Ouro**-style recurrent-depth Transformer: **one decoder block applied 4 times with shared weights**; hidden 32; SwiGLU intermediate 48; 2 query heads / 1 KV head (head dim 16); RoPE; pre+post RMSNorm; "value residual" (passes 2–4 mix value vectors 50/50 with pass 1); tied embeddings | HF card, `config.json` |
| Tokenizer / vocab | **378-token WordLevel** (word-level, not BPE/char), with numbered character-name markers that a helper maps to Alex/Robin/Casey | HF card |
| Context | `max_position_embeddings` = 2048; the generation example uses up to 2047 new tokens | `config.json`, card |
| Training data | **Not TinyStories.** Stories were "generated with a local Gemma model served through vLLM, using vocabulary constraints and numbered character markers", and Codex reviewed them | HF card |
| Training | 4,000 updates, batch 32, lr 3e-3, cosine with 5 % warmup, Muon on the 2-D decoder matrices + AdamW elsewhere, loss on the 4th recurrent pass, seed 456. **Hardware: one RTX 3090** (GPU, not CPU). Wall-clock time not stated. The whole workflow (38 experimental rounds) was run autonomously by Codex | HF card, `experiments/final_20k.json` |
| Reported loss / perplexity | **None reported.** | HF card |
| Reported evaluation | Codex-reviewed, 100 samples: goal resolved 98/100; no grammatical error 80/100; consistent characters/objects 91/100; length 100–300 words 82/100; **all four criteria 65/100** | HF card (self-reported) |
| CPU claim | AGI Hunt says it "runs fast on CPU with no GPU needed". That concerns inference; training was on a GPU | AGI Hunt |

**How comparable it is to our setup.** It is *TinyStories-style*, not a TinyStories-trained model.
It uses a custom distribution with a hard 378-word vocabulary, word-level tokens, and character
names collapsed to placeholders. That makes the task much easier than open TinyStories text. It
reports no likelihood metric, so it **cannot be placed on a loss/BPB axis** against anything below.
It shows what a *data-and-vocab-constrained* ~20K model can produce. It is not a benchmark number.

**Same author, earlier.** `raincandy-u/TinyStories-656K`
(<https://huggingface.co/raincandy-u/TinyStories-656K>, June 2024): Llama architecture, hidden 128,
2 layers, 8 heads / 4 KV heads, intermediate 384, tied embeddings, **2048-token BPE trained on
TinyStoriesV2**, context 512. The card gives no loss.

---

## 3. Other tiny reference points (TinyStories, char/byte-level, sub-100K)

### 3.1 Karpathy llama2.c `tinyllamas` (Llama-2 architecture, TinyStories)

From the README table (<https://github.com/karpathy/llama2.c>):

| model | dim | layers | heads | kv heads | ctx | params | val loss (nats/token) | vocab |
|---|---|---|---|---|---|---|---|---|
| stories260K | 64 | 5 | 8 | 4 | 512 | 260K | **1.297** | **512** (custom `tok512`) |
| stories15M | 288 | 6 | 6 | 6 | 256 | 15M | 1.072 | 32,000 (Llama 2) |
| stories42M | 512 | 8 | 8 | 8 | 1024 | 42M | 0.847 | 32,000 |
| stories110M | 768 | 12 | 12 | 12 | 1024 | 110M | 0.760 | 32,000 |

The stories260K readme (<https://huggingface.co/karpathy/tinyllamas/blob/main/stories260K/readme.md>)
gives its training config: batch 128, seq 512, lr 1e-3, dropout 0.05, wd 0.01, 100K iters,
`vocab_size=512`, `n_kv_heads=4` ("2X multiquery"). It "trained for ~10 minutes (?) on my A100"
and reaches val loss **1.2968**. The author's verdict on samples: "you can't expect too much from a
260K parameter model." The data is `TinyStories_all_data.tar.gz` (GPT-3.5 + GPT-4 stories), with
shard 0 held out (`tinystories.py`).

The README makes the vocabulary point directly. A 4096-token SentencePiece BPE trained on
TinyStories "creates integer sequences with about the same sequence length per example as the
default Llama 2 tokenizer of 32000 tokens."

**Converting to bits-per-byte (our computation).** Per-token loss is not comparable across
vocabularies. We tokenized the TinyStoriesV2-GPT4 validation file with each tokenizer:

- `tok512`: 2.06 UTF-8 bytes/token.
- Llama-2 32K: 3.77 bytes/token.

Using BPB = loss / ln 2 / (bytes per token), this gives approximately:

- stories260K ≈ **0.91 BPB**
- stories15M ≈ **0.41 BPB**
- stories42M ≈ **0.32 BPB**
- stories110M ≈ **0.29 BPB**

These are approximate because Karpathy's validation shard is from the v1 mix, not V2, and BOS tokens
are counted.

Embedding share in these models **(our computation; llama2.c ties input/output embeddings)**:

- stories260K: 512 × 64 = 32,768 embedding parameters, ≈ **13 %** of 260K.
- stories15M: 32,000 × 288 = 9.2M, ≈ **61 %** of 15M.

### 3.2 nanoGPT `shakespeare_char` (character level, not TinyStories)

From <https://github.com/karpathy/nanoGPT>:

- **GPU config:** 6 layers, 6 heads, `n_embd` 384, block size 256, dropout 0.2, lr 1e-3, 5000 iters.
  It trains in "about 3 minutes" on one A100 to a best val loss of **1.4697** nats/char, which is
  ≈ 2.12 bits/char **(our conversion)**.
- **CPU recipe:** 4 layers, 4 heads, `n_embd` 128, block 64, batch 12, 2000 iters, no dropout. It
  runs in "~3 minutes" and reaches loss **1.88** (≈ 2.71 bits/char).

This is the closest published "CPU in minutes" baseline, but on Tiny Shakespeare, not TinyStories.

### 3.3 SimpleStories (Finke et al., 2025)

<https://arxiv.org/abs/2504.09184>. A 2M-story synthetic dataset (English + Japanese) built as an
alternative to TinyStories. The model suite runs **1.25M to 35M parameters *including* embeddings**.
The smallest model has 4 layers, d_model 128, 4 heads, and a **4096-token custom WordPiece**
vocabulary. The abstract claims to "move the frontier regarding the fewest-parameter language model
that outputs grammatical natural language." The authors also replace GPT-2's 50,257 vocabulary with
the 4096 one and report large quality gains at a fixed parameter budget.

### 3.4 Community sub-1M / sub-100K TinyStories checkpoints (all self-reported, single seed)

| Model | Params | Tokenizer | Reported metric | BPB **(our conversion)** |
|---|---|---|---|---|
| [MicroT-test1-10K-TinyStories](https://huggingface.co/llaa33219/MicroT-test1-10K-TinyStories) (vanilla Transformer, RoPE) | 9,808 | bytes (256) | val PPL 4.31 (3 epochs) | ≈ 2.11 |
| MicroT-test1 50K / 100K / 300K / 1M (same card) | 49,888 / 97,872 / 297,680 / 996,736 | bytes | val PPL 2.60 / 2.28 / 1.90 / 1.69 | ≈ 1.38 / 1.19 / 0.93 / 0.76 |
| [MicroMixer-4-10K-TinyStories](https://huggingface.co/llaa33219/MicroMixer-4-10K-TinyStories) (attention-free MLP-mixer) | 9,666 | bytes | val PPL 4.06 | ≈ 2.02 |
| [wskan-10k-tinystories](https://huggingface.co/llaa33219/wskan-10k-tinystories) (KAN/SSM) | 12,546 | bytes | eval loss 1.4401 | ≈ 2.08 |
| wskan-100k / 1m (same card) | 113,612 / 985,956 | bytes | eval loss 0.8224 / 0.6135 | ≈ 1.19 / 0.89 |
| [eeny-tinystories-999k](https://huggingface.co/sprapp/eeny-tinystories-999k) (dim 88, 7 layers, GQA, distilled) | 999,328 | 4096 BPE | **0.625 BPB**; the card also reports **0.707 BPB for TinyStories-1M** on held-out val | — |

The MicroT/MicroMixer cards say they sampled "~200K stories" of TinyStories, flattened to 1024-byte
sequences, and trained for 3 epochs. Their numbers are therefore not trained to convergence on the
full corpus, and the validation set may differ from the official one. Treat this whole table as
order-of-magnitude anchors, not benchmarks.

### 3.5 Hardware-novelty ports (useful only as context)

The stories260K checkpoint has been ported to a Game Boy Color
(<https://github.com/maddiedreese/gbc-transformer>) and to a classic ESP32
(<https://github.com/serenustaken/tinyllm-esp32-no-psram>). JAM, a family of 2–80 KB language
models on an Atari 800 (<https://github.com/marspa73/atarijam>), is **not** TinyStories-trained and
publishes no parameter counts or losses.

### 3.6 BabyLM

BabyLM caps *training data* (10M/100M words), not parameter count. SimpleStories §6 notes that the
challenge "received many submissions with 10M-100M parameters." It is not directly relevant to a
sub-1M parameter study.

---

## 4. Scaling laws at small N; embeddings and vocabulary

**Kaplan et al. 2020** (<https://arxiv.org/abs/2001.08361>):

- They define model size N as **non-embedding parameters**:
  N ≈ 2·d_model·n_layer·(2·d_attn + d_ff), excluding vocabulary and positional embeddings (§2.1).
- The fit is L(N) = (N_c/N)^α_N with α_N ≈ 0.076 and N_c ≈ 8.8×10¹³ (Eq. 1.1).
- Model sizes range "from 768 to 1.5 billion non-embedding parameters" (§3). The smallest models
  are therefore in our regime.
- Fig. 6 shows that including embedding parameters makes the size trend depend on depth. Excluding
  them collapses the curves onto one line.
- Training compute is approximated as C ≈ 6·N·B·S (non-embedding).

**Hoffmann et al. 2022 (Chinchilla)** (<https://arxiv.org/abs/2203.15556>):

- Over 400 models from **70M** to over 16B parameters, trained on 5–500B tokens.
- Compute-optimal training scales N and D equally. Table 3 gives 400M params ↔ 8.0B tokens, i.e.
  the familiar **~20 tokens per parameter**.
- Their fits do not go below 70M, so applying them to 20K–1M models is extrapolation.

**Reconciling the two.** Pearce & Song 2024 (<https://arxiv.org/abs/2406.12907>) attribute much of
the Kaplan/Chinchilla gap (N_opt ∝ C^0.73 vs C^0.50) to "Kaplan counting non-embedding rather than
total parameters, combined with their analysis being performed at small scale." Porian et al. 2024
(<https://arxiv.org/abs/2406.19146>) point to last-layer compute cost, warmup duration and
scale-dependent optimizer tuning. They add that "tuning the AdamW β₂ parameter is essential at lower
batch sizes," which matches Karpathy's use of β₂ = 0.99 for tiny models.

**Very small N, recent.** Romanyukov et al. 2026 (<https://arxiv.org/abs/2609.27581>) test whether
"Step Law" learning-rate and batch-size optima transfer below 59M parameters. They use a
nanoGPT/TinyStories pipeline with a **2048-token BPE**, 935 runs and 29 (N, D) cells. The power-law
form holds, but with different coefficients. We have not read this paper beyond its abstract, so
any coefficients taken from it should be checked against the full text.

**Vocabulary as a scaling variable.** Tao et al. 2024, *Scaling Laws with Vocabulary*
(<https://arxiv.org/abs/2407.13623>):

- They split N = N_nv + N_v, with N_v = V·d, and train models from 33M to 3B parameters.
- They find N_v,opt ∝ N_nv^γ with **γ ≈ 0.83 < 1**, and the optimal vocabulary grows with compute.
- The corollary for us: at tiny compute, the optimal vocabulary is *small*. A 50K-row embedding in
  a ~1M model is badly over-allocated (see TinyStories-1M: 86 % of its parameters are token
  embedding).
- To compare vocabularies they use a unigram-normalized loss and report it correlates with **BPC**.

**Tokenizer-independent metric.** Gao et al. 2020, *The Pile* (<https://arxiv.org/abs/2101.00027>,
§4) define **bits per UTF-8 byte**: BPB = (L_T/L_B)·ℓ/ln 2, where ℓ is mean loss per token, L_T
the token count and L_B the byte count. They prefer it to bits-per-character because characters
are ill-defined across Unicode. For ASCII-only TinyStories, BPB and BPC coincide almost exactly.

---

## 5. Interpretability work at TinyStories scale

- **Eldan & Li §5** (<https://arxiv.org/abs/2305.07759>):
  - They analyse a 1-layer, d=1024, 16-head model and separate distance-based (positional) heads
    from semantic heads that attend to e.g. the subject or main topic.
  - MLP neurons in the 1M (d=64) model fire on interpretable token roles, such as the protagonist's
    introduction or adjectives. Analogous neurons in GPT-2-XL are less clean.
  - The authors call this preliminary.
- **Self-ablating Transformers** (Ferrao et al. 2025, <https://arxiv.org/abs/2505.00509>):
  - A k-winner-takes-all self-ablation applied during training of small TinyStories models.
  - It gives "more localized circuits … increased neuron specialization without compromising
    language modelling performance."
- **Matryoshka SAEs** (Bussmann, Nabeshima, Karvonen, Nanda 2025,
  <https://arxiv.org/abs/2503.17547>): SAEs trained on Gemma-2-2B *and TinyStories* models, with
  less feature absorption.
- **SimpleStories** (<https://arxiv.org/abs/2504.09184>): claims improved interpretability over
  TinyStories-trained models. Its probing analysis shows the 1.25M model "is often only able to
  achieve 50 % relative accuracy."
- **Explaining Attention with Program Synthesis** (Hayes, Li, Andreas 2026,
  <https://arxiv.org/abs/2606.19317>): fits executable programs to attention heads, reaching
  > 75 % IoU on TinyStories inputs. The models studied are GPT-2/TinyLlama/Llama-3B, not tiny
  models.

We found no mechanistic study of a sub-100K-parameter TinyStories model.

---

## 6. Implications for our experiment

**Reference numbers to quote (with their caveats).** Report everything in **BPB on the official
TinyStoriesV2-GPT4 validation file**.

| Reference | Params | ≈ BPB | Status |
|---|---|---|---|
| llama2.c stories110M | 110M | 0.29 | our conversion |
| stories42M | 42M | 0.32 | our conversion |
| stories15M | 15M | 0.41 | our conversion |
| eeny (distilled) | 1.0M total | 0.625 | self-reported |
| TinyStories-1M | 3.7M total / 0.4M non-embedding | 0.707 | self-reported by the eeny author |
| byte-level ~1M | ~1M | ≈ 0.76–0.89 | community, ~200K-story subset |
| stories260K | 260K | 0.91 | our conversion |
| byte-level ~100K | ~100K | ≈ 1.19 | community |
| byte-level ~10K | ~10K | ≈ 2.0–2.1 | community |

Bigram/unigram character baselines would sit above these; compute them ourselves as a floor.

**Where "coherent" starts.**

- stories260K (≈ 0.9 BPB) yields grammatical but plot-inconsistent stories.
- Eldan & Li report that consistency emerges between hidden size 64 and 128.
- MacroStories shows that ~20K parameters *can* produce rule-satisfying stories, but only on a
  378-word, placeholder-name, Gemma-generated distribution with no reported likelihood.
- A defensible statement for us: *on open TinyStories text, published sub-100K models stay above
  ~1.2 BPB and none has a coherence evaluation.* That leaves the 10K–300K range open for a
  controlled study.

**Tokenizer recommendation for a parameter-count study.**

1. Primary: **byte-level (V = 256)** or a **small BPE (V = 512)**, as in stories260K. Embedding cost
   is then V·d: 256 × 32 ≈ 8K parameters, so most of the budget goes to the mixing layers being
   compared (linear vs softmax attention).
2. A 4096-vocabulary BPE is a reasonable secondary arm at ≥ 1M parameters. It costs 4096·d
   parameters (131K at d = 32), which would swamp a 20K–100K model.
3. Never use the 50K GPT-2/GPT-Neo vocabulary below ~10M parameters.

**Pitfalls.**

1. **Parameter accounting.** Report total, embedding, and non-embedding counts separately, and
   state whether embeddings are tied. TinyStories model names follow none of these conventions
   (1M ≈ 3.7M total). Kaplan-style fits use non-embedding N; Chinchilla uses total N, and the choice
   alone changes fitted exponents at small scale (Pearce & Song).
2. **Cross-tokenizer comparison.** Per-token loss/perplexity is not comparable across vocabularies.
   Always convert to BPB using the byte count of the *same* text.
3. **Dataset version.** v1 (GPT-3.5 + GPT-4) and V2 (GPT-4 only) are different distributions; most
   community numbers do not say which they used. Pin one (V2 valid), and hash it in provenance.
4. **Subsets and epochs.** Several sub-1M checkpoints trained on ~200K-story subsets for 3 epochs.
   Chinchilla's ~20 tokens/param means a 100K model is "compute-optimal" at only ~2M tokens. In
   practice tiny models keep improving far past that, so fix a token budget, report it, and sweep
   it.
5. **Recurrent/shared-weight tricks change the meaning of "parameters".** MacroStories applies one
   block 4× and so spends 4× the FLOPs of its parameter count. Report FLOPs/token alongside
   parameters when comparing attention variants.
6. **Coherence judged by an LLM** (GPT-Eval, Codex review) is not reproducible across grader
   versions. Pair it with BPB, and with a fixed rubric and seed if used at all.
7. **Optimizer settings matter at tiny scale.** Use a high LR (1e-3 to 3e-3) and β₂ ≈ 0.99, per
   Karpathy and Porian et al. Tune them per size, or the size comparison will measure tuning rather
   than capacity.
8. **Seeds.** Every community number above is a single seed. Following this repository's protocol,
   our own numbers should be mean ± std over seeds 0–9.

---

## BibTeX

```bibtex
@article{eldan2023tinystories,
  title   = {TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author  = {Eldan, Ronen and Li, Yuanzhi},
  journal = {arXiv preprint arXiv:2305.07759},
  year    = {2023}
}
@article{kaplan2020scaling,
  title   = {Scaling Laws for Neural Language Models},
  author  = {Kaplan, Jared and McCandlish, Sam and Henighan, Tom and Brown, Tom B. and Chess, Benjamin and Child, Rewon and Gray, Scott and Radford, Alec and Wu, Jeffrey and Amodei, Dario},
  journal = {arXiv preprint arXiv:2001.08361},
  year    = {2020}
}
@article{hoffmann2022chinchilla,
  title   = {Training Compute-Optimal Large Language Models},
  author  = {Hoffmann, Jordan and Borgeaud, Sebastian and Mensch, Arthur and Buchatskaya, Elena and Cai, Trevor and Rutherford, Eliza and de Las Casas, Diego and Hendricks, Lisa Anne and others},
  journal = {arXiv preprint arXiv:2203.15556},
  year    = {2022}
}
@article{tao2024vocab,
  title   = {Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies},
  author  = {Tao, Chaofan and Liu, Qian and Dou, Longxu and Muennighoff, Niklas and Wan, Zhongwei and Luo, Ping and Lin, Min and Wong, Ngai},
  journal = {arXiv preprint arXiv:2407.13623},
  year    = {2024}
}
@article{pearce2024reconciling,
  title   = {Reconciling Kaplan and Chinchilla Scaling Laws},
  author  = {Pearce, Tim and Song, Jinyeop},
  journal = {arXiv preprint arXiv:2406.12907},
  year    = {2024}
}
@article{porian2024resolving,
  title   = {Resolving Discrepancies in Compute-Optimal Scaling of Language Models},
  author  = {Porian, Tomer and Wortsman, Mitchell and Jitsev, Jenia and Schmidt, Ludwig and Carmon, Yair},
  journal = {arXiv preprint arXiv:2406.19146},
  year    = {2024}
}
@article{gao2020pile,
  title   = {The Pile: An 800GB Dataset of Diverse Text for Language Modeling},
  author  = {Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and others},
  journal = {arXiv preprint arXiv:2101.00027},
  year    = {2020}
}
@article{finke2025simplestories,
  title   = {Parameterized Synthetic Text Generation with SimpleStories},
  author  = {Finke, Lennart and Sreedhara, Chandan and Dooms, Thomas and Allen, Mat and Zhang, Emerald and Rodriguez, Juan Diego and Nabeshima, Noa and Marshall, Thomas and others},
  journal = {arXiv preprint arXiv:2504.09184},
  year    = {2025}
}
@article{lee2025readability,
  title   = {Readability $\ne$ Learnability: Rethinking the Role of Simplicity in Training Small Language Models},
  author  = {Lee, Ivan and Berg-Kirkpatrick, Taylor},
  journal = {arXiv preprint arXiv:2510.13915},
  year    = {2025}
}
@article{romanyukov2026steplaw,
  title   = {Does Step Law Transfer to Small-Scale Language Models? An Empirical Recalibration Below 59M Parameters},
  author  = {Romanyukov, Egor and Novikov, Timofey and Shokarov, Timur and Zorkina, Elizaveta and Palienko, Anastasia and Dergachev, Stepan},
  journal = {arXiv preprint arXiv:2609.27581},
  year    = {2026}
}
@article{ferrao2025selfablating,
  title   = {Self-Ablating Transformers: More Interpretability, Less Sparsity},
  author  = {Ferrao, Jeremias and Mikaelson, Luhan and Pepper, Keenan and Perez-Campanero Antolin, Natalia},
  journal = {arXiv preprint arXiv:2505.00509},
  year    = {2025}
}
@article{bussmann2025matryoshka,
  title   = {Learning Multi-Level Features with Matryoshka Sparse Autoencoders},
  author  = {Bussmann, Bart and Nabeshima, Noa and Karvonen, Adam and Nanda, Neel},
  journal = {arXiv preprint arXiv:2503.17547},
  year    = {2025}
}
@article{hayes2026programsynthesis,
  title   = {Explaining Attention with Program Synthesis},
  author  = {Hayes, Amiri and Li, Belinda Z. and Andreas, Jacob},
  journal = {arXiv preprint arXiv:2606.19317},
  year    = {2026}
}
```

Software / model references (not papers): llama2.c <https://github.com/karpathy/llama2.c>;
nanoGPT <https://github.com/karpathy/nanoGPT>; MacroStories
<https://huggingface.co/raincandy-u/MacroStories>; TinyStories dataset
<https://huggingface.co/datasets/roneneldan/TinyStories>.
