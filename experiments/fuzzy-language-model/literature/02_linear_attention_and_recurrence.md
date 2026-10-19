# 02 — Linear attention and recurrent alternatives to softmax attention

Literature review for the tiny-model (≈5K–1M params, CPU, TinyStories) softmax vs. linear/recurrent
comparison, aimed at an eventual recurrent fuzzy-inference-system (FIS) language model.
Compiled 2026-10-08. Every arXiv ID below was checked against the arXiv API on that date;
equations and quoted numbers were read off the papers' HTML/PDF unless marked **(unverified)**.

Notation: causal step `t`, query/key/value `q_t, k_t, v_t`, feature map `φ`, matrix state `S_t`,
normaliser state `z_t`.

---

## 1. Attention as kernel smoothing

**Tsai et al. 2019, "Transformer Dissection"** ([arXiv:1908.11775](https://arxiv.org/abs/1908.11775))
recast attention as "applying kernel smoother over the inputs with the kernel scores being the
similarities between inputs", and used that view to study positional embeddings and to propose
attention variants built from different kernels.

The kernel smoother here is the **Nadaraya–Watson** estimator:

    ŷ(q) = Σ_j κ(q, k_j) v_j / Σ_j κ(q, k_j)

Softmax attention is this estimator with `κ(q,k) = exp(qᵀk/√d)`. The link to a **Gaussian kernel**
follows from expanding the square:

    exp(qᵀk/τ) = exp(-‖q−k‖²/(2τ)) · exp(‖q‖²/(2τ)) · exp(‖k‖²/(2τ))

The Performer paper uses the same identity, written `SM(x,y) = exp(‖x‖²/2) K_gauss(x,y) exp(‖y‖²/2)`
([arXiv:2009.14794](https://arxiv.org/abs/2009.14794)). The factor that depends on `q` cancels between
numerator and denominator. If keys are also held at a constant norm (QK-norm / L2-normalised q,k), the
factor that depends on `k` is constant too, so **softmax attention becomes exactly Nadaraya–Watson
regression with an isotropic Gaussian kernel of bandwidth √τ**. In FIS terms, each past token is a
rule with a Gaussian membership centred at `k_j` and a constant consequent `v_j`. The normalised
output is a zero-order TSK / normalised RBF network. (The algebra above is elementary; the
TSK reading is ours.)

## 2. Linear attention

### 2.1 Katharopoulos et al. 2020, "Transformers are RNNs" ([arXiv:2006.16236](https://arxiv.org/abs/2006.16236))
Replace `κ` with a factorisable kernel `φ(q)ᵀφ(k)` and use associativity, which turns causal attention
into an RNN with a matrix state:

    S_t = S_{t-1} + φ(k_t) v_tᵀ          z_t = z_{t-1} + φ(k_t)
    y_t = φ(q_t)ᵀ S_t / (φ(q_t)ᵀ z_t),   φ(x) = elu(x) + 1

This costs O(N) in time and O(1) memory per generated token. The paper reports quality "similar"
to vanilla transformers and up to 4000× faster autoregressive generation on very long sequences.

### 2.2 Performer / FAVOR+ (Choromanski et al. 2021) ([arXiv:2009.14794](https://arxiv.org/abs/2009.14794))
Gives an unbiased estimate of the softmax kernel using **positive orthogonal random features**:

    φ(x) = exp(−‖x‖²/2)/√m · [exp(ω_1ᵀx), …, exp(ω_mᵀx)],  ω_i ~ N(0, I_d), orthogonalised

The row normaliser `D̂⁻¹ = diag(Q′(K′ᵀ1))⁻¹` is kept. Positivity avoids the instability of
trigonometric features.

### 2.3 Random Feature Attention (Peng et al. 2021) ([arXiv:2103.02143](https://arxiv.org/abs/2103.02143))
Also uses random features for the softmax kernel, and adds an optional **gate** that gives a recency
bias: `S_t = g_t S_{t-1} + (1−g_t) φ(k_t)⊗v_t`, with the same update for `z_t`. The gate equation is
from memory of the paper; the abstract confirms "an optional gating mechanism" for recency bias
**(equation form unverified)**.

### 2.4 cosFormer (Qin et al. 2022) ([arXiv:2202.08791](https://arxiv.org/abs/2202.08791))
Argues that two properties of softmax matter: a non-negative attention matrix and a non-linear
re-weighting that concentrates it. It keeps `φ = ReLU` and adds a decomposable cosine locality
re-weighting, `ReLU(q_i)ᵀReLU(k_j) · cos(π/2 · (i−j)/M)` **(exact form unverified)**.

### 2.5 Fast weight programmers and the delta rule (Schlag, Irie, Schmidhuber 2021) ([arXiv:2102.11174](https://arxiv.org/abs/2102.11174))
- Shows linear attention is formally a 1990s fast weight programmer. Because a purely additive
  update has finite memory capacity, it proposes the **delta rule**:
  `W_t = W_{t-1} + β_t (v_t − v̄_t) ⊗ φ(k_t)`, where `v̄_t = W_{t-1}φ(k_t)` is the value currently
  retrieved.
- Introduces the DPFP feature map: products of ReLU-ed `[k; −k]` entries.
- Normalisation: the paper replaces the attention denominator with **sum normalisation**
  (`φ′(x) = φ(x)/Σ_j φ(x)_j`, applied to q and k). Its stated reason is that the accumulated
  denominator "always grows with the number of steps, and may result in instability".
- WikiText-103 (Table 2), small config (D=128, ~40M params), test perplexity: Transformer 34.1,
  linear transformer 38.3, delta network 35.5, Performer 39.6 (37.2 with the delta rule).
  Medium config (~90M): 29.6 / 33.0 / 31.5. The delta rule recovers most of the gap; the gap is
  roughly constant in absolute perplexity between 40M and 90M.

### 2.6 DeltaNet at scale and Gated DeltaNet (Yang et al. 2024; 2025)
- **DeltaNet** ([arXiv:2406.06484](https://arxiv.org/abs/2406.06484)) parallelises
  `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` over sequence length.
  - Swaps elu+1 for SiLU and L1 (sum) normalisation for **L2 normalisation** of q,k. With β=1,
    `I − kkᵀ` is then a projection.
  - Uses no denominator.
  - At 340M params / 15B tokens it is competitive with Transformer++, Mamba and GLA, and solves MQAR
    in the paper's settings.
- **Gated DeltaNet** ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464), ICLR 2025) adds a decay,
  `S_t = α_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`. The paper reports that "gating enables rapid
  memory erasure while the delta rule facilitates targeted updates", and that the model beats Mamba-2
  and DeltaNet at 400M/1.3B.

### 2.7 Gated Linear Attention (Yang et al. 2023) ([arXiv:2312.06635](https://arxiv.org/abs/2312.06635))
`S_t = (α_tᵀ1) ⊙ S_{t-1} + k_tᵀ v_t`, with a data-dependent gate `α_t = σ(x_t W_α1 W_α2)^{1/τ}`
(τ=16), identity φ, and **no normaliser**. The paper notes that "a linear kernel (i.e., setting ϕ to
be the identity) without a normalizer works well in practice". It is competitive with a LLaMA-style
Transformer++ at 340M/1.3B, and also introduced FlashLinearAttention.

### 2.8 RetNet (Sun et al. 2023) ([arXiv:2307.08621](https://arxiv.org/abs/2307.08621))
`S_n = γ S_{n-1} + K_nᵀV_n`, `out = Q_n S_n`. The decay γ is fixed per head,
`γ = 1 − 2^{−5−arange(0,h)}`, and GroupNorm replaces the softmax normaliser. **Small-scale caveat:**
"RetNet starts to outperform Transformer when the model size is larger than 2B". Below that it trails.

### 2.9 Hedgehog (Zhang et al. 2024) ([arXiv:2402.04347](https://arxiv.org/abs/2402.04347))
- Diagnoses why earlier feature maps lose to softmax. elu+1, Performer and cosFormer produce
  high-entropy (non-"spiky") weights and are not monotone in `qᵀk`.
- Fix: a learnable MLP feature map with exp activations,
  `φ(x) = [exp(Wx+b), exp(−Wx−b)]`, trained by distillation against softmax attention weights.
- Reported 125M WikiText-103 perplexities: Transformer 18.6, Performer 26.8, Hedgehog 20.8.

### 2.10 Zoology and Based (Arora et al. 2023; 2024) — the recall gap
- **Zoology** ([arXiv:2312.04927](https://arxiv.org/abs/2312.04927)):
  - Attention-free gated-convolution LMs trail attention by up to 2.1 perplexity on the Pile, and
    "82% of the gap is explained by" in-context **associative recall**.
  - "a 70M parameter attention model outperforms a 1.4 billion parameter gated-convolution model on
    associative recall."
  - Introduces the **MQAR** synthetic task.
- **Based** ([arXiv:2402.18668](https://arxiv.org/abs/2402.18668)):
  - Formalises the **state-size vs. recall trade-off**: fixed-state models (H3, Mamba, RWKV)
    "struggle at recall".
  - Combines linear attention with a 2nd-order **Taylor** feature map
    (`φ(q)ᵀφ(k) = 1 + qᵀk + (qᵀk)²/2`, with q,k projected to d′=16) and small sliding-window softmax
    attention (≤64 tokens).
  - Keeps the denominator.
  - On MQAR, Taylor, PosELU and ReLU feature maps sit on the Pareto frontier.

### 2.11 "The Devil in Linear Transformer" (Qin et al. 2022) ([arXiv:2210.10340](https://arxiv.org/abs/2210.10340))
Names two failure modes:
1. **Unbounded gradients caused by the scaling (denominator).** They show
   `|∂p_ij/∂s_ik| ≤ 1/(4|s_ik|)`, which has no upper bound.
2. **Attention dilution** over long sequences.

The fix (TransNormer) is `O = XNorm(Q(KᵀV))`: drop the denominator, apply LayerNorm/RMSNorm to the
output, and use local "diagonal" attention in early layers.

## 3. Recurrent / SSM alternatives

- **RWKV-4** (Peng et al. 2023) ([arXiv:2305.13048](https://arxiv.org/abs/2305.13048)): an RNN built
  from an exponentially decayed, *normalised* weighted average of past values:

      wkv_t = (Σ_{i<t} e^{−(t−1−i)w + k_i} v_i + e^{u+k_t} v_t) / (Σ_{i<t} e^{−(t−1−i)w + k_i} + e^{u+k_t})

  Models go down to 169M. Enwik8 (character-level) results are in its appendix J.3 (numbers not
  checked). **RWKV-7 "Goose"** ([arXiv:2503.14456](https://arxiv.org/abs/2503.14456), 2025)
  generalises the delta rule with vector-valued gating and in-context learning rates, and claims it
  can recognise all regular languages.
- **Mamba** (Gu & Dao 2023) ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752)): a selective SSM,
  `h_t = Ā_t h_{t-1} + B̄_t x_t`, `y_t = C_t h_t`, where Δ, B, C depend on the input (selection) and
  are discretised by zero-order hold.
- **Mamba-2 / SSD** (Dao & Gu 2024) ([arXiv:2405.21060](https://arxiv.org/abs/2405.21060)): with a
  scalar `A_t = a_t I`, the SSM is exactly causal linear attention with a cumulative-product decay
  mask, `Y = (L ∘ CBᵀ) X`.
  - Its kernel ablation (Fig. 12) reports Pile perplexity of 11.58 with no activation, 11.66 Swish,
    11.62 exp, 11.73 ReLU, 11.64 ReLU+normaliser, 11.97 cosFormer, 11.57 RFA, 12.21 Performer.
    Takeaway: **once decay/selection is present the feature map barely matters**, and the paper found
    "kernel approximation methods ... did not seem to improve over simple pointwise non-linear
    activation functions".
- **xLSTM** (Beck et al. 2024) ([arXiv:2405.04517](https://arxiv.org/abs/2405.04517)): exponential
  gating with stabilisation.
  - **mLSTM** has a matrix memory and a covariance update: `C_t = f_t C_{t-1} + i_t v_t k_tᵀ`,
    `n_t = f_t n_{t-1} + i_t k_t`, `h̃_t = C_t q_t / max{|n_tᵀ q_t|, 1}`. This is gated linear
    attention that **keeps a lower-bounded denominator**. The C_t/n_t lines are written from the
    paper's structure; the max-normaliser was confirmed in the text.
  - **sLSTM** has a scalar memory and recurrent memory mixing.
- **minGRU / minLSTM, "Were RNNs All We Needed?"** (Feng et al. 2024)
  ([arXiv:2410.01201](https://arxiv.org/abs/2410.01201)):
  - minGRU: `h_t = (1−z_t)⊙h_{t-1} + z_t⊙h̃_t`, `z_t = σ(Lin(x_t))`, `h̃_t = Lin(x_t)`. The gates do
    not depend on `h_{t-1}`, so training is a parallel scan.
  - minLSTM normalises its gates: `f′ = f/(f+i)`, `i′ = i/(f+i)`.
  - **Character-level Shakespeare** (nanoGPT, 3 layers, d=384) test loss: minGRU 1.548, minLSTM 1.555,
    Mamba 1.575, Transformer 1.547. The Transformer needed ~2.5× more steps to converge. No linear
    attention model was in this comparison.
- **LRU** (Orvieto et al. 2023) ([arXiv:2303.06349](https://arxiv.org/abs/2303.06349)): a linear,
  diagonal, complex-valued recurrence with stable exponential parameterisation and forward-pass
  normalisation. It matches deep SSMs on Long Range Arena. Its usefulness here is as a recipe for
  stable linear recurrences, not as an LM result.

**2025–2026 developments**
- **DeltaProduct** ([arXiv:2502.10297](https://arxiv.org/abs/2502.10297)) uses several Householder
  steps per token to improve state tracking.
- **Grazzi et al.** ([arXiv:2411.12537](https://arxiv.org/abs/2411.12537)) show that allowing negative
  eigenvalues in the transition unlocks state tracking such as parity.
- **Log-Linear Attention** ([arXiv:2506.04761](https://arxiv.org/abs/2506.04761)) gives a growing,
  log-size state.
- **Kimi Linear** ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) is a production hybrid built
  on a gated delta rule.
- **Mamba-3** ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569), ICLR 2026) adds a complex-valued
  state update and a MIMO formulation, and reports +0.6 points average accuracy over Gated DeltaNet at
  1.5B (from search summary; not read in full).

The 2025–26 trend is gated delta-rule recurrences and hybrids that keep a few softmax layers.

## 4. Evidence at small scale

Direct evidence below ~10M params is thin. Nearly all architecture comparisons start at 125M–400M.
What exists:

- **Tiny models are viable on TinyStories.**
  - TinyStories ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) trains GPT-Neo models from
    ~1M to 33M params (10k-token vocabulary, 1–8 layers) that produce fluent stories.
  - "Grammar can be mastered by relatively small models", while consistency and creativity emerge
    at larger size.
  - The embedding width matters most for word meaning and depth matters most for long-range
    dependencies.
  - No linear or recurrent variants were tested there.
- **The linear vs. softmax gap does not close with scale at small sizes:**
  - Schlag et al.: the plain linear transformer trails softmax by ~4.2 test perplexity at 40M and
    ~3.4 at 90M on WikiText-103. The delta rule shrinks this to ~1.4–1.9.
  - Hedgehog at 125M: Performer +8.2 perplexity over softmax, Hedgehog +2.2.
  - RetNet trails Transformer below ~2B.
- **Decay/gating is what makes linear models competitive.** The scaling-law study of linear models
  (Shen et al. 2024, [arXiv:2406.16690](https://arxiv.org/abs/2406.16690); 70M–7B) finds similar
  scaling to LLaMA overall, but cosFormer2 (no decay) "performs worse than TNL (which uses
  data-independent decay)" on perplexity. All linear models below 160M "struggle" on
  needle-in-a-haystack retrieval.
- **Recall is the main deficit, and it is worst when the state is small.** In Zoology, a 70M attention
  model beats a 1.4B gated convolution on associative recall, and Based ties recall to state size. At
  our sizes (d≈32–128) a linear-attention state of d×d is tiny, so in-context copying (character
  names in TinyStories) is the expected failure mode.
- **Character level and tiny gated RNNs.** minGRU, minLSTM and Mamba roughly match a small
  Transformer on char-level Shakespeare (loss 1.548–1.575 vs 1.547) and converge in fewer steps. This
  is the most relevant data point for a CPU-scale study. RWKV reports enwik8 results in an appendix
  **(not checked)**.
- **Feature-map choice matters less than gating** (Mamba-2 ablation, 130M scale). Without gating,
  spiky and monotone feature maps (Hedgehog, Taylor) matter more (Hedgehog, Based).

## 5. The denominator `z_t` and normalised TSK firing strengths

With a positive feature map, normalised linear attention

    y_t = Σ_{j≤t} [φ(q_t)ᵀφ(k_j)] v_j / Σ_{j≤t} φ(q_t)ᵀφ(k_j)

has exactly the form of a **normalised TSK / NW output** `Σ_r w_r f_r / Σ_r w_r`. Each past token is a
rule, the firing strength is `w_j = φ(q_t)ᵀφ(k_j) ≥ 0`, and the consequent is `v_j` (constant or
zero-order). Softmax attention is the Gaussian-membership case (§1). `z_t` is the running sum of
firing strengths. Where the denominator survives and where it is dropped:

| Keeps a normaliser | Drops it (and why) |
|---|---|
| Katharopoulos 2020 (`φ(q)ᵀz_t`) | TransNormer / Devil (Qin 2022): denominator causes unbounded gradients → output LayerNorm/RMSNorm |
| Performer (`D̂⁻¹`), RFA, cosFormer, Based | RetNet: GroupNorm instead; Mamba-2: denominator "introduced instabilities to most variants", only slightly helped ReLU |
| RWKV-4 (exact normalised exp-decay average) | GLA: identity φ "without a normalizer works well in practice" |
| xLSTM mLSTM: `max{|n_tᵀq_t|, 1}`, a denominator floored at 1 | DeltaNet / Gated DeltaNet: L2-normalised q,k, no denominator |
| minLSTM: gates normalised to a convex combination `f/(f+i)` | Schlag 2021: replaces it with sum-normalised φ; attention normalisation blew up for the Delta Net, but removing it from the plain linear transformer gave perplexity >1600 in the state-carry-over setting |

The pattern: **the denominator is needed when the state is a pure accumulation** (no decay). It
becomes unnecessary or harmful once decay, gating or the delta rule bounds the state; then a
post-hoc norm does the job. For a TSK reading, xLSTM's floored normaliser and minLSTM's normalised
gates are the closest well-behaved analogues of normalised firing strengths.

## Implications for our experiment

**Baselines to implement (all a few dozen lines in PyTorch, all O(N) recurrent on CPU):**
1. **Softmax attention** with QK-norm (L2-normalised q,k and a learnable temperature). This is the
   reference, and also the Gaussian-kernel NW model of §1, which makes it a direct FIS counterpart.
2. **Plain linear attention** (Katharopoulos), elu+1, **with** the `z_t` denominator plus a small ε,
   and no decay. This is the "normalised TSK firing" baseline; expect the largest gap.
3. **Gated linear attention / RetNet-style decay:** identity or SiLU φ, no denominator, output
   RMSNorm/GroupNorm. Start with a fixed per-head γ (RetNet), then data-dependent α_t (GLA).
4. **Gated DeltaNet** (L2-normalised q,k, sigmoid β_t, decay α_t). This is the current best
   linear-recurrent design for recall.

Optionally add **minGRU** as a cheap vector-state RNN floor; it has char-level evidence at tiny scale.

**Feature maps.** Compare elu+1 against Taylor-2 (Based) on the ungated model only. The Mamba-2
ablation suggests the choice washes out once gating is present. Skip Performer random features: they
add variance and their tiny-d approximation is poor.

**Decay/gating.** Include it. It is the single most consistent factor separating competitive linear
models from weak ones (Shen 2024, Mamba-2, Gated DeltaNet), and RetNet's fixed γ is a cheap first
step.

**Pitfalls**
- **Stability:**
  - An unbounded denominator gives exploding gradients (Qin 2022). An accumulating `z_t` grows with
    t (Schlag 2021).
  - Use fp32 on CPU, ε in the denominator, an output norm, and L2-normalised q,k for delta rules.
  - Keep β_t ∈ (0,1) via a sigmoid, or the delta rule is unstable.
- **Normalisation choice is a confound.** Report denominator vs. output-norm as a separate factor,
  not folded into "linear vs. softmax".
- **Recall:**
  - Expect linear variants to fail most on in-context copying (names, repeated entities).
  - Add a small MQAR probe alongside TinyStories loss so a recall gap is not misread as a general
    quality gap.
  - State size is d_k×d_v per head; match **state size**, not just parameter count, when comparing.
- **Scale confound.** RetNet loses below 2B and linear models fail NIAH below 160M, so a gap at 5K–1M
  params may be a small-scale effect, not an architectural ceiling. Sweep at least 3 sizes and
  10 seeds (repo protocol), and report the gap as a function of size.

---

## BibTeX

```bibtex
@article{tsai2019dissection, title={Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel}, author={Tsai, Yao-Hung Hubert and Bai, Shaojie and Yamada, Makoto and Morency, Louis-Philippe and Salakhutdinov, Ruslan}, journal={arXiv:1908.11775}, year={2019}}
@article{katharopoulos2020transformers, title={Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention}, author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois}, journal={arXiv:2006.16236}, year={2020}}
@article{choromanski2021performer, title={Rethinking Attention with Performers}, author={Choromanski, Krzysztof and others}, journal={arXiv:2009.14794}, year={2020}}
@article{peng2021rfa, title={Random Feature Attention}, author={Peng, Hao and others}, journal={arXiv:2103.02143}, year={2021}}
@article{qin2022cosformer, title={cosFormer: Rethinking Softmax in Attention}, author={Qin, Zhen and others}, journal={arXiv:2202.08791}, year={2022}}
@article{schlag2021fwp, title={Linear Transformers Are Secretly Fast Weight Programmers}, author={Schlag, Imanol and Irie, Kazuki and Schmidhuber, J{\"u}rgen}, journal={arXiv:2102.11174}, year={2021}}
@article{yang2024deltanet, title={Parallelizing Linear Transformers with the Delta Rule over Sequence Length}, author={Yang, Songlin and others}, journal={arXiv:2406.06484}, year={2024}}
@article{yang2024gateddeltanet, title={Gated Delta Networks: Improving Mamba2 with Delta Rule}, author={Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali}, journal={arXiv:2412.06464}, year={2024}}
@article{yang2023gla, title={Gated Linear Attention Transformers with Hardware-Efficient Training}, author={Yang, Songlin and others}, journal={arXiv:2312.06635}, year={2023}}
@article{sun2023retnet, title={Retentive Network: A Successor to Transformer for Large Language Models}, author={Sun, Yutao and others}, journal={arXiv:2307.08621}, year={2023}}
@article{zhang2024hedgehog, title={The Hedgehog \& the Porcupine: Expressive Linear Attentions with Softmax Mimicry}, author={Zhang, Michael and others}, journal={arXiv:2402.04347}, year={2024}}
@article{arora2023zoology, title={Zoology: Measuring and Improving Recall in Efficient Language Models}, author={Arora, Simran and others}, journal={arXiv:2312.04927}, year={2023}}
@article{arora2024based, title={Simple linear attention language models balance the recall-throughput tradeoff}, author={Arora, Simran and others}, journal={arXiv:2402.18668}, year={2024}}
@article{qin2022devil, title={The Devil in Linear Transformer}, author={Qin, Zhen and others}, journal={arXiv:2210.10340}, year={2022}}
@article{peng2023rwkv, title={RWKV: Reinventing RNNs for the Transformer Era}, author={Peng, Bo and others}, journal={arXiv:2305.13048}, year={2023}}
@article{peng2025rwkv7, title={RWKV-7 "Goose" with Expressive Dynamic State Evolution}, author={Peng, Bo and others}, journal={arXiv:2503.14456}, year={2025}}
@article{gu2023mamba, title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces}, author={Gu, Albert and Dao, Tri}, journal={arXiv:2312.00752}, year={2023}}
@article{dao2024ssd, title={Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality}, author={Dao, Tri and Gu, Albert}, journal={arXiv:2405.21060}, year={2024}}
@article{beck2024xlstm, title={xLSTM: Extended Long Short-Term Memory}, author={Beck, Maximilian and others}, journal={arXiv:2405.04517}, year={2024}}
@article{feng2024mingru, title={Were RNNs All We Needed?}, author={Feng, Leo and others}, journal={arXiv:2410.01201}, year={2024}}
@article{orvieto2023lru, title={Resurrecting Recurrent Neural Networks for Long Sequences}, author={Orvieto, Antonio and others}, journal={arXiv:2303.06349}, year={2023}}
@article{eldan2023tinystories, title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?}, author={Eldan, Ronen and Li, Yuanzhi}, journal={arXiv:2305.07759}, year={2023}}
@article{shen2024linearscaling, title={Scaling Laws for Linear Complexity Language Models}, author={Shen, Xuyang and Li, Dong and Leng, Ruitao and Qin, Zhen and Sun, Weigao and Zhong, Yiran}, journal={arXiv:2406.16690}, year={2024}}
@article{siems2025deltaproduct, title={DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products}, author={Siems, Julien and others}, journal={arXiv:2502.10297}, year={2025}}
@article{grazzi2024negeig, title={Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues}, author={Grazzi, Riccardo and Siems, Julien and Zela, Arber and Franke, J{\"o}rg K. H. and others}, journal={arXiv:2411.12537}, year={2024}}
@article{guo2025loglinear, title={Log-Linear Attention}, author={Guo, Han and others}, journal={arXiv:2506.04761}, year={2025}}
@article{kimi2025linear, title={Kimi Linear: An Expressive, Efficient Attention Architecture}, author={{Kimi Team}}, journal={arXiv:2510.26692}, year={2025}}
@article{lahoti2026mamba3, title={Mamba-3: Improved Sequence Modeling using State Space Principles}, author={Lahoti, Aakash and Li, Kevin Y. and Chen, Berlin and Wang, Caitlin and others}, journal={arXiv:2603.15569}, year={2026}}
```
