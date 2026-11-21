# 03 — Fuzzy ↔ neural equivalences, recurrent fuzzy networks, fuzzy attention, and a prior-art check for the FLM / FRLM

*Literature review, 2026-10-08. Scope: the claim that a causal language model built from
Takagi–Sugeno–Kang (TSK) fuzzy inference systems (FLM), and a recurrent variant (FRLM), would be
the first of its kind. Searches: arXiv API (title/abstract boolean queries), Crossref and OpenAlex
title/DOI lookups, general web search. Semantic Scholar returned HTTP 429 for every request in this
session, so it does not appear among the sources. Every DOI and arXiv ID below was resolved through
Crossref or the arXiv API, or fetched directly. Anything not confirmed that way is marked
**(unverified)**.*

---

## TL;DR: novelty verdict

* **There is no prior generative causal LM whose attention and feed-forward sublayers are TSK fuzzy
  systems.** I searched for one specifically and did not find it.
* **Several works come close, and the proposal must cite them.** Two of them are generative text
  models:
  1. **FuzzyS2S / GenFS** (Yang, Deng et al., IEEE TNNLS 2026; arXiv 2024) is a *generative fuzzy
     system for text*, covering machine translation, code generation and summarization. Its rules
     are sequence-level: each rule's consequent is a whole Transformer encoder–decoder, and the
     firing strengths mix their outputs. It is a fuzzy mixture of Transformers, not a Transformer
     built from TSK blocks, and it is seq2seq, not a decoder-only LM.
  2. **NC-FFN** (Oskin, arXiv June 2026) is a 125M-parameter decoder-only LM trained on OpenWebText
     whose FFN contains explicit *fuzzy set operators*: product t-norm, set difference A·(1−B), and
     causal fuzzy quantifiers with a learned forgetting rate, which is a recurrent fuzzy state. It is
     not TSK: there are no antecedent/consequent rules, and a GELU block is kept beside the fuzzy
     one. The abstract reports that a fully *Boolean* FFN (every unit a fuzzy-set operator, no GELU path) "diverges in training" (verified against the arXiv abstract, 2026-10-08).
* **"Attention is a TSK system" is not novel.** Jiang, …, Lin (IEEE TFS 2025) explicitly derive
  softmax attention as normalized TSK firing strength and build a "Fuzzy Attention Layer".
  FISformer (Haznedar & Karacan, IEEE TFS 2026) replaces self-attention with a first-order Sugeno
  FIS. Both target non-language tasks (fNIRS and time-series forecasting). Tsai et al. (EMNLP 2019)
  already read attention as a kernel smoother.
* **"Neural weights ≡ fuzzy rules" is not novel as a general idea.** It goes back to RBF≡FIS (1993),
  sigmoid MLP ≡ fuzzy rules (1997), TSK ≡ MoE (2019), and ReLU ≡ triangular-MF TSK (2023/2025). A
  *composed, multi-layer weight↔rule map for an LM block, attention included* does appear to be
  open.
* **Recurrent TSK networks are not novel.** RSONFIN 1999, RFNN 2000, TRFN 2002 and RSEIT2FNN 2009
  exist, but none of them has been applied to language modelling.

---

## 1. FIS ↔ neural network equivalences

**Jang & Sun (1993)** showed that a zero-order TSK system with Gaussian MFs, a product t-norm and a
weighted-average defuzzifier is functionally equivalent to a normalized RBF network, provided each
rule has one Gaussian per input and consequents are constants
([10.1109/72.182710](https://doi.org/10.1109/72.182710)). **Hunt, Haas & Murray-Smith (1996)**
extended the equivalence to first-order (local-linear) consequents and relaxed the conditions
([10.1109/72.501735](https://doi.org/10.1109/72.501735)). **ANFIS** (Jang 1993) is the standard
layered TSK network trained by hybrid least squares and gradient descent
([10.1109/21.256541](https://doi.org/10.1109/21.256541)). **Buckley, Hayashi & Czogała (1993)**
gave an early equivalence of neural nets and fuzzy expert systems
([10.1016/0165-0114(93)90167-G](https://doi.org/10.1016/0165-0114(93)90167-G)).

**Benítez, Castro & Requena (1997)**, *Are artificial neural networks black boxes?*, showed that a
one-hidden-layer sigmoid MLP equals a fuzzy rule base. Each hidden unit is one rule. The combining
operator is not a t-norm but the "interactive-or" (i-or) operator
([10.1109/72.623216](https://doi.org/10.1109/72.623216)). This is the closest classical precedent
for "every hidden unit of an FFN is a fuzzy rule". It also warns that the induced rule logic may be
non-standard.

**Mantas & Puche (2008)**, *Artificial Neural Networks are Zero-Order TSK Fuzzy Systems*
([10.1109/TFUZZ.2007.902016](https://doi.org/10.1109/TFUZZ.2007.902016), IEEE TFS 16(3):630–643;
abstract verified via OpenAlex, 2026-10-08, by the main session, not the review agent). The paper
proves that a *multilayer* feed-forward NN is functionally equivalent to a zero-order TSK rule
base. The rules use the product t-norm and take the network's own inputs. This is the closest
classical result to "the whole FFN stack is a zero-order TSK system", so it must be cited beside
Benítez and Bede et al. It covers feed-forward networks only, with no attention and no recurrence.

**Bede, Kreinovich & Toth** (already in `papers/nn-fis-equivalence/`):

* 1-D TSK with triangular MFs ≡ one-hidden-layer ReLU network (NAFIPS 2023, LNNS, pp. 44–56,
  [10.1007/978-3-031-46778-3_5](https://doi.org/10.1007/978-3-031-46778-3_5)).
* A 2024 NAFIPS "on the real line" follow-up.
* The n-D extension via simplicial (tetrahedral) MFs, stated as a *local* equivalence (IJCCC 20(4),
  2025, [10.15837/ijccc.2025.4.7127](https://doi.org/10.15837/ijccc.2025.4.7127)).

This is the most direct route from a ReLU FFN to rules. Their own conclusion lists "local → global"
as future work.

**TSK ≡ mixture of experts.** **Jacobs, Jordan, Nowlan & Hinton (1991)**
([10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79)) and **Jordan & Jacobs
(1994)**, the hierarchical MoE
([10.1162/neco.1994.6.2.181](https://doi.org/10.1162/neco.1994.6.2.181)), define a softmax gate over
local experts. A first-order TSK system has the same form: normalized firing strengths × linear
consequents. **Wu, Lin, Huang & Zeng (2019)** make this explicit: TSK ≡ NN, MoE, CART and stacking
under stated conditions ([arXiv:1903.10572](https://arxiv.org/abs/1903.10572)).

*Repo correction:* `papers/nn-fis-equivalence/references.bib` lists this paper's authors as "Wu, Yuan,
Huang, Tan". The arXiv record lists **Wu, Chin-Teng Lin, Jian Huang, Zhigang Zeng**. The listed
names belong to the MBGD-RDA paper below.

**Training TSK systems with deep-learning tools (Dongrui Wu's group).** These matter in practice
because an FLM is a very high-dimensional TSK system:

* **MBGD-RDA**: mini-batch GD with regularization, DropRule and AdaBound (IEEE TFS 28(5), 2020,
  [10.1109/TFUZZ.2019.2958559](https://doi.org/10.1109/TFUZZ.2019.2958559); rule-pruning extension
  [arXiv:2003.00608](https://arxiv.org/abs/2003.00608)).
* **FCM-RDpA**: FCM initialization, DropRule and Powerball AdaBelief (Information Sciences 574,
  2021, [10.1016/j.ins.2021.05.084](https://doi.org/10.1016/j.ins.2021.05.084)).
* **Cui, Wu & Xu (IJCNN 2021)** show that product-t-norm firing strengths underflow and softmax-
  saturate as dimension grows. This is the *curse of dimensionality for TSK* and directly concerns
  d_model-dimensional antecedents
  ([10.1109/ijcnn52387.2021.9534265](https://doi.org/10.1109/ijcnn52387.2021.9534265),
  [arXiv:2102.04271](https://arxiv.org/abs/2102.04271)). The FLM design must address it, for
  example with low-dimensional antecedent projections or a mean-of-log t-norm. Expect reviewers to
  ask about it.
* **Gu & Cheng (2020)** distil a DNN into a TSK FIS
  ([arXiv:2010.04974](https://arxiv.org/abs/2010.04974)). This is a post-hoc route, not an
  architecture.

## 2. Recurrent fuzzy neural networks

Each of the following feeds rule-layer outputs back as internal state. All target **dynamic-system
identification, control and time-series prediction**. None targets language.

| Model | Ref | Where the recurrence is |
|---|---|---|
| RSONFIN (Juang & Lin 1999) | IEEE TNN 10(4):828–845, [10.1109/72.774232](https://doi.org/10.1109/72.774232) | Self-organizing; internal-memory rules feed back as extra antecedent inputs |
| RFNN (Lee & Teng 2000) | IEEE TFS 8(4):349–366, [10.1109/91.868943](https://doi.org/10.1109/91.868943) | Self-feedback on membership-layer nodes |
| TRFN (Juang 2002) | IEEE TFS 10(2):155–170, [10.1109/91.995118](https://doi.org/10.1109/91.995118) | **TSK-type** consequents; global feedback of rule firing; NN + GA training |
| RSEIT2FNN (Juang, Huang & Lin 2009) | IEEE TFS 17(5):1092–1105, [10.1109/TFUZZ.2009.2021953](https://doi.org/10.1109/TFUZZ.2009.2021953) | Interval type-2, TSK consequents, recurrent internal states |
| FCM time-series (Stach, Kurgan & Pedrycz 2008) | IEEE TFS 16(1):61–72, [10.1109/TFUZZ.2007.902020](https://doi.org/10.1109/TFUZZ.2007.902020) | Fuzzy cognitive map iterated as a dynamical system |

TRFN is the obvious ancestor of an FRLM cell. An FRLM claim must say what is new relative to it:
discrete-token inputs, vocabulary-sized softmax outputs, and next-token training at scale. The
recurrence itself is not new. Recent *linear-attention-as-RNN* work (Katharopoulos et al. 2020,
[arXiv:2006.16236](https://arxiv.org/abs/2006.16236)) gives a direct bridge: a normalized-kernel
(TSK-like) attention can be run recurrently with a constant-size state. An FRLM can be framed as a
TSK-gated version of that state.

## 3. Fuzzy attention and fuzzy transformers

| Work | Fuzzy component | Task | Generative LM? |
|---|---|---|---|
| **Jiang, Ou, Chen, Ao, Chang, Do, Lin**, "A Fuzzy Logic-Based Approach to Predict Human Interaction by fNIRS", IEEE TFS 2025 ([10.1109/tfuzz.2025.3528376](https://doi.org/10.1109/tfuzz.2025.3528376); [arXiv:2409.17661](https://arxiv.org/abs/2409.17661)) | "Fuzzy Attention Layer": queries attend to *R learnable rule centres*. The paper writes normalized Gaussian TSK firing strength next to dot-product attention and states their relationship (Prop. 1) | fNIRS classification | No (encoder) |
| **FISformer** (Haznedar & Karacan; arXiv [2603.21724](https://arxiv.org/abs/2603.21724), IEEE TFS 34(8):2437–2450, Aug 2026; TFS DOI not found) | Replaces QK similarity with a **first-order Sugeno FIS** per query–key pair and feature; softmax over tokens | Multivariate time-series forecasting | No |
| **FANTF** (Chakraborty & Heintz 2025, [arXiv:2504.00070](https://arxiv.org/abs/2504.00070)) | Fuzzy MFs inside attention scoring | TS forecasting, classification, anomaly detection | No |
| **Fuzzformer** (Ožbot, Škrjanc & Štruc 2025, [arXiv:2510.00960](https://arxiv.org/abs/2510.00960)) | LSTM + MHSA encoder → Gaussian-cluster fuzzy local-model (ARIX) head | Stock forecasting | No |
| **Sparse Fuzzy Attention** (Peng, Li & Zhao 2021, [arXiv:2109.06719](https://arxiv.org/abs/2109.06719)) | "Fuzzy" sparse attention *scorer* for parsing | Structured sentiment analysis | No |
| **Rule-Based Spatial MoE U-Net** (Dogga, …, Cohen 2026, [arXiv:2602.05100](https://arxiv.org/abs/2602.05100)) | TSK fuzzy head plus MoE gating | Edge detection | No |
| **VLM-TSK-DA** (Shi, Lu, Fang & Zhang, FUZZ-IEEE 2024, [10.1109/fuzz-ieee60900.2024.10612077](https://doi.org/10.1109/fuzz-ieee60900.2024.10612077)) | TSK FIS as a residual image *adapter* in a vision-language model | Domain adaptation | No |
| **FMLC** (Zhou, AAAI 2026 doctoral abstract, [10.1609/aaai.v40i48.42178](https://doi.org/10.1609/aaai.v40i48.42178)) | DNN produces modulators of TSK linear consequents | Tabular/general | No |

Theoretical background: **Tsai et al. (EMNLP 2019)** read attention as a kernel smoother
([10.18653/v1/D19-1443](https://doi.org/10.18653/v1/D19-1443)). Softmax attention with an RBF kernel
is a Nadaraya–Watson estimator. By Jang & Sun it is therefore a zero-order Gaussian TSK system with
one rule per key. That makes **"attention ≡ zero-order TSK" a corollary of known results** (exact up
to the ‖q‖², ‖k‖² norm factors in exp(q·k)), and Jiang et al. already published it in the fuzzy
literature. The FLM's attention contribution has to be something beyond the identity: first-order
consequents, rule-centred keys, or interpretability that is shown to hold in a language model.

## 4. Fuzzy approaches to language modelling and text generation

**Generative text models with genuine fuzzy-inference structure (closest prior art):**

* **FuzzyS2S / GenFS.** Yang, Deng, Zhang, Zhao, Wang & Choi, "Generative Fuzzy System for
  Sequence-to-Sequence Learning via Rule-Based Inference", IEEE TNNLS 37(3):1435–1448 (issue 2026;
  online 2025), [10.1109/TNNLS.2025.3615650](https://doi.org/10.1109/TNNLS.2025.3615650). Preprint:
  "Generative Fuzzy System for Sequence Generation",
  [arXiv:2411.13867](https://arxiv.org/abs/2411.13867).

  The full text (read for this review) defines *generative fuzzy rules* in the following way:
  - Antecedents are obtained by fuzzy clustering (a DSMFCM variant) of *sequences*, using cosine
    similarity to rule prototypes.
  - Each rule consequent is a **full Transformer encoder–decoder** ("GenFS-Trans"), framed as a
    generalization of first-order TSK consequents.
  - Outputs are combined by a firing-strength-weighted average of word embeddings, followed by
    softmax.
  - A "fuzzy tokenizer" chooses among sub-word tokenizers of several granularities.

  Tasks are machine translation, code generation and summarization; the paper claims gains over a
  Transformer and parity with T5 and CodeT5 on some sets. **This is a prior "generative fuzzy
  system for text" and must be cited prominently.** It differs from the FLM in four ways:
  1. The rules are coarse, with K rules per *sequence*, not per token or per sublayer.
  2. The interior of each rule is a black-box Transformer.
  3. It is encoder–decoder seq2seq, not causal LM pretraining.
  4. It has no weight↔rule derivation.
* **NC-FFN with self-forgetting quantifiers.** Oskin, "Explicit Fuzzy Logic in the Feed-Forward
  Layer: Self-Forgetting Quantifiers Discover Legible Grammatical-Licensing Detectors",
  [arXiv:2606.31845](https://arxiv.org/abs/2606.31845), 30 June 2026 (preprint). Read in full for
  this review.
  - It is a GPT-2-small decoder: 125M parameters, trained on OpenWebText.
  - Part of the FFN is replaced by sigmoid-bounded fuzzy set operators: intersection A·B and
    set-difference A·(1−B).
  - Its headline adds causal *fuzzy quantifiers*: a soft existential
    E_t = max(M_t, γ⊙E_{t−1}) and a soft proportion P_t = (1−λ)M_t + λP_{t−1}, with learned decays.
    This is a **recurrent fuzzy state inside a causal LM**.
  - Results: perplexity ties GELU, LAMBADA improves, and units become readable as
    grammatical-licensing detectors.
  - Limits that matter for the FLM: a GELU majority must be kept, a fully Boolean FFN "diverges
    within the first ~16k steps", and there is no TSK rule structure (no antecedent partitions, no
    consequent functions).

  **It is the nearest prior art to both the FLM's FFN and the FRLM's state. Its stated novelty
  ("to our knowledge, novel is the specific assembly … in a decoder-only language model") shows the
  space is now contested.** Its related-work section names no TSK or ANFIS work.
* **Semantic Fusion with Fuzzy-Membership Features** (Huang & Raza 2025,
  [arXiv:2509.13357](https://arxiv.org/abs/2509.13357)). A causal LM receives a parallel channel of
  hand-specified per-token fuzzy memberships (POS cues, polarity) through a gated adapter. The
  backbone is not fuzzy.

**Older "fuzzy LM" work (different meaning).** Fuzzy class-based n-gram LMs soft-cluster words with
FCM or possibilistic c-means, so a word belongs to several classes. One example is Momtazi et al. on
Persian for ASR (**unverified**: only an abstract page without author or venue metadata could be
fetched, at
[coli.uni-saarland.de](https://www.coli.uni-saarland.de/projects/irtg/contents/Colloquium/WS-07/AbstractSMomtazi.txt)).
These are count-based models with fuzzy word classes, not neural-fuzzy generators. They should
still be cited so that "fuzzy language model" is not claimed as a coined term.

**Not the same thing:**

* **Zadeh's computing with words** ([10.1109/91.493904](https://doi.org/10.1109/91.493904)) uses
  "linguistic" in the linguistic-variable sense.
* **LLM-reasons-about-fuzzy-logic** work, for example FRoG,
  [arXiv:2407.01046](https://arxiv.org/abs/2407.01046).
* **Fuzzy layers attached to a frozen LLM:**
  - DFIL, a fuzzy ordinal *prediction head* on an LLM
    ([arXiv:2609.26113](https://arxiv.org/abs/2609.26113)).
  - Fuzzy-assisted contrastive *decoding* (IEEE TFS 2025,
    [10.1109/tfuzz.2025.3575060](https://doi.org/10.1109/tfuzz.2025.3575060)).
  - Fuzzy-rule in-context-learning debiasing
    ([arXiv:2412.19018](https://arxiv.org/abs/2412.19018)).
  - "Causal Graph Fuzzy LLMs" for time series
    ([arXiv:2507.17016](https://arxiv.org/abs/2507.17016)).
* **Fuzzy string matching and deduplication.**
* **Tarau's Arrow LM**, which uses *intuitionistic* (not fuzzy) implication as a "rule" reading of
  next-token prediction ([arXiv:2601.19915](https://arxiv.org/abs/2601.19915)). It is a useful
  non-fuzzy contrast.

## 5. Interpretability bridges on the neural side

* **FFN = key–value memory.** Geva et al. (EMNLP 2021,
  [10.18653/v1/2021.emnlp-main.446](https://doi.org/10.18653/v1/2021.emnlp-main.446)) read FFN
  keys as input-pattern detectors and values as output-vocabulary distributions. In fuzzy terms
  each hidden unit is a rule "IF x matches k_i THEN add v_i" with an *unnormalized* activation as
  firing strength. That is a zero-order TSK system without defuzzifier normalization, the same
  observation as Benítez 1997 and Bede et al. at token level.
* **Sparse autoencoders / dictionary learning.** Cunningham et al. 2023
  ([arXiv:2309.08600](https://arxiv.org/abs/2309.08600)) and Bricken et al. 2023, *Towards
  Monosemanticity* ([transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features/index.html))
  recover sparse, monosemantic features post hoc. A TSK antecedent bank is the *by-construction*
  counterpart. The FLM can be evaluated against SAEs: are its rules as monosemantic as SAE
  features?
* **MoE routing.** Shazeer et al. 2017 ([arXiv:1701.06538](https://arxiv.org/abs/1701.06538)) and
  the Switch Transformer ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961)) use a softmax/top-k
  router over FFN experts. That is TSK with top-k-truncated firing strengths and non-linear
  consequents (Wu et al. 2019). An FFN-as-TSK block is mathematically a *fine-grained MoE with
  linear experts*, and reviewers from ML will say so. Pre-empt it.
* **Induction heads** (Olsson et al. 2022, [arXiv:2209.11895](https://arxiv.org/abs/2209.11895))
  are the kind of attention mechanism that a rule reading of attention should recover.

---

## Novelty assessment

Claim under test: *"first generative causal language model built from TSK fuzzy inference systems
(FLM) / with recurrent TSK state (FRLM)."*

| Closest prior work | Generative? | Causal LM? | Fuzzy where | TSK? | Recurrent fuzzy state? | Weight↔rule theory? | Threat to claim |
|---|---|---|---|---|---|---|---|
| FuzzyS2S / GenFS (TNNLS 2026) | **Yes (text)** | No (enc–dec seq2seq) | Sequence-level rule gate over K whole Transformers; fuzzy tokenizer | TSK-inspired (consequents = Transformers) | No | No | **High** for any "first generative fuzzy system for text" wording |
| NC-FFN (Oskin, arXiv 2026) | **Yes** | **Yes (125M, OWT)** | Fuzzy set operators in part of FFN; fuzzy quantifiers with decay | **No** | **Yes (leaky max/mean quantifiers)** | No (empirical legibility) | **High** for "first fuzzy causal LM" or "first recurrent fuzzy LM"; low for "TSK" |
| Jiang et al. Fuzzy Attention (TFS 2025) | No | No | Attention to rule centres; TSK firing ≈ softmax attention | Yes (Gaussian TSK) | No | Partial (attention↔TSK identity) | **High** for "attention is a TSK FIS" as a contribution |
| FISformer (TFS 2026) | No (forecasting) | No | First-order Sugeno FIS replaces QK attention | Yes | No | No | Medium: TSK attention exists, not for language |
| Huang & Raza (arXiv 2025) | Yes | Yes | Side-channel fuzzy features | No | No | No | Low |
| TRFN / RSONFIN / RFNN / RSEIT2FNN | No | No | Whole network | Yes (TRFN, RSEIT2FNN) | **Yes** | Classical | Medium for "first recurrent TSK network", none for LM use |
| Bede–Kreinovich–Toth; Jang–Sun; Benítez; Mantas–Puche 2008; Wu 2019 | n/a | n/a | Equivalence theorems | Yes | No | **Yes**, single block | High for "weights ≡ rules" in general |

**What clearly is *not* novel:**

* fuzzy ↔ NN equivalence in general;
* TSK ≡ MoE;
* attention as normalized TSK firing strength;
* replacing attention with a Sugeno FIS;
* recurrent TSK networks;
* fuzzy components inside a causal LM's FFN (Oskin);
* a generative fuzzy system for text (FuzzyS2S);
* the phrase "fuzzy language model" (fuzzy class n-grams).

**What does appear novel (nothing found contradicts it):**

1. A **decoder-only causal LM in which every attention and FFN sublayer is an explicit TSK FIS**,
   with per-token antecedents, rule-centred keys, and first-order consequents, trained end-to-end on
   next-token prediction, with **no retained conventional (GELU) FFN path**. Oskin's result that a
   fully Boolean (all-fuzzy-operator) FFN diverges makes this a non-trivial feasibility claim.
2. An **LM whose recurrent state is updated by TSK rules** (a TRFN-style cell at token level). It
   must be distinguished from Oskin's quantifiers, which are fixed max/mean aggregators, not rule
   bases.
3. A **compositional weight↔rule correspondence across a full LM block** (attention + FFN + residual
   + normalization), together with rule extraction measured on language. The existing theorems are
   single-layer and regression-scale; Bede et al. are local, and Jang–Sun and Benítez are one
   hidden layer.

**Defensible phrasing.** Avoid "first fuzzy language model". Suggested wording:

> "To our knowledge, the FLM is the first decoder-only language model in which *every* token-mixing
> and channel-mixing sublayer is a Takagi–Sugeno–Kang fuzzy inference system, trained end-to-end
> on next-token prediction. It differs from generative fuzzy systems that gate whole Transformers at
> sequence level (Yang et al., 2026), from LMs that add fuzzy set operators alongside a conventional
> FFN (Oskin, 2026), and from TSK-based attention for non-language tasks (Jiang et al., 2025;
> Haznedar & Karacan, 2026). The FRLM extends TSK-type recurrent fuzzy networks (Juang, 2002) to
> token-level language modelling."

State "to our knowledge" and the search date. The 2025–2026 preprint rate means this needs
re-checking before the defense; a scheduled search for arXiv `fuzzy AND (language model OR
transformer)` is advisable.

---

## BibTeX

```bibtex
@article{mantas2008ann, author={Mantas, C. J. and Puche, J. M.}, title={Artificial Neural Networks are Zero-Order {TSK} Fuzzy Systems}, journal={IEEE Transactions on Fuzzy Systems}, volume={16}, number={3}, pages={630--643}, year={2008}, doi={10.1109/TFUZZ.2007.902016}}
@article{jang1993functional, author={Jang, J.-S. R. and Sun, C.-T.}, title={Functional Equivalence Between Radial Basis Function Networks and Fuzzy Inference Systems}, journal={IEEE Transactions on Neural Networks}, volume={4}, number={1}, pages={156--159}, year={1993}, doi={10.1109/72.182710}}
@article{hunt1996extending, author={Hunt, K. J. and Haas, R. and Murray-Smith, R.}, title={Extending the Functional Equivalence of Radial Basis Function Networks and Fuzzy Inference Systems}, journal={IEEE Transactions on Neural Networks}, volume={7}, number={3}, pages={776--781}, year={1996}, doi={10.1109/72.501735}}
@article{jang1993anfis, author={Jang, J.-S. R.}, title={{ANFIS}: Adaptive-Network-Based Fuzzy Inference System}, journal={IEEE Transactions on Systems, Man, and Cybernetics}, volume={23}, number={3}, pages={665--685}, year={1993}, doi={10.1109/21.256541}}
@article{buckley1993equivalence, author={Buckley, James J. and Hayashi, Yoichi and Czoga{\l}a, Ernest}, title={On the Equivalence of Neural Nets and Fuzzy Expert Systems}, journal={Fuzzy Sets and Systems}, volume={53}, number={2}, pages={129--134}, year={1993}, doi={10.1016/0165-0114(93)90167-G}}
@article{benitez1997blackboxes, author={Ben{\'i}tez, J. M. and Castro, J. L. and Requena, I.}, title={Are Artificial Neural Networks Black Boxes?}, journal={IEEE Transactions on Neural Networks}, volume={8}, number={5}, pages={1156--1164}, year={1997}, doi={10.1109/72.623216}}
@inproceedings{bede2023equivalence1d, author={Bede, Barnab{\'a}s and Kreinovich, Vladik and Toth, Peter}, title={Equivalence Between 1-{D} {Takagi--Sugeno} Fuzzy Systems with Triangular Membership Functions and Neural Networks with {ReLU} Activation}, booktitle={NAFIPS 2023}, series={Lecture Notes in Networks and Systems}, pages={44--56}, publisher={Springer}, year={2023}, doi={10.1007/978-3-031-46778-3_5}}
@article{bede2025equivalencend, author={Bede, Barnab{\'a}s and Kreinovich, Vladik and Toth, Peter}, title={On Equivalence between {Takagi--Sugeno--Kang} Fuzzy Systems with Triangular Membership Functions and Neural Networks with {ReLU} Activation in Two or More Dimensions}, journal={International Journal of Computers Communications \& Control}, volume={20}, number={4}, year={2025}, doi={10.15837/ijccc.2025.4.7127}}
@article{jacobs1991adaptive, author={Jacobs, Robert A. and Jordan, Michael I. and Nowlan, Steven J. and Hinton, Geoffrey E.}, title={Adaptive Mixtures of Local Experts}, journal={Neural Computation}, volume={3}, number={1}, pages={79--87}, year={1991}, doi={10.1162/neco.1991.3.1.79}}
@article{jordan1994hme, author={Jordan, Michael I. and Jacobs, Robert A.}, title={Hierarchical Mixtures of Experts and the {EM} Algorithm}, journal={Neural Computation}, volume={6}, number={2}, pages={181--214}, year={1994}, doi={10.1162/neco.1994.6.2.181}}
@misc{wu2019functional, author={Wu, Dongrui and Lin, Chin-Teng and Huang, Jian and Zeng, Zhigang}, title={On the Functional Equivalence of {TSK} Fuzzy Systems to Neural Networks, Mixture of Experts, {CART}, and Stacking Ensemble Regression}, year={2019}, eprint={1903.10572}, archivePrefix={arXiv}}
@article{wu2020mbgdrda, author={Wu, Dongrui and Yuan, Ye and Huang, Jian and Tan, Yihua}, title={Optimize {TSK} Fuzzy Systems for Regression Problems: Minibatch Gradient Descent With Regularization, {DropRule}, and {AdaBound} ({MBGD-RDA})}, journal={IEEE Transactions on Fuzzy Systems}, volume={28}, number={5}, pages={1003--1015}, year={2020}, doi={10.1109/TFUZZ.2019.2958559}}
@article{shi2021fcmrdpa, author={Shi, Zhenhua and Wu, Dongrui and Guo, Chenfeng and Zhao, Changming and Cui, Yuqi and Wang, Fei-Yue}, title={{FCM-RDpA}: {TSK} Fuzzy Regression Model Construction Using Fuzzy {C}-Means Clustering, Regularization, {DropRule}, and {Powerball AdaBelief}}, journal={Information Sciences}, volume={574}, pages={490--504}, year={2021}, doi={10.1016/j.ins.2021.05.084}}
@inproceedings{cui2021curse, author={Cui, Yuqi and Wu, Dongrui and Xu, Yifan}, title={Curse of Dimensionality for {TSK} Fuzzy Neural Networks: Explanation and Solutions}, booktitle={International Joint Conference on Neural Networks (IJCNN)}, year={2021}, doi={10.1109/IJCNN52387.2021.9534265}}
@misc{gu2020distilling, author={Gu, Xiangming and Cheng, Xiang}, title={Distilling a Deep Neural Network into a {Takagi-Sugeno-Kang} Fuzzy Inference System}, year={2020}, eprint={2010.04974}, archivePrefix={arXiv}}
@article{juang1999rsonfin, author={Juang, Chia-Feng and Lin, Chin-Teng}, title={A Recurrent Self-Organizing Neural Fuzzy Inference Network}, journal={IEEE Transactions on Neural Networks}, volume={10}, number={4}, pages={828--845}, year={1999}, doi={10.1109/72.774232}}
@article{lee2000rfnn, author={Lee, Ching-Hung and Teng, Ching-Cheng}, title={Identification and Control of Dynamic Systems Using Recurrent Fuzzy Neural Networks}, journal={IEEE Transactions on Fuzzy Systems}, volume={8}, number={4}, pages={349--366}, year={2000}, doi={10.1109/91.868943}}
@article{juang2002trfn, author={Juang, Chia-Feng}, title={A {TSK}-Type Recurrent Fuzzy Network for Dynamic Systems Processing by Neural Network and Genetic Algorithms}, journal={IEEE Transactions on Fuzzy Systems}, volume={10}, number={2}, pages={155--170}, year={2002}, doi={10.1109/91.995118}}
@article{juang2009rseit2fnn, author={Juang, Chia-Feng and Huang, Ren-Bo and Lin, Yang-Yin}, title={A Recurrent Self-Evolving Interval Type-2 Fuzzy Neural Network for Dynamic System Processing}, journal={IEEE Transactions on Fuzzy Systems}, volume={17}, number={5}, pages={1092--1105}, year={2009}, doi={10.1109/TFUZZ.2009.2021953}}
@article{stach2008fcmts, author={Stach, W. and Kurgan, L. A. and Pedrycz, W.}, title={Numerical and Linguistic Prediction of Time Series With the Use of Fuzzy Cognitive Maps}, journal={IEEE Transactions on Fuzzy Systems}, volume={16}, number={1}, pages={61--72}, year={2008}, doi={10.1109/TFUZZ.2007.902020}}
@misc{katharopoulos2020rnns, author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois}, title={Transformers are {RNNs}: Fast Autoregressive Transformers with Linear Attention}, year={2020}, eprint={2006.16236}, archivePrefix={arXiv}}
@article{jiang2025fuzzyattention, author={Jiang, Xiaowei and Ou, Liang Shiou and Chen, Yanan and Ao, Na and Chang, Yu-Cheng and Do, Thomas and Lin, Chin-Teng}, title={A Fuzzy Logic-Based Approach to Predict Human Interaction by Functional Near-Infrared Spectroscopy}, journal={IEEE Transactions on Fuzzy Systems}, year={2025}, doi={10.1109/TFUZZ.2025.3528376}, note={arXiv:2409.17661}}
@article{haznedar2026fisformer, author={Haznedar, Bulent and Karacan, Levent}, title={{FISformer}: Replacing Self-Attention With a Fuzzy Inference System in Transformer Models for Time Series Forecasting}, journal={IEEE Transactions on Fuzzy Systems}, volume={34}, number={8}, pages={2437--2450}, year={2026}, note={arXiv:2603.21724; journal DOI not verified}}
@misc{chakraborty2025fantf, author={Chakraborty, Sanjay and Heintz, Fredrik}, title={Enhancing Time Series Forecasting with Fuzzy Attention-Integrated Transformers}, year={2025}, eprint={2504.00070}, archivePrefix={arXiv}}
@misc{ozbot2025fuzzformer, author={O{\v{z}}bot, Miha and {\v{S}}krjanc, Igor and {\v{S}}truc, Vitomir}, title={A Neuro-Fuzzy System for Interpretable Long-Term Stock Market Forecasting}, year={2025}, eprint={2510.00960}, archivePrefix={arXiv}}
@misc{peng2021sparsefuzzy, author={Peng, Letian and Li, Zuchao and Zhao, Hai}, title={Sparse Fuzzy Attention for Structured Sentiment Analysis}, year={2021}, eprint={2109.06719}, archivePrefix={arXiv}}
@misc{dogga2026smoe, author={Dogga, Bharadwaj and Shankar, Kaaustaaub and Raju, Gibin and Louw, Wilhelm and Cohen, Kelly}, title={Rule-Based Spatial Mixture-of-Experts {U-Net} for Explainable Edge Detection}, year={2026}, eprint={2602.05100}, archivePrefix={arXiv}}
@inproceedings{shi2024vlmtsk, author={Shi, Kuo and Lu, Jie and Fang, Zhen and Zhang, Guangquan}, title={Enhancing Vision-Language Models Incorporating {TSK} Fuzzy System for Domain Adaptation}, booktitle={IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)}, year={2024}, doi={10.1109/FUZZ-IEEE60900.2024.10612077}}
@inproceedings{zhou2026fmlc, author={Zhou, Yumin}, title={Fusing Deep Learning and Fuzzy Logic: A Framework for Adaptive and Scalable Interpretability}, booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}, volume={40}, year={2026}, doi={10.1609/aaai.v40i48.42178}}
@inproceedings{tsai2019dissection, author={Tsai, Yao-Hung Hubert and Bai, Shaojie and Yamada, Makoto and Morency, Louis-Philippe and Salakhutdinov, Ruslan}, title={Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel}, booktitle={EMNLP-IJCNLP}, pages={4343--4352}, year={2019}, doi={10.18653/v1/D19-1443}}
@article{yang2026genfs, author={Yang, Hailong and Deng, Zhaohong and Zhang, Wei and Zhao, Zhuangzhuang and Wang, Guanjin and Choi, Kup-Sze}, title={Generative Fuzzy System for Sequence-to-Sequence Learning via Rule-Based Inference}, journal={IEEE Transactions on Neural Networks and Learning Systems}, volume={37}, number={3}, pages={1435--1448}, year={2026}, doi={10.1109/TNNLS.2025.3615650}, note={Preprint arXiv:2411.13867}}
@misc{oskin2026ncffn, author={Oskin, Mark}, title={Explicit Fuzzy Logic in the Feed-Forward Layer: Self-Forgetting Quantifiers Discover Legible Grammatical-Licensing Detectors}, year={2026}, eprint={2606.31845}, archivePrefix={arXiv}}
@misc{huang2025semanticfusion, author={Huang, Yongchao and Raza, Hassan}, title={Semantic Fusion with Fuzzy-Membership Features for Controllable Language Modelling}, year={2025}, eprint={2509.13357}, archivePrefix={arXiv}}
@misc{zhang2026dfil, author={Zhang, Zhen and Alanwar, Amr}, title={Differentiable Fuzzy Inference Layer: A Monotone, Compositional Ordinal Reasoning Head for Large Language Models}, year={2026}, eprint={2609.26113}, archivePrefix={arXiv}}
@article{wang2025fcd, author={Wang, Shuai and Ding, Liang and Zhan, Yibing and Luo, Yong and Liu, Shuai and Ding, Weiping}, title={Fuzzy-Assisted Contrastive Decoding Improving Code Generation of Large Language Models}, journal={IEEE Transactions on Fuzzy Systems}, year={2025}, doi={10.1109/TFUZZ.2025.3575060}}
@misc{tarau2026arrow, author={Tarau, Paul}, title={Modeling Next-Token Prediction as Left-Nested Intuitionistic Implication}, year={2026}, eprint={2601.19915}, archivePrefix={arXiv}}
@article{zadeh1996cww, author={Zadeh, Lotfi A.}, title={Fuzzy Logic = Computing with Words}, journal={IEEE Transactions on Fuzzy Systems}, volume={4}, number={2}, pages={103--111}, year={1996}, doi={10.1109/91.493904}}
@article{takagi1985fuzzy, author={Takagi, Tomohiro and Sugeno, Michio}, title={Fuzzy Identification of Systems and Its Applications to Modeling and Control}, journal={IEEE Transactions on Systems, Man, and Cybernetics}, volume={SMC-15}, number={1}, pages={116--132}, year={1985}, doi={10.1109/TSMC.1985.6313399}}
@inproceedings{geva2021ffn, author={Geva, Mor and Schuster, Roei and Berant, Jonathan and Levy, Omer}, title={Transformer Feed-Forward Layers Are Key-Value Memories}, booktitle={EMNLP}, pages={5484--5495}, year={2021}, doi={10.18653/v1/2021.emnlp-main.446}}
@misc{cunningham2023sae, author={Cunningham, Hoagy and Ewart, Aidan and Riggs, Logan and Huben, Robert and Sharkey, Lee}, title={Sparse Autoencoders Find Highly Interpretable Features in Language Models}, year={2023}, eprint={2309.08600}, archivePrefix={arXiv}}
@misc{bricken2023monosemanticity, author={Bricken, Trenton and others}, title={Towards Monosemanticity: Decomposing Language Models With Dictionary Learning}, howpublished={Transformer Circuits Thread}, year={2023}, url={https://transformer-circuits.pub/2023/monosemantic-features/index.html}}
@misc{shazeer2017moe, author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff}, title={Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer}, year={2017}, eprint={1701.06538}, archivePrefix={arXiv}}
@misc{fedus2021switch, author={Fedus, William and Zoph, Barret and Shazeer, Noam}, title={Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity}, year={2021}, eprint={2101.03961}, archivePrefix={arXiv}}
@misc{olsson2022induction, author={Olsson, Catherine and Elhage, Nelson and Nanda, Neel and others}, title={In-context Learning and Induction Heads}, year={2022}, eprint={2209.11895}, archivePrefix={arXiv}}
@misc{pereira2023fuzzyfingerprint, author={Pereira, Patr{\'i}cia and Ribeiro, Rui and Moniz, Helena and Coheur, Luisa and Carvalho, Joao Paulo}, title={Fuzzy Fingerprinting Transformer Language-Models for Emotion Recognition in Conversations}, year={2023}, eprint={2309.04292}, archivePrefix={arXiv}}
```
