# Prior art and reading list

Status column is honest about provenance. **[V]** = abstract page fetched and
title/authors/claims checked. **[S]** = seen in search results only, arXiv ID
plausible but not independently confirmed. Nothing here should reach a document
with a committee's name on it while still marked [S].

## The name collision — read first

| status | ref |
|---|---|
| [V] | Holsman, Huang, Dhingra. **Fuzzy Speculative Decoding for a Tunable Accuracy-Runtime Tradeoff.** arXiv:2502.20704 |

"Fuzzy" here is an English adjective for a divergence-bounded acceptance test.
No membership functions, no rule base, no fuzzy sets. Anyone searching the
obvious phrase finds this first, so the introduction has to distinguish it
explicitly and early.

## 1. Exact speculative decoding

| status | ref | note |
|---|---|---|
| [V] | Leviathan, Kalman, Matias. **Fast Inference from Transformers via Speculative Decoding.** ICML 2023. arXiv:2211.17192 | Algorithm 1 is the acceptance rule this project's §2.1 argument depends on — read the primary notation |
| [V] | Chen et al. (DeepMind). **Accelerating LLM Decoding with Speculative Sampling.** arXiv:2302.01318 | quote the "within hardware numerics" caveat precisely |
| [V] | Cai et al. **Medusa.** arXiv:2401.10774 | "typical acceptance" — precedent for deliberately abandoning exactness |
| [V] | Li et al. **EAGLE.** arXiv:2401.15077 · **EAGLE-2** arXiv:2406.16858 (3.05–4.26×) · **EAGLE-3** arXiv:2503.01840 (up to 6.5×) | |
| [V] | Fu et al. **Lookahead Decoding.** ICML 2024. arXiv:2402.02057 | |
| [V] | Kim et al. **Big Little Decoder.** NeurIPS 2023. arXiv:2302.07863 | ~2×; fallback/rollback thresholds |
| [S] | Sharma. **When Is a Draft Accepted?** arXiv:2606.30265 | six acceptance regimes with KL certificates; the theoretical spine if the work ever goes lossy |

## 2. Entropy-gated adaptive drafting — the crowded neighbour

This is the finding that most constrains novelty. The move "predict a cheap
statistic of distribution shape, gate a decoding decision on it" is taken, done
with small neural probes. None of it uses fuzzy logic.

| status | ref | reported |
|---|---|---|
| [V] | Agrawal, Jeon, Lee. **AdaEDL.** NeurIPS 2024 ENLSP workshop. arXiv:2410.18351 | 10–57% over static γ |
| [V] | Sen, Dasgupta, Waghela. **Confidence-Modulated Speculative Decoding.** IEEE INDISCON 2025. arXiv:2508.15371 | entropy + margin → γ and strictness |
| [S] | **SpecKV.** arXiv:2605.02888 | MLP on confidence/entropy → per-step γ; 56% over fixed γ=4 |
| [S] | **EntMTP.** arXiv:2606.27550 | 1.09–1.36× |
| [S] | **EASD.** arXiv:2512.23765 | claims to beat the target LLM outright — scrutinise the eval before citing |
| [S] | **Learning to Draft (RL).** arXiv:2603.01639 | |

## 3. Cheap and non-neural drafters — the benchmark set

| status | ref | reported |
|---|---|---|
| [V] | He et al. **REST: Retrieval-Based Speculative Decoding.** NAACL 2024. arXiv:2311.08252 | 1.62–2.36× |
| — | **Prompt Lookup Decoding** (Saxena) | GitHub + blog only, **not a paper**; claimed 2–4× on input-grounded tasks. Cite as an engineering artefact or not at all |
| [S] | **Adaptive N-gram Parallel Decoding.** arXiv:2404.08698 | |

Known failure mode, and the one V2 must be tested against: retrieval/n-gram
drafters depend on lexical overlap between context and continuation. They do
well on code and summarization and degrade on open-ended generation. Any claim
for an FIS drafter has to report both regimes.

## 4. Structure of the output distribution — grounding for the "shape" premise

| status | ref |
|---|---|
| [V] | Yang, Dai, Salakhutdinov, Cohen. **Breaking the Softmax Bottleneck.** ICLR 2018. arXiv:1711.03953 |
| [S] | **Sequences of Logits Reveal the Low Rank Structure of Language Models.** arXiv:2510.24966 |
| [V] | Noarov et al. **Foundations of Top-k Decoding for Language Models.** arXiv:2505.19371 |
| [V] | Plaksin et al. **SlimSpec: Low-Rank Draft LM-Head.** arXiv:2605.10453 |
| [S] | **Future Lens.** CoNLL 2023. arXiv:2311.04897 |
| [S] | **Semantic Entropy Probes.** arXiv:2406.15927 |
| [S] | **On the Entropy Calibration of Language Models.** arXiv:2511.11966 |

Note against arXiv:2402.14740: the widely-repeated "~60% of mass on top-1,
~90% on top-16" figure is attributed to it by secondary summaries and was
**not** confirmed inside the paper. Our own run measures 0.561 and 0.848 (top-10)
on SmolLM2-135M — directionally consistent, but cite our measurement, not theirs.

## 5. Vocabulary-scale engineering — adjacent, do not conflate

| status | ref | reported |
|---|---|---|
| [V] | Zhao et al. **FR-Spec.** arXiv:2502.14856 | 75% LM-head compute cut, 1.12× over EAGLE-2 |
| [V] | Goel et al. **VocabTrim.** ICML 2025 workshop. arXiv:2506.22694 | 16% memory-bound speedup |
| [S] | **DynaSpec.** arXiv:2510.13847 | up to 2.18× |

These solve "the drafter's output projection is too expensive", not "is shape a
sufficient statistic". Cite for awareness; keep separate from the contribution.

## 6. Type-2 fuzzy and probability

| status | ref |
|---|---|
| [V] | Mendel, John. **Type-2 Fuzzy Sets Made Simple.** IEEE TFS 10(2):117–127, 2002 |
| [S] | Wu, Mendel. **Uncertainty measures for interval type-2 fuzzy sets.** Information Sciences, 2007 |
| [V] | Pan, Bester. **Fuzzy Bayesian Learning.** IEEE TFS 26(3), 2018. arXiv:1610.09156 |
| [V] | Cui, Wu, Xu. **Curse of Dimensionality for TSK Fuzzy Neural Networks.** IJCNN 2021. arXiv:2102.04271 |

Wu & Mendel draw the FoU ↔ pdf analogy explicitly and define fuzzy
entropy/variance/skewness — useful vocabulary. But no work was found using
type-1 or type-2 fuzzy sets to *parameterise a categorical distribution over a
large discrete vocabulary for sampling*. That slot is unoccupied.

Cui et al. is a hazard to test for, not just cite: TSK defuzzification saturates
in high input dimension, and the failure is silent.

## 7. FIS + LLM generally — exists, different layer of the stack

| status | ref |
|---|---|
| [S] | **An uncertainty-aware framework integrating LLM and FIS for commonsense reasoning.** Expert Systems with Applications, S0957417426001867 — paywalled, needs institutional access. Closest FIS+LLM work in spirit |
| [V] | Chen et al. **Fuzzy Reasoning Chain.** arXiv:2509.22054 |
| [V] | Figueiredo. **A Fuzzy Logic Prompting Framework for LLMs.** arXiv:2508.06754 |
| [V] | Huang, Raza. **Semantic Fusion with Fuzzy-Membership Features.** arXiv:2509.13357 |

All use fuzzy logic as an input feature channel or a post-hoc reasoning layer.
None touch the output distribution or decoding.

## Gap statement

FIS as a speculative-decoding drafter is unoccupied — confirmed across multiple
search angles. But the adjacent slot (cheap scalar proxy for shape, used to gate
a decoding decision) is crowded, so the defensible contribution is narrower than
"nobody has done this": it is *an interpretable rule base substituted into a
known mechanism, with the accuracy cost against an MLP measured and stated.*
