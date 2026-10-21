# 04 — Compact and structured embeddings / output heads, and what maps onto fuzzy inference

*Literature review, 2026-10-20. Scope: the open item in `../PLAN_FUZZYLOCAL.md` (how to fuzzify the head of
the token table, where `emb_fall` / `e2e_all` cost +0.130 BPC alone and +0.277 on top of the fuzzy blocks at
30M characters). Method: web search plus fetches of arXiv / ACL Anthology pages. Every entry below was
returned by a search result or page that showed its title, authors, year and venue; what could not be checked
is marked **(unverified)** in place and listed in section 6. Small-scale numbers (<1M parameters) are given only
where a source showed them. For most methods none is published, and that is said.*

Current layout (from `flm/fuzzy_tokens.py` and the plan): V = 3,072, d = 48. 384 head tokens have crisp rows
(384 x 48 = 18,432). 2,688 tail tokens have E = 12 logits each (32,256) plus a 12 x 48 up-projection (576). Total
about 51.3K of the 99K. The tied output is `h . (row)`. A fully crisp table would be 3,072 x 48 = 147,456, so the
tiering is already forced, not optional.

---

## TL;DR

* **The tiered table is already adaptive input / adaptive softmax** (Baevski & Auli 2018; Grave et al. 2017):
  frequency-ordered tiers, a narrower width for the tail, a projection up. What we add is the fuzzy
  (softmax-over-prototypes) reading of the tier. That is new at the fuzzy level, not at the neural level.
* **I found nothing that fuzzifies a vocabulary softmax or embedding table.** Fuzzy c-means has been applied
  to word vectors after training (Atakishiyev & Reformat 2019) and to soft word classes for n-gram LMs
  (a Persian abstract, **unverified** author/date). Fuzzy-membership features exist on the input side of a tied
  LM (Huang & Raza 2025, in 03). State it as "I found none", not "none exists".
* **The convex-combination head has structural defects the literature names.** A score that is a convex
  combination of K prototype scores is bounded by the extreme prototype scores, has no per-token prior, and
  lives in a rank-d space (softmax bottleneck: Yang et al. 2018; Chang & McCallum 2022). The standard fixes are
  cheap: a per-token bias initialised at log-unigram (Meister et al. 2023), a learned scale (NormFace line), a
  class/tier gate (Goodman 2001), a mixture of softmaxes (Yang et al. 2018).
* **A Gaussian-RBF head is algebraically a dot-product head with a norm-dependent bias** (derivation in 3.4, mine).
  The crisp tied head is therefore already a zero-order TSK system with one rule per token. That reading is free.

---

## 1. Published methods that make the embedding / output layer compact or structured

| method | what it does | fuzzy-inference reading | cost / gain, and at small scale |
|---|---|---|---|
| **Adaptive input** (Baevski & Auli, arXiv:1809.10853, 2018) | Frequency tiers with shrinking widths, each projected to d. | Tiers = coarse partition of the vocabulary; projection = consequent map. | WikiText-103 perplexity 18.7, Billion Word 23.02 (abstract). Large scale only. |
| **Adaptive softmax** (Grave et al., arXiv:1609.04309, ICML 2017) | Head cluster of frequent words plus tail clusters scored in lower dimension; P(word) = P(cluster) P(word given cluster). | Product t-norm of a class firing and a within-class firing. | Near full-softmax accuracy, big speed-up. Designed for speed, not small-vocabulary parameter budgets. |
| **Class-based / hierarchical softmax** (Goodman, arXiv:cs/0108006, ICASSP 2001; Morin & Bengio, AISTATS 2005; Mnih & Hinton, NIPS 2008) | Predict class, then word given class. Morin & Bengio's expert tree was fast but "considerably worse"; Mnih & Hinton's learned tree recovered quality (per the latter's abstract). | Hard-class special case of a soft-class head. | The documented risk is a poor class assignment. |
| **Sampled softmax** (Jean et al., arXiv:1412.2007, 2014) | Importance sampling over a huge target vocabulary. | None (training-time device). | Irrelevant at V = 3K: the full softmax is one 3,072 x 48 matmul. Skip. |
| **Weight tying** (Press & Wolf, arXiv:1608.05859, EACL 2017) | Output matrix = input embedding. | One rule per token, shared antecedent and consequent. | Fewer parameters and lower perplexity at their scales. Idea 6 below relaxes it. |
| **Factorized embedding** (ALBERT, Lan et al., arXiv:1909.11942, ICLR 2020) | V x E then E x d. | Low-rank consequent map. | E = 128 best in their shared setting; BERT scale. Our tail table is this with E = 12. |
| **DeFINE** (Mehta et al., arXiv:1911.12385, ICLR 2020) | Deep factorized input embedding with skips. | Small rule layer between table and block. | 6-20% perplexity reduction over adaptive input (abstract). Input side only. |
| **Tensor-train embeddings** (Hrinchuk et al., arXiv:1901.10787, Findings of EMNLP 2020) | Table replaced by TT cores. | Separable (tensor-product) antecedents. | "Negligible drop" at large compression (abstract), mid-size models. No sub-1M data seen. |
| **Product quantization / multi-codebook** (Shu & Nakayama, arXiv:1711.01068, ICLR 2018; DPQ, Chen, Li & Sun, arXiv:1908.09756, ICML 2020) | Token = G discrete codes into G codebooks, learned with Gumbel-softmax / softmax relaxation. | Token = conjunction of G fuzzy sets, one per subspace (product t-norm); the relaxation is a membership. | 94-99% compression (NMT) and 14-238x (DPQ) at negligible loss (abstracts). Large vocabularies; not tested at V = 3K. |
| **Hash / Bloom / compositional embeddings** (Svenstrup et al., arXiv:1709.03933, NeurIPS 2017; Serrà & Karatzoglou, arXiv:1706.03993, RecSys 2017; Shi et al., arXiv:1909.02107, 2019, venue not checked) | Tokens pick k rows from a shared pool by hashing, with learned per-token importance weights. | Shared fuzzy sets with token-specific weights (soft union). | On par with full tables for million-token vocabularies. Collisions cost more when V is already small. |
| **Anchor & Transform** (Liang et al., arXiv:2003.08197, ICLR 2021) | Embedding = sparse combination of shared anchors. | **Closest published relative of our prototype table**: anchors = prototypes, sparse transform = memberships. | LM and text classification; large vocabularies. |
| **Subword-composed vectors** (fastText, Bojanowski et al., arXiv:1607.04606, 2016) | Word vector = sum of character n-gram vectors. | Token = conjunction of character-level memberships. | Handles unseen words; no output-head use reported in what I saw. |
| **Product-key memory, PEER, memory layers** (Lample et al., arXiv:1907.05242, NeurIPS 2019; He, arXiv:2407.04153, 2024; Berges et al., arXiv:2412.09764, ICML 2025) | Huge sparse key-value memory with product-key lookup. | Rule base with sparse top-k firing; keys = antecedent centres. | Add parameters; relevant to the FFN, not to this budget. |
| **kNN-LM** (Khandelwal et al., arXiv:1911.00172, ICLR 2020) | Interpolate the LM with a nearest-neighbour distribution from a datastore. | Instance-based rule base interpolated with the parametric model. | Perplexity 15.79 on WikiText-103 (abstract). Needs a datastore: outside a 99K budget. |
| **Engram / n-gram memory** (Cheng et al., arXiv:2601.07372, 2026) | "Modernised N-gram embeddings" with O(1) lookup, a sparsity axis beside MoE; 27B scale. | Hashed n-gram antecedent. | Mechanics (hashing, gate) **not read**, abstract only. Over-Tokenized Transformer (Huang et al., arXiv:2501.16975, ICML 2025) enlarges only the *input* n-gram vocabulary and reports a larger *output* vocabulary may hurt small models (search summary, **unverified against the paper**). Both add parameters; skip at 99K. |

**Read-across.** The compression methods target vocabularies of 10^5-10^6 and 10-100x savings. At V = 3,072 and
d = 48 the table is 147K and we need about 3x. Nobody reports <1M-parameter results for these, so the predicted
effects in section 4 are my estimates, not literature values.

---

## 2. Fuzzy and soft-cluster vocabulary layers: what exists

* **Soft word classes for n-gram LMs.** A Saarland colloquium abstract applies fuzzy c-means and possibilistic
  c-means to the 20,000 most frequent Persian words, builds class n-gram models, and reports a perplexity reduction
  against other fuzzy approaches and a speech-recognition gain
  ([page](https://www.coli.uni-saarland.de/projects/irtg/contents/Colloquium/WS-07/AbstractSMomtazi.txt)).
  The page states no title, author or date (the file name suggests a Momtazi abstract, 2007; **unverified**).
  Closest prior art to "fuzzy classes for the vocabulary"; count-based, not neural.
* **FCM on trained embeddings.** Atakishiyev & Reformat, *Analysis of Word Embeddings Using Fuzzy Clustering*,
  arXiv:1907.07672 (2019): FCM and Gustafson-Kessel on GloVe; fuzzy clustering is sensitive to high dimension and
  the fuzzifier must be tuned to work up to about 100 dimensions. **A warning for initialising prototypes by FCM at
  d = 48** (plan item 6): memberships drift toward 1/K unless m is tuned.
* **Fuzzy features on the input of a tied LM.** Huang & Raza, arXiv:2509.13357 (2025), in 03: head stays the
  transposed table; nothing on the output side is fuzzified.
* **Neuro-fuzzy with large label sets.** A search for Gaussian / RBF output layers on many-class problems returned
  only patents and generic overviews. RBF networks date to Broomhead & Lowe (1988, **unverified**, not opened) and
  the Jang & Sun 1993 equivalence in 03. No neural-LM use found.
* **Fuzzy tokenization.** Nothing found.

Verdict: a fuzzy-set token table inside an all-fuzzy LM is open as far as this search goes. Cite as nearest:
Anchor & Transform, the DPQ / Shu-Nakayama relaxed codebooks, and the soft word classes.

---

## 3. Why a softmax-over-prototypes head can lose likelihood, and the standard fixes

1. **Rank / softmax bottleneck.** A tied head makes the log-probability matrix (contexts x tokens) rank at most
   d + 1 = 49 (Yang et al. 2018, ICLR). Chang & McCallum (ACL 2022) show one hidden state cannot represent
   multi-mode distributions because other tokens' embeddings interfere. The prototype head does not lower the
   rank (K = 48 prototypes span the same 48 dimensions) but worsens the geometry, see 2. Fixes: mixture of softmaxes
   (Yang et al.), multi-facet softmax (Chang & McCallum), sigsoftmax (Kanai et al., arXiv:1805.10829, NeurIPS 2018:
   exponential times sigmoid, no added parameters; beat softmax and mixture of softmaxes in their LM runs).
2. **Bounded logits.** Token logit = sum_k mu_vk s_k with mu on the simplex, so it lies between min_k s_k and
   max_k s_k. A crisp row reaches |logit| up to ||e_v|| ||h|| and can be scaled per token. The convex head reaches
   a large logit gap only by going one-hot, the sharp few-rule regime that a softmax over logits trains slowly
   (ratios, not magnitudes). This fits the 300M result that the gap shrinks with data: a learning-speed problem,
   not a ceiling. Fixes: learned output scale / temperature; normalised heads with a learned scale (NormFace,
   Wang et al., arXiv:1704.06369, 2017; I did not see its scale values).
3. **No per-token bias, no frequency prior.** The tied dot-product head has no bias, and a convex head cannot say
   "token v is 100x more frequent" except through its position on the simplex. Meister et al., *A Natural Bias for
   Language Generation Models* (ACL 2023, short, pp. 243-255), set the final-layer bias to the log-unigram
   distribution and report faster learning and better performance (NMT, not LM; full author list **unverified**).
   The head tokens carry most characters, so a prior matters most where the fuzzy head hurt most.
4. **RBF identity.** A Gaussian firing head logit_v = -||h - e_v||^2/(2 s^2) + log pi_v expands to
   (h . e_v)/s^2 - ||e_v||^2/(2 s^2) - ||h||^2/(2 s^2) + log pi_v. The last term is constant over tokens, so a softmax
   over it is a dot-product head with scale 1/s^2 and bias b_v = log pi_v - ||e_v||^2/(2 s^2). The crisp tied head is
   already a normalised-RBF (zero-order TSK) system, and the bias carries both the rule weight and the row norm.
   (Algebra mine, checked by hand, not run.)
5. **Residual crisp part, mixtures, signed memberships, sharper memberships.** The plan lists the first, third and
   fourth; mixtures (Jacobs et al. 1991, in 03) appear as idea 7.

---

## 4. Ranked ideas (30M characters, five seeds; none repeats OPEN items 1-7)

Notation: N_h = 384 head tokens, N_t = 2,688 tail tokens, d = 48, E = 12, K prototypes. Cost = extra parameters over the
current embedding. Predicted effects are **my estimates** in BPC on the head-fuzzy arm (`e2e_all`, +0.332 vs the
hybrid recipe at 30M), not literature values.

**1. Per-token prior bias (rule weight), log-unigram initialised.**
*Reading:* each token is one rule; b_v is its certainty factor; by 3.4 a crisp head already carries it in the row norm.
logit_v = defuzzified score + b_v. *Cost:* head only 384; all tokens 3,072 (3.1%); membership-tied b_v = w . mu_v only
K = 48 + 12 = 60. *Predicted:* removes 30-60% of the head-fuzzy penalty (-0.10 to -0.20 on `e2e_all`); about -0.01 on
`e2e_tail`. Cheapest arm; run first. The tied variant separates "prior" from "per-token freedom".

**2. Output tier gate: a soft two-level (class-based) head.**
*Reading:* P(token) = P(tier | h) x P(token | tier), a product t-norm of a tier rule and a within-tier rule (Goodman;
Grave et al.). The tier firing is a free scalar function of h, so the head/tail mass split stops being a by-product
of bounded prototype scores. *Cost:* two tiers 48 + 1 = 49; three tiers (head / mid / tail) 2 x 48 + 2 = 98.
*Predicted:* -0.01 to -0.03 on `e2e_tail`, more on `e2e_all`. Test that probabilities still sum to one.

**3. Amplitude: learned output gain plus per-token gain.**
*Reading:* output scale = inverse defuzzifier width; per-token gain a_v = rule confidence, a free positive scalar that
restores the magnitude freedom the simplex removes. logit_v = g a_v sum_k mu_vk s_k. *Cost:* g: 1 (decouples output
scale from input scale); a_v on the 384 head tokens: 384. *Predicted:* -0.03 to -0.08 on `e2e_all`. Overlaps with 1,
so run 1 and 3 as a 2 x 2 factorial.

**4. Subword-composed tail table (byte / character memberships).**
*Reading:* a tail token is a conjunction (product t-norm) of per-position character fuzzy sets, fastText-style, with
softmax memberships over character prototypes; the same composition on the output side, so no per-token parameters.
*Cost:* 256 byte rows x E = 3,072 + per-position gains 8 x 12 = 96, about 3.2K against 32,256 now, **freeing about
29K parameters** (over half the block budget). *Predicted:* high variance. Token-level loss +0.02 to +0.06 before
reinvesting; net win only if 29K in the blocks buys more (depth did not; width did, b1_d64). About 35% chance of a net gain.
The plan's coherence sets already show the tail table costs about a point.

**5. Product-quantised head: G groups of fuzzy sets.**
*Reading:* token = AND over G subspaces of one fuzzy set each (G-conjunct antecedent, product t-norm); consequent =
sum over groups of the group's prototype score. Each group has its own softmax, so the logit range is about G times
that of one convex combination (the bounded-logit fix). *Cost* (G = 4, 12 prototypes of dimension 12 per group): logits
384 x 48 = 18,432 (unchanged), prototypes 4 x 12 x 12 = 576 vs 48 x 48 = 2,304, **-1,728**. *Predicted:* -0.05 to -0.12
on `e2e_all`; supported only by Shu-Nakayama and DPQ at large V.

**6. Untied prototypes: tied memberships, separate input and output prototypes.**
*Reading:* different antecedent space (input) and consequent singletons (output) for the same fuzzy token sets.
*Cost:* head K x d = 2,304 (2.3%); tail 12 x 48 = 576. *Predicted:* -0.02 to -0.06 on `e2e_all`. Differs from OPEN
item 5 (output-only fuzzy): memberships stay tied, only the prototype matrix splits. Press & Wolf's tying gains were
measured at larger scale; at d = 48 the saving may not pay.

**7. Two-component mixture head (mixture of fuzzy softmaxes).**
*Reading:* a TSK system whose consequents are distributions: M = 2 rules, gate = normalised firing (Jacobs et al. 1991;
Yang et al. 2018). *Cost:* second context vector from a rank-16 map 48 x 16 + 16 x 48 = 1,536, plus gate 48, about
1.6K. *Predicted:* -0.01 to -0.03. V/d = 64, so the bottleneck is real, but MoS gains were shown at PTB / WikiText
scale. Orthogonal to the membership issue; run after 1-3.

**8. Hashed multi-bucket tail (hash embeddings).**
*Reading:* a tail token = soft union of 2 shared bucket fuzzy sets with learned weights. *Cost:* 2 hashes into B = 256
buckets x E = 12 = 3,072, plus 2 x 2,688 weights, about 8.4K, **-24K** vs now. *Predicted:* worse than 4 (collisions are
arbitrary, subword overlap is linguistic), +0.05 or more before reinvesting. Kept as the negative control for 4.

Order of work: 1, 2, 3 (cheap; one at a time; add the log-unigram bias to the baseline before measuring anything else),
then 5, then 4. Each needs a manipulation check: print the learned b_v, g, a_v and the per-band loss (the plan's
diagnostics) to show the new knob moved.

---

## 5. Notes for the proposal

* The sub-100K, V = 3K setting is not covered by any paper above; claim it as unexplored, not as a replication.
* Frame the tiered table as adaptive input + adaptive softmax with fuzzy-set tiers; cite Baevski & Auli and Grave et al. first.
* Define "fuzzy head" carefully: by 3.4 the crisp tied head already is a normalised-RBF TSK system. The arms vary how the rule
  centres are parameterised (free rows vs memberships over prototypes), plus how the consequents are scaled.
* Re-run the prior-art search before quoting novelty (as README says for 03). Anchor & Transform and the soft-class
  n-gram LM are the nearest items.

## 6. Could not verify

* The Persian fuzzy-class abstract: no title, author or date on the page (author inferred from the file name only).
* Meister et al. 2023: first author and Stokowiec appeared in results; full author list not seen.
* Engram (2601.07372): authors and abstract verified; hashing and gating mechanics not read.
* Over-Tokenized Transformer: the output-vocabulary claim is from a search summary, not the paper text.
* Broomhead & Lowe (1988): not opened. Shi et al. (1909.02107): venue not checked.
* NormFace scale values; Morin & Bengio / Mnih & Hinton details (from search summaries of abstracts).
* Jang & Sun 1993 and Jacobs et al. 1991 are verified in 03, not re-opened here.
* Two arXiv ids I first guessed (1909.04120, 1911.00359) opened unrelated papers and were replaced by ids shown in
  search results; neither is cited.

## BibTeX

```bibtex
@misc{baevski2018adaptive, author={Baevski, Alexei and Auli, Michael}, title={Adaptive Input Representations for Neural Language Modeling}, year={2018}, eprint={1809.10853}, archivePrefix={arXiv}}
@inproceedings{grave2017adaptive, author={Grave, {\'E}douard and Joulin, Armand and Ciss{\'e}, Moustapha and Grangier, David and J{\'e}gou, Herv{\'e}}, title={Efficient Softmax Approximation for {GPUs}}, booktitle={ICML}, series={PMLR 70}, pages={1302--1310}, year={2017}, eprint={1609.04309}}
@misc{goodman2001classes, author={Goodman, Joshua}, title={Classes for Fast Maximum Entropy Training}, howpublished={ICASSP 2001}, year={2001}, eprint={cs/0108006}, archivePrefix={arXiv}}
@inproceedings{morin2005hierarchical, author={Morin, Frederic and Bengio, Yoshua}, title={Hierarchical Probabilistic Neural Network Language Model}, booktitle={AISTATS}, series={PMLR R5}, pages={246--252}, year={2005}}
@inproceedings{mnih2008hlbl, author={Mnih, Andriy and Hinton, Geoffrey}, title={A Scalable Hierarchical Distributed Language Model}, booktitle={NIPS 21}, pages={1081--1088}, year={2008}}
@misc{jean2014vocab, author={Jean, S{\'e}bastien and Cho, Kyunghyun and Memisevic, Roland and Bengio, Yoshua}, title={On Using Very Large Target Vocabulary for Neural Machine Translation}, year={2014}, eprint={1412.2007}, archivePrefix={arXiv}}
@inproceedings{press2017tying, author={Press, Ofir and Wolf, Lior}, title={Using the Output Embedding to Improve Language Models}, booktitle={EACL (short)}, pages={157--163}, year={2017}, eprint={1608.05859}}
@inproceedings{lan2020albert, author={Lan, Zhenzhong and Chen, Mingda and Goodman, Sebastian and Gimpel, Kevin and Sharma, Piyush and Soricut, Radu}, title={{ALBERT}: A Lite {BERT} for Self-supervised Learning of Language Representations}, booktitle={ICLR}, year={2020}, eprint={1909.11942}}
@inproceedings{mehta2020define, author={Mehta, Sachin and Koncel-Kedziorski, Rik and Rastegari, Mohammad and Hajishirzi, Hannaneh}, title={{DeFINE}: Deep Factorized Input Token Embeddings for Neural Sequence Modeling}, booktitle={ICLR}, year={2020}, eprint={1911.12385}}
@inproceedings{hrinchuk2020tt, author={Hrinchuk, Oleksii and Khrulkov, Valentin and Mirvakhabova, Leyla and Orlova, Elena and Oseledets, Ivan}, title={Tensorized Embedding Layers}, booktitle={Findings of EMNLP}, year={2020}, eprint={1901.10787}}
@inproceedings{shu2018dccl, author={Shu, Raphael and Nakayama, Hideki}, title={Compressing Word Embeddings via Deep Compositional Code Learning}, booktitle={ICLR}, year={2018}, eprint={1711.01068}}
@inproceedings{chen2020dpq, author={Chen, Ting and Li, Lala and Sun, Yizhou}, title={Differentiable Product Quantization for End-to-End Embedding Compression}, booktitle={ICML}, series={PMLR 119}, pages={1617--1626}, year={2020}, eprint={1908.09756}}
@inproceedings{svenstrup2017hash, author={Svenstrup, Dan and Hansen, Jonas Meinertz and Winther, Ole}, title={Hash Embeddings for Efficient Word Representations}, booktitle={NeurIPS}, year={2017}, eprint={1709.03933}}
@inproceedings{serra2017bloom, author={Serr{\`a}, Joan and Karatzoglou, Alexandros}, title={Getting Deep Recommenders Fit: Bloom Embeddings for Sparse Binary Input/Output Networks}, booktitle={RecSys}, year={2017}, eprint={1706.03993}}
@misc{shi2019qr, author={Shi, Hao-Jun Michael and Mudigere, Dheevatsa and Naumov, Maxim and Yang, Jiyan}, title={Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems}, year={2019}, eprint={1909.02107}, archivePrefix={arXiv}, note={venue not checked}}
@inproceedings{liang2021ant, author={Liang, Paul Pu and Zaheer, Manzil and Wang, Yuan and Ahmed, Amr}, title={Anchor \& Transform: Learning Sparse Embeddings for Large Vocabularies}, booktitle={ICLR}, year={2021}, eprint={2003.08197}}
@misc{bojanowski2016fasttext, author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas}, title={Enriching Word Vectors with Subword Information}, year={2016}, eprint={1607.04606}, archivePrefix={arXiv}}
@inproceedings{lample2019pkm, author={Lample, Guillaume and Sablayrolles, Alexandre and Ranzato, Marc'Aurelio and Denoyer, Ludovic and J{\'e}gou, Herv{\'e}}, title={Large Memory Layers with Product Keys}, booktitle={NeurIPS}, year={2019}, eprint={1907.05242}}
@misc{he2024peer, author={He, Xu Owen}, title={Mixture of A Million Experts}, year={2024}, eprint={2407.04153}, archivePrefix={arXiv}}
@inproceedings{berges2025memory, author={Berges, Vincent-Pierre and O{\u{g}}uz, Barlas and Haziza, Daniel and Yih, Wen-tau and Zettlemoyer, Luke and Ghosh, Gargi}, title={Memory Layers at Scale}, booktitle={ICML}, series={PMLR 267}, pages={3831--3842}, year={2025}, eprint={2412.09764}}
@inproceedings{khandelwal2020knnlm, author={Khandelwal, Urvashi and Levy, Omer and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike}, title={Generalization through Memorization: Nearest Neighbor Language Models}, booktitle={ICLR}, year={2020}, eprint={1911.00172}}
@misc{cheng2026engram, author={Cheng, Xin and others}, title={Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models}, year={2026}, eprint={2601.07372}, archivePrefix={arXiv}}
@inproceedings{huang2025overtok, author={Huang, Hongzhi and Zhu, Defa and others}, title={Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling}, booktitle={ICML}, year={2025}, eprint={2501.16975}}
@inproceedings{yang2018mos, author={Yang, Zhilin and Dai, Zihang and Salakhutdinov, Ruslan and Cohen, William W.}, title={Breaking the Softmax Bottleneck: A High-Rank {RNN} Language Model}, booktitle={ICLR}, year={2018}, eprint={1711.03953}}
@inproceedings{chang2022mfs, author={Chang, Haw-Shiuan and McCallum, Andrew}, title={Softmax Bottleneck Makes Language Models Unable to Represent Multi-mode Word Distributions}, booktitle={ACL}, pages={8048--8073}, year={2022}}
@inproceedings{kanai2018sigsoftmax, author={Kanai, Sekitoshi and Fujiwara, Yasuhiro and Yamanaka, Yuki and Adachi, Shuichi}, title={Sigsoftmax: Reanalysis of the Softmax Bottleneck}, booktitle={NeurIPS}, year={2018}, eprint={1805.10829}}
@inproceedings{meister2023natural, author={Meister, Clara and others}, title={A Natural Bias for Language Generation Models}, booktitle={ACL (short)}, pages={243--255}, year={2023}, note={author list not fully verified}}
@misc{wang2017normface, author={Wang, Feng and Xiang, Xiang and Cheng, Jian and Yuille, Alan L.}, title={{NormFace}: {L2} Hypersphere Embedding for Face Verification}, year={2017}, eprint={1704.06369}, archivePrefix={arXiv}}
@misc{atakishiyev2019fuzzyemb, author={Atakishiyev, Shahin and Reformat, Marek Z.}, title={Analysis of Word Embeddings Using Fuzzy Clustering}, year={2019}, eprint={1907.07672}, archivePrefix={arXiv}}
```
