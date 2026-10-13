# Literature review — tiny, linear-attention, and fuzzy language models

Three reviews, written 2026-10-08. Each carries its own BibTeX block. Every
citation was resolved against arXiv, Crossref or OpenAlex, or fetched directly;
anything that could not be checked is marked **(unverified)** in place. Fuzzy-LM
preprints are appearing quickly, so **re-run the prior-art search (03) before
quoting any novelty claim at the defense.**

| file | covers | the result that matters most |
|---|---|---|
| [`01_tiny_language_models.md`](01_tiny_language_models.md) | TinyStories; the ~19K-param Reddit model; llama2.c and community sub-1M models; small-scale scaling laws; vocabulary vs size | The Reddit model is **MacroStories** (19,969 params, one shared block ×4, d=32, 378-word vocab). It was trained on its own Gemma-generated stories, **not** TinyStories, on a GPU, and reports no loss. Community sub-1M TinyStories points reach ≈2.0 bits/byte at ~10K params, ≈1.2 at ~100K, ≈0.8 at ~1M (single-seed, self-reported). |
| [`02_linear_attention_and_recurrence.md`](02_linear_attention_and_recurrence.md) | Attention as a kernel smoother; linear attention (Katharopoulos, Performer, RFA, cosFormer, RetNet, GLA, DeltaNet, Hedgehog, Based); SSMs/RNNs (RWKV, Mamba, xLSTM, minGRU) | At small scale, softmax still beats plain linear attention. Decay/gating and the delta rule close most of the gap, and the choice of feature map matters less. Recall is the main deficit of linear attention. |
| [`03_fuzzy_neural_and_prior_art.md`](03_fuzzy_neural_and_prior_art.md) | FIS↔NN equivalences; recurrent fuzzy NNs; fuzzy attention; fuzzy LMs; **novelty assessment** | **"First fuzzy language model" is not defensible.** FuzzyS2S/GenFS (generative fuzzy seq2seq) and NC-FFN (fuzzy-operator FFN in a 125M decoder LM) exist. "Attention is a TSK system" is published (Jiang…Lin, TFS 2025). The open claim is narrower: a causal LM whose every sublayer is a TSK system, with a token-level recurrent TSK state. |

Corrections made by hand after the agents' drafts:

* 03: Oskin (arXiv 2606.31845) reports that a fully **Boolean** FFN diverges; the
  draft said "fully fuzzy". Corrected against the abstract.
* 03: added **Mantas & Puche (2008)**, which proves multilayer FFN ≡ zero-order TSK. It
  turned up during a Crossref check and the agent had missed it.
* Found by the 03 review and fixed 2026-10-08: `papers/nn-fis-equivalence/references.bib`
  had the wrong authors for arXiv 1903.10572. It now lists Wu, Lin, Huang and Zeng, per
  arXiv and Crossref, and cites the published version, IEEE TFS 28(10):2570–2580, 2020.

Additional sources checked directly for [`../theory.md`](../theory.md):

* Vyas, Katharopoulos & Fleuret, *Fast Transformers with Clustered Attention*,
  [arXiv:2007.04825](https://arxiv.org/abs/2007.04825) (2020). It clusters queries
  and computes attention per centroid. This is the nearest neural analogue of the
  FRLM's "compress per-token rules into R prototype rules", without the fuzzy
  partition and without recurrence.
* One targeted search (2026-10-08) found no prior statement that the delta rule on
  normalized fuzzy-membership features is in-context NLMS training of zero-order
  TSK consequents. The delta rule's reading as online regression is standard
  (Schlag et al. 2021). Treat the fuzzy reading as **not yet checked for
  priority**, not as novel.
