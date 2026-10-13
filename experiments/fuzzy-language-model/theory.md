# Theory: sequence mixers as Takagi–Sugeno–Kang systems

This note states the correspondences the experiment is built on. Each is an
identity, not an approximation, unless marked otherwise, and each one is pinned
numerically by a test in [`test_models.py`](test_models.py). Prior art for each
step is in [`literature/`](literature/); §6 says what is and is not new.

Notation: a zero-order TSK system with rules $r = 1..R$, antecedent membership
$\mu_r(\cdot)$, rule weight (certainty factor) $w_r$ and constant consequent
$\theta_r$ outputs

$$ y(x) = \frac{\sum_r w_r\,\mu_r(x)\,\theta_r}{\sum_r w_r\,\mu_r(x)}. $$

A Gaussian antecedent with per-dimension widths is the product-t-norm conjunction
of one-dimensional terms, $\mu_r(x) = \prod_j \exp\!\big(-(x_j - c_{rj})^2 / 2s_{rj}^2\big)$:
"IF $x_1$ is about $c_{r1}$ AND … AND $x_d$ is about $c_{rd}$".

## 1. Softmax attention is a weighted zero-order TSK system with a dynamic rule base

For one query $q$ and past tokens $s \le t$ with keys $k_s$ and values $v_s$, use

$$ q\cdot k = \tfrac12\big(\|q\|^2 + \|k\|^2 - \|q-k\|^2\big). $$

Then

$$ \operatorname{softmax}_s\!\Big(\frac{q\cdot k_s}{\tau}\Big)
   = \frac{\mu_s(q)\,w_s}{\sum_{s'} \mu_{s'}(q)\,w_{s'}},\qquad
   \mu_s(q) = e^{-\|q-k_s\|^2/2\tau},\quad w_s = e^{\|k_s\|^2/2\tau}. $$

The $e^{\|q\|^2/2\tau}$ factor is common to every term and cancels. **Attention
output is therefore exactly a zero-order TSK system.**

* There is **one rule per context token**: IF query is about $k_s$ THEN output $v_s$.
* The rule weight grows with the key norm.
* The memberships are isotropic Gaussians of variance $\tau$ ($=\sqrt{d_h}$ in a
  standard transformer).

RoPE keeps the identity, because rotation preserves norms. The quadratic cost of
attention is the cost of a rule base that grows by one rule per token.
*(test: `test_softmax_attention_is_weighted_tsk`)*

The `gauss` mixer is this TSK system with the rule weight removed ($w_s \equiv 1$)
and with learned per-dimension widths, tied within each RoPE rotation pair so that
scaling commutes with rotation. It is implemented with the same fused kernel as
softmax attention, by augmenting $q \to [q/s, 1]$ and $k \to [k/s, -\|k/s\|^2/2]$.
*(test: `test_gauss_attention_is_explicit_tsk`)*

The kernel-smoother reading of attention (Tsai et al. 2019) and the
Gaussian–softmax link (Choromanski et al. 2021, Lemma 1) are prior art. What this
section adds is the TSK bookkeeping: the rule weight, the per-dimension widths, and
the dynamic rule base.

## 2. Linear attention with a non-negative feature map is a TSK system with a fixed rule base and recurrent consequents

Normalized linear attention (Katharopoulos et al. 2020) with feature map
$\phi:\mathbb R^{d_h}\to\mathbb R^F_{\ge 0}$ and decay $\alpha_t$ is

$$ S_t = \alpha_t S_{t-1} + \phi(k_t)v_t^\top,\quad z_t = \alpha_t z_{t-1} + \phi(k_t),\quad
   y_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}. $$

Read each coordinate $r$ of $\phi$ as the membership of a vector in rule $r$, and
define $\bar\theta_r(t) = S_{t,r}/z_{t,r}$, the decayed membership-weighted mean of
the values written into rule $r$. Then

$$ y_t = \sum_r \frac{\phi_r(q_t)\, z_{t,r}}{\sum_{r'} \phi_{r'}(q_t)\, z_{t,r'}}\;\bar\theta_r(t). $$

This is a zero-order TSK system with $F$ **fixed** rules. Its consequents are
**state**: rule $r$'s output is "what followed things like $A_r$ in this context".
Its rule weights $z_{t,r}$ are the accumulated evidence for rule $r$. The state size
is $F\times(d_v+1)$ whatever the context length, which is why the cost is linear.
*(test: `test_fuzzy_mixer_reads_as_tsk_with_recurrent_consequents`)*

Two readings follow.

* **Katharopoulos' elu+1 map is a fuzzy system with one rule per feature.** Here
  $\phi_j(x) = \mathrm{elu}(x_j)+1$ depends on coordinate $j$ alone, so rule $j$ is
  "$x_j$ is LARGE". Its membership is monotone, unbounded, and has a single
  antecedent. These are poor fuzzy sets: they are not normal, not bounded, and do
  not form a partition.
* **The FRLM `fuzzy` mixer chooses $\phi$ to be proper fuzzy sets.** $\phi$ is
  normalized Gaussian memberships over $R$ learned rules, $\phi_r(u) = \mu_r(u)/\sum_{r'}\mu_{r'}(u)$.
  The same antecedents fuzzify keys (write) and queries (read), so the implied
  attention kernel $K(q,k)=\sum_r \phi_r(q)\phi_r(k)$ is symmetric positive
  semidefinite. Normalization makes each written token a unit of mass distributed
  over the rules, as in fuzzy c-means.

**Compression view (§1 → §2).** Quadratic attention keeps one rule per token, and
the FRLM keeps $R$ rules. Replacing the per-token antecedents $\{k_s\}$ by their
fuzzy assignment to $R$ prototypes merges the dynamic rule base of §1 into a fixed
one. Each prototype rule inherits the membership-weighted mean consequent of the
tokens assigned to it. This is fuzzy-clustering compression of attention, which
links the language-model work to the scalable-clustering half of the dissertation.
(Clustered attention, Vyas et al. 2020, clusters queries for a related purpose with
no fuzzy reading; see `literature/02`.)

## 3. The delta rule on fuzzy features is online normalized-LMS training of TSK consequents

DeltaNet (Schlag et al. 2021; Yang et al. 2024) replaces accumulation with
error-correcting writes:

$$ S_t = \alpha_t S_{t-1} + \beta_t\big(v_t - \alpha_t S_{t-1}\phi(k_t)\big)\phi(k_t)^\top,\qquad y_t = S_t\,\phi(q_t). $$

With $\phi$ the normalized rule firing ($\sum_r\phi_r = 1$), column $r$ of $S$ is a
consequent $\theta_r$. The read $y_t = \sum_r \phi_r(q_t)\theta_r$ is *literally* a
zero-order TSK output, with no separate denominator. The write

$$ \theta_r \leftarrow \alpha\theta_r + \beta\,\phi_r(k_t)\Big(v_t - \textstyle\sum_{r'}\phi_{r'}(k_t)\,\alpha\theta_{r'}\Big) $$

is one step of normalized least-mean-squares on the consequents, fit to the pair
(input $k_t$, target $v_t$). That is the consequent-learning half of ANFIS's hybrid
rule (Jang 1993), run inside the context window. It is stable for $\beta\in(0,1)$
because $\|\phi\|_2 \le \|\phi\|_1 = 1$.

**So the `fuzzydelta` FRLM is a TSK system that trains its own consequents in
context.** Its antecedents (what situations exist) are learned across the corpus by
backprop. Its consequents (what to say in each situation) are learned within each
story by LMS. *(tests: `test_parallel_equals_recurrent[FuzzyDeltaMixer-*]`)*

Training uses a parallel form. Writing $S_t=\sum_{s\le t}\Gamma_{ts}u_s\phi(k_s)^\top$
gives the unit-lower-triangular system
$(I+\operatorname{diag}(\beta)(\operatorname{tril}(\Phi_K\Phi_K^\top,-1)\odot\Gamma))U=\operatorname{diag}(\beta)V$,
which takes one triangular solve per head (see `_DeltaMixer`).

## 4. The feed-forward layer as a TSK system

A two-layer ReLU MLP is a piecewise-linear function, and in one dimension it is
exactly a TSK system with triangular memberships (Bede, Kreinovich & Toth; see
`papers/nn-fis-equivalence/` and `experiments/fis-to-neural-net/`). In $d$
dimensions that exact equivalence needs simplicial memberships.

This experiment does not convert MLPs. It **replaces** them with a zero-order TSK
layer, $y = \sum_r \operatorname{softmax}_r(-\|(Px - c_r)/s_r\|^2/2)\,a_r$. That is a
normalized Gaussian RBF network (Jang & Sun 1993) on a learned antecedent projection
$P$. It is equivalently a key–value memory with distance keys and softmax
competition, the distance analogue of the "FFN layers are key–value memories"
reading (Geva et al. 2021).

## 5. What the model families are

| arm | sequence mixer | channel mixer | rule base | cost |
|---|---|---|---|---|
| `softmax`+`mlp` | weighted TSK, 1 rule/token, isotropic MFs | ReLU-ish MLP | dynamic | quadratic |
| `gauss`+`tsk` (**FLM**) | TSK, 1 rule/token, learned widths, no rule weight | TSK | dynamic | quadratic |
| `linear`+`mlp` | one-feature monotone "rules" (elu+1) | MLP | fixed | linear |
| `fuzzy`+`tsk` (**FRLM**, accumulating) | R Gaussian rules, mean consequents | TSK | fixed | linear |
| `fuzzydelta`+`tsk` (**FRLM**, delta) | R Gaussian rules, LMS consequents | TSK | fixed | linear |
| `delta`+`mlp` | DeltaNet | MLP | — | linear |
| `gru` | gated RNN | — | — | linear |

## 6. Novelty: what is and is not claimed

Source: [`literature/03_fuzzy_neural_and_prior_art.md`](literature/03_fuzzy_neural_and_prior_art.md).
Its closest-prior-art citations were re-checked by hand on 2026-10-08.

**Not new; cite, don't claim:**

* §1, *softmax attention as a TSK system*. Jiang, …, Lin (IEEE TFS 2025) derive softmax
  attention as normalized TSK firing and build a fuzzy attention layer. FISformer (IEEE
  TFS 2026) replaces self-attention with a Sugeno FIS for time series. The kernel reading
  is Tsai et al. (2019). What §1 adds is bookkeeping: the explicit rule weight
  exp(‖k‖²/2τ), the learned per-dimension widths that commute with RoPE, and the
  one-rule-per-token reading of the quadratic cost.
* *Neural weights ≡ fuzzy rules* in general: Jang & Sun 1993 (RBF), Benítez et al. 1997
  (sigmoid MLP), Mantas & Puche 2008 (multilayer FFN ≡ zero-order TSK), Wu et al. 2020
  (TSK ≡ MoE/CART/stacking), Bede, Kreinovich & Toth (ReLU ≡ triangular TSK).
* *Recurrent TSK networks*: RSONFIN (1999), RFNN (2000), TRFN (2002). None of them was
  applied to language.
* *Fuzzy generative text models*: FuzzyS2S/GenFS (IEEE TNNLS 2026) uses sequence-level
  rules whose consequents are whole Transformers. NC-FFN (Oskin, arXiv 2606.31845) puts
  fuzzy set operators and decaying fuzzy quantifiers in the FFN of a decoder-only LM. It
  keeps a GELU path, and reports that a fully Boolean FFN diverges. "Fuzzy language
  model" as a phrase also predates this (fuzzy class n-gram LMs).

**Appears open (as of 2026-10-08; re-check before the defense):**

1. A decoder-only causal LM in which **every** sequence-mixing and channel-mixing
   sublayer is a TSK system, with no conventional MLP path (the FLM / FRLM arms here).
   Given Oskin's divergence report, *that it trains at all* is a claim that needs the
   evidence in `RESULTS.md`.
2. A **token-level recurrent TSK state**: the §2 reading of linear attention with a
   fuzzy-partition feature map as a TSK system whose consequents are state. Clustered
   attention (Vyas et al. 2020) is the nearest neural analogue; it has no fuzzy
   partition and no recurrence.
3. §3: the delta rule on fuzzy-partition features as **in-context NLMS training of TSK
   consequents**. The delta rule's online-regression reading is standard (Schlag et al.
   2021). Only the fuzzy reading may be new, and only one targeted search has been run
   for it.

**Defensible phrasing:** "a causal language model in which every sublayer is a
Takagi–Sugeno–Kang fuzzy system, including a recurrent variant whose rule consequents
are fit in context by least mean squares". **Not** "the first fuzzy language model".
