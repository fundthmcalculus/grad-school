# Chapter 6 — Hierarchical and Refined Fuzzy Models

*This chapter is partly built and partly proposed, and I mark which is which as I go. The shared solver, the fuzzy trees, the one-shot hierarchical mixture, the declarative structure specification, the Ruspini export, and the memory-augmented variant are implemented and evaluated. The EM refinement of the mixture and the full comparison suite are proposed for completion.*

## 6.1 Introduction

The flat model of Chapter 4 is fast and readable, but it is flat: every rule sees every feature, and there is no explicit sense in which some decisions come before others. For some problems that is fine. For others I want two things the flat model does not give me. The first is an explicit *hierarchy* of decisions over named variables — the kind of "first check this, then within that check that" structure a domain expert actually reasons with. The second is a little more accuracy on problems where the relationship is genuinely local, by letting different regions of the input space be handled by different experts. And whatever model I build, I may want the option to refine it.

All three of these turn out to rest on a single technical observation, which is the organizing idea of this chapter. For a Takagi–Sugeno–Kang system with fixed firing strengths, the output is *linear in the consequent coefficients*. That means I never have to search for the consequents: given the antecedents, the best consequents are the solution of a regularized weighted least-squares problem, in closed form. I use that one solver — a firing-weighted ridge least-squares — as a shared primitive everywhere: in the flat FIS of Chapter 4, in the leaves of a soft fuzzy tree, and in the experts of a hierarchical mixture. Reusing one well-understood primitive across three model shapes is most of what makes this chapter hang together.

The contributions, then, are: the shared ridge-TSK consequent solver (built); soft fuzzy trees whose leaves are exact ridge-TSK models (built); a hierarchical mixture of fuzzy experts, fit one-shot today and by EM as proposed; a declarative way to specify and steer the model structure, plus an export to an explicit triangular rule base (built); an antecedent-refinement stage that reuses the solver as its inner objective (in progress); and a memory-augmented variant that extends the whole apparatus to dynamical systems (built).

## 6.2 Background and Prior Art

The closest competitor, and the one I engage most carefully, is Medina-Chico et al. (2001), which builds a soft decision tree with *linear* models in the leaves. That is very nearly what my fuzzy tree does, so I have to be precise about the difference: their leaves are fit by backpropagation, and mine are the exact firing-weighted ridge solution in closed form. The daylight is "closed-form exact leaf" versus "iteratively fit leaf," not the tree idea itself, which I concede. On the mixture side, Wu et al. (2020) show that a single-layer TSK system is already equivalent to a mixture of experts, which narrows what I can claim for a *hierarchical* mixture, and I engage that directly rather than around. The rest of the lineage is standard: Jordan and Jacobs (1994) for the hierarchical mixture of experts and its EM; Janikow (1998) and Yuan and Shaw (1995) for fuzzy decision trees and the ambiguity split criterion; Olaru and Wehenkel (2003) for the finding that soft splits keep accuracy while staying interpretable.

There is one objection I have to meet head-on, because a good committee will raise it, and it is due to Magdalena (2018): a hierarchical fuzzy system is *not* automatically more interpretable, because the intermediate variables it introduces can be meaningless. My answer, which the method makes good on by construction, is that every split and every gate in my hierarchies is over an *original, named input* — never a synthetic intermediate. That is exactly the condition Magdalena requires for the interpretability claim to hold, and I hold to it deliberately.

One caution I impose on myself, from checking my own earlier reasoning: I do **not** motivate the ridge solver by claiming that ANFIS's least-squares step overfits. That premise did not survive verification. The honest motivation for the ridge solver is numerical conditioning and the value of one reusable primitive, not a deficiency in ANFIS.

## 6.3 Methodology

### 6.3.1 The shared ridge-TSK solver (built)

Because the TSK output is linear in the consequent coefficients for fixed firing strengths, fitting the consequents is a weighted least-squares problem, and I solve it with ridge-regularized normal equations — leaving the intercept and bucket-mean columns unpenalized. Two details matter in practice. I let the polynomial basis be orthogonal (Legendre or Chebyshev) rather than raw monomials, which conditions the problem far better at higher orders, and I select the order and the regularization strength by cross-validation. This one solver replaces the per-bucket pseudo-inverse and the iterative L-BFGS fit that came before it, and it is the primitive the next two subsections reuse.

### 6.3.2 Soft fuzzy trees (built)

The fuzzy tree is a CART-style recursive partition, with two differences from an ordinary decision tree. The splits are soft, so a point flows down multiple paths with graded membership rather than being sent left or right, and each leaf holds a full ridge-TSK model rather than a constant. The split criterion is firing-weighted variance reduction for regression, and a fuzzy ambiguity or information-gain measure for classification. The payoff is readability: the tree renders as a short list of IF–THEN rules, one per root-to-leaf path, each mentioning only the variables on that path and ordered by importance. On the Concrete dataset the tree splits first on cement content and then on age, right at the standard 28-day curing mark — which is to say it recovers domain knowledge nobody told it, and a materials engineer can read that and nod.

**[FIGURE 6.1 — placeholder]** *A trained fuzzy tree on Concrete, rendered as text rules, with the cement→age(28-day) split highlighted. Show the same for PhiUSIIL (HasSocialNet, HasCopyrightInfo, URLSimilarityIndex).*
`![fuzzy-tree](fig/06-fuzzy-tree.png)`

### 6.3.3 A hierarchical mixture of fuzzy experts (one-shot built; EM proposed)

The mixture generalizes the tree: instead of a single model per leaf, a tree of fuzzy partition-of-unity gates routes each point, softly, to leaves that are themselves full multi-rule TSK sub-models, with overlapping (soft-inclusion) training sets so the experts share boundary data. Today I fit this greedily, one shot. I have designed — but not yet implemented — a full EM refinement, with an E-step that assigns responsibilities from the product of gate probability and expert likelihood, and an M-step that updates each expert with a √h-weighted ridge solve and updates the gates in closed form, guarded against the usual underflow and starved-component pathologies. I am careful here: the novelty of this work does **not** rest on the EM estimator, which is standard; it rests on the composition. The EM is a proposed deliverable, and if it slips, the one-shot mixture and the trees stand on their own.

**[FIGURE 6.2 — placeholder]** *The hierarchical mixture structure: fuzzy gates over named inputs routing to TSK sub-experts. Emphasize that gates split only on original variables (the Magdalena condition).*
`![hme-structure](fig/06-hme-structure.png)`

### 6.3.4 Declarative structure and Ruspini export (built)

Two supporting pieces. The structure of a tree or mixture can be specified declaratively — a serializable plan with a clear precedence order (a pinned path beats a level ordering beats an automatic criterion), which lets an expert steer which variable gates where without touching code. And a trained model can be exported to an explicit Ruspini partition: the implicit per-class Gaussian mixture becomes a shared triangular strong partition of unity, after which I refine only the apex knots — a partition-preserving, piecewise-linear tuning that keeps the "memberships sum to one" property intact. The export is the interpretable-by-construction artifact: a clean triangular rule base a person can read directly.

### 6.3.5 Antecedent refinement, and what it taught me (in progress)

The one thing the fast construction leaves untuned is the antecedent parameters — the centers and widths of the membership functions. I refine them against a held-out-fold objective, using the ridge solver from §6.3.1 as the inner fitness, and I tried the obvious population methods: differential evolution and a real-valued genetic algorithm, with an optional gradient-descent polish. The honest finding is that a plain local optimizer (L-BFGS-B) usually beats the population methods here, because the population methods overfit the cross-validation estimate. I report this not as a disappointment but as evidence: it is a direct, small confirmation of the *structure before search* thesis — once the model is built from the data's structure, a global search has little left to find and mostly finds noise.

### 6.3.6 Memory for dynamical systems (built)

The last piece extends the same machinery to time. I augment each feature with a few summaries of its recent history — its current value, a short-term average, and a long-term average, plus an optional time index — which gives the model an LSTM-like sense of temporal context without any recurrent machinery. The model then predicts either one step ahead or, iterated, a whole trajectory. I validate this on dynamical systems — a double pendulum and an Atwood machine — which is a deliberately aerospace-flavored testbed, and the memory clearly helps: on the double pendulum it improves the coefficient of determination from about 0.92 to 0.96 and roughly cuts the trajectory error by a third.

**[FIGURE 6.3 — placeholder]** *Double-pendulum trajectory rollout: ground truth vs. memoryless FIS vs. memory-augmented FIS, showing the error reduction over the horizon.*
`![mimo-rollout](fig/06-mimo-rollout.png)`

## 6.4 Results and Proposed Experiments

> **TODO — repeatable performance (board-wide standard):** the training-time, accuracy, and speedup numbers here need the fixed reproducibility protocol and the full baseline suite before citation (see `ACTION_ITEMS.md` §A/§C and Ch 7 Goal G4/G3).

**What is measured today.** On Concrete regression, the flat model's $R^2$ is 0.658 (RMSE 9.38 MPa), the first-order fuzzy tree improves it to 0.746 (RMSE 8.09), and the hierarchical mixture reaches 0.791 (RMSE 7.34) — the most accurate of the three. On PhiUSIIL classification, the flat model is at 0.998 accuracy and the mixture at 0.996, with the tree splitting on interpretable signals. Antecedent refinement lifts the Concrete $R^2$ from roughly 0.88 to 0.92 (a number I flag for re-verification). And the memory result above.

> *Note on the flat Concrete baseline:* the flat $R^2 = 0.658$ reported here comes from the tree/mixture experiment and is not the same configuration as the flat MoG-TSK figures in Chapter 4 (0.44/0.77/0.87 at orders 0/1/2) — different split, preprocessing, and order selection. A single consistent Concrete benchmark, so the flat baseline reads identically across chapters, is a reconciliation TODO (see `ACTION_ITEMS.md` §A).

**Table 6.1 — Model family on Concrete and PhiUSIIL.** Model-family columns are measured; baselines are proposed (Goal G3) on identical splits.

| Dataset | metric | flat | fuzzy tree | mixture (HME) | ANFIS / CART / M5 |
|---|---|---:|---:|---:|:--:|
| Concrete | R² | 0.658 | 0.746 | **0.791** | _TODO_ |
| Concrete | RMSE (MPa) | 9.38 | 8.09 | **7.34** | _TODO_ |
| PhiUSIIL | accuracy | **0.998** | ~0.968 | 0.996 | _TODO_ |

**The honest scope.** I want to be plain about what the hierarchy buys and what it does not. On raw accuracy the tree and mixture do *not* beat the flat model in general — on PhiUSIIL the mixture is a hair *behind* the flat model — and they do not shrink the rule count below the already-compact flat model. What they buy is an explicit decision hierarchy over named variables and a readable path structure, and that payoff is real only at shallow depth and few terms, which is why I cap depth and leaf count. This is an interpretability-for-accuracy trade, made deliberately, and I would rather state it than let a reviewer discover it.

**What I propose to add.** Three things. Implement and evaluate the EM refinement of the mixture. Add the baselines a reviewer will demand — ANFIS, CART/C4.5, M5 model trees, flat TSK, and the recent Fumanal-Idocin (2025) and D-TSK-FC methods — on identical splits. And broaden the benchmark set beyond Concrete and PhiUSIIL to the other domains already scaffolded (turbine, wave-energy, wine, and the IoT sets), so the accuracy–interpretability trade is characterized across more than two problems. I also owe two literature searches — on knot/breakpoint optimization and on fuzzy mixtures-of-experts — to bound the novelty claims, and a small attribution fix in the references.

## 6.5 Discussion and Contributions

The defensible contribution here is architectural, not any single algorithm: one closed-form ridge primitive, reused across a flat FIS, a soft fuzzy tree, and a hierarchical mixture, with a clean export to a triangular rule base and an extension to temporal data. Every building block is prior art, and I credit each; the integration is the thing. The interpretability payoff is quantified against its real accuracy cost, and the Magdalena objection is answered by construction, not by assertion. This chapter is also where the pipeline closes: Chapter 5 supplies the antecedents when the data has no coordinates, this chapter supplies the consequents, the hierarchy, and the optional refinement, and Appendix A supplies the optimizers that do the polishing when polishing is wanted. What remains — the EM, the baselines, and the integrated end-to-end demonstration — is the subject of Chapter 7.

---

*Draft — Chapter 6 prose, in the author's voice; built vs. proposed marked throughout. Citations in bracketed shorthand pending the consolidated `references.bib`. Three figures and one table placeholder marked inline. Source outline in `../chapters/06-hierarchical-refined-fis.md`; open items in `../ACTION_ITEMS.md`.*
