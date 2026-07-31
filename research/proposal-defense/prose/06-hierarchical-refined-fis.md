# Chapter 6 — Hierarchical and Refined Fuzzy Models

*This chapter is partly built and partly proposed, and I mark which is which as I go. The shared solver, the fuzzy trees, the one-shot hierarchical mixture, the declarative structure specification, the Ruspini export, and the memory-augmented variant are implemented and evaluated. The antecedent refinement is built but still being broadened. The EM refinement of the mixture and the full comparison suite are proposed for completion.*

## 6.1 Introduction

The flat model of Chapter 4 is fast and readable, but it is flat: every rule sees every feature, and there is no explicit sense in which some decisions come before others. For some problems that is fine. For others I want two things the flat model does not give me. The first is an explicit *hierarchy* of decisions over named variables — the kind of "first check this, then within that check that" structure a domain expert actually reasons with. The second is a little more accuracy on problems where the relationship is genuinely local, by letting different regions of the input space be handled by different experts. And whatever model I build, I may want the option to refine it.

All three of these turn out to rest on a single technical observation, which is the organizing idea of this chapter. For a Takagi–Sugeno–Kang system with fixed firing strengths, the output is *linear in the consequent coefficients*. That means I never have to search for the consequents: given the antecedents, the best consequents are the solution of a regularized weighted least-squares problem, in closed form. I use that one solver — a firing-weighted ridge least-squares — as a shared primitive everywhere: in the flat FIS of Chapter 4, in the leaves of a soft fuzzy tree, and in the experts of a hierarchical mixture. Reusing one well-understood primitive across three model shapes is most of what makes this chapter hang together.

The contributions, then, are: the shared ridge-TSK consequent solver (built); soft fuzzy trees whose leaves are exact ridge-TSK models (built); a hierarchical mixture of fuzzy experts, fit one-shot today and by EM as proposed; a declarative way to specify and steer the model structure, plus an export to an explicit triangular rule base (built); an antecedent-refinement stage that reuses the solver as its inner objective (built; being extended); and a memory-augmented variant that extends the whole apparatus to dynamical systems (built).

## 6.2 Background and Prior Art

The closest competitor, and the one I engage most carefully, is Medina-Chico et al. (2001), which builds a soft decision tree with *linear* models in the leaves. That is very nearly what my fuzzy tree does, so I have to be precise about the difference: their leaves are fit by backpropagation, and mine are the exact firing-weighted ridge solution in closed form. The daylight is "closed-form exact leaf" versus "iteratively fit leaf," not the tree idea itself, which I concede. On the mixture side, Wu et al. (2020) show that a single-layer TSK system is already equivalent to a mixture of experts, which narrows what I can claim for a *hierarchical* mixture, and I engage that directly rather than around. The rest of the lineage is standard: Jordan and Jacobs (1994) for the hierarchical mixture of experts and its EM; Janikow (1998) and Yuan and Shaw (1995) for fuzzy decision trees and the ambiguity split criterion; Olaru and Wehenkel (2003) for the finding that soft splits keep accuracy while staying interpretable.

There is one objection I have to meet head-on, because a good committee will raise it, and it is due to Magdalena (2018): a hierarchical fuzzy system is *not* automatically more interpretable, because the intermediate variables it introduces can be meaningless. My answer, which the method makes good on by construction, is that every split and every gate in my hierarchies is over an *original, named input* — never a synthetic intermediate. That is exactly the condition Magdalena requires for the interpretability claim to hold, and I hold to it deliberately.

One caution I impose on myself, from checking my own earlier reasoning: I do **not** motivate the ridge solver by claiming that ANFIS's least-squares step overfits. That premise did not survive verification. The honest motivation for the ridge solver is numerical conditioning and the value of one reusable primitive, not a deficiency in ANFIS.

## 6.3 Methodology

### 6.3.1 The shared ridge-TSK solver (built)

Because the TSK output is linear in the consequent coefficients for fixed firing strengths, fitting the consequents is a weighted least-squares problem, and I solve it with ridge-regularized normal equations — leaving the intercept and bucket-mean columns unpenalized. Two details matter in practice. I let the polynomial basis be orthogonal (Legendre or Chebyshev) rather than raw monomials, which conditions the problem far better at higher orders, and I select the order and the regularization strength by cross-validation. This one solver replaces the per-bucket pseudo-inverse and the iterative L-BFGS fit that came before it, and it is the primitive the next two subsections reuse.

### 6.3.2 Soft fuzzy trees (built)

The fuzzy tree is a CART-style recursive partition, with two differences from an ordinary decision tree. The splits are soft, so a point flows down multiple paths with graded membership rather than being sent left or right, and each leaf holds a full ridge-TSK model rather than a constant. The split criterion is firing-weighted variance reduction for regression, and a fuzzy ambiguity or information-gain measure for classification. The payoff is readability: the tree renders as a short list of IF–THEN rules, one per root-to-leaf path, each mentioning only the variables on that path and ordered by importance. On the Concrete dataset the tree splits first on cement content and then on age, right at the standard 28-day curing mark — which is to say it recovers domain knowledge nobody told it, and a materials engineer can read that and nod.

That readability is the reason the tree is fit on *raw*, untransformed features while the flat model of Chapter 4 is not. A threshold of "cement ≥ 350 kg/m³" is something an engineer can check; "cement ≥ 0.42" after standardization is not. The tree can afford the choice, because axis-aligned splits are rank-based and therefore invariant to monotone transforms — Chapter 4 §4.3 measures this directly and finds the transform worth exactly nothing to CART and Random Forest, against as much as +0.15 $R^2$ to the Gaussian models. The mixture is the interesting middle case: its gates are tree-like but its experts are Gaussian, and it gains +0.089 from the transform, which is roughly what one would predict from that split of machinery.

**[FIGURE 6.1 — placeholder]** *A trained fuzzy tree on Concrete, rendered as text rules, with the cement→age(28-day) split highlighted. Show the same for PhiUSIIL (HasSocialNet, HasCopyrightInfo, URLSimilarityIndex).*
`![fuzzy-tree](fig/06-fuzzy-tree.png)`

### 6.3.3 A hierarchical mixture of fuzzy experts (one-shot built; EM proposed)

The mixture generalizes the tree: instead of a single model per leaf, a tree of fuzzy partition-of-unity gates routes each point, softly, to leaves that are themselves full multi-rule TSK sub-models, with overlapping (soft-inclusion) training sets so the experts share boundary data.

Today I fit this greedily, in one shot: the gates are chosen by the same split criteria as the tree, then each expert is fit to the points that reach it. That is fast and it works, but it is not a joint optimum — the gates never get to reconsider themselves in light of how well the experts they created actually perform. The point of the EM refinement is precisely to close that loop, which is why it is the chapter's main proposed deliverable rather than a nicety.

I have designed it but not implemented it. The E-step assigns each point a responsibility $h_{i\ell}$ for expert $\ell$, proportional to the product of that expert's gate probability and its likelihood under the expert's own model. The M-step then re-solves each expert by the same ridge primitive as §6.3.1, with every row scaled by $\sqrt{h_{i\ell}}$ so that a weighted least-squares solve implements the responsibility weighting exactly, and updates the Gaussian gate parameters in closed form. The design includes the guards this class of algorithm always needs: log-sum-exp accumulation against underflow, variance floors, and pruning of components that starve. What I expect it to buy is a better-conditioned fit than the greedy pass and a principled treatment of the boundary overlap that soft-inclusion currently handles by heuristic.

I am careful about one thing here: the novelty of this work does **not** rest on the EM estimator, which is entirely standard since Jordan and Jacobs. It rests on the composition — that the M-step reduces to the same shared ridge primitive used everywhere else in the chapter. The EM is a proposed deliverable, and if it slips, the one-shot mixture and the trees stand on their own as completed contributions.

**[FIGURE 6.2 — placeholder]** *The hierarchical mixture structure: fuzzy gates over named inputs routing to TSK sub-experts. Emphasize that gates split only on original variables (the Magdalena condition).*
`![hme-structure](fig/06-hme-structure.png)`

### 6.3.4 Declarative structure and Ruspini export (built)

Two supporting pieces, both about putting a human in control of the result.

The structure of a tree or mixture can be specified declaratively — a serializable plan with a clear precedence order, in which a pinned path beats a level ordering, which in turn beats an automatic criterion. This lets a domain expert dictate which variable gates where without touching code, and it is the mechanism by which the Magdalena condition of §6.2 is *enforced* rather than merely respected: the plan can only name original inputs, so no synthetic intermediate can appear in a gate even by accident.

Separately, a trained model can be exported to an explicit Ruspini partition. Here the input is the per-class Gaussian mixture of Chapter 4 rather than the tree itself — the export applies to the flat model, and I note that because it is easy to assume otherwise in a chapter about hierarchies. The implicit mixture becomes a shared triangular strong partition of unity; I then refine only the apex knots, which is a partition-preserving, piecewise-linear tuning that keeps the memberships summing to one throughout. The result is the interpretable-by-construction artifact of the whole dissertation: a clean triangular rule base over named variables that a person can read, check, and edit by hand.

### 6.3.5 Antecedent refinement, and what it taught me (built; being extended)

The one thing the fast construction leaves untuned is the antecedent parameters — the centers and widths of the membership functions. I refine them against a held-out-fold objective, using the ridge solver from §6.3.1 as the inner fitness, and I tried the obvious population methods: differential evolution and a real-valued genetic algorithm, with an optional gradient-descent polish. This machinery runs and produces the improvement quoted in §6.4; what remains is broadening it beyond the one dataset and re-verifying the number under the repeatability protocol, which is why I label it built rather than finished.

The honest finding is that a plain local optimizer (L-BFGS-B) usually beats the population methods here, because the population methods overfit the cross-validation estimate — they find configurations that score well on the folds used to select them and worse on held-out data. I report this not as a disappointment but as evidence, and it is worth being clear about what it is evidence *for*. It is a small, direct confirmation of the *structure before search* thesis: once the model has been built from the data's own structure, a global search has very little left to find, and what it does find is substantially noise. The negative result is the point. Had a genetic algorithm produced a large gain here, it would have suggested the structure-first construction was leaving real accuracy on the table.

### 6.3.6 Memory for dynamical systems (built)

The last piece extends the same machinery to time. I augment each feature with a few summaries of its recent history — its current value, a short-term average, and a long-term average, plus an optional time index — which gives the model an LSTM-like sense of temporal context without any recurrent machinery. The model then predicts either one step ahead or, iterated, a whole trajectory. I validate this on dynamical systems — a double pendulum and an Atwood machine — which is a deliberately aerospace-flavored testbed, and the memory clearly helps: on the double pendulum it improves the coefficient of determination from about 0.92 to 0.96 and roughly cuts the trajectory error by a third.

**[FIGURE 6.3 — placeholder]** *Double-pendulum trajectory rollout: ground truth vs. memoryless FIS vs. memory-augmented FIS, showing the error reduction over the horizon.*
`![mimo-rollout](fig/06-mimo-rollout.png)`

## 6.4 Results and Proposed Experiments

> **Reproduction.** Tables 6.1–6.3 regenerate from `reproduce/tables/table_6_1_model_family.py`, which emits Markdown and CSV with mean ± standard deviation across a fixed seed set and fills the baseline columns from scikit-learn (CART, Random Forest) plus an optional M5 adapter. Cells marked *pending* are those whose adapter was not yet wired up; the harness reports what it could not run rather than substituting a guess.
>
> **TODO — repeatable performance (board-wide standard):** the training-time, accuracy, and speedup numbers here need the fixed reproducibility protocol and the full baseline suite before citation (see `ACTION_ITEMS.md` §A/§C and Ch 7 Goal G4/G3).

**What is measured today.** On Concrete regression, the flat model's $R^2$ is 0.658 (RMSE 9.38 MPa), the first-order fuzzy tree improves it to 0.746 (RMSE 8.09), and the hierarchical mixture reaches 0.791 (RMSE 7.34) — the most accurate of the three. On PhiUSIIL classification, the flat model is at 0.998 accuracy and the mixture at 0.996, with the tree splitting on interpretable signals. And the memory result above.

> *A necessary caution about Concrete numbers, because three different figures for "the flat model" appear across this document and they are not comparable.* Chapter 4 reports flat MoG-TSK at $R^2$ 0.44 / 0.77 / 0.87 for TSK orders 0 / 1 / 2. This chapter's Table 6.1 reports a flat baseline of 0.658. The antecedent-refinement experiment of §6.3.5 reports a lift from roughly 0.88 to 0.92. All three are real measurements of different configurations — different splits, preprocessing, consequent order, and in the refinement case a different objective — and none of them is directly comparable to the others.
>
> This matters most for one apparent contradiction a careful reader will notice: refinement reaching 0.92 looks like it beats the hierarchical mixture's 0.791, which would make the whole chapter pointless. It does not. The 0.92 is a refined *flat, higher-order* model measured in the Chapter 4 configuration, while 0.791 is the mixture measured in the tree/mixture configuration whose flat baseline is 0.658. Comparing them is meaningless. Running one consistent Concrete benchmark, so that every model in the document is measured the same way and the numbers can be read against each other, is the single most important housekeeping item I owe (`ACTION_ITEMS.md` §A). Until it is done I quote the refinement result only as a *relative* improvement — roughly four points of $R^2$ over its own baseline — and not as an absolute to be ranked against the others.

**Table 6.1 — The model family on Concrete, under one protocol.** Regenerated by the reproduction harness at the configuration `demo_concrete.py` specifies, with the preprocessing of Chapter 4 §4.3 applied uniformly; 3 seeds, shared splits.

| Model | R² | RMSE (MPa) |
|---|---:|---:|
| Flat MoG-TSK (full 2nd) | **0.881 ± 0.001** | **5.54** |
| Mixture of experts (HME) | 0.862 ± 0.022 | 6.30 |
| Fuzzy tree | 0.717 ± 0.039 | 8.92 |
| CART (reference) | 0.797 ± 0.029 | 7.19 |
| Random Forest (reference) | 0.904 ± 0.014 | 4.94 |

These supersede the figures this chapter previously carried (flat 0.658, tree 0.746, mixture 0.791), which came from three different configurations and could not be read against one another. Two things change with the correction, and I would rather report both than the flattering one.

The mixture does *not* beat the flat model. At 0.862 against 0.881 it is close — within about one standard deviation — but the earlier ordering, in which the hierarchy improved on the flat baseline, does not survive a common protocol. What survives is the weaker and more defensible claim: **the hierarchy reaches comparable accuracy while producing a readable decision structure**, which is the trade this chapter has argued for throughout and is now measured rather than asserted. The fuzzy tree remains clearly behind both, and a random forest still beats everything here.

The correction also cuts the other way, in the method's favor. Hyperparameters matter enormously: run at library defaults the mixture scores 0.638, and at the settings `demo_concrete.py` actually specifies it scores 0.862 — a swing of more than 0.22 in $R^2$. Any comparison that leaves the model at its defaults is measuring the defaults, not the method. The same caution applies to the baselines, which are reported here at *their* defaults and could no doubt be tuned further.

**Table 6.2 — External baselines** *(structure fixed; cells to be filled by the reproduction harness — Goal G3).* Run on identical splits, multi-seed with error bars.

| Method | Concrete R² | Concrete RMSE | PhiUSIIL accuracy |
|---|---:|---:|---:|
| **Fuzzy tree (this work)** | 0.717 | 8.92 | ~0.968 |
| **Mixture of experts (this work)** | 0.862 | 6.30 | 0.996 |
| CART | 0.797 | 7.19 | *pending* |
| Random Forest (reference) | 0.904 | 4.94 | *pending* |
| M5 model tree | *pending* | *pending* | — |
| ANFIS | *pending* | *pending* | *pending* |
| Flat TSK (= Ch 4 flat MoG) | 0.881 | 5.54 | 0.998 |

**Table 6.3 — The interpretability side of the trade** *(structural today; the counts are pending).* The intent of this table is to make the trade-off legible rather than asserted — the hierarchy's value is the readable decision path, not a smaller rule base. I should be straight that in its current form it describes the *shape* of each model rather than measuring it; the numbers that would make it an argument are the pending row, and until they exist this table sets up the claim rather than settling it.

| Model | Rules / leaves | Variables per rule | Reads as |
|---|---|---|---|
| Flat FIS (Concrete) | 3 output buckets | all 8 | one weighted rule set |
| Fuzzy tree (Concrete) | shallow, depth-capped | only the path variables | root→leaf IF–THEN path |
| Mixture of experts | one sub-FIS per gate leaf | path gates + expert inputs | gated hierarchy |
| **Exact counts at matched accuracy** | *pending* | *pending* | — |

**Table 6.4 — Memory augmentation on dynamical systems.** The clearest single result in the chapter, and the one least entangled with the Concrete configuration problem above.

| System | Model | R² | RMSE |
|---|---|---:|---:|
| Double pendulum | standard FIS | 0.92 | 0.045 |
| Double pendulum | **memory-augmented** | **0.96** | **0.028** |
| Atwood machine | standard / memory-augmented | *pending* | *pending* |

The error reduction is roughly 38%, achieved by adding short- and long-term feature averages rather than any recurrent machinery.

**The honest scope.** I want to be plain about what the hierarchy buys and what it does not. On raw accuracy the tree and mixture do *not* beat the flat model in general — on PhiUSIIL the mixture is a hair *behind* the flat model — and they do not shrink the rule count below the already-compact flat model. What they buy is an explicit decision hierarchy over named variables and a readable path structure, and that payoff is real only at shallow depth and few terms, which is why I cap depth and leaf count. This is an interpretability-for-accuracy trade, made deliberately, and I would rather state it than let a reviewer discover it.

**What I propose to add.** Four things. Implement and evaluate the EM refinement of the mixture. Add the baselines a reviewer will demand — ANFIS, CART/C4.5, M5 model trees, flat TSK, and the recent Fumanal-Idocin (2025) and D-TSK-FC methods — on identical splits. And broaden the benchmark set beyond Concrete and PhiUSIIL to the other domains already scaffolded (turbine, wave-energy, wine, and the IoT sets), so the accuracy–interpretability trade is characterized across more than two problems. Fourth, and cutting across all of it, run the single consistent Concrete benchmark that makes every number in this chapter comparable with Chapter 4's. I also owe two literature searches — on knot/breakpoint optimization and on fuzzy mixtures-of-experts — to bound the novelty claims, and a small attribution fix in the references.

## 6.5 Discussion and Contributions

The defensible contribution here is architectural, not any single algorithm: one closed-form ridge primitive, reused across a flat FIS, a soft fuzzy tree, and a hierarchical mixture, with a clean export to a triangular rule base and an extension to temporal data. Every building block is prior art, and I credit each; the integration is the thing. The Magdalena objection is answered by construction rather than by assertion, and enforced by the declarative plan rather than left to discipline.

This chapter is also where the pipeline closes, and it is worth being concrete about how. Chapter 3 finds the structure at scale. Chapter 5 turns that structure into antecedents for the cases where no coordinates exist. This chapter is where those antecedents become a working model: the ridge solver supplies the consequents, the tree or mixture supplies the hierarchy, and the Ruspini export supplies the readable artifact at the end. That handoff — Chapter 5's membership functions consumed by this chapter's inference machinery — is exactly the integration that Chapter 7 names as the capstone, and it is the one link in the chain I have specified but not yet demonstrated.

I should be equally clear about the interpretability claim, since I have used the word freely. The trade against accuracy is stated honestly above and the mechanism is real, but Table 6.3 shows that the *quantification* is still pending: I have described why the hierarchy is more readable and not yet measured it. Calling that quantified would be overstating. What remains — the EM, the baselines, the interpretability counts, and the end-to-end demonstration — is the subject of Chapter 7.

---

*Draft — Chapter 6 prose, in the author's voice; built vs. proposed marked throughout. Citations in bracketed shorthand pending the consolidated `references.bib`. Four tables (6.1–6.4) and three figure placeholders (6.1–6.3) inline. Source outline in `../chapters/06-hierarchical-refined-fis.md`; open items in `../ACTION_ITEMS.md`.*
