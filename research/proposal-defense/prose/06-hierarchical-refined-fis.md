# Chapter 6 — Hierarchical and Refined Fuzzy Models

*This chapter is partly built and partly proposed, and I mark which is which as I go. The shared solver, the fuzzy trees, the one-shot hierarchical mixture, the declarative structure specification, the Ruspini export, and the memory-augmented variant are implemented and evaluated. The antecedent refinement is built but still being broadened. The EM refinement of the mixture and the full comparison suite are proposed for completion.*

## 6.1 Introduction

The flat model of Chapter 4 is fast and readable, but it is flat: every rule sees every feature, and there is no explicit sense in which some decisions come before others. For some problems that is fine. For others I want two things the flat model does not give me. The first is an explicit *hierarchy* of decisions over named variables — the kind of "first check this, then within that check that" structure a domain expert actually reasons with. The second is a little more accuracy on problems where the relationship is genuinely local, by letting different regions of the input space be handled by different experts. And whatever model I build, I may want the option to refine it.

All three of these turn out to rest on a single technical observation, which is the organizing idea of this chapter. For a Takagi–Sugeno–Kang system with fixed firing strengths, the output is *linear in the consequent coefficients*. That means I never have to search for the consequents: given the antecedents, the best consequents are the solution of a regularized weighted least-squares problem, in closed form. I use that one solver — a firing-weighted ridge least-squares — as a shared primitive everywhere: in the flat FIS of Chapter 4, in the leaves of a soft fuzzy tree, and in the experts of a hierarchical mixture. Reusing one well-understood primitive across three model shapes is most of what makes this chapter hang together.

The contributions, then, are: the shared ridge-TSK consequent solver (built); soft fuzzy trees whose leaves are exact ridge-TSK models (built); a hierarchical mixture of fuzzy experts, fit one-shot today and by EM as proposed; a declarative way to specify and steer the model structure, plus an export to an explicit triangular rule base (built); an antecedent-refinement stage that reuses the solver as its inner objective (built; being extended); and a memory-augmented variant that extends the whole apparatus to dynamical systems (built).

## 6.2 Background and Prior Art

The closest competitor, and the one I engage most carefully, is Medina-Chico et al. (2001), which builds a soft decision tree with *linear* models in the leaves. That is very nearly what my fuzzy tree does, so I have to be precise about the difference: their leaves are fit by backpropagation, and mine are the exact firing-weighted ridge solution in closed form. The daylight is "closed-form exact leaf" versus "iteratively fit leaf," not the tree idea itself, which I concede. On the mixture side, Wu et al. (2020) show that a single-layer TSK system is already equivalent to a mixture of experts, which narrows what I can claim for a *hierarchical* mixture, and I engage that directly rather than around. The rest of the lineage is standard: Jordan and Jacobs (1994) for the hierarchical mixture of experts and its EM; Janikow (1998) and Yuan and Shaw (1995) for fuzzy decision trees and the ambiguity split criterion; Olaru and Wehenkel (2003) for the finding that soft splits keep accuracy while staying interpretable.

There is one objection I have to meet head-on, because a good committee will raise it, and it is due to Magdalena (2018): a hierarchical fuzzy system is *not* automatically more interpretable, because the intermediate variables it introduces can be meaningless. My answer, which the method makes good on by construction, is that every split and every gate in my hierarchies is over an *original, named input* — never a synthetic intermediate. That is exactly the condition Magdalena requires for the interpretability claim to hold, and I hold to it deliberately.

**An unresolved tension, which I would rather raise than have found.** Chapter 5 proposes as a stretch goal (G8) admitting *joint* two-feature membership functions for clusters that have no faithful axis-aligned description — a ring being the standard case, since it is not the intersection of per-axis intervals. Taken literally, a joint membership is close to the construction the Magdalena condition exists to forbid: it is a derived region rather than a statement about one named input, and "this point lies in an annulus in (cement, water) space" is not a clause an engineer reads the way "cement ≥ 350" is.

I do not think the two are irreconcilable, but I have not yet done the work to reconcile them, and I would rather mark that plainly than paper it over. The distinction I expect to rest on is that Magdalena's objection is to *synthetic intermediate variables* — quantities invented by the hierarchy, with no meaning outside it — whereas a joint membership is still over original named inputs, just two of them at once, and it is introduced only where the data demonstrably has no axis-aligned description. That is a weaker claim than the one this section currently makes, and it needs stating precisely rather than asserted.

Three things would settle it. Whether a two-feature membership over named inputs satisfies Magdalena's condition or merely evades it, which is a question about the literature and not about my code. Whether such clusters are rare enough on real data that the exception stays an exception — if most clusters need one, the interpretability argument does not survive and G8 should be abandoned rather than defended. And whether a rule base mixing 1-D and 2-D antecedents reads coherently to a domain expert, which only a person can answer and which Goal G6's interpretability study is the natural place to ask. Until those are settled, this chapter's claim should be read as holding for the hierarchy as built today, with G8 flagged as the one place the dissertation proposes to spend interpretability on purpose.

One caution I impose on myself, from checking my own earlier reasoning: I do **not** motivate the ridge solver by claiming that ANFIS's least-squares step overfits. That premise did not survive verification. The honest motivation for the ridge solver is numerical conditioning and the value of one reusable primitive, not a deficiency in ANFIS.

## 6.3 Methodology

### 6.3.1 The shared ridge-TSK solver (built)

Because the TSK output is linear in the consequent coefficients for fixed firing strengths, fitting the consequents is a weighted least-squares problem, and I solve it with ridge-regularized normal equations — leaving the intercept and bucket-mean columns unpenalized. Two details matter in practice. I let the polynomial basis be orthogonal (Legendre or Chebyshev) rather than raw monomials, which conditions the problem far better at higher orders, and I select the order and the regularization strength by cross-validation. This one solver replaces the per-bucket pseudo-inverse and the iterative L-BFGS fit that came before it, and it is the primitive the next two subsections reuse.

### 6.3.2 Soft fuzzy trees (built)

The fuzzy tree is a CART-style recursive partition, with two differences from an ordinary decision tree. The splits are soft, so a point flows down multiple paths with graded membership rather than being sent left or right, and each leaf holds a full ridge-TSK model rather than a constant. The split criterion is firing-weighted variance reduction for regression, and a fuzzy ambiguity or information-gain measure for classification. The payoff is readability: the tree renders as a short list of IF–THEN rules, one per root-to-leaf path, each mentioning only the variables on that path and ordered by importance. On the Concrete dataset the tree splits first on cement content and then on age, right at the standard 28-day curing mark — which is to say it recovers domain knowledge nobody told it, and a materials engineer can read that and nod.

That readability is the reason the tree is fit on *raw*, untransformed features while the flat model of Chapter 4 is not. A threshold of "cement ≥ 350 kg/m³" is something an engineer can check; "cement ≥ 0.42" after standardization is not. The tree can afford the choice, because axis-aligned splits are rank-based and therefore invariant to monotone transforms — Chapter 4 §4.3 measures this directly and finds the transform worth exactly nothing to CART and Random Forest, against as much as +0.13 $R^2$ to the Gaussian models. The mixture is the interesting middle case: its gates are tree-like but its experts are Gaussian, and it gains +0.099 from the transform, which is roughly what one would predict from that split of machinery.

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

> **Reproduction.** Table 6.1 is assembled from *two* generators, and it is worth saying which supplies what. `reproduce/tables/table_concrete_reconciliation.py` produces the flat-model rows, including the antecedent-refinement arm, which only it runs. `table_hyperparam_normalization.py` produces the demo-tuned column for the tree and the mixture, which only it runs. Combining them is safe rather than convenient: the two scripts share splits and seeds, and they agree to three decimals on every row they both compute — flat 2nd order, the fuzzy tree and mixture at library defaults, CART and Random Forest — so the join is cross-validated rather than assumed. Neither is the similarly named `table_6_1_model_family.py`, which deliberately runs the fuzzy arms at raw preprocessing and library defaults and instead supplies Table 6.2's external baselines (CART, Random Forest, and an optional M5 adapter). Table 6.3 is structural and has no generator. Table 6.4 comes from the double-pendulum experiment in `tribble-fis`. Both harness scripts emit Markdown and CSV with mean ± standard deviation across a fixed seed set; cells marked *pending* are those whose adapter was not yet wired up, and the harness reports what it could not run rather than substituting a guess. Per-cell provenance is tracked in `reproduce/PROVENANCE_MAP.md`.
>
> **TODO — repeatable performance (board-wide standard):** the training-time, accuracy, and speedup numbers here need the fixed reproducibility protocol and the full baseline suite before citation (see `ACTION_ITEMS.md` §A/§C and Ch 7 Goal G4/G3).

**What is measured today.** On Concrete regression the three models are compared under a single protocol in Table 6.1 below; the ordering they fall into is not the one an earlier draft of this chapter reported, and I take that up there rather than twice. On PhiUSIIL classification the mixture is at 0.999 ± 0.001 accuracy and the flat model at 0.997 ± 0.001, with the tree behind at 0.970 and splitting on interpretable signals — though as Table 6.2 notes, every method saturates on that dataset and the gaps there carry no weight. And the memory result above.

**Antecedent refinement, measured.** §6.3.5 described refining the membership-function parameters against a held-out objective; the harness now runs it, and the result is more interesting than the improvement I had claimed.

| TSK order | closed-form only | refined | Δ |
|---|---:|---:|---:|
| 0th | −0.334 | **0.580** | **+0.914** |
| 1st | 0.772 | **0.844** | +0.072 |
| 2nd | 0.824 | **0.861** | +0.037 |

Refinement helps most where the consequent model has least capacity to begin with. At zeroth order the closed-form solve is actively worse than predicting the mean — a negative $R^2$ — and refinement lifts it to 0.580; at first and second order it buys a far more modest seven and four points. The trend is monotone in the wrong direction for the refinement stage: the more expressive the consequents, the less the extra search is worth. An earlier draft of this chapter reported refinement lifting Concrete from roughly 0.88 to 0.92; that figure is not reproduced, and 0.92 does not appear anywhere in a controlled run. I am striking it.

The shape of the correction is worth more than the number, and I should be careful about how far I push it. Refinement's value decays sharply with the capacity of the consequent model: worth 0.914 at zeroth order, 0.072 at first, 0.037 at second — a factor of twenty-five across the range. That is the same direction §6.3.5 reported for the population methods, where differential evolution and a genetic algorithm overfit the cross-validation estimate and a plain local optimizer beat them. Both observations point the same way: once the structure has been recovered from the data, additional search buys progressively less. An earlier draft went further and claimed refinement actively *hurts* at high capacity, on the strength of a full-second-order row that lost 0.027. Under this protocol I cannot support that — refinement's contribution shrinks toward zero but stays positive at every order measured, and the negative row came from a configuration that is not in the uniform sweep. The honest version of the *structure before search* argument here is diminishing returns, not damage.

**Table 6.1 — The model family on Concrete: architecture × configuration.** All arms at the log-and-standardize preprocessing of Chapter 4 §4.3; 10 seeds, shared splits. Each cell is $R^2$ with RMSE in MPa beneath it, both as mean ± standard deviation. The columns are the hyperparameter setting; the flat model has one pipeline configuration, so its rows vary the consequent basis instead.

| Model | default settings | demo-tuned |
|---|---:|---:|
| Flat MoG-TSK, 2nd order | 0.824 ± 0.043 <br> *6.84 ± 0.94 MPa* | — |
| Flat MoG-TSK, full 2nd order | 0.859 ± 0.039 <br> *6.10 ± 0.91* | — |
| Flat MoG-TSK, 2nd + antecedent refinement | **0.861 ± 0.044** <br> *6.01 ± 0.79* | — |
| Mixture of experts (HME) | 0.781 ± 0.068 <br> *7.54 ± 0.98* | **0.833 ± 0.024** <br> *6.67 ± 0.56* |
| Fuzzy tree | 0.688 ± 0.056 <br> *9.09 ± 0.58* | 0.741 ± 0.051 <br> *8.28 ± 0.63* |
| CART (reference) | 0.826 ± 0.047 <br> *6.73 ± 0.74* | — |
| Random Forest (reference) | **0.909 ± 0.019** <br> *4.90 ± 0.31* | — |

These supersede the figures this chapter previously carried (flat 0.658, tree 0.746, mixture 0.791), which came from three different configurations and could not be read against one another. The table is deliberately a grid rather than a ranking, because the honest answer to "does the hierarchy beat the flat model?" turns entirely on what one holds fixed — and an earlier draft of this chapter picked a favourable answer without saying so, then a later one picked an unfavourable answer for the same bad reason.

Matched on architecture alone — the same second-order consequents, no post-hoc refinement — the mixture is *nominally ahead*: 0.833 against 0.824, a difference of 0.009 against standard deviations of 0.024 and 0.043. Give the flat model its full second-order basis and it leads by 0.026; add the antecedent refinement of §6.3.5, which has no counterpart in the tree or the mixture, and it leads by 0.028. Every one of those gaps is smaller than the spread on at least one of the two models, so the defensible reading is not an ordering at all: **the hierarchy and the flat model are level on this problem, and the hierarchy additionally produces a readable decision structure.** That is exactly the trade this chapter has argued for throughout, and it is better supported than either the "hierarchy improves accuracy" claim of the first draft or the "hierarchy loses" claim of the second. The fuzzy tree stays clearly behind both, and a random forest still beats all of it.

There is a second finding in the right-hand column that I did not expect, and it matters more than the means. Tuning does not merely raise the mixture from 0.781 to 0.833 — it cuts its standard deviation from 0.068 to 0.024, by nearly a factor of three, making the tuned mixture the *most stable* model in the table, flat models included. The fuzzy tree shows nothing comparable (0.056 to 0.051). So what `demo_concrete.py`'s settings buy the mixture is mostly reliability rather than accuracy: at library defaults it is not a slightly worse model, it is an erratic one, and the divergence described next is the extreme tail of that same behaviour. A mean without a spread would have hidden the entire effect.

**The mixture had a rare catastrophic failure mode, and finding it is the reason this table can be quoted at all.** At ten seeds one split — seed 9, under normalized features — produced a model whose predictions ran to 10,536 MPa on a target that never exceeds about 82. The other nine were unremarkable. A five-seed protocol did not contain the offending split and reported a clean $0.813 \pm 0.039$; the failure was always there and simply had not been sampled.

The cause was in the consequent solver, not the hierarchy: the closed-form ridge solve formed the normal equations, which squares the condition number, and applied no regularization to the rule intercepts — so two rules with nearly collinear firing strengths left a singular, unregularized block. `numpy.linalg.solve` does not raise on that; it returns finite coefficients of order $10^{24}$. Both halves are now fixed upstream, by solving least squares on the design rather than the normal equations and by giving the ridge term a non-zero default, and the row above is a clean ten-seed mean with no divergence.

I keep the episode in the text because the lesson outlived the bug. A five-seed mean did not merely give a slightly wrong number here — it certified as stable a model that fails one time in ten, and no amount of care in the *reporting* would have caught it, because the failing configuration was never run. That is the argument for the seed floor in Goal G4, and it is worth more to this dissertation than the corrected cell.

The configuration effect the grid makes visible is worth one more paragraph, because an earlier draft both overstated it and misattributed it. I once put the swing between a default-configured and a tuned mixture at "more than 0.22 in $R^2$." Measured under one protocol it is **0.052** at fixed preprocessing — 0.781 to 0.833, both normalized — and the larger part of what I had charged to hyperparameters is really the normalization, worth +0.099 to the library-default mixture and +0.065 to the demo-tuned one. Taken end to end, from library defaults on raw features to tuned settings on normalized ones, the swing is 0.151, and roughly two-thirds of that is preprocessing rather than tuning. Any comparison that leaves a model at its defaults is still measuring the defaults, but the effect is smaller than I asserted and differently caused. The same caution applies to CART and the random forest, reported here at *their* defaults.

**Table 6.2 — External baselines** *(structure fixed; cells to be filled by the reproduction harness — Goal G3).* Run on identical splits, multi-seed with error bars.

| Method | Concrete R² | Concrete RMSE | PhiUSIIL accuracy |
|---|---:|---:|---:|
| **Fuzzy tree (this work)** | 0.580 ± 0.067 | 10.58 | 0.970 ± 0.003 |
| **Mixture of experts (this work)** | 0.682 ± 0.064 | 9.17 | 0.999 ± 0.001 |
| CART | 0.825 ± 0.047 | 6.74 | 1.000 ± 0.000 |
| Random Forest (reference) | 0.909 ± 0.018 | 4.90 | 1.000 ± 0.000 |
| M5 model tree | *pending* | *pending* | — |
| ANFIS | *pending* | *pending* | *pending* |
| Flat TSK (= Ch 4 flat MoG) | 0.650 ± 0.056 | 9.63 | 0.997 ± 0.001 |

This table exists for two things Table 6.1 does not cover: the external baselines a reviewer will demand (M5 and ANFIS, both still owed), and the PhiUSIIL column. It runs every model at **raw features and library defaults**, which is why its Concrete numbers sit below Table 6.1's throughout — that is the third axis, and the two tables must not be read as one series. The Concrete column here says what these models do untuned; Table 6.1 says what they do under the tuned, normalized protocol the chapter argues for. The PhiUSIIL column is the more interesting one now that it is filled: on that dataset every method saturates, the two tree baselines reach a perfect score, and the fuzzy models sit a fraction behind — so PhiUSIIL discriminates between these methods hardly at all and should not carry any weight in the comparison.

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

**The honest scope.** I want to be plain about what the hierarchy buys and what it does not. On raw accuracy the tree and mixture do not *reliably* beat the flat model — on Concrete the tuned mixture and the flat model are level within their spreads, and on PhiUSIIL the mixture's nominal lead sits on a dataset where every method saturates and nothing is separable — and they do not shrink the rule count below the already-compact flat model. What they buy is an explicit decision hierarchy over named variables and a readable path structure, and that payoff is real only at shallow depth and few terms, which is why I cap depth and leaf count. This is an interpretability-for-accuracy trade, made deliberately, and I would rather state it than let a reviewer discover it.

**What I propose to add.** Four things. Implement and evaluate the EM refinement of the mixture. Add the baselines a reviewer will demand — ANFIS, CART/C4.5, M5 model trees, flat TSK, and the recent Fumanal-Idocin (2025) and D-TSK-FC methods — on identical splits. And broaden the benchmark set beyond Concrete and PhiUSIIL to the other domains already scaffolded (turbine, wave-energy, wine, and the IoT sets), so the accuracy–interpretability trade is characterized across more than two problems. Fourth, and cutting across all of it, run the single consistent Concrete benchmark that makes every number in this chapter comparable with Chapter 4's. I also owe two literature searches — on knot/breakpoint optimization and on fuzzy mixtures-of-experts — to bound the novelty claims, and a small attribution fix in the references.

## 6.5 Discussion and Contributions

The defensible contribution here is architectural, not any single algorithm: one closed-form ridge primitive, reused across a flat FIS, a soft fuzzy tree, and a hierarchical mixture, with a clean export to a triangular rule base and an extension to temporal data. Every building block is prior art, and I credit each; the integration is the thing. The Magdalena objection is answered by construction rather than by assertion, and enforced by the declarative plan rather than left to discipline.

This chapter is also where the pipeline closes, and it is worth being concrete about how. Chapter 3 finds the structure at scale. Chapter 5 turns that structure into antecedents for the cases where no coordinates exist. This chapter is where those antecedents become a working model: the ridge solver supplies the consequents, the tree or mixture supplies the hierarchy, and the Ruspini export supplies the readable artifact at the end. That handoff — Chapter 5's membership functions consumed by this chapter's inference machinery — is exactly the integration that Chapter 7 names as the capstone, and it is the one link in the chain I have specified but not yet demonstrated.

I should be equally clear about the interpretability claim, since I have used the word freely. The trade against accuracy is stated honestly above and the mechanism is real, but Table 6.3 shows that the *quantification* is still pending: I have described why the hierarchy is more readable and not yet measured it. Calling that quantified would be overstating. What remains — the EM, the baselines, the interpretability counts, and the end-to-end demonstration — is the subject of Chapter 7.

---

*Draft — Chapter 6 prose, in the author's voice; built vs. proposed marked throughout. Citations in bracketed shorthand pending the consolidated `references.bib`. Four tables (6.1–6.4) and three figure placeholders (6.1–6.3) inline. Source outline in `../chapters/06-hierarchical-refined-fis.md`; open items in `../ACTION_ITEMS.md`.*
