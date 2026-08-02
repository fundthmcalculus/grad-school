# Results: C rules on a Ruspini grid, antecedents chosen combinatorially

Run of record: `outputs/results-main.json` (free subsets), `outputs/results-convex.json`
(interval antecedents), `outputs/results-{tnorm-product,disj-max}.json` (operator
ablations), `outputs/landscape*.md` (exhaustive enumeration),
`outputs/summary.md` (the cross-run tables quoted below). Ten seeds, stratified
70/30 splits, `min` t-norm and `sum` disjunction unless stated. Full sweep: 141 s.

Setup is in [`README.md`](README.md).

## Headline

| dataset | best C-rule model | unconstrained decision tree | nearest centroid (C prototypes) |
|---|---|---|---|
| iris | **0.958 ± 0.029** (anneal, k=5) | 0.947 ± 0.027 | 0.898 ± 0.032 |
| wine | **0.954 ± 0.028** (greedy, k=7) | 0.917 ± 0.040 | 0.700 ± 0.037 |
| glass | 0.614 ± 0.059 (anneal, k=7, unweighted) | **0.654 ± 0.050** | 0.422 ± 0.056 |

Three rules on iris and three on wine beat an unconstrained CART, which is a
real result for a model this constrained — each rule is one conjunction of
linguistic terms, and there are exactly as many as there are classes. Six rules
on glass do not: glass has six classes in 214 samples with several of them
multi-modal, and one conjunction per class cannot cover two disjoint lumps.
That is the honest limit of the C-rule constraint, not a search failure — the
search reaches the enumerated optimum of its objective and still lands at 0.61.

## Hypotheses, scored

**H1 (MST over membership functions beats marginal mass) — confirmed, and it is
the cheapest thing that does.** `mst_mf` improves the one-vs-rest margin over
the `mass` baseline in 8 of 9 dataset/k cells: mean gap to the enumerated
optimum falls from 0.176 to 0.119 on iris k=3, and on wine k=3 the objective
goes 1.518 → 1.795. The exception is glass k=3, where it ties (−0.002). It does
this from **11–17 objective evaluations** out of 923 521 combinations (0.0015%
of the space). Co-firing is a real signal: MFs
that light up on the same class-c samples do belong in the same antecedent.

**H2 (the MST-of-samples core is the better of the two trees) — wrong on
objective, right on stability.** `mst_core` gets a better margin than `mst_mf`
on iris (gap 0.095 vs 0.119 at k=3) but is *worse* on wine (1.564 vs 1.795 at
k=3), at comparable cost (0.6–2.4x). Where it earns its place is glass, the
dataset with outliers and tiny classes: at k=7 it is the best of the two MSTs on
test accuracy (0.438 vs 0.354) and the best non-search selector overall. Cutting
straggler samples before counting mass is worth something exactly when the class
supports are ragged, and nothing when they are not.

**H3 (greedy hill climbing will leave real objective on the table) — wrong, and
this is the main finding.** Steepest ascent over single MF flips, starting from
all-don't-care, reaches the *exact* enumerated optimum on 9 of 10 seeds at iris
k=3 and 5 of 10 at k=5, with mean gaps of 0.0019 and 0.0287. It gets there in
**122–142 objective evaluations out of 923 521** — 0.013% of the space — and in
0.013 s against 3.67 s for enumeration (280x). Simulated annealing from the best
heuristic start hits the optimum on **10 of 10 seeds at every enumerable
setting**, in 0.56 s, still 6.5x faster than enumerating.

The objective is a difference of two monotone set functions, so none of that was
guaranteed. It appears to hold because the landscape is peaked but smooth: only
**0.004–0.007% of combinations are within 1% of the optimum** at iris k=5
(0.08–0.17% at k=3), yet single-flip ascent walks to the top of it from a fixed
start. See `figures/03-landscape.png`.

**H4 (optimising the margin harder keeps improving test accuracy) — wrong, it
saturates almost immediately.** Enumeration and annealing find identical optima
and score identically on test (iris k=5: both 0.958). Greedy, 0.029 short on the
training margin, scores 0.953 — inside the seed spread. And it inverts on wine
k=7, where greedy has the *worse* margin (2.687 vs 2.748) and the *better*
accuracy (0.954 vs 0.944). Past the first few points of margin the objective
stops carrying information about generalisation, which is why the practical
recommendation is `greedy`, not `exhaustive`: the extra 280x buys nothing.

**H5 (the inverse-mass rule weight is a straight improvement) — wrong; it is a
precision/recall trade, and its sign depends on class balance.** On balanced
data with a coarse partition it is decisive — iris k=3 `mass` goes 0.769 → 0.933
— because with k=3 every rule is broad and the argmax would otherwise go to
whichever rule is broadest. On glass it *costs* accuracy: anneal at k=7 is 0.614
unweighted against 0.555 weighted. But the same weighting *raises* glass macro-F1
(0.561 vs 0.514): boosting rules that rarely fire is exactly what the rare
classes need, and it over-predicts them. Both columns are reported for every
cell in `outputs/summary.md`; the weighting is a decision-rule knob to be picked
on a validation split, not a default.

**H6 (freely chosen subsets stay linguistically sensible) — wrong, and it gets
worse with k.** Nothing in the formulation requires a selected subset to be a
contiguous run, and the search does not keep it one. At k=3 essentially all
antecedents are intervals; by k=7 on glass only **65%** are, so a third of the
rules read "Fe is very low or low or med-low **or very high**" — a union with a
hole, which is two rules wearing one rule's clothes (`figures/05-convexity.png`).

## The convexity constraint

Restricting every antecedent to one contiguous run shrinks each variable's
choice from `2^k - 1` subsets to `k(k+1)/2` intervals — at k=7, 127 → 28, which
is what makes iris k=7 enumerable at all (614 656 combinations per class instead
of 2.6e8).

The constraint is close to free **if the optimiser can handle it**. Annealing
loses 2.5% of the training margin on glass k=7 (4.155 → 4.050), and its accuracy
moves by at most 2.2 points in *either* direction with no consistent sign
(glass k=7 −2.2, glass k=3 +1.1, iris unchanged at every k). It still hits the
enumerated optimum 10/10 on iris at k=3, 5 **and 7**.

Greedy is a different story: its optimum-hit rate collapses from 9/10 to 3/10 at
k=3 and from 5/10 to 0/10 at k=5, and on glass k=7 its accuracy drops 0.528 →
0.437. The reason is structural rather than statistical — the single-flip
neighbourhood restricted to intervals is far less connected, because reaching a
better interval often requires passing through a non-convex intermediate that
the constraint forbids. **The convexity constraint costs little in the optimum
and a great deal in the ease of finding it.** If you want convex rules, pay for
annealing.

## Operator ablation

`product` t-norm and `max` disjunction were each run across the full sweep
(`outputs/summary.md`). Neither moves anything consistently: **79 of 90 cells
differ by less than the pooled seed spread**. Ten of the eleven that exceed it
belong to `max`, and they point in both directions — it costs iris k=7 up to 7.6
points on the weaker selectors (`mst_mf` 0.944 → 0.869) and gains glass k=3
`mst_mf` 11.2 points (0.375 → 0.488). A change worth ±10 points depending on
which dataset it lands on is not a mechanism. The single `product` outlier
(wine k=3 `mst_core`, +4.4) is one cell out of 45.

That is a mildly interesting null. The reason `sum` is still the default is not
accuracy but semantics: on a Ruspini partition it is an exact t-conorm and makes
a full row exactly a don't-care, where `max` caps the same row at
`max_j mu_ij(x) ∈ [0.5, 1]`.

## Silent samples

With exactly C rules and no default rule, a test point can fall outside every
antecedent, and the argmax then decides on the tie-break alone. This is rare but
not negligible, and it tracks partition fineness: 0% on iris k=3 and ~0.2% at
k=5, up to **7.2% on glass k=7 with `mst_core`** and 5.8% with `mass`. The search-based selectors
suppress it — greedy is at 1.2% on the same cell — because a rule that fails to
fire scores nothing on its own in-class term. Worth watching if this model is
ever deployed anywhere: it is the failure mode that a C-rule base has and a
grid-of-all-combinations rule base does not.

## What the rules look like

`python run_experiment.py --rules iris:7:exhaustive --seeds 0 --convex`, the
globally optimal interval rule base, 0.978 on that split:

```
R0: IF sepal length is very low or low or med-low or medium
    AND petal length is very low or low
    AND petal width is very low or low or med-low
    THEN class = setosa
R1: IF sepal length is low or med-low or medium or med-high or high
    AND sepal width is very low or low or med-low or medium or med-high
    AND petal length is med-low or medium or med-high
    AND petal width is med-low or medium or med-high
    THEN class = versicolor
R2: IF sepal length is low or med-low or medium or med-high or high or very high
    AND sepal width is low or med-low or medium or med-high or high
    AND petal length is medium or med-high or high or very high
    AND petal width is med-high or high or very high
    THEN class = virginica
```

Note what the experiment's premise predicted and the fitted model confirms:
`petal length is medium` appears in both R1 and R2, `sepal length is low` in all
three, and the rules use 3, 4 and 4 variables with 2–6 terms each. The subsets
overlap, the counts differ per variable, and R0 drops sepal width entirely.

## Practical recommendation

`greedy`, k=5, weighting chosen on a validation split. It is within noise of the
global optimum, 280x cheaper than finding it, and 43x cheaper than annealing.
Use `anneal` if the antecedents must be convex, `mst_core` if the fit must cost
under 50 objective evaluations, and `exhaustive` only to check the others — which
is what it was for here.

## Loose ends

- Rules are optimised **independently** per class, one-vs-rest. That is what
  makes enumeration tractable, and it is also a modelling choice that leaves
  something on the table: nothing coordinates the rules against each other, and
  the argmax is where they finally meet. A joint objective over the argmax is
  the obvious next experiment, and it is not enumerable at any k.
- The exhaustive column exists for iris only. Establishing that greedy tracks
  the optimum on a 4-feature problem does not establish it on a 13-feature one;
  the wine and glass claims rest on the objective values, not on a known
  optimum.
- `lambda = 1` throughout. It was not swept, and it sets the precision/recall
  balance of every rule before the weighting knob ever gets a say.
