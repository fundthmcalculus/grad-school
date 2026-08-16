# Neural-network / fuzzy-inference-system equivalence (Bede, Kreinovich & Toth)

The source material for `experiments/fis-to-neural-net/`. Two papers, one result,
extended once:

| | |
|---|---|
| **NAFIPS 2023** | Bede, Kreinovich & Toth, *Equivalence Between 1-D Takagi–Sugeno Fuzzy Systems with Triangular Membership Functions and Neural Networks with ReLU Activation*. In *Fuzzy Information Processing 2023* (NAFIPS 2023), Lecture Notes in Networks and Systems **751**, Springer, pp. 44–56. DOI [`10.1007/978-3-031-46778-3_5`](https://doi.org/10.1007/978-3-031-46778-3_5). Outstanding-paper award at that meeting. |
| **IJCCC 2025** | Bede, Kreinovich & Toth, *On equivalence between Takagi–Sugeno–Kang fuzzy systems with triangular membership functions and Neural Networks with ReLU activation in two or more dimensions*. *International Journal of Computers Communications & Control* **20**(4). DOI [`10.15837/ijccc.2025.4.7127`](https://doi.org/10.15837/ijccc.2025.4.7127). Open access. |

## The PDFs are not in this directory

Nothing was downloaded. This session's egress proxy denies every scholarly host
— `link.springer.com`, `univagora.ro` (the IJCCC publisher, which serves this
article open access), `scholarworks.utep.edu` (where Kreinovich mirrors
essentially everything as a UTEP CS technical report), `doi.org`,
`researchgate.net`, and the Crossref/OpenAlex/Semantic Scholar APIs — with a
`403` on `CONNECT`. Only GitHub hosts and PyPI are reachable. That is an
organization egress policy, not a transient failure, so it is reported rather
than worked around.

Run `./fetch_paper.sh` from any machine with normal network access and both
PDFs land here. The IJCCC one needs no subscription; the NAFIPS chapter needs
either a Springer subscription, a UTEP ScholarWorks preprint search, or — most
directly — asking Dr. Kreinovich, who mirrors his own work.

Everything below and everything in `references.bib` is therefore assembled from
search-result metadata, not from a fetched copy. Bibliographic fields carry the
same `[V]`/`[?]` verification markers this repository already uses in
`papers/least-action-review/references.bib`. **Confirm the two `[?]` fields
against a real copy before either entry ships in a bibliography.**

## What the result says

The one-dimensional statement, which is what the experiment builds on:

> A Takagi–Sugeno fuzzy system over one input, whose antecedents are triangular
> membership functions and whose consequents are singletons, computes exactly
> the same function as a feedforward neural network with one hidden layer of
> ReLU units — and conversely.

The mechanism is a single identity. A triangular membership function with feet
`a`, `c` and apex `b` is a sum of three ReLUs of the input:

```
T(x; a, b, c) = s_a * relu(x - a) - (s_a + s_c) * relu(x - b) + s_c * relu(x - c)
s_a = 1 / (b - a),   s_c = 1 / (c - b)
```

so the fuzzy term *is* a hidden-layer motif, and the membership function's
knots *are* the ReLU biases. Both sides of the equivalence are the class of
continuous piecewise-linear functions of one variable; the fuzzy system and the
network are two coordinate systems on it. The direction the papers emphasize is
network → rules (an explainability result: a trained ReLU network can be *read*
as a rule base). `experiments/fis-to-neural-net/` runs it the other way, rules
→ network, to use a constructed FIS as an initialization.

Two preconditions do real work, and are exactly what the n-dimensional sequel
had to engineer around:

1. **Partition of unity.** The terms must sum to 1 everywhere — a Ruspini
   partition — so that the TSK firing-strength normalization is a division by 1
   and disappears. Division is not in the piecewise-linear class, so without
   this the equivalence is false.
2. **Singleton (or affine) consequents,** so the output is a fixed linear
   combination of the terms.

The 2025 paper lifts this to two or more dimensions by replacing triangles with
**tetrahedral** membership functions — a simplicial partition of the input
space, whose barycentric coordinates are piecewise linear and sum to 1 by
construction. That choice is forced: with a product t-norm over per-feature
triangles, firing strengths become *piecewise multilinear*, not piecewise
linear, and no finite ReLU network reproduces them exactly.

## What this repository does with it

`experiments/fis-to-neural-net/fis2nn.py` implements the identity directly, and
`test_fis2nn.py` pins it: a Ruspini partition built by `tribblefis.ruspini`'s
own `build_triangular_partition`, with singleton consequents, converts to a
one-hidden-layer ReLU network agreeing to better than `1e-10` over a
24,001-point grid, with one hidden unit per apex knot and no data touched. The
theorem is executable here, not just cited.

The precondition analysis above is also why that experiment reports a *warm
start* rather than an exact conversion in more than one dimension: TRIBBLE's
regressor uses the probabilistic-sum/product norm pair over per-feature
Gaussians, so both preconditions fail, and the experiment measures the size of
the resulting gap instead of assuming it away.

## Related work worth having alongside these

The equivalence has been rediscovered in several forms; the entries in
`references.bib` marked "context" are the ones worth reading next to the above,
in particular Jang & Sun's radial-basis-function/FIS functional equivalence
(1993), Buckley, Hayashi & Czogała's earlier neural-net/fuzzy-expert-system
equivalence (1993), and Wu et al.'s survey of TSK functional equivalences
(arXiv:1903.10572).
