# Neural-network / fuzzy-inference-system equivalence (Bede, Kreinovich & Toth)

The source material for `experiments/fis-to-neural-net/`. **Three** papers, one
result, extended twice:

| | |
|---|---|
| **NAFIPS 2023** | *Equivalence Between 1-D Takagi–Sugeno Fuzzy Systems with Triangular Membership Functions and Neural Networks with ReLU Activation*, pp. 44–56, Springer (LNNS 751). DOI [`10.1007/978-3-031-46778-3_5`](https://doi.org/10.1007/978-3-031-46778-3_5). |
| **NAFIPS 2024** | *Equivalence between TSK fuzzy systems with triangular membership functions and neural networks with ReLU activation on the real line*, Springer. |
| **IJCCC 2025** | *On equivalence between Takagi–Sugeno–Kang fuzzy systems with triangular membership functions and Neural Networks with ReLU activation in two or more dimensions*, **20**(4), August 2025, article 7127. DOI [`10.15837/ijccc.2025.4.7127`](https://doi.org/10.15837/ijccc.2025.4.7127). **PDF in this directory** (`bede-kreinovich-toth-2025-ijccc-nd.pdf`, supplied by the author). |

**A citation correction worth keeping:** a web search for the "on the real
line" title returns the 2023 conference and its outstanding-paper award, which
makes it look like the 2023 and 2024 papers are one paper. The 2025 paper's own
bibliography settles it — the 1-D paper is [2], 2023, pp. 44–56; "on the real
line" is [3], 2024. `references.bib` records both separately.

The two NAFIPS PDFs are still missing. This session's egress proxy answers 403
on CONNECT to every scholarly host (Springer, doi.org, UTEP ScholarWorks,
Crossref/OpenAlex/Semantic Scholar); only GitHub and PyPI are reachable.
`./fetch_paper.sh` will retrieve what it can from a normal network.

## What the results say

**One dimension (2023).** For every TS system with triangular membership
functions there is a one-hidden-layer ReLU network computing the same function,
and conversely. The mechanism is a single identity, quoted from the paper:

```
A(x) = [phi(x-a) - phi(x-b)] / (b-a)  -  [phi(x-b) - phi(x-c)] / (c-b)
```

which expands to `s_a*relu(x-a) - (s_a+s_c)*relu(x-b) + s_c*relu(x-c)` with
`s_a = 1/(b-a)`, `s_c = 1/(c-b)`. A fuzzy term is a hidden-layer motif and its
knots are the ReLU biases. The paper also gives the degenerate `a = b` and
`b = c` forms, which matter because a Ruspini partition's end terms are
shoulders. `fis2nn.triangle_to_relu` implements exactly this.

**Two or more dimensions (2025).** Triangles are replaced by **tetrahedral**
membership functions. The argument runs network → rules:

1. A ReLU network is a composition of piecewise-linear maps, so it is affine on
   each of finitely many polyhedral cells.
2. Every polyhedron can be triangulated into simplices, and inside a simplex the
   barycentric coordinates `c_i(x)` are affine and sum to 1.
3. So a linear function on a simplex is `p+1` TSK rules, `If c_i(x) then z = L(v_i)`.
4. Collecting, for each vertex `p`, every `c_i` that equals 1 there gives one
   piecewise-linear membership function `M_p` — a **polyhedral pyramid**, the
   multi-dimensional analogue of a triangle — and one rule per vertex:

   > **If `M_p(x)` then `z = f(p)`**, where `f` is the network's function.

Theorem 2 states this as a *local* equivalence — exact in the interior of a
triangle, extended to any bounded polygonal domain — and the conclusions name
"extend the results from local to global" as future work.

## What this repository does with it, and where it diverges

`experiments/fis-to-neural-net/` runs the map **backwards**, rules → network, to
use a constructed FIS as a neural network's initialization. That direction is
not symmetric with the paper's, and the asymmetry is the whole story of the
experiment's results:

* **The paper's triangulation is induced by the network's own linear regions**,
  so step 3's interpolation is exact by construction — there is nothing to
  approximate, because the function really is affine on each cell.
* **A TRIBBLE FIS is not piecewise linear** (Gaussian memberships, product
  t-norm, a normalization step), so no canonical triangulation exists to
  convert. `simplicial.py` imposes a regular **Freudenthal/Kuhn** lattice
  instead, and the equivalence becomes an interpolation with a measurable error.

Two consequences the experiment measures rather than assumes:

* The paper's rule `z = f(p)` — implemented as `consequents_from_fis("vertex")`
  — is exactly right when `f` is the network being decompiled, and is the *worst*
  of three estimators when `f` is a FIS being converted on a grid: in 8 or 12
  dimensions the lattice vertices sit off the data manifold, where the FIS's
  output is extrapolation. Measured fidelity on WEC: 5.8–8.7, against 0.82 for
  projecting onto the same basis instead.
* The papers do not discuss how many rules the construction needs.
  `simplicial.py` shows the count is governed by the data rather than the
  lattice — `n+1` rules fire at any point, found by a sort, and only the
  vertices the data reaches are ever built — and `run_simplicial.py` shows the
  binding constraint is statistical, not computational.

`simplicial.py`'s closed form for the Freudenthal hat,

```
phi_v(x) = relu( 1 - relu(max_i d_i) - relu(max_i (-d_i)) ),   d = (x - v)/h
```

is checked against Kuhn interpolation at **zero** error in dimensions 1, 2, 3, 5
and 8 (`test_simplicial.py`). Since `max(a,b) = a + relu(b-a)`, it is a ReLU
circuit of `O(n)` units at depth `O(log n)` — the n-dimensional analogue of "a
triangle is three ReLUs", and the form the polyhedral pyramid takes on this
particular triangulation.
