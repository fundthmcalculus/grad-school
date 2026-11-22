# Review: Backwards Construction of Fuzzy Inference Systems

Review of `paper.md` (2026-10-05 pass). No edits were made to `paper.md`; the
three build-breaking defects below are worked around by preprocessing in
`build_pdf.py` and flagged here so the author can fix them at the source.
Line numbers refer to `paper.md` as reviewed.

## Strengths

The core idea is real and well-motivated: invert the FIS construction --
memberships per variable per output class, combined by t-conorm -- so the rule
count is fixed at $N_{class}$ instead of scaling with $\prod N_{mf,i}$
(rulebase explosion). The regression derivation (L67-130) is the paper's best
content: defuzzification as a weighted average, linear in the consequent
coefficients, stacked into $\Phi\boldsymbol\theta = \mathbf{y}$, normal
equations, ridge variant. That is a genuinely publishable thread; it needs the
notation tightened (below).

## Correctness / math issues

- **L93 `\gte` and L106 stray `##` are build-breakers.** `\gte` is the MathJax
  spelling; LaTeX's is `\ge` -- xelatex dies with "Undefined control sequence".
  The stacked-matrix display opened at L94 is closed with a bare `##` line
  instead of `$$`; pandoc then leaves the opening `$$` as literal text and
  turns the `##` into an empty `\subsection{}`, so the matrix renders as raw
  LaTeX source. Fix both at the source.
- **L93 `$ M \gte R(N+1) $` is a third source defect:** pandoc's tex_math_dollars
  requires a non-space character immediately after the opening `$` (and
  before the closing `$`) for inline math, so this span is emitted as literal
  `\$ ... \$` and the TeX engine then dies on `\ge` in text mode ("Missing
  $ inserted" -- which xelatex papers over and still exits 0). Write
  `$M \ge R(N+1)$`. Also the rank condition itself: $\Phi$ needs $M \ge R(N+1)$
  *and* the samples to actually span it; as written it is necessary but not
  sufficient -- worth stating.
- **L54** `$$ \mu_{anomaly} = complement(tconorm(\forall rules)) + boost$$`:
  `complement` and `tconorm` render as italic products, `\forall rules` as
  "forall-rules". Define the operators and use `\operatorname`; the prose words
  in math mode are not a definition.
- **L28** `$O(N)=N^2$` is nonsense notation -- say what the dedup step costs.
- **L150** appendix triple `$(\top,\perp\,\neg)$` is missing the comma after
  `\perp` (the thin space reads as a pair, not a triple).
- **L153** t-conorm is stated twice and garbled:
  `$\perp(a,b) = \neg\top(a,b)$ such that $\perp(a,b) = 1-\top(1-a,1-b)$` --
  "such that" sits between two math spans, and the two identities are not
  equivalent as written (the first requires negation to distribute over
  `\top`). This is the De Morgan triplet definition that the anomaly section
  leans on; it must be exact.
- **Notation drift:** $W_{all}$ is defined at L68 but $W_{sum}$ is used at
  L70/L73/L76; $\phi$ vs $\Phi$ (L108 "Expand $\phi$" and L129 "rank-deficient
  $\phi$" both mean $\Phi$); coefficient indexing $a_{rule\text{-}num,coeff\text{-}num}$
  (L68) vs $a_{j-1,i}$ (L76/79) vs $a_{0,0}$ (L70). Pick one convention and
  state it.
- **L17** $\prod_1^{N_{input}} N_{mf,i} \approx N_\mu^{N_{input}}$ -- the
  $\approx$ hides that heterogeneous per-variable counts are not a power;
  fine as an illustration, mark it as such.

## Missing content

- No abstract body, **no references at all** (L20 literally says "**TODO
  citations**"). ANFIS, the TSK/Takagi-Sugeno formulation, De Morgan triplets,
  and the "discussed in the literature" claim (L43) all need citations; there
  is no related-work section despite positioning against ANFIS and
  consequent-first methods.
- **Key Results (L134-135) is empty** -- the paper currently asserts "far
  superior" (L17) and "provably minimal" (L31) with zero numbers.
- The author line style ("Vladik Kreinovich PhD") needs normalizing.

## Claims needing evidence

- "Provably minimal number of rules" / "not possible to simplify the FIS to
  have fewer rules" (L7/L31): true only relative to the fixed one-rule-per-
  output-class structure; as stated it is a global minimality claim. Needs a
  one-paragraph proof sketch with the constraint made explicit.
- L25 discarding GMM weights: the consequence (what is lost, what breaks) is
  unexplained.
- L43-45 "gaussian correlation" via `wasserstein` + `bhattacharyya` composed
  by "averaging the arithmetic _and_ geometric mean" is an undefined metric;
  L47 then mixes in *Pearson* correlation with a $<0.85$ threshold -- same
  score or a different one? As written the feature-selection step is not
  reproducible.
- L56 boost ~ 0.95: "**TODO evidence**" plus a stray `)` and
  `$0.95 \in [0.9,0.99]$` (a value is not "in" a tuning range as written).
- L56 membership > 1 is justified only by the argmax defuzz; L59's uniform-
  vs-quantile binning claim is unevidenced -- L143's own TODO admits it.
- L132 "our testing has found that regularization is helpful" -- no
  experiment, no numbers.

## Typos

"repoerted" (L52), "paritioning" (L59), "t-cnorm" (L61),
`bhattacharya` vs `bhattacharyya` inconsistent (L43/L44), "will be address
later" (L23), "FIS's" (L132), "PhiURII" (L52 -- the repo dataset is
**PhiUSIIL**), L56 stray `)` in "**TODO evidence**) indicates", and the L16-18
numbered-list-then-paragraph will renumber oddly in the render.

## Open TODOs (8)

Citations (L20), Ruspini-partition future work (L37), PhiUSIIL dataset
(L52), boost evidence (L56), Key Results (L136), fuzzy-trees paper "with
Hugo" (L141), quantile-bin issues (L143), admissibility (L145). These render
as-is in the draft PDF -- deliberate; strip only at submission.
