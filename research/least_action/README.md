# A Least-Action Formulation for Continuous, Differentiable FIS

Working notes on Dr. Cohen's handwritten TSK/least-action derivation
(`Handwritten_20250217`), extended to cover the classifier case and to answer
the rule-orthogonality question.

Code: `fis_action.py`. Experiments: `demo_regression.py`, `demo_classifier.py`.

---

## 1. The model and the action

On a scalar input universe $X=[x_{lo},x_{hi}]$, an order-$p$ TSK system is

$$y_c(x;a,b,B) \;=\; \sum_{i=1}^{N}\varphi_i(x)f_i(x),
\qquad \varphi_i(x)=\frac{\mu_i(x)}{\sum_j \mu_j(x)},
\qquad f_i(x)=\sum_{k=0}^{p}B_{ik}x^k .$$

The $\varphi_i$ form a **partition of unity**: $\varphi_i\ge 0$, $\sum_i\varphi_i\equiv 1$.

Cohen's notes propose (p. 4, eq. 12) minimizing both value and slope error. Written
as a functional that is the object of a least-action principle:

$$\boxed{\;S[y_c]\;=\;\int_{x_{lo}}^{x_{hi}} \underbrace{\Big[(y_d-y_c)^2+\lambda\,(y_d'-y_c')^2\Big]}_{\mathcal{L}(x,\,y_c,\,y_c')}\,dx,\qquad \lambda=\ell^2>0. \;}$$

$\lambda$ carries units of $x^2$; $\ell$ is a correlation length setting how far a
slope error is worth trading against a value error. This is the $H^1$ (Sobolev)
norm of the error $e = y_d - y_c$, and $S = \|e\|_{H^1}^2$.

### Why the Euler–Lagrange equation in the notes is degenerate

Page A4 of the notes takes $F = y_d - y_c$ and writes
$\partial F/\partial y_c - \tfrac{d}{dx}\!\left(\partial F/\partial y_c'\right)=0$.
With $F$ *linear* in $y_c$ and independent of $y_c'$ this gives $-1 = 0$: no content.
The functional has to be the quadratic $\mathcal{L}$ above for the variational
machinery to say anything. With that $\mathcal{L}$,

$$\frac{\partial\mathcal{L}}{\partial y_c}-\frac{d}{dx}\frac{\partial\mathcal{L}}{\partial y_c'}=0
\;\;\Longleftrightarrow\;\;
\boxed{\;\lambda\,e'' - e = 0\;}$$

a **screened-Poisson (modified Helmholtz) equation** in the error field, with
solutions $e = Ae^{x/\ell}+Be^{-x/\ell}$. Under either essential boundary
conditions ($e=0$ at both ends) or natural ones ($e'=0$ at both ends), the only
solution is $e\equiv 0$.

So the *unconstrained* least-action problem is trivial: perfect interpolation.
**All the content is in the constraint $y_c\in\mathcal{M}$**, the manifold of
functions a FIS with $N$ rules can represent.

### The real stationarity condition: Galerkin orthogonality

Restricting to $\mathcal{M}$ and differentiating $S$ with respect to a parameter
$\theta_m$, then integrating the slope term by parts:

$$\frac{\partial S}{\partial\theta_m}
= -2\int_{x_{lo}}^{x_{hi}}\big(e-\lambda e''\big)\frac{\partial y_c}{\partial\theta_m}\,dx
\;-\;2\lambda\Big[e'\,\frac{\partial y_c}{\partial\theta_m}\Big]_{x_{lo}}^{x_{hi}} \;=\;0 .$$

**The Euler–Lagrange residual $e-\lambda e''$ must be $L^2$-orthogonal to the tangent
space of the FIS manifold, up to a boundary term.** The FIS cannot annihilate the
E–L operator pointwise; it can only annihilate its projection onto the directions
it is able to move in. Equivalently, in weak form,

$$\langle e,\ \psi_m\rangle_{H^1}\;=\;\int\big(e\,\psi_m+\lambda\,e'\psi_m'\big)dx\;=\;0
\quad\text{for every regressor } \psi_m .$$

Verified numerically to $\sim10^{-5}$ in `demo_regression.py` §G.

---

## 2. Constraints required for the principle of least action to apply

These are the conditions the notes leave implicit. Each one is load-bearing.

| # | Constraint | Why it is needed | Enforced in code by |
|---|---|---|---|
| C1 | $\lambda>0$ | Legendre–Clebsch: $\partial^2\mathcal{L}/\partial y_c'^2 = 2\lambda>0$ is necessary for a minimum rather than a maximum or saddle. | `lam` argument |
| C2 | $\delta$-coverage: $\inf_x\sum_j\mu_j(x)\ge\delta>0$ | Without it $\varphi_i$ is $0/0$ and $y_c\notin H^1$; the action is not even defined. | `DecouplingReport.min_coverage` |
| C3 | $\mu_i\in C^1$ | $y_c'$ must exist for the slope term. Rules out triangular/trapezoidal MFs. | Gaussian, Cauchy-$\pi$, or $C^\infty$ bump |
| C4 | Transversality: fix endpoints **or** accept natural BCs $e'(x_{lo})=e'(x_{hi})=0$ | The boundary term above is not automatically zero; ignoring it biases the fit at the ends. | boundary term reported |
| C5 | Identifiability: $a_{i+1}-a_i\ge\gamma$ and $b_i\le b_{\max}$ | Otherwise membership functions collapse onto one another, the rule basis loses rank, and no second-order test can succeed. | `min_gap`, `width_bounds` |
| C6 | (classifier) output sets pairwise disjoint on $Y$ | Makes Mamdani $\vee$ equal $+$ — see §5. | `OutputPartition` |

**Jacobi condition.** The accessory (Jacobi) equation for this $\mathcal{L}$ is
$\lambda h''-h=0$, whose solutions are real exponentials — non-oscillatory, so
**there are no conjugate points on any interval**. The second variation is

$$\delta^2 S[h]\;=\;2\int\big(h^2+\lambda h'^2\big)dx\;\ge\;0,$$

strictly positive for $h\not\equiv0$ whenever $\lambda\ge0$. Hence:

> **The action functional is convex in function space. Every bit of
> non-convexity comes from the parameterization of $\mathcal{M}$, not from the
> physics.**

---

## 3. Proving local optimality

Split the parameters: consequents $B$ (linear) and antecedents $(a,b)$ (nonlinear).

### 3a. The consequent block is globally convex — for free

With antecedents fixed, define the **rule regressors**

$$\psi_{ik}(x)=\varphi_i(x)\,x^k,\qquad y_c=\sum_{i,k}B_{ik}\psi_{ik}.$$

$S$ is then a quadratic form with Hessian $2G$, where $G_{mn}=\langle\psi_m,\psi_n\rangle_{H^1}$
is a **Gram matrix** — automatically positive semidefinite, and positive definite
exactly when the $\psi_m$ are linearly independent. So the consequents have a
unique global minimizer, obtainable in closed form:

$$G\,\theta=r,\qquad r_m=\langle y_d,\psi_m\rangle_{H^1}.$$

This is **variable projection**: eliminate $B$ analytically and search only over
$(a,b)$. For $N=3$, $p=1$ that is 6 parameters instead of 12. (`demo_regression.py` §A, §F.)

### 3b. The antecedent block: second-order conditions done properly

Three things are easy to get wrong here, and all three bit during this work:

1. **Test the reduced Hessian, not the full one.** The full-parameter Hessian is
   never definite — consequent rescaling and rule relabeling guarantee flat and
   symmetric directions. Only the variable-projected objective has a meaningful
   Hessian.
2. **Fitted solutions sit on the identifiability constraints.** At an active
   constraint the correct second-order condition is definiteness on the
   **critical cone** — the null space of the active constraint normals — not on
   all of $\mathbb{R}^{2N}$. Likewise the first-order condition is that the
   *projected* gradient vanishes; the raw gradient is legitimately non-zero,
   balanced by the KKT multipliers.
3. **Curvature is only informative at a stationary point.** Galerkin
   orthogonality (§1) certifies the *consequent* block only — variable projection
   makes it hold by construction — so it says nothing about $(a,b)$. The
   antecedent gradient must be checked separately.

The certificate in `optimality_certificate()` therefore reports: KKT residual on
the critical cone, projected-Hessian eigenvalues, and a **falsification probe**
that walks along the most-negative eigenvector (with the first-order term
subtracted) to test whether the action actually drops.

### 3c. Small-residual sufficiency

Writing the reduced Hessian in Gauss–Newton form,
$H = J^\top G J + \sum_m e_m\nabla^2 y_c^{(m)}$, the first term is PSD and the
second is indefinite. So a **sufficient** condition for a strict local minimum:

$$\|e\|_{H^1}\cdot\kappa(\mathcal{M})\;<\;\sigma_{\min}(J)^2$$

where $\kappa(\mathcal{M})$ bounds the curvature of the FIS manifold. In words:
**a good fit is automatically a certifiable local optimum**; only poor fits can
be saddles. This is why driving the residual down and certifying optimality are
the same activity, not two separate ones.

### 3d. The analytic reduced gradient

With $M=G+\varepsilon I$, $\theta=M^{-1}r$ and $S=\langle y_d,y_d\rangle-r^\top\theta$,

$$dS=-2(dr)^\top\theta+\theta^\top(dM)\theta .$$

Write $A=\partial\Psi/\partial p$ and $a\equiv A^\top\theta$. Because $\theta$ is held
fixed, $a=\partial y_c/\partial p$. The first term is $-2\langle y_d,a\rangle_{H^1}$ and the
second is $2\langle a,y_c\rangle_{H^1}$, and they fuse:

$$\boxed{\;\frac{\partial S}{\partial p}\;=\;-2\big\langle e,\ \partial y_c/\partial p\big\rangle_{H^1}
\;+\;\frac{c}{n}\operatorname{tr}\!\big(\partial G/\partial p\big)\,\|\theta\|^2 ,\qquad e=y_d-y_c .\;}$$

This is the **envelope theorem**: $\theta$ is already optimal, so its implicit
dependence on $p$ contributes nothing. Note it is the *same* $H^1$ pairing that
expresses stationarity in §1 — there tested against the regressors themselves
(where it vanishes by construction, giving Galerkin orthogonality), here against
the directions the antecedents can move the model. One formula covers both
parameter blocks. The trailing term is the derivative of the trace-scaled ridge;
negligible in magnitude but included so the gradient is exact for the objective
actually minimized rather than an idealized one.

**All membership-function derivatives collapse to one shape triple.** Every MF
here is $\mu=F(z)$ with $z=(x-a)/b$, so

$$\frac{\partial\mu}{\partial x}=\frac{F'}{b},\quad
\frac{\partial\mu}{\partial a}=-\frac{F'}{b},\quad
\frac{\partial\mu}{\partial b}=-\frac{zF'}{b},\quad
\frac{\partial^2\mu}{\partial x\,\partial a}=-\frac{F''}{b^2},\quad
\frac{\partial^2\mu}{\partial x\,\partial b}=-\frac{F'+zF''}{b^2}.$$

Adding a membership function means supplying $F,F',F''$ and nothing else. Then,
with $\Sigma=\sum_k\mu_k$ and $g=\partial\mu_j/\partial p$, $h=\partial^2\mu_j/\partial x\partial p$,

$$\frac{\partial\varphi_i}{\partial p}=\frac{g\,(\delta_{ij}-\varphi_i)}{\Sigma},
\qquad
\frac{\partial\varphi_i'}{\partial p}=\frac{d}{dx}\frac{\partial\varphi_i}{\partial p}
=\frac{h(\delta_{ij}-\varphi_i)-g\varphi_i'}{\Sigma}-\frac{g(\delta_{ij}-\varphi_i)\Sigma'}{\Sigma^2},$$

using that $x$ and $p$ are independent so the derivatives commute.

**Validation, and what it reveals.** Against finite differences with the step
swept (a single fixed $h$ measures the *reference's* round-off, not the
gradient's error):

| MF | order | coverage | $\kappa(G)$ | best rel. err |
|---|---|---|---|---|
| Gaussian | 0 | 8.4e−1 | 1.7e1 | 4.4e−10 |
| Gaussian | 1 | 8.4e−1 | 1.3e5 | 1.1e−6 |
| Gaussian | 2 | 8.4e−1 | 2.8e10 | 4.2e−5 |
| Cauchy-$\pi$ | 0 | 8.7e−1 | 1.5e1 | 9.8e−11 |
| Cauchy-$\pi$ | 1 | 8.7e−1 | 5.8e2 | 1.1e−8 |
| bump | 1 | 3.1e−1 | 4.2e2 | 1.5e−9 |
| bump | 1 | 1.3e−1 | 5.5e1 | 1.2e−9 |
| bump | 1 | **0.0** | 2.9e2 | **4.5e−1** |

Agreement degrades with consequent order only because $\kappa(G)$ grows and the FD
reference loses digits — the error *rises as $h$ shrinks*, the signature of
round-off in the reference rather than truncation in the gradient.

The bump rows are the interesting ones. The analytic gradient is exact to ~1e−8
**exactly where $\delta$-coverage holds** and meaningless where it fails — because
where $\sum_j\mu_j=0$ the model is not differentiable at all. **C2 is the
differentiability condition, not bookkeeping**, and the gradient computation
detects its violation independently of the theory that predicted it.

### 3e. Numerical caveat found the hard way

Before the analytic gradient existed, this cost real time and is worth recording.
The variable-projected objective is accurate only to $\sim\kappa(G)\varepsilon_{\text{mach}}$;
with $\kappa(G)\sim10^5$ that is $\sim10^{-11}$, which is *above* the noise floor of
SciPy's default finite-difference step. Consequence: **L-BFGS-B terminates after
zero iterations and reports success**, and SLSQP returns points with a *higher*
objective than it was handed, also reporting success. The symptom is a
"converged" fit whose gradient is visibly non-zero. Supplying the analytic
gradient removes the failure mode entirely rather than papering over it — and
tightened the certified solutions' KKT residual from 5.2e−5 to 2.8e−7.

---

## 4. The orthogonality question

> *How do we optimize/constrain domains so that this holds? Thought: ensure all
> rules are orthogonal, $\int f_i f_j\,dx=0$ for $i\neq j$. Sequentially identify
> and fit rules using residuals of previously-identified rules.*

The instinct is right; two corrections make it precise.

### 4a. Orthogonality belongs to the regressors, not the consequents

The model is **not** $\sum_i f_i$. It is $\sum_i \varphi_i f_i$, so the basis
functions are $\psi_{ik}=\varphi_i x^k$. Two consequences:

- $\int f_i f_j\,dx=0$ is a constraint on the consequent *coefficients* and has
  essentially nothing to do with the geometry of the rule partition. Two rules
  with disjoint supports are already decoupled no matter what $f_i,f_j$ are.
- The relevant inner product is the **Sobolev** one, $\langle u,v\rangle_{H^1}=\int(uv+\lambda u'v')dx$,
  because that is the metric the action uses.

### 4b. The exact condition for sequential residual fitting

**Claim.** Fitting rules one at a time against the running residual reproduces the
joint least-action solution *exactly* if and only if the $H^1$ Gram matrix is
block diagonal across rules:

$$\big\langle \varphi_i x^k,\ \varphi_j x^l\big\rangle_{H^1}=0
\qquad \forall\, i\neq j,\ \ 0\le k,l\le p .$$

**Corollary (the sharp form of the user's intuition).** For non-negative
membership functions this is equivalent to **disjoint supports**. Take $k=l=0$
and $\lambda=0$: $\int\varphi_i\varphi_j\,dx=0$ with $\varphi_i,\varphi_j\ge0$
forces $\varphi_i\varphi_j=0$ almost everywhere.

So: *rule orthogonality $\equiv$ non-overlapping activation domains.* There is no
weaker condition that buys exactness.

### 4c. …and that is exactly why it cannot be imposed on the input

Disjoint supports plus partition of unity forces $\varphi_i$ to be **indicator
functions**, and $y_c$ becomes piecewise polynomial with jumps at the seams —
destroying the $C^1$ property the whole action formulation depends on. Worse,
$\delta$-coverage (C2) fails: $\sum_j\mu_j\to0$ between the blocks.

Measured (`demo_regression.py` §C), 3 rules on $[-15,15]$:

| MF | width | coherence | off-block energy | min coverage | best action | greedy vs joint |
|---|---|---|---|---|---|---|
| Gaussian | 7.0 | 4.9e−1 | 4.1e−1 | 6.1e−1 | 1.85e4 | 6.4e0 |
| Gaussian | 5.0 | 2.5e−1 | 2.0e−1 | 3.7e−1 | 7.79e4 | 4.2e−1 |
| Gaussian | 3.0 | 3.1e−2 | 2.1e−2 | 6.2e−2 | 1.37e5 | 1.2e−2 |
| Gaussian | 2.0 | 7.6e−2 | 7.1e−2 | 1.9e−3 | 1.53e5 | 6.8e−4 |
| **bump** | **5.0** | **0.0** | **0.0** | **0.0** | **3.28e6** | **4.0e−9** |

The compactly-supported bumps at width 5 tile $[-15,15]$ with exactly touching
supports: off-block coupling is identically zero and greedy fitting equals the
joint fit to machine precision — **and the achievable action is 42× worse than
the Gaussian of the same width** (7.79e4 → 3.28e6), with coverage collapsed to
zero at the seams. Widening the bumps to 6.0 restores overlap and recovers the
action, but immediately reintroduces off-block coupling (1.0e−1) — the trade-off
is continuous and unavoidable.

> **Exact input-side rule orthogonality and a usable differentiable model are
> mutually exclusive.** The trade-off is not an artifact; it is the geometry.

Two ways out, and they are the practical answer:

**(i) $\varepsilon$-orthogonality.** Accept overlap only in thin transition bands.
The Gram is then $G=G_{\text{block}}+E$ and the greedy/joint gap is controlled by
the **coherence** $\max_{i\neq j}|\hat G_{mn}|$, exactly as in matching pursuit.
The table shows the gap tracking coherence over four orders of magnitude. Pick
the width that puts coherence under your tolerance.

**(ii) Orthogonalize the basis instead of the supports.** Run $H^1$ Gram–Schmidt
on the rule blocks before fitting — this is orthogonal least squares (OLS/ROLS),
the standard tool for RBF and fuzzy rule selection. Sequential residual fitting
becomes exact for *any* overlap, at the cost of consequents that are only
interpretable after back-substitution. Measured (§D):

| width | naive greedy gap | OLS greedy gap |
|---|---|---|
| 5.0 | 4.2e−1 | 2.1e−2 |
| 3.0 | 1.2e−2 | 4.6e−5 |
| 2.0 | 6.8e−4 | 1.7e−7 |

**Recommendation.** Keep Gaussian (or $\pi$-shaped) antecedents with genuine
overlap for smoothness and coverage; get the decoupling from $H^1$-OLS, not from
disjointness. Use clustering (FCM, as in `AEEM6097/fuzzy-symphony.py`) to seed
centers and OLS-with-forward-selection to pick which rules to keep — that is the
"identify rules from residuals" scheme, made exact.

---

## 5. The classifier case

> *Without a recurrence relation, a classifier ranking is just a regression model
> with range $[0,1]$; ensure output rule activation domains do not overlap so
> De Baets' first/last/mean-of-maxima defuzzification can be approximated by
> summation.*

This is correct, and it resolves the §4c tension — because **it applies to the
output universe $Y$, not the input universe $X$.**

- On $X$: keep antecedents overlapping, $C^\infty$, $\delta$-covering. Smooth
  firing strengths.
- On $Y$: make the consequent sets $B_i$ pairwise **disjoint**. Nothing about
  smoothness in $x$ is harmed, because $Y$-side disjointness is a statement about
  the output partition, not the input one.

### 5a. Disjoint output supports make $\vee$ equal $+$

If the $B_i$ have pairwise disjoint supports then at any $y$ at most one term of
$\bigvee_i(\alpha_i\wedge B_i(y))$ is non-zero, so

$$B'(y)=\max_i\big(\alpha_i\wedge B_i(y)\big)=\sum_i\big(\alpha_i\wedge B_i(y)\big).$$

Measured over 400 random firing vectors (`demo_classifier.py` §H): sup-norm
difference **0.0** for disjoint sets, **4.8e−1** for overlapping ones. The
identity is exactly the disjointness condition — nothing weaker works.

This matters because $\max$ is non-differentiable and $+$ is analytic.

### 5b. The score really is a regression onto $[0,1]$

With disjoint $B_i$ of area $A_i$ and centroid $c_i$, centroid defuzzification
collapses algebraically:

$$y=\frac{\int y\,B'(y)\,dy}{\int B'(y)\,dy}=\frac{\sum_i\alpha_iA_ic_i}{\sum_i\alpha_iA_i}$$

— an **order-0 TSK model**. Verified exact to 2.2e−16 with product implication
(§I). Because the weights are non-negative and normalized, $y$ is a convex
combination of the cores, so **the $[0,1]$ range constraint is free** — no
clipping, no sigmoid, nothing non-smooth. This is the precise sense in which
"a classifier ranking is just a regression model with range $[0,1]$."

Consequently **the entire §1–§3 action machinery applies verbatim to the
classifier**, which was the point of the exercise.

### 5c. MOM is still discontinuous — and the fix

Disjointness alone does *not* make mean-of-maxima smooth. With disjoint sets the
maximum of $B'$ lives in the core of $B_{i^*}$, $i^*=\arg\max_i\alpha_i$, so
$\text{MOM}=c_{i^*}$ — piecewise constant, jumping by the core spacing whenever
two firing strengths cross. Measured: a $10^{-9}$ change in one $\alpha$ moves
MOM by **0.25** (§J).

The repair is a **temperature-annealed** defuzzifier:

$$y_\beta=\frac{\sum_i\alpha_i^\beta c_i}{\sum_i\alpha_i^\beta}$$

- $\beta=1$: centroid/height defuzzification of the summed aggregate.
- $\beta\to\infty$: MOM (and FOM/LOM if $c_i$ is replaced by $\inf$/$\sup$ of the core).
- **$C^\infty$ in $\alpha$ for every finite $\beta$.**

With $r=\alpha_{(2)}/\alpha_{(1)}$ the runner-up ratio and $D$ the output
diameter,

$$\big|y_\beta-\text{MOM}\big|\;\le\;\frac{(N-1)\,D\,r^\beta}{1+(N-1)r^\beta}$$

— geometric convergence in the classification **margin**. Measured at $r=0.671$
the bound holds at every $\beta$ tested, with the gap falling from 2.1e−1 at
$\beta=1$ to 2.0e−12 at $\beta=64$ (§J).

So $\beta$ is a **continuation parameter**: optimize the action at small $\beta$
where the landscape is smooth, then anneal $\beta$ upward to recover crisp
De Baets defuzzification. Differentiability is exact throughout; only the
$\beta\to\infty$ limit is crisp, and it is never actually taken.

Sanity check (§K): across $x$, MOM jumps by 0.25 between adjacent samples while
the annealed score's largest jump is 0.001–0.015, and its analytic derivative
matches central differences to 1.1e−10.

### 5d. The margin constraint

The one extra constraint the classifier needs: a **margin bound**
$\alpha_{(1)}/\alpha_{(2)}\ge1+\gamma$ on the training set. It does double duty —
it is what makes the annealing error bound in §5c decay, and it keeps the model
away from the tie set where MOM is discontinuous, so the small-residual
sufficiency argument of §3c stays valid.

---

## 6. Summary of what is settled and what is not

**Settled (derived and numerically verified):**

- The correct Lagrangian is $\mathcal{L}=e^2+\lambda e'^2$; the E–L equation is
  $\lambda e''-e=0$; the notes' $F=y_d-y_c$ version is degenerate.
- The action is convex in function space with no conjugate points; all
  non-convexity is parameterization.
- Stationarity $\equiv$ $H^1$-Galerkin orthogonality of the error against the rule
  regressors.
- Consequents solve globally in closed form (variable projection): $2N$ search
  parameters instead of $(p+3)N$.
- The reduced gradient is exactly $-2\langle e,\partial y_c/\partial p\rangle_{H^1}$ by
  the envelope theorem — the same $H^1$ pairing as the stationarity condition,
  now against the antecedent directions. Verified to 1e−9 wherever
  $\delta$-coverage holds; it correctly reports garbage where coverage fails,
  because the model genuinely is not differentiable there.
- Sequential residual fitting is exact **iff** the $H^1$ Gram is block diagonal,
  **iff** rules have disjoint support — which is unattainable on $X$ without
  destroying differentiability and coverage. Use $H^1$-OLS instead.
- Output-side disjointness makes Mamdani $\vee$ equal $+$ exactly, collapses
  centroid defuzzification to an order-0 TSK model, and gives the $[0,1]$ range
  for free.
- MOM stays discontinuous; the $\beta$-annealed surrogate is $C^\infty$ and
  converges geometrically in the margin.

**Open / next:**

- Analytic *Hessian* of the reduced action. The gradient is now exact (§3d); the
  second-order certificate still uses finite differences, and its noise floor is
  what forces the tolerance-based verdict in §3b. The same envelope argument
  differentiated once more should give it in closed form.
- Multi-input case: the tensor-product rule basis makes the Gram block structure
  richer, and disjointness on $X\subset\mathbb{R}^n$ is even more costly.
- Choosing $\lambda$ principled rather than by sweep — it is a correlation length,
  so a spectral criterion on $y_d$ should set it.
- Whether the annealing schedule in $\beta$ can be tied to a trust region on the
  action, giving a single unified continuation method.
- Empirical validation on the sonar dataset already wired up in
  `AEEM6097/fuzzy-symphony.py`.
