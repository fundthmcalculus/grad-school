# A Least-Action Formulation for Continuous, Differentiable FIS

Working notes on Dr. Cohen's handwritten TSK/least-action derivation
(`Handwritten_20250217`), extended to cover the classifier case and to answer
the rule-orthogonality question.

Code: `fis_action.py`. Experiments: `demo_regression.py`, `demo_classifier.py`.
Recorded output: `results.txt`. Reproduction instructions: §7.

### Notation

| symbol | meaning |
|---|---|
| $X=[x_{lo},x_{hi}]$ | input universe (all experiments use $[-15,15]$) |
| $Y$ | output universe; $[0,1]$ in the classifier case |
| $N$, $p$ | number of rules, consequent polynomial order |
| $\mu_i(x)$ | membership function of rule $i$, centre $a_i$, width $b_i$ |
| $\varphi_i=\mu_i/\sum_j\mu_j$ | normalized firing strength (partition of unity) |
| $f_i(x)=\sum_kB_{ik}x^k$ | rule consequent |
| $\psi_{ik}=\varphi_i x^k$ | **rule regressor** — the actual model basis |
| $\theta$ | flattened consequent coefficients, $B$ raveled |
| $y_d$, $y_c$, $e=y_d-y_c$ | target, FIS output, error field |
| $\lambda=\ell^2$ | slope weight; $\ell$ a correlation length, units of $x$ |
| $\langle u,v\rangle_{H^1}$ | $\int(uv+\lambda u'v')dx$ |
| $G$, $r$ | $H^1$ Gram matrix $\langle\psi_m,\psi_n\rangle$, and $r_m=\langle y_d,\psi_m\rangle$ |
| $S$ | the action $\|e\|_{H^1}^2$ |
| $\alpha_i$ | rule firing strength in the classifier |
| $\beta$ | defuzzification annealing temperature |

Note the code uses a rescaled monomial $t=(x-\text{mid})/\text{half}$ in place of
$x^k$. This spans the identical function space — the model is unchanged — but
keeps $\kappa(G)$ from running to $10^7$ on $[-15,15]$.

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

Measured on Cohen's cubic $y_d=0.3x^3+0.2x^2-5x-3$, $N=3$, $p=1$:

| $\lambda$ | $\|e\|_{L^2}$ | $\|e'\|_{L^2}$ | action | dof searched |
|---|---|---|---|---|
| 0 | 0.577 | 0.990 | 0.914 | 6 |
| 0.25 | 0.511 | 0.745 | 1.096 | 6 |
| 1 | 0.468 | 0.529 | 1.382 | 6 |
| 4 | 0.971 | 0.730 | 3.666 | 6 |
| 16 | 0.286 | 0.251 | 3.126 | 6 |

Slope error falls ~75% from $\lambda=0$ to $\lambda=16$, and value error falls too —
the gradient term regularizes the antecedent search rather than merely trading
one error for the other. The trend is **not monotone** ($\lambda=4$ is worse on
both), because each row is an independent multistart on a non-convex antecedent
landscape and individual rows land in different basins. That scatter is the
content of §3b, not noise to be smoothed away. Actions at different $\lambda$ are
not comparable to each other — they are different functionals.

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

**Proposition 1.** Greedy residual fitting reproduces the joint least-action
solution for *every* target $y_d$ **iff** the $H^1$ Gram matrix is block diagonal
across rules:

$$G_{ij}=0\ \ (i\neq j),\qquad
\big(G_{ij}\big)_{kl}=\big\langle \varphi_i x^k,\ \varphi_j x^l\big\rangle_{H^1},
\qquad 0\le k,l\le p .$$

*The "for every $y_d$" quantifier is not decoration* — for one fixed target the two
schemes can agree by coincidence.

**Proof.** ($\Leftarrow$) If $G$ is block diagonal the joint normal equations
$G\theta=r$ decouple into $G_{ii}\theta_i=r_i$. Greedy gives block 1 the same
equation; block 2 sees residual right-hand side $r_2-G_{21}\theta_1=r_2$ since
$G_{21}=0$; induction on $i$ closes it.

($\Rightarrow$) Take $N=2$. Joint: $G_{11}\theta_1+G_{12}\theta_2=r_1$. Greedy:
$G_{11}\tilde\theta_1=r_1$. Agreement for all $y_d$ forces
$G_{11}^{-1}G_{12}\theta_2=0$ for all attainable $\theta_2$; as $y_d$ ranges over
$H^1$, $\theta_2$ ranges over all of $\mathbb{R}^{p+1}$, so $G_{12}=0$. Blocks
$>2$ follow by applying the same argument to consecutive pairs. $\square$

**Proposition 2 ($\lambda=0$).** In the pure-$L^2$ case, block diagonality is
**equivalent to disjoint supports**. Taking $k=l=0$, $\int\varphi_i\varphi_j\,dx=0$
with $\varphi_i,\varphi_j\ge0$ forces $\varphi_i\varphi_j=0$ a.e. $\square$

**But Proposition 2 does *not* extend to $\lambda>0$**, and C1 requires $\lambda>0$.
The $H^1$ pairing is

$$\langle\varphi_i,\varphi_j\rangle_{H^1}=\underbrace{\int\varphi_i\varphi_j\,dx}_{>0\ \text{if supports overlap}}
+\ \lambda\underbrace{\int\varphi_i'\varphi_j'\,dx}_{<0\ \text{for adjacent rules}}$$

and the slope term is **negative** for neighbouring rules — one is rising exactly
where the other falls. The two can therefore cancel. This is not hypothetical;
see §4d. Disjoint support remains *sufficient* for block diagonality at any
$\lambda$ (both integrands vanish identically), but it is no longer necessary.

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

> **At fixed $\lambda$, exact input-side rule orthogonality and a usable
> differentiable model are mutually exclusive.** The trade-off is not an
> artifact; it is the geometry. (The qualifier matters — §4d shows that letting
> $\lambda$ vary opens a narrow third door.)

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

### 4d. A third route: $\lambda$-tuned orthogonality

Because the slope term is negative for adjacent rules (§4b), there is a positive
$\lambda^\*$ at which the value and slope contributions cancel exactly. For $N=2$
order-0 rules it is available in closed form:

$$\lambda^\*=-\frac{\int\varphi_0\varphi_1\,dx}{\int\varphi_0'\varphi_1'\,dx}\;>\;0 .$$

This gives **exact $H^1$ orthogonality between fully overlapping, $C^\infty$,
positively-covering membership functions** — all three properties that §4c said
were incompatible with exactness. Measured (`demo_regression.py` §H), Gaussians
at $\pm5$:

| width | $\int\varphi_0\varphi_1$ | $\int\varphi_0'\varphi_1'$ | $\lambda^\*$ | off-block at $\lambda^\*$ |
|---|---|---|---|---|
| 2.0 | 2.0000e−1 | −8.3333e−1 | 0.2400 | 3.7e−18 |
| 4.0 | 8.0000e−1 | −2.0833e−1 | 3.8400 | 1.5e−17 |
| 6.0 | 1.7991e0 | −9.2593e−2 | 19.4307 | 0.0 |
| 10.0 | 4.5257e0 | −3.2898e−2 | 137.5701 | 5.9e−17 |

and greedy fitting becomes exact there (gap 4.6e−10 at $\lambda^\*$ vs 6.5e−4 at
$\lambda=1$).

**Scope, stated precisely.** One free parameter $\lambda$ can satisfy one scalar
condition. So the route is exact only when the number of independent
orthogonality conditions is one:

| configuration | conditions | exact? | best off-block |
|---|---|---|---|
| $N=2$, order 0 | 1 | **yes** | ~1e−17 |
| $N=2$, order 1 | $(p{+}1)^2=4$ | no | 1.1e−2 |
| $N=3$, order 0 | 3 | no | 9.7e−3 (from 2.5e−1) |
| $N=4$, order 0 | 6 | no | 1.2e−2 (from 2.3e−1) |
| $N=5$, order 0 | 10 | no | 1.3e−2 (from 2.1e−1) |

For $N\ge3$ it still reduces coupling by roughly $20\times$, so it is worth tuning
even when it cannot be exact.

The binary-classifier case is where it lands cleanly: §5b shows a classifier
score *is* an order-0 TSK model, so $N=2$, order 0 is exactly the binary
classification setting, and there $\lambda^\*$ is exact and in closed form.

### 4d′. Evaluating $\lambda^\*$: closed form, meaning, and price

**Closed form.** For two equal-width Gaussians at $\pm c$ the log-ratio
$\log(\mu_1/\mu_0)=4cx/b^2$ is *linear in $x$*, so the normalized weights collapse
to a logistic:

$$\varphi_1=\sigma(kx),\qquad \varphi_0=1-\sigma(kx),\qquad k=\frac{4c}{b^2}.$$

Then $\varphi_0\varphi_1=\sigma(1-\sigma)$ and $\varphi_i'=\pm k\,\sigma(1-\sigma)$, and both
integrals are elementary. With $u=kx$ and then $s=\sigma(u)$, $ds=s(1-s)\,du$:

$$\int\varphi_0\varphi_1\,dx=\frac1k\int_{\mathbb R}\sigma'(u)\,du=\frac1k,
\qquad
\int\varphi_0'\varphi_1'\,dx=-k\int_{\mathbb R}\sigma'(u)^2du=-k\int_0^1 s(1-s)\,ds=-\frac{k}{6}.$$

$$\boxed{\;\lambda^\*=\frac{6}{k^2}=\frac{3\,b^4}{8\,c^2}\;}$$

Verified to all printed digits (`demo_regression.py` §I): e.g. $c=5,b=4\Rightarrow k=1.25$,
$\int\varphi\varphi=0.800000=1/k$, $\int\varphi'\varphi'=-0.208333=-k/6$, $\lambda^\*=3.84000=6/k^2$.

**Meaning — and the answer to the open question.** Let $w=1/k=b^2/(4c)$, the
crossover width of the rule handover. Then

$$\ell^\*=\sqrt{\lambda^\*}=\sqrt6\,w$$

confirmed numerically as $2.44949$ in every configuration tested. So $\lambda^\*$ is
fixed **entirely by the partition geometry**. $y_d$ appears nowhere in it.

> $\lambda^\*$ is **not** a distinguished correlation length of the target. It is the
> correlation length of the *rule crossover region*. The open question posed in
> the previous revision is answered, and answered negatively.

**Why $N\ge3$ cannot work — structurally.** Since $\lambda^\*\propto 1/c^2$, on a
uniform partition of pitch $d$ the adjacent pairs (separation $d$) and the
next-nearest pairs (separation $2d$) demand values in ratio $(2d/d)^2=4$.
Measured: 4.0000 in every case. One scalar $\lambda$ cannot satisfy two conditions
that differ by a fixed factor of 4 — this is sharper than the earlier
"generically unsatisfiable" and explains exactly *why* the residual coupling
plateaus around $10^{-2}$ for $N\ge3$.

**Price.** $L^2$ error of the $H^1$-optimal consequents at $\lambda^\*$, against the
$L^2$-optimal ($\lambda=0$) fit of the same partition, target $\tanh(x/3)$:

| $c$ | $b$ | $w$ | $\lambda^\*$ | $\ell^\*$ | $L^2$ at $\lambda{=}0$ | $L^2$ at $\lambda^\*$ | cost |
|---|---|---|---|---|---|---|---|
| 5 | 1 | 0.050 | 0.015 | 0.12 | 3.2540 | 3.2542 | 0.01% |
| 5 | 2 | 0.200 | 0.240 | 0.49 | 1.0992 | 1.1036 | 0.40% |
| 5 | 4 | 0.800 | 3.840 | 1.96 | 0.5191 | 0.5394 | 3.91% |
| 5 | 6 | 1.800 | 19.44 | 4.41 | 0.1799 | 0.1938 | 7.72% |
| 5 | 8 | 3.200 | 61.44 | 7.84 | 0.7769 | 0.8204 | 5.60% |
| 3 | 4 | 1.333 | 10.67 | 3.27 | 0.1101 | 0.1174 | 6.59% |

**This corrects the caveat in the previous revision.** I wrote that buying
decoupling this way "distorts the very fit the model exists to perform." It
does not, or not much: the price stays under ~8% across the sweep and under 2%
for sharp crossovers. The reason is now clear from the closed form —
$\ell^\*=\sqrt6\,b^2/(4c)$ is **sub-rule-scale** for any sensible partition, and
weighting slopes at a short correlation length barely perturbs the $L^2$
solution. $\lambda^\*$ is cheaper than I claimed.

The residual caveat is narrower: $\lambda^\*$ grows like $b^4$, so for heavily
overlapping rules it becomes large ($137.6$ at $b=10$, $c=5$) and then it *is*
a real modelling constraint rather than a free choice.

**Robustness.** $\lambda^\*$ is not an artifact of the logistic collapse — it exists
for unequal widths (3.84 → 3.69 → 3.29 as widths go $4,4\to3,5\to2,7$),
asymmetric centres (4.74), and Cauchy-$\pi$ membership functions, though for
Cauchy it is enormous ($\approx1258$) because heavy tails make
$\int\varphi'\varphi'$ tiny. Only the *closed form* is Gaussian-specific.

### 4e. Recommendation

Keep Gaussian (or $\pi$-shaped) antecedents with genuine overlap for smoothness
and coverage; get the decoupling from $H^1$-OLS, not from disjointness. Use
clustering (FCM, as in `AEEM6097/fuzzy-symphony.py`) to seed centers and
OLS-with-forward-selection to pick which rules to keep — that is the "identify
rules from residuals" scheme, made exact. Reach for $\lambda^\*$ only in the binary
order-0 case, where it is exact and costs nothing extra.

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

**Proposition 3.** $\max_i(\alpha_i\wedge B_i)=\sum_i(\alpha_i\wedge B_i)$ for all
firing vectors $\alpha\in[0,1]^N$ **iff** the $B_i$ have pairwise disjoint supports
(up to null sets).

*Proof.* ($\Leftarrow$) At any $y$ at most one $B_i(y)$ is non-zero, so at most one
summand is non-zero and $\max=\sum$ trivially.
($\Rightarrow$) Suppose $\operatorname{supp}B_i\cap\operatorname{supp}B_j$ has
positive measure for some $i\neq j$. Take $\alpha\equiv1$. On that intersection
both $B_i(y)>0$ and $B_j(y)>0$, so
$\sum\ge B_i(y)+B_j(y)>\max(B_i(y),B_j(y))=\max$. $\square$

Measured over 400 random firing vectors (`demo_classifier.py` §H): sup-norm
difference **0.0** for disjoint sets, **4.8e−1** for overlapping ones.

This matters because $\max$ is non-differentiable and $+$ is analytic.

### 5b. The score really is a regression onto $[0,1]$

**Proposition 4.** With pairwise disjoint $B_i$, areas $A_i=\int B_i$, centroids
$c_i=A_i^{-1}\int yB_i$, and product implication $B'=\sum_i\alpha_iB_i$, centroid
defuzzification is *exactly* an order-0 TSK model.

*Proof.* Disjointness makes the aggregate a plain sum, and both integrals split:

$$y=\frac{\int y\,B'(y)\,dy}{\int B'(y)\,dy}
=\frac{\sum_i\alpha_i\int yB_i\,dy}{\sum_i\alpha_i\int B_i\,dy}
=\frac{\sum_i\alpha_iA_ic_i}{\sum_i\alpha_iA_i}
=\sum_i w_i c_i,\quad w_i=\frac{\alpha_iA_i}{\sum_j\alpha_jA_j}. \ \square$$

Verified exact to 2.2e−16 (§I). With Gödel implication ($\wedge$ instead of
product) clipping at $\alpha_i$ rescales each area but leaves each centroid
unchanged for symmetric $B_i$, so the same form holds with $A_i$ replaced by the
clipped area; the residual in the measured table (6.8e−2) is precisely that
area correction.

Since $w_i\ge0$ and $\sum_iw_i=1$, $y$ is a **convex combination of the cores**, so
$y\in[\min_ic_i,\max_ic_i]\subseteq[0,1]$: **the range constraint is free** — no
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

**Proposition 5 (annealing error bound).** Let $\alpha_{(1)}\ge\alpha_{(2)}\ge\dots$
be the sorted firing strengths, $r=\alpha_{(2)}/\alpha_{(1)}<1$ the runner-up ratio
and $D=\max_ic_i-\min_ic_i$ the output diameter. Then

$$\big|y_\beta-\text{MOM}\big|\;\le\;\frac{(N-1)\,D\,r^\beta}{1+(N-1)r^\beta}.$$

*Proof.* Write $w_i=\alpha_i^\beta/\sum_j\alpha_j^\beta$ and let $i^\*$ be the
argmax, so $\text{MOM}=c_{i^\*}$. Then
$y_\beta-c_{i^\*}=\sum_{i\neq i^\*}w_i(c_i-c_{i^\*})$, giving
$|y_\beta-\text{MOM}|\le D\sum_{i\neq i^\*}w_i$. Normalising by
$\alpha_{(1)}^\beta$, each off-peak weight satisfies
$\alpha_i^\beta/\alpha_{(1)}^\beta\le r^\beta$, so with $K=(N-1)r^\beta$,

$$\sum_{i\neq i^\*}w_i\;=\;\frac{\sum_{i\neq i^\*}(\alpha_i/\alpha_{(1)})^\beta}
{1+\sum_{i\neq i^\*}(\alpha_i/\alpha_{(1)})^\beta}\;\le\;\frac{K}{1+K},$$

the last step because $t\mapsto t/(1+t)$ is increasing. $\square$

Convergence is therefore **geometric in the classification margin**. Measured at
$r=0.671$ the bound holds at every $\beta$ tested, with the gap falling from
2.1e−1 at $\beta=1$ to 2.0e−12 at $\beta=64$ (§J).

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
- Sequential residual fitting is exact (for all targets) **iff** the $H^1$ Gram is
  block diagonal (Prop. 1). At $\lambda=0$ that is **equivalent** to disjoint
  support (Prop. 2), which is unattainable on $X$ without destroying
  differentiability and coverage. Use $H^1$-OLS instead.
- **Correction to the above for $\lambda>0$**: disjointness is sufficient but *not*
  necessary. The $H^1$ slope term is negative for adjacent rules, so value and
  slope contributions can cancel at a positive $\lambda^\*$, giving exact
  orthogonality between overlapping, smooth, covering rules. It is exact only
  when there is one independent condition to satisfy — $N=2$, order 0, which is
  precisely the binary-classifier setting — and reduces coupling ~20× otherwise.
- $\lambda^\*$ has the closed form $3b^4/(8c^2)$, equivalently $\ell^\*=\sqrt6\times$ the
  rule crossover width. It depends only on the partition, never on the target,
  so it has no variational reading as a target correlation length. Its $1/c^2$
  scaling forces adjacent and next-nearest rule pairs to demand values in ratio
  exactly 4, which is why $N\ge3$ cannot be zeroed. Fit cost is under ~8%.
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
- ~~Whether $\lambda^\*$ has a variational reading.~~ **Answered in §4d′**, negatively:
  $\lambda^\*=3b^4/(8c^2)$ in closed form, $\ell^\*=\sqrt6\times$ the rule crossover
  width, and $y_d$ appears nowhere — it is a property of the partition, not of
  the target. The same scaling explains the $N\ge3$ obstruction (adjacent and
  next-nearest pairs demand values in ratio exactly 4) and shows the fit cost is
  under ~8%, correcting the earlier caveat.
- Whether $\lambda^\*$ generalizes usefully to the *multi-input* case, where the
  crossover geometry is a hypersurface rather than a point and $w$ becomes
  direction-dependent.

---

## 7. Reproduction

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy scipy matplotlib
cd research/least_action
../../.venv/bin/python demo_regression.py    # ~12 min, sections A-I
../../.venv/bin/python demo_classifier.py    # ~15 s, sections H-K
```

Both are deterministic — seeded RNG, no wall-clock dependence — so the output
should match `results.txt` exactly. Only `numpy` and `scipy` are imported by
`fis_action.py`; `matplotlib` is not required.

### Every numerical constant that matters

Recorded because several of these were tuned against observed failures, and a
reader changing one should know what it was protecting against.

| quantity | value | why |
|---|---|---|
| quadrature | Gauss–Legendre, 400 nodes (2000 in §H) | exact for degree $<2n$; the integrands are smooth so this is far past convergence |
| consequent ridge | $10^{-9}\cdot\operatorname{tr}(G)/m$ | trace-relative so it scales with the problem; a ridge rather than a truncating pseudo-inverse so the map $(a,b)\mapsto\theta$ stays smooth — a rank-flipping SVD leaves the reduced action non-differentiable and corrupts the FD Hessian |
| outer optimizer | SLSQP + L-BFGS-B, alternating, up to 8 rounds | they stall in different places; SLSQP handles the gap constraint, L-BFGS-B drives the gradient harder |
| outer gradient | analytic (§3d) | SciPy's default FD step is round-off dominated here — see §3e |
| multistart | 24 restarts, seed 0 | the antecedent landscape is multimodal (§3b) |
| saddle-escape polish | ≤6 rounds along the most-negative eigenvector | first-order methods stall at saddles on this objective |
| Hessian FD step | $10^{-2}\cdot\max(\lvert p\rvert,\,0.1\,\text{span})$ | large on purpose: the mixed second difference divides by $4h_ih_j$, so a small $h$ is pure round-off |
| active-constraint tol | $10^{-3}$ relative | loose enough to catch bounds SLSQP only approached to its own tolerance; a missed bound shows up as spurious negative curvature in an infeasible direction |
| definiteness tol | estimated noise floor $\approx\varepsilon_f/h^2$ | judging against exact zero reads numerical dust as a saddle |
| default `min_gap` | $0.6\times$ rule pitch | identifiability (C5) |
| default `width_bounds` | $[0.25,\,1.5]\times$ pitch | identifiability (C5) |
| bump $u$-floor | $10^{-8}$ | $F=e^{-1/u}$ underflows to 0 long before the $1/u^4$ factors overflow, but without a floor $u\to0$ gives $\infty\cdot0=$ NaN |

### API map

| function | does |
|---|---|
| `Quadrature.legendre` | Gauss–Legendre rule on $[lo,hi]$ |
| `MF_SHAPES` | $(F,F',F'')$ per membership-function family — the only place to add one |
| `mf_param_derivatives` | $(\mu,\ \partial_x\mu,\ \partial_a\mu,\ \partial_b\mu,\ \partial^2_{xa}\mu,\ \partial^2_{xb}\mu)$ |
| `normalized_weights` | $\varphi_i$, $\varphi_i'$, and $\sum_j\mu_j$ (the coverage check) |
| `rule_regressors` | $\psi_{ik}$ and $\psi_{ik}'$ on a grid |
| `regressor_jacobian` | $\partial\psi/\partial p$, $\partial\psi'/\partial p$ for all $2N$ antecedent parameters |
| `h1_gram`, `h1_project` | $G$ and $r$ |
| `decoupling_report` | coherence, off-block energy, $\kappa(G)$, min coverage |
| `solve_consequents` | closed-form consequents + reduced action |
| `reduced_action_and_gradient` | the same, plus the exact gradient (§3d) |
| `fit` | full variable-projected fit with constraints, multistart, saddle escape |
| `galerkin_residual` | $\langle e,\psi_m\rangle_{H^1}$ — the weak E–L form |
| `optimality_certificate` | KKT residual on the critical cone, projected Hessian, descent probe |
| `sequential_fit` | greedy residual fitting, optionally $H^1$-orthogonalized (OLS) |
| `OutputPartition` | disjoint consequent sets on $Y$ |
| `aggregate_max` / `aggregate_sum` | Mamdani $\vee$ and its summation surrogate |
| `defuzz_mom` / `defuzz_annealed` | crisp MOM and the $C^\infty$ $\beta$-annealed score |
| `mom_gap_bound` | the Proposition 5 bound |

### Where each claim is checked

| claim | §here | demo section |
|---|---|---|
| variable projection, $\lambda$ sweep | 3a | regression A, B |
| decoupling vs. overlap trade-off | 4c | regression C |
| $H^1$-OLS rescues overlap | 4c(ii) | regression D |
| identifiability ⇒ certifiability | 3b | regression E |
| consequent block globally convex | 3a | regression F |
| analytic gradient exact | 3d | regression G′ |
| Galerkin orthogonality | 1 | regression G |
| $\lambda^\*$ orthogonality | 4d | regression H |
| $\vee=+$ iff disjoint | 5a | classifier H |
| centroid $=$ order-0 TSK | 5b | classifier I |
| MOM discontinuous, annealing bound | 5c | classifier J |
| score is $C^1$ in $x$ | 5c | classifier K |
