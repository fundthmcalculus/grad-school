# A Least-Action Formulation for Continuous, Differentiable FIS

Working notes on Dr. Cohen's handwritten TSK/least-action derivation
(`Handwritten_20250217`), extended to cover the classifier case and to answer
the rule-orthogonality question.

Code: `fis_action.py` (approximation), `fis_control.py` (control).
Experiments: `demo_regression.py`, `demo_classifier.py`, `demo_control.py`.
Recorded output: `results.txt`. Reproduction instructions: §10.

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

**Open / next:** consolidated and prioritized in **§9**, which supersedes the
list that used to sit here. §8-§9 added substantially to it.

---

## 7. Control: provably optimal fuzzy control parameters

Code: `fis_control.py`. Experiments: `demo_control.py`.

The original notes framed this as "optimal control via the principle of least
action." Sections 1–5 built the approximation machinery; this section asks what
can actually be *proved* about a fuzzy controller.

### 7a. The trivial case, stated so it can be set aside

**Theorem C1.** For $\dot x=Ax+Bu$ with $J=\int(x^\top Qx+u^\top Ru)\,dt$, let
$K=R^{-1}B^\top P$ from the Riccati equation. Any TSK FIS with **affine
consequents all set to $f_i(x)=-Kx$** reproduces $u^*$ exactly, for *any*
partition of unity:

$$u_{\text{fis}}(x)=\sum_i\varphi_i(x)\,(-Kx)=-Kx\sum_i\varphi_i(x)=-Kx .$$

Measured: $\max|u_{\text{fis}}-u_{\text{LQR}}|=7.1\times10^{-15}$ over 200 random
partitions (2–6 rules, random centres and widths) × 4000 states.

This is a genuinely provable global optimality result, and it is also a warning:
**the membership functions cancel identically, so fuzzy structure buys nothing
on an LTI/LQR problem.** Any claimed benefit there is an artifact. The value has
to come from nonlinearity or constraints, which is what the rest of this section
uses.

### 7b. The central result: suboptimality is exactly a weighted $L^2$ error

**Theorem C2.** For a control-affine plant $\dot x=f(x)+g(x)u$ with cost
$\int (q(x)+u^\top Ru)\,dt$, optimal value function $V^*$ and optimal law
$u^*=-\tfrac12R^{-1}g^\top\nabla V^{*\top}$, **any admissible** $u$ satisfies

$$\boxed{\;J(x_0)-V^*(x_0)\;=\;\int_0^\infty\big(u-u^*\big)^\top R\,\big(u-u^*\big)\,dt\;}$$

the integral taken along the closed loop driven by $u$ itself.

*Proof.* HJB gives $\nabla V^*f=-q+u^{*\top}Ru^*$ and $\nabla V^*g=-2u^{*\top}R$. Along
the closed loop,
$\tfrac{d}{dt}V^*=\nabla V^*(f+gu)=-q+u^{*\top}Ru^*-2u^{*\top}Ru$, so

$$q+u^\top Ru+\tfrac{d}{dt}V^*=u^\top Ru-2u^{*\top}Ru+u^{*\top}Ru^*=(u-u^*)^\top R(u-u^*).$$

Integrating on $[0,\infty)$ and using $x(\infty)\to0$, $V^*(0)=0$. $\square$

Three things follow, and they reshape the whole approach:

1. **This is a certificate, not a bound.** There is no constant to estimate and
   nothing to be conservative about. Fit a controller, integrate once, and you
   know *exactly* how much more than optimal it costs.
2. **Fitting the control law is not a surrogate for optimal control.**
   Minimizing the right weighted approximation error *is* minimizing true excess
   cost, identically.
3. **The right weight is the closed-loop occupation measure**, not Lebesgue
   measure on a box (§7d).

### 7c. Verifying it, including the hypothesis

Ground truth comes from **inverse optimal control**: choose $V$, $f$, $g$, $R$
first and *define* $q$ from the HJB equation, rather than solving for $V$. Then
$V$ is the exact value function by construction. The benchmark used is

$$\dot x=-x+u,\quad V^*=x^2+\tfrac12x^4,\quad u^*=-(x+x^3),\quad q=3x^2+4x^4+x^6\ (\ge0),$$

chosen so $u^*$ is genuinely nonlinear — a linear controller cannot represent it.

| controller | $x_0$ | $J$ | $V^*$ | gap | $\int(u-u^*)^2dt$ | residual |
|---|---|---|---|---|---|---|
| optimal $u^*$ | 1.5 | 4.781250 | 4.781250 | −1.8e−10 | 0.0 | 1.8e−10 |
| linear $-1.5x$ | 1.5 | 5.146875 | 4.781250 | 3.656e−1 | 3.656e−1 | 3.6e−11 |
| linear $-3x$ | −2.0 | 12.666667 | 12.000000 | 6.667e−1 | 6.667e−1 | 1.3e−12 |
| $0.7\,u^*$ | −2.0 | 12.531794 | 12.000000 | 5.318e−1 | 5.318e−1 | 4.3e−10 |
| $u^*+0.4$ | 1.5 | 14.342006 | 4.781250 | 9.561e0 | 9.600e0 | **3.9e−2** |

The last row misses by 3.9e−2, and that is not numerical error — it is the
**admissibility hypothesis showing its teeth**. $u^*+0.4$ settles at
$x(\infty)=0.196222$ rather than the origin, so the boundary term the proof
discards is not zero. The exact statement is

$$J(x_0)-V^*(x_0)+V^*\big(x(\infty)\big)=\int_0^\infty(u-u^*)^\top R(u-u^*)\,dt$$

and indeed $V^*(0.196222)=3.924448\times10^{-2}$ against a residual of
$3.924448\times10^{-2}$ — agreement to 1.9e−10. **A controller with a steady-state
offset has no certificate at all**; admissibility must be established separately,
which is what §7e does.

### 7d. Consequence: fit under the occupation measure

Theorem C2 weights control error by closed-loop occupation time, so fitting $u^*$
uniformly over a box optimizes the wrong functional. On this benchmark the
occupation density concentrates sharply near the origin
($\rho(0)/\rho(\pm3)\approx6.8\times10^4$).

| $N$ | weight | $\sup|u-u^*|$ | mean certified gap | max certified gap |
|---|---|---|---|---|
| 2 | uniform $L^2$ | 4.23 | 1.61e2 | 1.62e2 (not admissible) |
| 2 | occupation $\rho$ | 12.78 | **1.39e−1** | 3.56e−1 |
| 3 | uniform $L^2$ | 1.83 | 1.60e−1 | 1.88e−1 |
| 3 | occupation $\rho$ | 6.55 | **1.69e−2** | 4.38e−2 |
| 4 | uniform $L^2$ | 1.11 | 1.86e−1 | 1.97e−1 |
| 4 | occupation $\rho$ | 4.06 | **6.16e−3** | 1.50e−2 |

$\rho$-weighting produces a **worse** sup-norm fit and a **1–3 orders of magnitude
better** certified cost, at identical rule count. That inversion is the point:
uniform accuracy is not the control objective, and optimizing for it is actively
counterproductive. Note also that the 2-rule uniform fit is not even admissible,
so its "gap" is not a certificate.

### 7e. What ships with a fitted controller

| $N$ | weight | ROA radius | certified sublevel $V^*$ | worst $\dot V$ | mean gap |
|---|---|---|---|---|---|
| 2 | uniform $L^2$ | **0.0** | 0.0 | — | not admissible |
| 2 | occupation $\rho$ | 3.0 | 49.5 | −3.0e−7 | 1.39e−1 |
| 3 | occupation $\rho$ | 3.0 | 49.5 | −3.8e−7 | 1.69e−2 |
| 4 | occupation $\rho$ | 3.0 | 49.5 | −3.5e−7 | 6.16e−3 |

The region of attraction is an inner estimate obtained by testing $\dot V^*<0$
along the *fuzzy* closed loop, walking outwards from the origin and stopping at
the first failure — conservative by construction, never optimistic. The 2-rule
uniform controller correctly reports radius 0: it does not hold the origin, so
there is nothing to certify.

So each controller ships with **a region it provably stabilizes** and **exactly
how much more than optimal it costs**. That is the strongest sense of "provably
optimal fuzzy control parameters" available without exact representation.

### 7f. What $\lambda$ costs here

Theorem C2 needs $L^2(\rho)$ and nothing else, so $\lambda>0$ is a deliberate
*deviation* from the certified-optimal objective:

| $\lambda$ | $\sup|u-u^*|$ | $\sup|u'-u^{*\prime}|$ | mean certified gap |
|---|---|---|---|
| 0 | 6.55 | 15.10 | **1.69e−2** |
| 0.01 | 2.19 | 10.21 | 8.18e−2 |
| 0.1 | 1.77 | 8.15 | 1.76e−1 |
| 1.0 | 1.36 | 7.51 | 2.47e−1 |

$\lambda$ buys feedback-*gain* accuracy — $\partial u/\partial x$ sets the closed-loop
linearization and hence local pole placement — and pays for it in certified cost,
roughly $15\times$ over this range. **For pure cost-optimality use $\lambda=0$;
raise it only when the gain profile itself matters** (robustness, actuator rate
limits, gain scheduling). This is a cleaner story than the regression case, where
$\lambda$ had no comparably exact accounting.

### 7g. Honest limits

- The benchmark is scalar and inverse-optimal by construction. That is what makes
  ground truth exact, but it also means $u^*$ was *chosen* to be representable-ish.
  A plant with a genuinely awkward $u^*$ would tell more.
- $\rho$ depends on the controller, so §7d used $\rho$ from the *optimal* closed
  loop. In practice this must be iterated (fit → simulate → re-weight); the fixed
  point is what policy iteration converges to, and that is not done here.
- The ROA uses $V^*$ as the Lyapunov candidate, which is available only because
  the benchmark is inverse-optimal. For a real plant one must either solve for a
  candidate (SOS programming) or accept a smaller certified region.
- No input constraints. Saturation is where fuzzy control usually earns its keep,
  and $u^*$ under saturation is not smooth, so §1's C3 would need revisiting.

---

## 8. Two-cart benchmark: how fast does it converge, with no refinement?

Code: `fis_twocart.py`. Experiments: `demo_twocart.py`.

Plant: the classic two-mass-spring non-collocated benchmark. Force on cart 1,
cart 2 is the one that must be stilled, $m_1=m_2=k=1$, $|u|\le1$. Open-loop poles
sit on the imaginary axis ($\pm1.4142j$ plus a double integrator), so nothing
decays on its own and every number below is entirely the controller's doing.

Objectives: **settling time** (2% of $\|z_0\|$, taken as the *last* exit from the
tolerance ball, not the first entry), **peak force**, **energy** $\int u^2dt$.
Scored equal-weight, normalized against the best LQR in a sweep.

### 8a. The objectives are what make this a fair contest

Theorem C1 (§7a) applies to this plant: it is LTI, so with a *quadratic* cost a
partition-of-unity TSK reproduces $-Kx$ exactly and the membership functions
cancel. A fuzzy controller cannot beat LQR at LQR's own objective, and any
result claiming otherwise would be an artifact.

What breaks the degeneracy is the **cost, not the plant**: settling time and peak
force are not quadratic, and $|u|\le u_{\max}$ makes the true optimum non-smooth.
That is the only reason there is anything to measure.

### 8b. Convergence — the answer

Recipe applied *once*, with no tuning: weighted $k$-means on the open-loop optimal
trajectories → variable-project the consequents (one linear solve, globally
optimal for those rule positions, §3a) → simulate. Nothing else.

| rules | params | settle | peak $|u|$ | energy | score | shift | settled |
|---|---|---|---|---|---|---|---|
| LQR | 4 | 19.797 | 1.0000 | 0.5375 | 1.0000 | — | yes |
| open-loop ref | — | 10.972 | 0.8615 | 0.5585 | 0.8183 | — | yes |
| 1 | 5 | 26.773 | 0.9575 | 0.4214 | 1.0313 | 3.05 | yes |
| **2** | **10** | 17.830 | 0.9088 | 0.4248 | **0.8666** | 3.07 | yes |
| **3** | **15** | 17.217 | 0.9023 | 0.4390 | **0.8629** | 3.05 | yes |
| 4 | 20 | 18.470 | 0.9088 | 0.4465 | 0.8908 | 3.07 | yes |
| 6 | 30 | 18.257 | 0.8560 | 0.4798 | 0.8903 | 3.07 | yes |
| 8 | 40 | 19.040 | 0.5531 | 0.6121 | ∞ | 3.07 | **no** |
| 12 | 60 | 14.313 | 1.0000 | 1.2041 | 1.3211 | 4.67 | yes |
| 16 | 80 | 34.990 | 1.0000 | 7.8944 | ∞ | 6.73 | **no** |

**It converges in two rules.** Ten parameters take the score from 1.031 (worse
than LQR) to 0.867; the third rule adds 0.4% and that is the end of it. Two rules
close **75%** of the gap between the best LQR and the open-loop reference
$(1.000-0.863)/(1.000-0.818)$.

Where the win comes from is specific: at $N=3$ the fuzzy controller uses **18%
less energy than the best LQR and 21% less than the open-loop reference it was
trained on**, and 10% less peak force — while being *worse* on settling time
(17.2 vs 11.0). It trades response speed for actuator economy, which is a
reasonable reading of an equal-weight score but should not be mistaken for
uniform dominance.

### 8c. …and then it gets worse, for a diagnosable reason

Past $N\approx3$ the score degrades monotonically and by $N=8$ the closed loop
stops settling at all. This is **not** a numerical artifact and not a failure of
the variable projection (which is a globally optimal linear solve at every rule
count). It is **distribution shift**.

The `shift` column is the maximum distance from a closed-loop state to the
nearest training sample, in units of the training set's own per-axis spread. It
tracks the failure exactly: 3.05 while the method works, 4.67 at $N=12$, 6.73 at
$N=16$. The controller is fitted **only on optimal trajectories**, so it has no
data off them. More rules mean each rule is supported by fewer samples and
extrapolates harder, so the closed loop leaves the training manifold and the
consequents are evaluated where nothing constrained them.

This is the standard imitation-learning failure, and it means:

> **Rule count and closed-loop quality are not monotonically related.** The
> least-action fit is globally optimal *for the data it was given*, and the data
> is the binding constraint — not the model class, not the optimizer.

The fix is one round of aggregation (simulate, collect the off-trajectory states,
refit — DAgger), or off-trajectory sampling around the optimal tube. Both are
*refinement*, which the question explicitly excluded, so they are not done here.

### 8d. The nonlinear case — where the framework predicted a win, and didn't get one

§7a argued that fuzzy structure can only earn its keep through nonlinearity,
since Theorem C1 makes it vacuous on LTI plants. Adding a hardening cubic spring
($k_{nl}=2$) voids the exact-representation argument and should therefore be
where the method shines. It is not.

| controller | settle | peak $|u|$ | energy | score |
|---|---|---|---|---|
| best linear LQR | 14.563 | 1.0000 | 1.1887 | 1.0000 |
| open-loop reference | 9.167 | 0.7998 | 0.4225 | 0.5949 |
| fuzzy $N=1$ | 49.600 | 0.5386 | 0.1935 | ∞ (doesn't settle) |
| fuzzy $N=2$ | 33.960 | 0.8140 | 0.3090 | 1.1353 |
| fuzzy $N=4$ | 31.216 | 0.8976 | 0.4747 | ∞ |
| fuzzy $N=8$, $N=16$ | — | — | — | ∞ |

**No rule count beats the linear LQR**, and everything past $N=4$ fails outright.
Meanwhile the open-loop reference scores 0.595, so roughly 40% of the objective
is genuinely available — the room exists and the method cannot reach it.

This is a negative result for the hypothesis, and it should be read as one. The
nonlinearity that removes the C1 degeneracy also amplifies the §8c failure mode:
a hardening spring makes off-trajectory states produce disproportionately large
restoring forces, so extrapolation error is punished far harder than in the
linear case. The binding constraint is the same one — training data confined to
optimal trajectories — but nonlinearity raises the price of leaving them.

So the honest summary is narrower than §7a's framing suggested: nonlinearity is a
*necessary* condition for fuzzy structure to matter, not a sufficient one. On
this benchmark, going nonlinear made the problem harder for the method faster
than it created advantage for it.

### 8e. Does off-trajectory data recover it?

§8c blamed the post-$N{=}3$ breakdown on distribution shift. Testing the fix
splits that claim in two, and only half of it survives.
Code: `demo_augment.py`. Expert labels come from a short-horizon re-solve
(~1.7 s each), so both schemes are affordable at a few hundred labels.

**V — tube augmentation** (perturb the optimal trajectories by 0.35 of the
per-axis spread, re-label, add; controller-independent, computed once):

| rules | on-traj score | shift | +tube score | shift |
|---|---|---|---|---|
| 3 | **0.8629** | 3.05 | 0.9049 | 3.04 |
| 4 | 0.8908 | 3.07 | 0.8887 | 3.01 |
| 8 | **∞** | 3.07 | **0.8879** | 2.98 |
| 12 | 1.3211 | 4.67 | **0.9259** | 3.06 |
| 16 | **∞** | 6.73 | 1.6630 | 4.48 |

**W — DAgger** (fit → roll out → label the states the controller *actually*
visits → refit; the fixed-point iteration that makes the training measure equal
the deployment measure, as §7d requires):

| rules | round 0 | round 1 | round 2 |
|---|---|---|---|
| 3 | **0.8629** | 0.9355 | 0.9357 |
| 8 | ∞ | ∞ | **0.9407** |
| 16 | ∞ | 1.2228 | ∞ |

Three findings, and the second is the one that matters:

1. **Stability recovers, decisively.** $N=8$ goes from unusable to 0.888 (tube)
   or 0.941 (DAgger); $N=12$ goes 1.321 → 0.926. The shift metric moves with it
   — 4.67 → 3.06 at $N=12$, 6.73 → 4.48 at $N=16$ — confirming §8c's diagnosis of
   *the collapse*.
2. **The plateau does not move.** The best score anywhere in either experiment is
   0.8879 (tube, $N=8$, 40 parameters), still worse than **0.8629 from 3 rules /
   15 parameters on trajectory data alone**. No amount of off-trajectory data,
   under either scheme, beats what two or three rules already achieved.
3. **Augmentation actively hurts the controller that was already working.**
   $N=3$ degrades 0.863 → 0.905 under tube and → 0.936 under DAgger, and DAgger
   converges to the worse value rather than wandering there.

Finding 3 has a clean explanation, and it is the framework's own: Theorem C2 says
weight by the deployed closed loop's occupation measure. Tube and DAgger samples
add mass where the *working* controller spends little time, so they buy support
at the cost of fidelity where it actually matters. **Robustness and
occupation-weighted accuracy are in direct tension here**, and for a controller
that is already stable the trade is a losing one.

So §8c's conclusion needs splitting:

> Distribution shift explains the **collapse at high rule count** — that part is
> confirmed, and fixable with off-trajectory data. It does **not** explain the
> **0.863 plateau**, which survives every augmentation tried. Two different
> limits were conflated; only one of them is a data problem.

What the plateau actually is is not left open: §8f rules out the model class and
§8g names the imitation objective, which §8h then confirms by removing it.

One caveat on W: this is the $\beta=0$ variant of DAgger, rolling out the learner
only. The original algorithm mixes in the expert on a decaying schedule, and its
absence is the likely reason $N=16$ oscillates (∞ → 1.22 → ∞) instead of
converging.

### 8f. Does a richer consequent class move the plateau?

§8e left two candidates for the 0.8629 plateau: the model class and the training
targets. This isolates the first. Code: `demo_order.py`.

Raising the consequent polynomial order enlarges the model class while keeping
the consequents **linear in the parameters**, so variable projection still
returns the global optimum per rule placement (§3a). Nothing else changes — same
rule placement, same occupation weighting, same single linear solve. Basis sizes
for four states are 1, 5, 15, 35 at orders 0–3.

**Y — score vs rule count, by order:**

| order | basis | N=1 | N=2 | N=3 | N=4 | N=6 | N=8 | N=12 |
|---|---|---|---|---|---|---|---|---|
| 0 (constants) | 1 | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ |
| 1 (affine) | 5 | 1.0313 | 0.8666 | **0.8629** | 0.8908 | 0.8903 | ∞ | 1.3211 |
| 2 (quadratic) | 15 | 0.9756 | **0.8487** | 1.3395 | ∞ | ∞ | ∞ | ∞ |
| 3 (cubic) | 35 | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ |

Order 0 never works: constant consequents cannot produce state-proportional
feedback, and an undamped oscillator needs it. Order 3 never works either, for
the opposite reason — 35 parameters per rule against 186 samples.

**Z — the fair comparison, at matched parameter budget:**

| params | order | rules | score |
|---|---|---|---|
| 15 | 1 | 3 | **0.8629** |
| 15 | 2 | 1 | 0.9756 |
| 30 | 1 | 6 | 0.8903 |
| 30 | 2 | 2 | **0.8487** |
| 45 | 2 | 3 | 1.3395 |
| 60 | 1 | 12 | 1.3211 |

**The plateau moves, and barely: 0.8629 → 0.8487, a 1.6% improvement, bought
with double the parameters.** At a matched 15-parameter budget affine wins
decisively (0.8629 vs 0.9756). And richer consequents destabilize *sooner* —
order 1 survives to $N=6$, order 2 only to $N=2$, order 3 never.

**Z′ — and here is why.** Weighted RMS fit to $u^*$ on the training set, against
closed-loop score:

| order | rules | params | RMS fit | score |
|---|---|---|---|---|
| 0 | 3 | 3 | 0.109721 | ∞ |
| 0 | 8 | 8 | 0.084124 | ∞ |
| **1** | **3** | **15** | **0.025332** | **0.8629** |
| 1 | 8 | 40 | 0.016583 | ∞ |
| 2 | 3 | 45 | 0.015875 | 1.3395 |
| 2 | 8 | 120 | 0.003684 | ∞ |
| 3 | 3 | 105 | 0.004774 | ∞ |
| 3 | 8 | 280 | 0.003339 | ∞ |

Fit quality improves **monotonically and by 33×** (0.1097 → 0.0033). Closed-loop
score gets monotonically *worse*. The best-scoring configuration has nearly the
**worst** fit of any that works, and **every configuration that fits better than
0.0253 fails to settle at all**.

> **The relationship between open-loop fit accuracy and closed-loop performance
> is inverse.** The model class was never the binding constraint — fitting $u^*$
> more accurately makes the controller worse.

### 8g. What the plateau actually is

Three experiments now bracket it:

| candidate | test | result |
|---|---|---|
| distribution shift / data | §8e, tube + DAgger | fixes the collapse, plateau unmoved |
| model class | §8f, orders 0–3 | 1.6% at 2× params, inverse fit/score relation |
| training targets | — | **not excluded** |

By elimination, the binding constraint is the **imitation objective itself**. The
targets are an open-loop optimum computed with full knowledge of $z_0$, and a
more faithful reproduction of an open-loop control *law* is not a better
*feedback* law — it is a better fit to something that was never a feedback law to
begin with. Extra capacity spends itself reproducing trajectory-specific detail
that does not transfer, which is exactly the inverse relation in Z′.

That also explains why §8b's two-rule result is so good: with 10 parameters the
model is too coarse to imitate the open-loop law faithfully, and is forced into
something closer to a genuine feedback policy.

The implied fix is a different algorithm, not a bigger model or more data:
**optimize the closed-loop objective directly over the FIS parameters** rather
than fitting sampled targets. That abandons the variable-projection guarantee
(the objective stops being quadratic in the consequents), which is precisely the
trade §7b's certificate was designed to avoid — and is why it was worth
establishing that the cheap route genuinely caps out first. §8h does it, and the
diagnosis holds: 22.9% at ten parameters.

### 8h. Direct policy optimization — the plateau breaks

Code: `demo_policyopt.py`. §8g concluded by elimination that the binding
constraint was the imitation objective. This tests the implied fix: stop fitting
sampled targets, and optimize the closed-loop objective directly over the FIS
parameters.

**The price is paid up front.** The objective is not quadratic in the
consequents, so variable projection no longer applies — no globally optimal
linear solve, only derivative-free search (Powell) over a non-convex landscape.
Every guarantee from §3a and §7b is surrendered. The true score is also unusable
as a search objective: settling time is discontinuous, peak force is a max, and
failures score ∞, which tells an optimizer nothing. `shaped_cost` replaces each
term with an always-finite analogue on the same normalization — soft time outside
the tolerance ball, log-sum-exp peak, unchanged energy, plus a terminal-norm term
that makes a non-settling controller *improvable* rather than merely rejected.

**AA — warm-started from the imitation fit, 600 closed-loop evaluations each:**

| rules | params | imitation | direct | improvement |
|---|---|---|---|---|
| **2** | **10** | 0.8666 | **0.6547** | **24.5%** |
| 3 | 15 | 0.8629 | 0.6729 | 22.0% |
| 4 | 20 | 0.8908 | 0.7565 | 15.1% |

**AD — where the gain comes from** (best configuration, 2 rules / 10 parameters):

| | settle | peak $|u|$ | energy | score |
|---|---|---|---|---|
| best LQR | 19.797 | 1.0000 | 0.5375 | 1.0000 |
| imitation | 17.830 | 0.9088 | 0.4248 | 0.8666 |
| **direct opt** | **13.563** | **0.6481** | **0.3391** | **0.6547** |

Unlike every earlier result, this improves **all three objectives at once** —
31% faster settling, 35% less peak force, 37% less energy than the best LQR. The
imitation controllers had to trade settling time for actuator economy (§8b);
direct optimization does not.

- **34.5% better than the best LQR**
- **22.9% better than the best imitation result** (0.8487, which needed 30 parameters)
- **20.0% better than the open-loop reference** (0.8183)

Beating the open-loop reference is not a contradiction even though that
reference had full knowledge of $z_0$: it is open-loop, so it cannot correct
along the way, and it was only a surrogate local optimum (§8b).

**AB — but the warm start is load-bearing.** The same 3-rule structure started
from zero consequents instead of the imitation fit:

> **cold start: ∞ → ∞** (601 evaluations, no stabilizing controller found)

Direct optimization could not find a stabilizing controller from scratch within
the same budget. So the two stages are complementary rather than competing:

> **The least-action fit is not the final method — it is the initializer that
> makes the final method reachable.** On its own it caps out at 0.863; without
> it, direct optimization does not start. Neither stage reaches 0.655 alone.

That rehabilitates §3a's variable projection in a specific role. Its value here
is not that it produces the best controller — §8f showed it cannot — but that it
produces a *stabilizing* one in a single globally optimal linear solve, with no
search and no initialization problem of its own.

**AC — tuning the antecedents too** (15 + 24 = 39 parameters) gave 0.6728 against
0.6547 for consequents only, at equal evaluation budget. With 2.6× the
parameters and the same 600 evaluations that is a statement about budget, not
capacity; it is not evidence that antecedent tuning cannot help.

### 8i. Held-out check — is 0.6547 real or memorized?

Every score in §8 up to this point is measured on the same six initial
conditions used for fitting. For the imitation fits that is a mild concern; for
direct policy optimization it is a real one, since 10 parameters tuned against 6
trajectories could simply be memorizing them. Code: `demo_holdout.py`. Six
held-out initial conditions, different directions and magnitudes, two of them
larger than anything in training. Each split is normalized against LQR *on that
split*, so the two columns are directly comparable.

| controller | train | test | test settled | degradation |
|---|---|---|---|---|
| best LQR | 1.0000 | 1.0000 | 6/6 | +0.0% |
| imitation $N{=}2$ | 0.8666 | 0.8632 | 6/6 | −0.4% |
| imitation $N{=}3$ | 0.8629 | 0.8621 | 6/6 | −0.1% |
| **direct opt $N{=}2$** | **0.6547** | **0.6905** | **6/6** | **+5.5%** |

The imitation fits generalize essentially perfectly — expected, since they have
few parameters and never see the objective. **Direct optimization does overfit,
by a measurable 5.5%**, which is the honest cost of tuning against a finite set
of initial conditions.

But the conclusion survives intact: the held-out score of 0.6905 still beats the
best imitation result (0.8487) by 19% and the best LQR by **31%**, on initial
conditions it never saw, including two outside the training envelope, settling
from all six. The headline in §8h should be read as **34.5% in-sample, 31%
out-of-sample**.

### 8j. The plateau, resolved

| candidate | test | verdict |
|---|---|---|
| distribution shift / data | §8e | fixes the collapse, plateau unmoved |
| model class | §8f | 1.6% at 2× params; fit/score inversely related |
| **imitation objective** | **§8h** | **confirmed — 22.9% once removed** |

The elimination in §8g was correct. Optimizing the objective you actually care
about, rather than fitting samples from something that optimized it once
open-loop, is worth more than every other lever tried combined — and it is worth
it at *ten parameters*, the smallest model in the entire study.

### 8k. Honest limits

- Six initial conditions, all modest. A wider $z_0$ set would stress
  extrapolation harder and probably lower the rule count at which shift bites.
- §8d is a single nonlinear plant at one stiffness. It refutes "nonlinearity is
  sufficient" but does not establish that fuzzy structure never helps on
  nonlinear plants — a milder $k_{nl}$, or a nonlinearity the affine consequents
  are better matched to, could go the other way.
- The open-loop reference is a local optimum of a **smooth surrogate** of the
  objective, not a certified optimum. The fuzzy controller beats it on the true
  score (0.863 vs 0.818 is against it, but $N{=}3$ beats it on energy and the
  $N{=}2$/$N{=}3$ rows beat several LQRs), which is direct evidence the surrogate
  is not the objective. Scores are therefore normalized against the best LQR,
  which can be computed exactly.
- Theorem C2's exact certificate covers the **energy** term only; settling time
  and peak force are outside its quadratic form, so §8's suboptimality
  certificate does not transfer to this three-objective score. That gap is real
  and is the main thing separating §9 from §8.
- §8i uses six held-out initial conditions from one plant. It bounds the
  overfitting at 5.5% for this configuration; it does not establish that the
  recipe generalizes across plants or across much wider state envelopes.
- §8h used a fixed 600-evaluation budget and one optimizer (Powell). No claim is
  made that 0.6547 is the best reachable; a larger budget, CMA-ES, or restarts
  would likely go further. The claim is only that the plateau is not a limit of
  the model or the data.
- Two numerical guards were needed and are worth flagging as reusable: a
  divergence event, and a hard RHS-evaluation budget. Without the second, a
  marginally-stable oscillatory closed loop drives the stiff solver to
  ever-smaller steps and the run neither fails nor returns — it simply hangs.

---

## 9. Next steps

Consolidated from the open threads in §6, §7g and §8k, ordered by value per unit
of work rather than by section. Rough costs assume this machine.

### 9a. Do first — cheap, and they close claims already made

**1. Certify the direct-optimized controller.** §8h surrendered every guarantee
in the framework to derivative-free search, and nothing was put back. But §7e's
machinery is controller-agnostic: a Lyapunov/ROA certificate can be computed
*post hoc* for the 0.6547 controller exactly as it was for the scalar ones. Right
now the best-performing controller in the study is the only one with no
stability certificate at all, which is backwards. (~1 hour; low risk, high value
— it either certifies or reveals the controller is quietly fragile.)

**2. Direct policy optimization on the nonlinear plant.** §8d is the study's
sharpest negative result: on the cubic-spring plant no imitation controller beat
a linear LQR, and everything past $N{=}4$ failed. §8h then showed imitation was
never the right objective. **The obvious question is whether §8h's fix rescues
§8d**, and it directly tests whether that negative result was about the *method*
or about the *plant*. If direct optimization wins there, §8d's conclusion —
"nonlinearity is necessary but not sufficient" — needs revising toward "the
imitation route cannot exploit nonlinearity, but the direct route can."
(~30 min; the single highest-information experiment left.)

**3. Widen the state envelope.** All of §8 uses six modest initial conditions
and §8i bounds overfitting at 5.5% against six more. A substantially wider
envelope would stress the distribution-shift finding (§8c) and probably lower
the rule count at which it bites. (~1 hour.)

### 9b. Theory gaps that are actually load-bearing

**4. Extend the Theorem C2 certificate beyond quadratic cost.** This is the
largest hole. C2 gives an *exact* suboptimality certificate — no constants, no
conservatism — but only for $\int u^\top Ru$. Settling time and peak force fall
outside it, so §8's entire three-objective study is measured rather than
certified, and that gap is what separates §8 from §7. An $L^\infty$ or
exit-time functional will not admit the same clean identity, but a one-sided
bound may be reachable, and would upgrade the whole control half of this work.
(Days; genuinely open, possibly not achievable in the clean form.)

**5. Analytic Hessian of the reduced action.** §3d made the gradient exact and
that removed a real failure mode. The second-order certificate still uses finite
differences, and its noise floor is what forces the tolerance-based verdict in
§3b. The same envelope argument differentiated once more should give it in
closed form. (~half a day; low risk, mostly algebra already half-done.)

**6. $\lambda^\*$ in the multi-input case.** §4d′ gives $\lambda^\*=3b^4/(8c^2)$ in
closed form for two scalar Gaussians, with $\ell^\*=\sqrt6\times$ the crossover
width. In $\mathbb{R}^n$ the crossover is a hypersurface and the width becomes
direction-dependent, so the scalar formula cannot carry over unchanged. Whether
a useful anisotropic version exists is unknown. (Open-ended.)

### 9c. Validation the work has not had

**7. The classifier side has never met real data.** §5 proves the max-equals-sum
identity, the order-0 TSK collapse, and the annealing bound, all on synthetic
partitions. The sonar dataset is already wired up in
`AEEM6097/fuzzy-symphony.py` and has been the intended target since the start.
Everything in §5 is untested against anything with real class overlap. (~half a
day; the most obvious unfinished business in the whole document.)

**8. Input constraints in §7.** §7g flags this and it remains true: saturation
is where fuzzy control usually earns its keep, and under saturation $u^*$ is not
smooth, so constraint C3 would need revisiting. §8 has saturation but no
certificate; §7 has certificates but no saturation. Closing that gap would join
the two halves.

**9. A second plant.** Every control result rests on two plants, one of them
inverse-optimal by construction. A cart-pole or a flexible link would test
whether the two-stage recipe (imitate to stabilize, then optimize directly) is a
pattern or a coincidence.

### 9d. Lower priority

**10. Better optimizer for §8h.** Powell at 600 evaluations is not a serious
budget. CMA-ES with restarts would establish how much of the remaining gap is
algorithmic. Worth doing only after item 1, since a better controller with no
certificate is not obviously progress.

**11. Annealing schedule for $\beta$** (§5c) tied to a trust region on the
action, giving a single continuation method rather than two separate knobs.

### 9e. What would change the story

Two results would force real revision:

- **If item 2 fails** — direct optimization also loses to linear LQR on the
  nonlinear plant — then the problem is not the imitation objective but the
  TSK structure itself on that plant, and §8g's elimination argument would need
  a fourth candidate.
- **If item 1 finds no certificate** — the 0.6547 controller has a small or
  empty region of attraction — then §8h's headline is a performance number
  bought with fragility, and the honest recommendation flips back toward the
  imitation fit, which is slower but was never in doubt.

---

## 10. Reproduction

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy scipy matplotlib
cd research/least_action
../../.venv/bin/python demo_regression.py    # ~12 min, sections A-I
../../.venv/bin/python demo_classifier.py    # ~15 s, sections H-K
../../.venv/bin/python demo_control.py       # ~3 min, sections L-P
../../.venv/bin/python demo_twocart.py       # ~25 min, sections Q-T
../../.venv/bin/python demo_augment.py       # ~20 min, sections V-X
../../.venv/bin/python demo_order.py         # ~10 min, sections Y-Z'
../../.venv/bin/python demo_policyopt.py     # ~50 min, sections AA-AD
../../.venv/bin/python demo_holdout.py       # ~12 min, generalization check

Total ~2 hours. `demo_order.py`, `demo_policyopt.py` and `demo_holdout.py` reuse
a cached training set (`.twocart_train.npz`, gitignored); delete it to rebuild.
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

`fis_control.py`:

| function | does |
|---|---|
| `LqrProblem` | Riccati solve for the LTI benchmark (Theorem C1) |
| `InverseOptimalScalar` | control-affine plant whose `q` is defined FROM the HJB, making `V*` and `u*` exact |
| `cubic_benchmark` | the nonlinear benchmark used throughout §8 |
| `simulate` | closed loop with cost and control-error carried as ODE states |
| `ClosedLoopResult.gap` | the exact suboptimality certificate |
| `ClosedLoopResult.identity_residual` | Theorem C2 check |
| `occupation_density` | closed-loop occupation measure — the correct fitting weight |
| `stability_certificate` | inner region-of-attraction estimate from `Vdot < 0` |

`fis_twocart.py`:

| function | does |
|---|---|
| `TwoCart` | two-mass-spring plant, optional cubic spring and damping, saturating input |
| `Metrics` / `simulate` | settling time, peak force, energy; divergence event and RHS-evaluation budget |
| `optimal_trajectory` | direct-transcription open-loop reference (smooth surrogate of the objective) |
| `TskController` | multi-input TSK, product-Gaussian antecedents, affine consequents |
| `place_rules` | occupation-weighted k-means rule placement (multi-input form of §7d) |
| `fit_consequents` | occupation-weighted variable projection — one globally optimal linear solve |
| `distribution_shift` | max closed-loop distance to the nearest training sample |
| `label_state` | expert label from a short-horizon re-solve (~1.7 s) |
| `augment_tube` | perturb the optimal trajectories and re-label |
| `dagger_states` | states the current controller actually visits, occupation-weighted |
| `poly_basis` / `basis_size` | consequent monomials up to a given order (1, 5, 15, 35 for orders 0-3) |
| `shaped_cost` | always-finite smooth surrogate of the three-objective score |
| `policy_optimize` | derivative-free search on the FIS parameters against the closed loop |

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
| TSK represents LQR exactly | 8a | control L |
| Theorem C2 + admissibility | 8b, 8c | control M |
| occupation-weighted fitting | 8d | control N |
| stability / ROA certificate | 8e | control O |
| price of $\lambda$ in control | 8f | control P |
| objectives break C1 degeneracy | 9a | twocart Q |
| LQR sweep + open-loop reference | 9b | twocart R |
| convergence in 2 rules | 9b | twocart S |
| distribution-shift breakdown | 9c | twocart S |
| nonlinear case fails | 9d | twocart T |
| augmentation recovers stability | 9e | augment V, W |
| plateau survives augmentation | 9e | augment V, W |
| higher order barely moves it | 9f | order Y, Z |
| fit accuracy vs score is inverse | 9f | order Z' |
| direct policy opt breaks plateau | 9h | policyopt AA, AD |
| warm start is load-bearing | 9h | policyopt AB |
| held-out generalization | 9i | holdout |
