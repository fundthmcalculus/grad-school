# Certified Fuzzy Control by Least Action — what is proven, and why it matters

**Scott Phillips, Kelly Cohen** · University of Cincinnati, AEEM

---

## The problem

Fuzzy controllers are easy to build and hard to trust. A TSK controller can be tuned to excellent closed-loop performance, but the tuning produces *numbers*, not *statements*: nothing in a typical design says "this controller is within X of optimal" or "this controller is stable on this set, and here is the proof." That gap is why fuzzy control is under-used in the places where it would help most — systems where a designer wants nonlinear, interpretable feedback but has to defend it.

This work asks what happens if the fitting problem is posed variationally, and if every step is required to produce a checkable statement rather than a converged solver.

## What we prove

**1. The hard half of fuzzy fitting is free.** Posing controller fitting as minimization of an $H^1$ action, the consequent parameters have a *closed-form global optimum* — one linear solve, no iteration, no local minima. All non-convexity in the problem is confined to rule placement. The reduced problem over rule placement then has an *exact* analytic gradient via the envelope theorem, which matters practically: with finite differences the optimizer terminated after zero iterations and reported success.

**2. When sequential rule identification is legitimate.** Greedy fitting (fit rule 1, fit rule 2 to the residual, …) equals the joint fit for every target **iff** the $H^1$ Gram matrix is block-diagonal. For two rules we give the slope weight achieving this in closed form, $\lambda^{*}=3b^{4}/(8c^{2})$, and show it is a property of the *rule geometry*, not of the data. We also prove a factor-of-four obstruction: no single $\lambda$ can decouple three or more uniformly spaced rules. This turns a common heuristic into a condition with a known validity boundary.

**3. An exact suboptimality certificate, for any convex control cost.** For a control-affine plant, the closed-loop performance gap is *exactly* an integral of a Bregman divergence:
$$J(x_0)-V^{*}(x_0)=\int_0^{\infty} D_c\big(u\,\|\,u^{*}\big)\,dt.$$
No inequalities, no unknown constants. Non-negativity of the gap *is* convexity of the cost. The familiar quadratic identity is one special case; this holds for costs that are not even polynomial. It also tells us precisely what is **out of reach**: peak force and settling time are a supremum and an exit time, not integral functionals, so no argument of this form can certify them.

**4. Machine-checked stability, with no floating-point step in the chain.** Using $\pi$-shaped (rational) membership functions, the closed loop becomes a rational vector field and the Lyapunov condition becomes a polynomial inequality a sum-of-squares program can decide. We then take the extra step most SOS work does not: the floating-point certificate is rounded to rationals, projected onto the exact polynomial identity, and verified positive-definite by exact integer arithmetic. The result — the Lyapunov function strictly decreases on a specified sublevel set — is **proved, not measured**.

## The result that surprised us

We put the certified region *into* the design objective, so performance and provability compete explicitly. The expectation was a strict trade-off. Instead:

> **A Lyapunov certificate acts as a regularizer.** A small certificate penalty produced a controller that was better on the training objective, better on held-out initial conditions, *and* had a larger certified region than the unpenalized optimum.

The explanation is that unpenalized optimization buys performance through aggression, aggression is what a certificate charges for, and it is also what fails to generalize. Requiring provability suppressed the overfitting.

## Turning the certificate into a specification

A certified radius is only useful as a design input if you can *ask* for one. The SOS program is too slow to sit inside a search (47 s per controller against a 601-evaluation optimization), so we optimize against a sampled *necessary* condition for it — safe because any state where the Lyapunov function fails to decrease **proves** the certificate must fail there.

The first estimator was badly biased. Diagnosing why was the useful part: it was not the SOS relaxation being conservative (the usual suspect, and the expensive one to fix), it was the *estimator* — a minimum over random samples, with a 12.5% seed-to-seed spread. The obvious repair is worse: it is structurally degenerate, because the origin always satisfies the constraint and any descent method walks to it. Reorganizing the search by direction fixed both, cutting the bias from 18% to 4.4% and the spread to 1.5%. What remains is a single constant, $\kappa=0.962$ — and 2.5 of those 3.8 points are a deliberate safety margin, so genuine relaxation conservatism is about **1.3%**.

The payoff: **state a certified radius, get a controller the solver certifies at or above it.** The best controller found this way carries a certified region of 0.679 at a performance score of 0.847 — dominating the imitation-fit baseline on *both* axes.

## Why it is valuable

| For | Value |
|---|---|
| **Fuzzy control practice** | Turns "tune until it looks good" into "specify a certified region and a performance budget, and get a controller that provably meets both." |
| **Certification and airworthiness** | The stability argument is a finite exact-integer computation, auditable end to end. Not a solver's word. |
| **Optimal control theory** | The Bregman identity widens the exact-certificate envelope from quadratic to *all convex* control costs, and cleanly delimits what such certificates can never cover. |
| **Learning-based control** | An empirical finding with reach beyond fuzzy systems: requiring a stability certificate improved out-of-sample performance. Certificates may be worth adding to learned-controller objectives for reasons *other* than safety. |

## What we do not claim

Rule placement has no global optimality guarantee — the landscape is multimodal and we bound nothing. Certification does not scale: the SDP grows from $35\times35$ at two rules to $715\times715$ at eight, so the certified story is confined to low rule count (which is, fortunately, where the method converges anyway). The certified radius is a maximum over two fixed Lyapunov candidates, which produces a bimodal achievable set — asking for 0.45 delivers 0.678, so a specification is *met, not matched*. And results so far are on a single plant at nominal parameters. Each of these has an identified next step.
