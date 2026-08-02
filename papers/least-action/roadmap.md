# Roadmap: next steps and comparison benchmarks

Companion to `paper.md`. Part 1 is a prioritized work plan; Part 2 identifies
benchmark control problems; Part 3 identifies the methods we should be measured
against.

> **Citation caveat.** Author/topic attributions below are from memory and are
> *pointers, not citations*. Every one needs verification against the actual
> paper before it appears in a submission.

---

## Part 1 — Next steps, in priority order

Priority is by **how much certified ground each buys**, not by how interesting
it is. Effort estimates assume the existing code base.

### Tier A — removes a stated limitation

**A1. Search $V$ jointly with $\theta$ (bilinear / alternating SOS).**
*The single highest-value item.* Every certified radius we report is a maximum
over two fixed quadratic Lyapunov candidates (Riccati, closed-loop
linearization). That one choice causes **both** unresolved defects in §9:

- the calibration outlier (the proxy's max is attained at a candidate the
  degree-4 S-procedure cannot certify *at all* — `SOS Ric = 0.0000`), and
- the bimodal achievable set (every result lands near 0.35 or 0.68 according to
  which candidate wins), which is why asking for 0.45 delivers 0.678.

Standard approach: alternate between (i) fixing $\theta$ and searching $(V,\sigma)$
by SOS, and (ii) fixing $V$ and improving $\theta$. Each half is convex; the
alternation is not, so it needs a monotone acceptance test and honest reporting
that it finds a local solution. **Effort ~1 week. Risk: moderate** (bilinear
alternation can stall). **Payoff: makes the specification of §9.4 *matched*
rather than merely *met*.**

**A2. Restate the benchmark objectives as integral costs.**
§10 (G14) shows peak force and settling time are structurally outside the
Theorem 8 envelope — a supremum and an exit time are not integral functionals.
Replacing them with $\int\phi(u)\,dt$ ($\phi$ a convex barrier approximating the
saturation limit) and $\int w(t)\|x\|^{2}dt$ (a time-weighted state cost) puts
**the entire benchmark inside the certified envelope**: every controller would
then carry an exact suboptimality certificate rather than a measured score.
This changes the problem, which must be said plainly — but it changes it in the
direction the framework can certify. **Effort ~1–2 days. Risk: low.**

**A3. Relative form of Theorem 8 for unknown $V^{*}$.**
The certificate presupposes $V^{*}$, which is exactly what a forward problem
lacks; today it is usable only on inverse-optimal benchmarks, i.e. problems
whose answer was constructed in advance. For any $\hat V$ satisfying HJB with
residual $\varepsilon$, the same algebra should bound the true gap in terms of
$\varepsilon$ and $\|\hat V-V^{*}\|$. The derivation looks routine; the useful
part is determining **in which norm $\varepsilon$ must be measured** for the
bound to be tight. **Effort ~1–2 days. Risk: moderate — if no useful bound
exists, that is itself a result worth stating, and it bounds the practical reach
of Theorem 8.**

**A4. Encode input saturation inside the certificate.**
We currently report an "unsaturated radius" separately, which is an honest
dodge: the certificate is about the unsaturated closed loop and we note where
saturation begins. Saturation is piecewise, so the clean route is either a
sector/deadzone bound handled by an extra S-procedure multiplier, or certifying
the saturated region separately and intersecting. **Effort ~3 days. Risk:
moderate.** Without this the certified region can never exceed the saturation
boundary, which is currently the binding limit on some rows.

### Tier B — scaling and generalization

**B1. Push past the rule-count wall.**
$\deg Q=2(N-1)$ drives the Gram from $35\times35$ at $N=2$ to $715\times715$ at
$N=8$. Three routes, cheapest first:
- **Newton-polytope / sparsity reduction** on the monomial basis — often removes
  a large fraction of the basis for free.
- **DSOS/SDSOS** (diagonally dominant / scaled diagonally dominant relaxations,
  Ahmadi & Majumdar): replaces the SDP by an LP or SOCP. Weaker, so the certified
  region shrinks — but it *scales*, and we can quantify the loss exactly by
  comparing on the $N=2$ case we can already solve both ways.
- **Exploit the partition-of-unity structure** — $\sum_i\varphi_i=1$ is an
  algebraic identity the solver is not currently told about.

**Effort ~1 week. Risk: low** (worst case we measure and report the wall).

**B2. A second and third plant.** See Part 2. Currently every control claim
rests on one plant at nominal parameters, which is the most obvious objection a
reviewer will raise.

**B3. Robustness of the certified region under parameter uncertainty.**
The two-mass–spring system is conventionally posed *with* a spring-constant
uncertainty range, which we do not exercise. Certifying a region valid for all
$k$ in an interval is a natural and well-posed extension: add $k$ as an
indeterminate with its own S-procedure multiplier. **This is the step that makes
the work interesting to a robust-control audience.**

### Tier C — completes the analytic story

**C1. Analytic Hessian of the reduced action.** Upgrades G11 from "certified
numerically" to "certified analytically"; the finite-difference noise floor is
what currently forces a tolerance-based verdict. Same envelope argument
differentiated once more. **Effort ~half a day. Risk: low.**

**C2. $\lambda^{*}$ in the multi-input case.** The crossover becomes a
hypersurface and the width becomes direction-dependent; there may be no clean
closed form. Worth a bounded attempt, and a negative result is publishable as
a limitation of Theorem 5.

**C3. Lean/Mathlib formalization of Theorem 8.** Blocked in our current
environment by network policy, not by difficulty. Its hypotheses (admissibility,
convexity, differentiability of $V^{*}$) are exactly the kind that get quietly
dropped — we dropped one empirically. Mathlib has enough (`MeasureTheory`,
`intervalIntegral`, `deriv`, convexity) to state and prove it.

### What would change the story

- **If A1 fails** — joint $(V,\theta)$ search does not converge usefully — then
  the certified radius stays a max over hand-picked candidates, the achievable
  set stays bimodal, and the honest framing is "specify a radius, receive a
  controller certified at or above it, with the overshoot uncontrolled."
- **If B1 fails** — no relaxation scales past 3–4 rules — then the method is
  permanently confined to low rule count. That is survivable (§6 converges at 2–3
  rules) but it must be stated as a property of the approach, not a temporary
  limitation.
- **If A3 fails** — no useful bound in terms of the HJB residual — then
  Theorem 8 is a strong theoretical result with narrow practical reach, and we
  should say so rather than imply generality we cannot deliver.

---

## Part 2 — Benchmark control problems

Selection criteria, in order: (i) the plant must admit a *rational* closed loop
so the SOS machinery applies without approximation, or admit an exact rational
reformulation; (ii) it should be recognized, so results are comparable; (iii)
the set should span the failure modes we claim to handle — unstable open loop,
non-polynomial nonlinearity, higher dimension, and parameter uncertainty.

### Tier 1 — do these next

| # | Plant | States | Why this one |
|---|---|---|---|
| **P1** | **Two-mass–spring, with uncertainty** | 4 | Already implemented at nominal $k$. Its standard form carries a spring-constant uncertainty range; certifying across that interval is the natural extension (B3) and needs no new plant code. |
| **P2** | **Van der Pol, reverse time** | 2 | *The* calibration target. Its region of attraction is known to high accuracy in the literature, so it converts our §9 conservatism estimate from a self-consistency check into a comparison against **ground truth**. Polynomial, 2 states, cheap — every SOS experiment should be run here first. |
| **P3** | **Inverted pendulum on a cart** | 4 | The canonical fuzzy-control benchmark; a reviewer will expect it. Open-loop unstable, so it tests a regime the two-mass–spring does not. Trig nonlinearity handled exactly by the Weierstrass/half-angle substitution $s=\sin\vartheta$, $c=\cos\vartheta$ with the algebraic constraint $s^2+c^2=1$ as an S-procedure equality — no Taylor approximation needed. |

### Tier 2 — breadth

| # | Plant | States | Why this one |
|---|---|---|---|
| **P4** | **TORA / RTAC** (translational oscillator, rotational actuator) | 4 | A standard nonlinear benchmark with a substantial ROA literature, so certified-region numbers are directly comparable to published ones. Same trig treatment as P3. |
| **P5** | **Magnetic levitation** | 2–3 | Nonlinearity is *natively rational* ($\propto i^2/x^2$), which is an unusually good structural match for $\pi$-shaped membership functions and the exact-rational verification chain. Open-loop unstable, hard input constraint (current $\ge0$) — a genuine test of A4. |
| **P6** | **Duffing / cubic-spring two-mass** | 4 | Already implemented (`k_nl`). Keeps the plant familiar while making it genuinely nonlinear, so it isolates *nonlinearity* from *dimension* and *instability*. |
| **P7** | **Ball and beam** | 4 | Classic fuzzy benchmark; relative-degree structure makes naive designs fail, so it discriminates. |

### Tier 3 — stress the scaling claim

| # | Plant | States | Why this one |
|---|---|---|---|
| **P8** | **Planar quadrotor (PVTOL)** | 6 | Deliberately chosen to break things. With 6 states the monomial basis explodes and this is where B1 either works or the wall is confirmed. A negative result here is worth publishing as a scaling law. |

**Recommended immediate set: P2 for calibration, P1 for uncertainty, P3 for
recognizability.** P2 first — it is two states and will run in seconds, and it
is the only one that tells us whether our 1.3% conservatism estimate is real.

---

## Part 3 — Methods to compare against

Performance alone is not the comparison that matters; the claim is
*performance with a certificate*. So the baselines split.

### Certified baselines (the ones that matter)

1. **PDC with LMI design on a T–S model** (Tanaka & Wang, and the large
   literature following). This is *the* standard certified fuzzy control method
   and the most important comparison in the paper. It gives a quadratic-Lyapunov
   guarantee by construction, where we *fit first and certify after*. The honest
   framing of our contribution against it: PDC guarantees stability by
   construction but ties the controller to the local models; we start from a
   least-action fit or a directly optimized policy and then certify, which
   admits controllers PDC cannot express — at the cost of the certificate not
   being automatic.
2. **Polynomial fuzzy systems with SOS** (Tanaka, Yoshida, Wang and successors).
   The closest adjacent work: also SOS, also fuzzy, also beyond LMI. Key
   differences to establish carefully — they *synthesize* via SOS, we certify a
   fitted/optimized controller and, in §9, make the certified radius a design
   *specification*. We should be explicit that the certificate-as-objective idea,
   not the certificate itself, is what is new.
3. **Direct SOS/Lyapunov controller synthesis** (non-fuzzy). Establishes what
   the certified region would be if one designed for it directly, which upper-
   bounds what any certify-after-the-fact method can claim.

### Performance baselines

4. **LQR** — already used, and the normalizer for our score.
5. **Open-loop optimal (direct transcription)** — already used; a *local*
   performance reference with full knowledge of $x_0$.
6. **MPC** — the practical performance ceiling. Expected to beat us on score; the
   argument is cost and certification, not raw performance.
7. **ANFIS / gradient-trained TSK** — the standard uncertified fuzzy baseline.
   Tests whether the least-action formulation buys anything over ordinary
   training *at equal parameter count*, which is a question the current results
   do not answer.

### The comparison table we should be able to fill

| method | score | certified region | certificate strength | design cost |
|---|---|---|---|---|
| LQR | 1.000 | (quadratic, global for the linearization) | analytic | trivial |
| PDC / LMI | ? | ? | LMI feasibility | one convex solve |
| polynomial fuzzy + SOS | ? | ? | SOS | one SOS solve |
| ANFIS | ? | none | — | training |
| MPC | ? | none (as posed) | — | per-step optimization |
| **least action (fit only)** | 0.867 | **0.568** | **exact rational** | one linear solve |
| **least action + certified policy opt** | **0.847** | **0.679** | **exact rational** | ~600 evals + SDP |

Filling the unknown rows is a concrete, bounded piece of work and it is what
converts the paper from "here is a method" into "here is a method that is better
than the alternatives in a stated respect."
