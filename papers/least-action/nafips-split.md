# Breaking this into NAFIPS papers

The combined write-up is ~18 pages of dense material with four independent
analytic results. That is two to three conference papers' worth of content, and
submitting it as one would bury the strongest contribution.

> **Check before relying on this:** NAFIPS page limits, format (recent editions
> have gone through Springer LNNS rather than IEEE), and the submission
> deadline. The plan below assumes roughly 8–12 pages; adjust the cut lines if
> the limit is tighter.

---

## Recommendation: three papers, in this order

The split is by **what each paper proves**, not by chronology. Each stands alone
and each has one headline claim a reviewer can evaluate.

---

### Paper 1 — *Certificate-Aware Design of Fuzzy Controllers* **(submit first)**

**Headline claim.** The certified region of attraction can be made a **design
specification** rather than an outcome: state a radius, receive a controller a
sum-of-squares program certifies at or above it. And, unexpectedly, requiring
the certificate *improved* out-of-sample performance.

**Content.** Paper §7 (SOS certification for $\pi$-membership TSK, including the
$u(0)=0$ requirement, the degree-1 basis requirement, and the exact rational
verification), §8 (certificate-aware optimization, the regularization finding),
§9 (the estimator diagnosis, the ray search, $\kappa$, the specification
results).

**Why first.** It is the most novel and the most complete. The
regularization result — a Lyapunov certificate acting as a regularizer,
improving held-out performance — is the kind of finding that gets remembered,
and it generalizes past fuzzy systems to learned controllers generally. The
exact-rational verification chain is a strong differentiator against the SOS
fuzzy-control literature, which typically stops at the solver's word.

**What it needs before submission.** Van der Pol (P2 in the roadmap) so the
conservatism estimate is checked against a known ROA rather than only against
itself. Ideally also a PDC/LMI comparison row.

**Risk.** Reviewers close to the polynomial-fuzzy-SOS literature will ask what is
new versus SOS synthesis. The answer must be stated in the abstract, not
buried: we do not synthesize by SOS, we make the certified radius an
*optimization objective* via a calibrated necessary condition, which is what
makes it a specification.

---

### Paper 2 — *Least Action as an Initializer for Fuzzy Control*

**Headline claim.** Posing TSK fitting as $H^1$ least action makes the consequent
block a closed-form global optimum, and that fit's value is not that it is best
— it is that it is the **initializer without which direct policy optimization
does not start**.

**Content.** Paper §2 (action, Euler–Lagrange, Galerkin stationarity), §3
(variable projection, envelope gradient, second-order conditions on the critical
cone), §6 (the whole two-mass–spring study: convergence in two rules,
distribution-shift diagnosis, the plateau, direct policy optimization, the
cold-start failure).

**Why it stands alone.** It has a clean narrative arc with a genuine negative
result in the middle (the method caps out at 0.863, diagnosably) and a
resolution at the end (0.655 with a warm start; ∞ without). Negative results
that are *diagnosed* are more useful to a conference audience than another
tuning success.

**What it needs.** An ANFIS or gradient-trained-TSK comparison at equal
parameter count. Without it, "least action beats ordinary training" is
unsupported — we have never actually run that comparison, and the paper should
either run it or avoid the claim.

**Optional inclusion: §4 ($\lambda^{*}$ and rule decoupling).** It fits
thematically — it answers how to *choose* $\lambda$, which this paper otherwise
leaves as a free parameter — but it is self-contained enough to be cut for space
or spun out (see "possible fourth paper" below).

---

### Paper 3 — *Exact Suboptimality Certificates for Fuzzy Control via Bregman Divergence*

**Headline claim.** For any **convex** control cost, the closed-loop
suboptimality gap is exactly $\int D_c(u\|u^{*})\,dt$ — an identity, not a bound,
with non-negativity following from convexity alone. And a matching negative
result: peak force and settling time are structurally outside any certificate of
this form.

**Content.** Paper §5 (Theorem 7 and why it is vacuous, Theorem 8, Corollary 9,
the non-quadratic numerical verification), §10 (G14 and the structural
exclusion), plus the machine-verification tier (§11) as evidence.

**Why it stands alone.** It is a short, self-contained theory paper — probably
the shortest of the three — and its content is independent of both the fitting
method and the stability machinery. It applies to *any* control-affine problem;
fuzzy control is the application, not the hypothesis.

**What it needs.** The relative form for unknown $V^{*}$ (roadmap A3) would
substantially strengthen it. Without A3 the certificate applies only to
inverse-optimal benchmarks, which a reviewer will notice — so either do A3 first,
or state the restriction prominently and frame A3 as the stated next step.

**Alternative packaging.** If A3 lands, this becomes a stronger standalone paper
and possibly a journal submission rather than a conference one.

---

## Possible fourth paper

### *$H^1$ Orthogonality and the Validity of Sequential Rule Identification*

**Content.** Paper §4 in full: Proposition 3 (greedy = joint iff block-diagonal
Gram), Proposition 4 (overlap is not the obstruction), Theorem 5
($\lambda^{*}=3b^{4}/(8a^{2})$, $\ell^{*}=\sqrt6\,\omega$), Corollary 6 (the
factor-of-four obstruction for $N\ge3$), the cost table, and the OLS alternative.

**Why it might be separate.** It answers a question practitioners actually ask —
"can I identify rules one at a time?" — with an exact condition and a sharp
negative result. It is also the piece least tied to control, which makes it the
natural bridge to the classifier paper.

**Why it might not.** On its own it is thin for a full paper, and it is more
useful as §4 of Paper 2 where it tells the reader how to pick $\lambda$.

**Recommendation:** keep it inside Paper 2 initially. Split it out only if
Paper 2 runs over length, or if the multi-input extension (roadmap C2) succeeds
and gives it enough independent substance.

---

## The classifier paper (separate track)

Per the current scope decision, the classifier development — Mamdani
aggregation reducing to summation iff output supports are disjoint, centroid
defuzzification equalling an order-0 TSK model, and the $\beta$-annealed
mean-of-maxima error bound — is its own paper. Note that it currently has a
weakness the control side does not: **those results have never met real data.**
They are proved and symbolically verified but exercised only on synthetic
partitions. Running them on a real dataset should be a precondition for
submitting it.

---

## Sequencing and shared material

```
Paper 1 (certificate-aware)  ──┐
                               ├── cite each other; Paper 1 is the flagship
Paper 2 (least action)       ──┤
                               │
Paper 3 (Bregman)            ──┘

Paper 4 (orthogonality)     ← only if Paper 2 overflows
Classifier paper            ← separate track, needs real data first
```

**Shared material each paper needs its own version of:**
- The TSK model and normalized-firing notation (~half a page). Unavoidable
  duplication; keep it identical across papers so a reader following all three
  is not re-learning notation.
- The two-mass–spring plant description. Papers 1 and 2 both need it; Paper 3
  does not (it uses the inverse-optimal benchmarks).
- The reproducibility statement and the pointer to the code.

**What must NOT be duplicated:** the benchmark results tables. Paper 2 owns the
convergence and policy-optimization tables; Paper 1 owns the certification and
specification tables. Paper 1 may quote a single number from Paper 2 as context
(the 0.773 starting point) with a citation, but should not reproduce the table.

---

## Honest assessment of readiness

| paper | content ready | blocking work | realistic status |
|---|---|---|---|
| **1 — certificate-aware** | ~95% | Van der Pol calibration (roadmap P2, ~1 day) | closest to submittable |
| **2 — least action** | ~90% | ANFIS comparison, or drop the implied claim | submittable with a scope caveat |
| **3 — Bregman** | ~85% | A3 would transform it; without it, state the restriction | submittable, but weaker than it could be |
| **4 — orthogonality** | 100% of what exists | needs C2 (multi-input) to justify standing alone | fold into Paper 2 |
| classifier | proofs done | real dataset | not ready |

The gating item across all of them is **breadth of plants** (roadmap Part 2). One
plant is the objection every reviewer will reach for, and Van der Pol plus the
inverted pendulum would answer it for roughly a week of work.
