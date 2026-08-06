# A fuzzy inference system as a TSP strategy engine — findings

A Lin-Kernighan solver spends its time unevenly and its parameters uniformly. Some cities sit on
edges already as short as the instance allows; others carry a long edge or a sharp kink where a
deep gain chain pays for itself many times over. A conventional LK cannot tell them apart, so it
picks one candidate breadth and one chain depth and applies them everywhere.

This work replaces three of those fixed rules with small fuzzy inference systems — `CONSTRUCT`
(which city next — a measured failure), `EFFORT` (how hard to search each city) and `CHAIN`
(deepen this gain chain or cut it) — and measures what that buys.

## 1. What is claimed

One claim, and it is narrow:

> A **small, readable, scale-free** rule base can allocate Lin-Kernighan search effort per
> decision. It **transfers across instance families without refitting**. Against a properly
> configured LKH it loses on the instances LKH finds easy, and wins a bounded window on the
> instances LKH cannot solve — where an `EFFORT`-aimed perturbation reaches tours that uniform
> kicking cannot reach at any budget.

The supporting evidence, and the bound on each piece:

| | evidence | bound |
|---|---|---|
| **transfers** | fitting per instance family costs −0.0006 in q against a ±0.0006 noise floor; on 2 of 4 families a family's own fit scored *worse* on it than a foreign one (§4.2) | 4 test instances per family, one fitting seed |
| **readable** | 5 inputs, 15 rules per base, 87 fitted parameters; the **hand-written** rules are best or near-best on 3 of 4 synthetic families, beating every fitted variant (§4.2) | fitting still wins on the pool it was fitted for |
| **wins somewhere** | where uniform kicking has **plateaued** — four times the budget buying the same tour — `EFFORT`-aimed kicks are the best of a 2x2 factorial on **2 of 2**, by 5.3x on fl1577; on d2103 a shorter tour than any LKH configuration below 39 s (§6.3) | 13 instances, 2 plateaued, one seed; three earlier readings of this table were wrong (§6.3) |
| **loses elsewhere** | on 6 of 7 easy instances LKH reaches 10–100x better quality *inside our own time budget* (§6.2) | — |

**What is not claimed.** That this beats LKH in general — it does not, and §6.2 is explicit. That
it is competitive with the learned-LKH line: NeuroLKH reports 0.05–0.09% where the arms here sit
at 0.3–1.3%, under a stricter protocol (`references/BENCHMARKS.md`). That fuzzy control of a
metaheuristic is new — it is a subfield with its own surveys (`references/PRIOR_ART.md`).

**Three things this document got wrong, left in place as retractions.** The reason each was
believed is the useful part. LKH's "floor" was an `elkai` default and a speed advantage of up to
1208x was claimed against it (§6.2). Aimed kicks were reported first as not working, then as
working above n ≈ 4000, when the variable is instance hardness (§6.3). And fitting was reported
as beating the hand-written rules, which holds only on the pool it was fitted for (§4.2).

---
## 2. How everything here is measured

Beating one configuration of a tunable solver proves nothing — there is always a slow one
and always a weak one. So the baseline is LK swept into a time-versus-quality **frontier**
over the parameters that trade one for the other: 11 configurations, k ∈ {16, 24, 32, 48,
64}, chain depth 2…10, deep breadth 4…32.

The reported number is **q**, computed per instance:

> **q** = (this arm's tour length) ÷ (the tour length the baseline sweep's own frontier
> reaches **at the same wall clock on that instance**), averaged over instances.

**q < 1 means the arm is outside LK's frontier** — a shorter tour than every LK
configuration that spends what it spent. Three properties make this the right measure, and
each was arrived at by getting it wrong first:

* **Per instance, not aggregated.** The conventional pair — mean gap against total seconds —
  puts an unweighted mean against a sum, so a few high-gap instances decide the quality axis
  while the largest instance decides the time axis. Under that pair the hand-written rule
  base appears to dominate most of the sweep; under q it does not. The aggregate was largely
  measuring instance mix.
* **Against the frontier, not one reference configuration.** A solver whose entire premise
  is choosing its own operating point cannot be judged at someone else's. A sweep also has
  to include a *cheap* point: an earlier sweep started at depth 4 and so had none, and the
  fuzzy arm scored well by landing in the gap. LK at depth 2 reaches 4.05% in 1.86 s.
* **Lengths, not gaps.** A ratio of gaps is undefined on instances the baseline solves
  exactly, where the gap is 0. Since `gap = 100(L/L* − 1)`, the length ratio is just
  `(1 + gap/100) / (1 + bar/100)` — finite everywhere, and it reads directly as "how much
  longer is this tour".

**q = 1 is a demanding bar.** The frontier is the envelope over all 11 configurations *per
instance*, so no single fixed configuration reaches it either — the best fixed LK sits at
q = 1.0035 over the test set. Only an oracle picking the best configuration per instance and
per budget would sit at 1. Comparisons below are therefore between arms, not against 1.

Reference values are the published TSPLIB optima. The 20 test instances (n = 52…18512) are
disjoint from everything used for fitting; `tune_opt.py` asserts that at import rather than
trusting the lists to stay right.

---

## 3. The engine

### 3.1 What the rule bases do

Averaged over city searches on the test instances, `EFFORT` and `CHAIN` run the solver at
**mean chain depth 6.4** where the baseline is pinned at 10, and **mean first-level breadth
25.0** where the baseline is 32.

The rule base spends more than the previous one did (depth 4.4, breadth 16.5) and gets a
better result, which is the depth-1 probe of §3.2 doing its job: with a look-ahead saying
whether gain is visible at all, the base can afford to go deep where it is rather than
staying globally timid.

That *depth* is the parameter worth being clever about was not the obvious guess. Sweeping
first-level breadth from 2 to 32 barely moves the clock: the sequential positive-gain
criterion truncates most candidate scans long before the breadth cap bites. Sweeping chain
depth from 4 to 10 costs 2.6×. So the rule bases earn their keep by deciding *which cities
deserve a deep chain*, and `CHAIN` — which cuts a chain mid-flight from its own gain
trajectory rather than at a fixed depth — is the most valuable of the three.

### 3.2 Choosing antecedents by measurement, not by argument

Adding an input costs twice: runtime in the innermost loop, and generalisation, because every
extra input multiplies the parameters the GA must fit. Features had been chosen by argument
and judged by whether the whole system improved — which conflates *is this signal
informative* with *did the GA manage to exploit it*.

**The reframing that made this measurable.** `EFFORT` was being asked "how much effort does
this city deserve?", which has no ground truth: there is no label for the right breadth at a
city. Asked instead **"will searching this city pay off, and by how much?"** the target is
directly observable — run the search and record whether the city yielded an improving move.
Effort allocation is then a monotone response to predicted payoff, and a candidate feature can
be scored on the prediction task alone, for the cost of one instrumented run.

`experiments/features_probe.py` does that over **12 278 city scans** on six instances, three TSPLIB and
three synthetic, of which 10.9% yielded an improving move. AUC is for predicting that event;
ρ is the rank correlation with realised gain over the paying cities only, which is a different
question and worth separating.

#### The master table

Every antecedent tried, its measured score, what computing it costs, the verdict, and which
rule-base scale includes it. `feature_registry.py` holds this as data and `FINDINGS.md` renders
it, so the two cannot drift apart; `python feature_registry.py --check` verifies the scale
columns against what `fis.py` actually builds.

| feature | AUC | ρ (paying) | cost | verdict | legacy | small | large |
|---|---|---|---|---|---|---|---|
| `probe_frac` (EFFORT) | 0.858 | +0.327 | a scan that breaks at the first failing candidate; usually 1-3 iterations | kept |  | ● | ● |
| `rank` (EFFORT) | 0.833 | +0.301 | a scan of an ascending list, breaks at the edge length | kept |  | ● | ● |
| `probe` (EFFORT) | 0.795 | +0.197 | free alongside probe_frac — same loop | kept |  | ● | ● |
| `excess` (EFFORT) | 0.759 | +0.153 | two distance evaluations and a divide | kept | ● | ● | ● |
| `edge_asym` (EFFORT) | 0.741 | +0.157 | free — both distances are already computed | kept |  | ● | ● |
| `turn` (EFFORT) | 0.691 | -0.116 | two hypots — the most expensive feature here | dropped | ● |  | ● |
| `peak` (EFFORT) | 0.589 | +0.231 | two loads and a divide, both precomputed per instance | dropped | ● |  | ● |
| `progress` (EFFORT) | 0.579 | +0.018 | a divide | dropped | ● |  | ● |
| `pos_spread` (EFFORT) | 0.547 | +0.174 | a full k-iteration scan, no early break | rejected |  |  |  |
| `cand_step` (EFFORT) | 0.520 | -0.106 | two loads and a divide | rejected |  |  |  |
| `fails` (EFFORT) | 0.488 | -0.123 | one load | dropped | ● |  |  |
| `in_degree` (EFFORT) | 0.449 | -0.031 | one load, precomputed per instance | rejected |  |  |  |
| `nbr_active` (EFFORT) | 0.426 | -0.100 | a full k-iteration scan, no early break | rejected |  |  |  |
| `credit` (CHAIN) | — | — | free — the chain already has it | kept | ● | ● | ● |
| `depth` (CHAIN) | — | — | free | kept | ● | ● | ● |
| `banked` (CHAIN) | — | — | free | kept | ● | ● | ● |
| `trade` (CHAIN) | — | — | free — computed while choosing the next candidate | kept | ● | ● | ● |
| `revcost` (CHAIN) | — | — | free — `reverse` already returns its swap count | kept |  | ● | ● |

The scales:

* **`legacy`** — features by argument; 6 inputs, 27 rules, 9 interactions. EFFORT reads `excess`, `turn`, `peak`, `progress`, `fails`.
* **`small`** — AUC >= 0.74 only; 5 inputs, 15 rules, no interactions. EFFORT reads `probe_frac`, `rank`, `probe`, `excess`, `edge_asym`.
* **`large`** — small + the AUC 0.55-0.70 band + interactions; 8 inputs, 36 rules. EFFORT reads `probe_frac`, `rank`, `probe`, `excess`, `edge_asym`, `turn`, `peak`, `progress`.

`CHAIN`'s inputs are deliberately unscored. The screen answers a *per-city* question — "will
searching this city pay off" — and the chain cut-off is a decision taken many times within one
city, whose outcome is "would one more level have improved the best closing gain". That is a
different label, and the per-city one does not answer it. Those five are justified by the
chain's own arithmetic and by ablation instead, which is a weaker footing and worth saying.

Four of seven new proposals were rejected, which is the point: without this they would have been
argued into the rule base and judged only by whether the system got better.

**The winners are a look-ahead probe.** Run one level of search *before* committing to any, and
read how many candidates pass the positive-gain test and how large the best gain is. It is
nearly free for the same reason the search itself is: candidate distances ascend, so the loop
breaks at the first candidate that fails, which for a city already sitting on short edges is
the first one. This is the shift in perspective made concrete — decide effort *after* a cheap
probe rather than from static geometry before.

**`fails` at AUC 0.488 is the instructive result.** It was an existing input, and it is
indistinguishable from noise for this purpose — the don't-look-bit queue already removes
settled cities structurally, so the count adds nothing on top of the mechanism that generates
it. `turn` is interesting differently: AUC 0.691 says it predicts *whether* a city pays, while
ρ = −0.116 says that among paying cities sharper turns pay *less*. A single monotone rule
cannot serve both, which is a reason to drop it rather than a reason to weight it.

**The bound on what this shows.** The labels come from a *fixed-parameter* search, so this
measures "would effort here have paid off at these settings", not "what is the optimal effort
here". It screens out useless features and ranks plausible ones. It does not prove a kept
feature earns its runtime — §5 and §3.1 are where that is tested, and the depth/breadth increase
in §3.1 is the evidence that the probe changed behaviour rather than just adding cost.

### 3.3 Making inference cheap enough to be worth consulting

A rule base consulted once per city scan must cost far less than the scan. It did not.
`costmodel.py` (§3.4) priced the pieces: a chain-continuation decision cost **494 ns** against
the **~120 ns** chain level it was deciding about.

Two fixes, in the order they were found:

1. **Memberships were recomputed inside the rule loop** — 19 rules over 4 inputs evaluating
   up to 76 membership functions when only 12 distinct values exist. Hoisting them into a
   per-evaluation table should have been a clear win.
2. **It wasn't, because that table was a heap allocation.** numba allocates a small array per
   call, and the allocation cost about what the saved exponentials did: measured improvement
   494 → 494 ns. Every scratch buffer is now owned by the caller and threaded down.

Then the membership bank was compiled to a **lookup table over [0,1]**, which every fuzzy
input already maps into. That removed the exponentials and had a larger second consequence:
*membership shape became data.* Centres, widths and functional form are now all things an
optimiser can move at no hot-path cost, because the hot path never learns which it got.

| | chain decision | effort decision | city scan |
|---|---|---|---|
| original | 494 ns | 578 ns | 866 ns |
| caller-owned buffers | 210 ns | 578 ns | 866 ns |
| + lookup-table bank | **161 ns** | **344 ns** | 760 ns |

The residual **349 ns effort decision against a ~493 ns city scan** is the mechanism behind
§5's size dependence, and the depth-1 probe of §3.2 adds a little more on top of it. It is a fixed cost per city, so it is amortised only once each city's
search does enough work to hide it — which is why the fuzzy arm loses below n ≈ 1000 and
leads above n ≈ 2000, and it says where the crossover has to be rather than leaving it as an
empirical curiosity.

---

### 3.4 A deterministic cost proxy, so the search is reproducible

The objective must include runtime, but wall clock is a bad thing to optimise against: it is
noisy, irreproducible, and — decisively — unmeasurable under parallel evaluation, because
concurrent workers contend for cores.

`costmodel.py` fits the solver's own work counters to measured time by **non-negative least
squares** (a counter cannot make the solver faster; a model that says it does is fitting
noise) in **relative** space (times span four orders of magnitude, so a plain fit is
dominated by the largest instance and over-predicts every small one by more than 100%).

Current fit, 224 (instance, configuration) samples: **R² 0.9990, rank correlation 0.9995,
mean relative error 3.0%, p90 6.3%.** Per-unit costs: city scan 493 ns, accepted move
1151 ns, effort decision 349 ns, chain decision 135 ns, chain level 91 ns, candidate
evaluation 7.5 ns, reversal element 4.4 ns.

Two things came out of building it, beyond serving the tuner:

* **A reversal-element counter had to be added** before the fit was usable. Without it, `n`
  absorbed the missing cost at 6684 ns/city and mean relative error was 146%. Segment
  reversal is real memory traffic that no move count sees.
* **It found the inference overhead in §4.** The model was built for the objective; its
  largest payoff was diagnostic.

`benchmark.py` still reports real wall clock. The proxy is what the *search* spends; it is
not what any result is measured with.

---

## 4. Fitting, and why the design matters more than the parameters

### 4.1 GA, then derivative-free descent

`tune_opt.py` fits **87 parameters** — 75 rule consequents plus membership centres and widths
— over **20 training and 14 validation instances**, all n ≥ 1000. Triangular membership
functions throughout; they beat gaussian on every run that compared them (validation q 1.017
against 1.069 at matched budget) and are cheaper to tabulate.

Three changes made fitting transfer to unseen instances, where before it did not:

**1. Fewer parameters, because the rules are simpler.** `EFFORT` and `CHAIN` are now three
rules per input with no interactions — 15 rules each, 75 consequents against the previous 180.
The interactions were where most of the parameters lived and therefore most of the
overfitting. `CONSTRUCT` is no longer fitted at all: it is a measured failure and appears in
no reported arm, so every parameter spent on it was one the GA could overfit with. 170 → 87.

**2. Four times the training instances.** `synth.py` generates uniform, clustered,
jittered-grid and mixed-density instances, taking the pool from 9 training instances to 20 and
7 validation to 14. This is possible because the objective is a ratio of two tours on *one*
instance, so the optimum cancels and synthetic instances need none — which required pushing
tour lengths rather than gaps through the frontier machinery to be true in the code and not
only in the argument. The mixed-density family exists because no TSPLIB instance isolates the
case this engine should be best at: one instance containing both a dense and a sparse regime,
where no single global effort setting is right.

**3. Antecedents chosen by measured predictive power** rather than by argument (§3.2).

**The result:**

| | validation q | test q, n ≥ 5000 |
|---|---|---|
| hand-written | 1.0042 | 1.0048 |
| GA-fitted | **0.9997** | **1.0014** |

Fitting is now better than the hand-written rules on unseen data, which reverses the earlier
finding. It is better in the bands it was fitted for (n ≥ 1000) and clearly *worse* below
n = 1000 — 1.0182 against 1.0032 — which is the expected consequence of restricting fitting to
n ≥ 1000 and of the probe features costing per-city time that only amortises at scale.

#### The second stage: derivative-free stepwise refinement

The GA is a global search whose moves are recombination and mutation, neither of which is a
descent step. The evidence is direct: its own final vector scored q = 1.0025 on validation
where the best vector it had *passed through* scored 0.9997. So `refine.py` adds a **compass
search** — try `±step` on each coordinate, keep only improvements, halve the step when a whole
sweep finds nothing.

Derivative-free is the correct choice here rather than a concession: the objective is
piecewise constant in every parameter, because nudging a consequent either changes some city's
*rounded* breadth or does nothing at all, so a finite-difference gradient is either exactly
zero or an artefact of the step size. This is the deterministic sibling of the `optimizers`
library's own `local_perturb_optim`, which does the same one-variable-at-a-time thing with a
random step; both are available.

It behaves exactly as the theory says, in both directions:

| polish budget | validation before | after | kept? |
|---|---|---|---|
| 301 evaluations | 1.0006 | **1.0002** | yes |
| 2500 evaluations (16 sweeps) | 0.9997 | 1.0011 | **no** |

At a small budget it improves the answer. At a large budget it descends far enough into the
*training* objective to lose validation, and the validation gate rejects it. That is the same
budget-versus-generalisation relationship the GA shows, arriving faster because a descent
method is more efficient at overfitting than a population method is.

#### More search still makes generalisation worse

| GA evaluations | training q | validation q |
|---|---|---|
| 140 | — | 0.990 |
| 380 (current) | 1.0033 | **0.9997** |
| 1440 | 0.9938 | 1.057 |
| 4000 | 0.9520 | 1.069 |

Training score improves monotonically while the answer does not. The default budget is small
deliberately, and the polish is gated on validation rather than trusted.

#### What remains unsound

Selection still runs through validation — the best of 24 pooled candidates is *chosen* on it,
and so is the accept/reject decision on the polish. That makes validation part of the fitting
procedure and its numbers optimistic. Only the test set is untouched, and §5 is what it says.

#### Practical notes

* **Thread parallelism buys nothing.** numba holds the GIL: `n_jobs=4` measured 18.0 s against
  17.2 s for `n_jobs=1`. The deterministic cost proxy of §3.4 was built partly to enable
  parallel evaluation, which turned out to be unavailable for an unrelated reason.
* **Evaluation cost depends on the candidate.** A vector telling the rule bases to use full
  depth and breadth everywhere makes the solver several times slower than the baseline, and
  bound-seeking optimisers walk into it: one PSO generation cost 99 s against the GA's 12.5 s.
  Evaluation now abandons a candidate once it exceeds twice the dearest baseline
  configuration's cost, ranked by how far over so the search keeps a gradient. PSO then
  completed 720 evaluations in 73 s.
* GA is the reported optimiser; PSO and ACO drivers remain selectable but are not part of the
  comparison.

### 4.2 Does one rule base transfer? Yes, and that is the result

This is the load-bearing claim of the whole design and it had never been measured directly.
"One rule base, fitted once, works on a held-out test set spanning n = 52…18512 and four
structural families" is a *test score*. It becomes a *generalisation* result only against a
contrast: what would fitting per family have bought?

`experiments/transfer.py` measures it. Fit on one structural family alone; test on all of them.
The diagonal is home field — fitted and tested on the same family, different instances — and the
off-diagonal is transfer. Affordable because the objective is a ratio of two tours on one
instance, so the optimum cancels and `synth.py` can generate the pools freely.

Four fitting instances per family (n = 1300…5300), four test instances per family
(n = 1500…4100), plus the held-out TSPLIB instances in 1000 ≤ n ≤ 4000 as a fifth column:

| fitted on | uniform | clustered | grid | mixed | tsplib |
|---|---|---|---|---|---|
| **hand-written** | **0.9993** | 0.9991 | **0.9995** | **1.0028** | 1.0031 |
| shipped (all families) | 1.0023 | 0.9996 | 1.0031 | 1.0031 | **0.9993** |
| uniform | 0.9992 | **0.9966** | 1.0011 | 1.0023 | 1.0031 |
| clustered | 1.0018 | 1.0070 | 1.0008 | 1.0033 | 1.0079 |
| grid | 1.0032 | 1.0029 | 1.0010 | 1.0023 | 1.0064 |
| mixed | 1.0025 | 1.0071 | 1.0031 | 1.0040 | 1.0026 |

#### Transfer is free

| fitted on | home | mean away | penalty |
|---|---|---|---|
| uniform | 0.9992 | 1.0000 | +0.0008 |
| clustered | 1.0070 | 1.0020 | **−0.0050** |
| grid | 1.0010 | 1.0028 | +0.0018 |
| mixed | 1.0040 | 1.0042 | +0.0002 |

Mean penalty **−0.0006**, against a re-run noise floor of ±0.0006 (§5). Knowing the instance
family in advance is worth nothing measurable.

Stronger than that: on **two of four families, fitting on that family produced a worse score on
it than fitting on a different one**. The clustered-fitted base scores 1.0070 on clustered —
the worst cell in its own row — while the uniform-fitted base scores 0.9966 there, a gap of
0.0104 that is well outside noise. There is little family-specific signal to exploit, and the
budget spent looking for it goes into overfitting instead.

This is what the scale-free antecedent design was for and it is the first direct evidence that
it works. Every `EFFORT` input is a ratio — excess over the nearest-neighbour distance,
candidate rank, the fraction of probed candidates that pass — so nothing in the rule base
carries the units, the density or the diameter of the instance it was fitted on. `CHAIN`'s
inputs are dimensionless by construction, being internal to the search.

#### The uncomfortable half: fitting buys little over the hand-written rules

**The hand-written rule base is the best or near-best arm on three of the four synthetic
families**, beating every fitted variant on `uniform` and `grid` and losing to the
uniform-fitted base on `clustered` only. Fitting wins on exactly one column — TSPLIB, 0.9993
against 1.0031 — which is the pool the shipped base was fitted for.

That is a real tension with §4.1, which reports fitting beating the hand-written rules on
validation (0.9997 against 1.0042). Both are true and they are measuring different things:
§4.1's validation pool is the same TSPLIB-plus-synthetic mixture the fitting used, and fitting
helps *there*. On structurally pure families it does not.

The honest reading is that **the design generalises and the fitted parameters do not add much**:
the scale-free ratios, the three-terms-per-input structure and the antecedent selection of §3.2
are doing the work, and the GA is a marginal refinement tied to its pool. For a claim about a
*simple, general* effort model that is a better result than the alternative — it means the rule
base can be read, and the reader is not being asked to trust 87 fitted numbers.

**Bounds.** Four test instances per family, one fitting seed, one machine, GA budget of 12
generations x 20 population. Differences below about 0.002 here should be treated as noise; the
0.0104 clustered gap and the 0.0038 TSPLIB gap should not.

---

### 4.3 Does a bigger rule base help? Only at the top of the size range

Two scales were fitted with an identical pipeline — same GA budget, same compass polish, same
20/14 instance pools, same seed — and differ only in the rule base:

| | inputs | rules (EFFORT + CHAIN) | fitted parameters |
|---|---|---|---|
| `small` | 5 (AUC ≥ 0.74) | 15 + 15 | 87 |
| `large` | 8 (adds AUC 0.55–0.70) | 30 + 25 | 157 |

Test-set q for the two best arms, by size band:

| arm | scale | q, all 20 | n<1000 | 1000–2000 | 2000–5000 | **n ≥ 5000** |
|---|---|---|---|---|---|---|
| FIS + deferred verification | small | **1.0059** | 1.0164 | 1.0075 | 1.0033 | 0.9997 |
| FIS + deferred verification | large | 1.0077 | 1.0189 | 1.0124 | 1.0056 | **0.9991** |
| FIS effort + chain | small | **1.0070** | 1.0192 | 1.0063 | 1.0033 | 1.0013 |
| FIS effort + chain | large | 1.0081 | 1.0186 | 1.0112 | 1.0077 | **0.9994** |

The pattern is consistent across both arms and it inverts at around n = 5000. Below that the
large base is clearly worse — at 1000–2000 it loses by 0.005, which is large by the standards
of everything else here. Above it, the large base produces the two best figures in the study
and is below 1.0 on **five of seven** instances against the small base's four.

**Why, mechanically.** The three extra inputs are not free: `turn` needs two square roots and
is the most expensive feature in the system. That is a fixed cost per city scan, so it lands on
exactly the same amortisation curve as §3.3's inference overhead — it hurts wherever the work per
city is small and disappears wherever it is large. The extra discrimination has to beat that
cost, and it only does so at the top of the range. Mean chain depth confirms the base is
behaving differently rather than just costing more: 7.5 for `large` against 6.4 for `small`,
so the wider base is finding more places it judges worth a deep chain.

**The other half of the answer is about the search, not the rule base.** At a matched
380-evaluation GA budget, `large` fits its *training* set **worse** than `small` — q 1.0046
against 1.0033 — despite having 70 more parameters to do it with. More capacity did not
produce a better fit because the same budget spread over nearly twice the dimensions explores
each of them less. So the large base is not being shown at its best here, and the honest
reading of its n ≥ 5000 advantage is that it is achieved *despite* being under-fitted. A budget
scaled to the dimension count is the obvious next experiment, and it is the one that would
decide whether the middling-AUC features are worth their runtime or merely tolerable.

**What is kept.** `small` remains the default: better over the whole test set, cheaper, and
half the parameters. `large` is the better choice above n ≈ 5000 and is selectable with
`--scale large`. That both are worth keeping — rather than one being simply right — is itself
the result.

---

## 5. Against the same LK, swept into a frontier

Test set, 20 instances, n = 52…18512, wall clock, best of 3 repetitions.

| arm | mean gap | total s | **q** |
|---|---|---|---|
| `lk_32_2_4` — best fixed LK | 4.048% | 2.18 | **1.0029** |
| `lk_16_4_8` | 4.598% | 4.08 | 1.0033 |
| `lk_32_6_32` | 3.485% | 3.47 | 1.0035 |
| **FIS effort, GA-fitted** | 3.671% | 3.03 | **1.0044** |
| **FIS effort + chain, hand-written** | 3.671% | 2.61 | **1.0044** |
| `lk_32_3_8` | 3.876% | 2.51 | 1.0049 |
| `lk_48_10_32` | 3.305% | 5.38 | 1.0056 |
| FIS + deferred verification | 3.852% | 2.42 | 1.0057 |
| FIS effort + chain, GA-fitted | 4.269% | 2.04 | 1.0061 |
| FIS + fuzzy construction | 5.535% | 1.89 | 1.0178 |

Over the whole test set no fuzzy arm beats the best fixed LK configuration. Half these
instances are below n = 1000, where §3.3 explains why the fuzzy arms cannot win.

### The size dependence is the result

Mean q by size band. The fuzzy arms improve monotonically with size; the fixed LK
configurations do not, because there is no reason they should — a fixed configuration is
equally fixed at every scale.

| arm | n < 1000 (5) | 1000–2000 (3) | 2000–5000 (5) | n ≥ 5000 (7) |
|---|---|---|---|---|
| **FIS + deferred verification** | 1.0163 | 1.0071 | 1.0031 | **0.9994** |
| FIS effort + chain, GA-fitted | 1.0182 | **1.0011** | 1.0034 | 1.0014 |
| FIS effort + chain, hand-written | **1.0032** | 1.0093 | 1.0023 | 1.0048 |
| FIS effort, GA-fitted | 1.0068 | 1.0056 | 1.0026 | 1.0034 |
| `lk_32_2_4` | 1.0062 | 1.0045 | 1.0026 | 1.0000 |
| `lk_32_3_8` | 1.0139 | 1.0038 | **1.0009** | 1.0019 |
| `lk_48_10_32` | 1.0169 | 1.0000 | 1.0045 | 1.0008 |
| `lk_32_6_32` | 1.0042 | 1.0047 | 1.0013 | 1.0040 |

At n ≥ 5000 the fuzzy arms are the only ones outside the frontier. Ranked over those seven
instances, at the **`large`** scale (§4.3 — the better choice in this band):

| arm (n ≥ 5000) | mean gap | total s | q |
|---|---|---|---|
| **FIS + deferred verification** | 3.196% | 2.40 | **0.9991** |
| **FIS effort + chain, GA-fitted** | 3.533% | 1.92 | **0.9994** |
| `lk_32_2_4` | 3.782% | 1.87 | 1.0000 |
| `lk_48_10_32` | 2.824% | 4.75 | 1.0008 |
| `lk_32_6_8` | 3.297% | 2.99 | 1.0012 |

**What that is and is not.** Per instance the deferred arm is below 1.0 on **five of seven**
(0.9943, 0.9965, 0.9979, 0.9986, 0.9993) and above on two, the worst being pla7397 at 1.0063.
A sign test on five of seven gives p ≈ 0.23; the mean margin is nine parts in ten thousand of
tour length; and re-running the benchmark moves the same figure by ±0.0006, which is most of
the margin. The standard errors in every band overlap between arms.

So: adaptive effort is **at** LK's frontier at the top of this size range, with a consistent
but unestablished suggestion of crossing it, and inside the frontier below. The size trend is
an order of magnitude larger than the between-arm differences and is the solid part.

---

## 6. Against LKH

### 6.1 The move repertoire was the binding constraint

Every section above tunes *how much* search to spend and *where*. None of it changes **what
moves exist**, and that turns out to have been the ceiling the whole time.

The repertoire was sequential 2-opt chains plus Or-opt relocation. Both are improving moves, so
the search converges to a local optimum of that neighbourhood and stops. The rule bases can
decide how fast to get there and how cheaply; they cannot decide to leave. The evidence that
this bound everything is in the sweep itself: on pr1002 **every** fixed configuration tops out
at 2.473%, and spending more time — `k32/d10/b32`, `k48/d10/b32` — buys nothing or loses. That
is not a tuning plateau, it is the neighbourhood being exhausted.

#### The double bridge

A non-sequential 4-opt move reconnecting A-B-C-D as A-C-B-D. What makes it the right addition is
precisely that **it reverses no segment**, so no sequence of improving 2-opt steps can reach it —
which is why it escapes the optimum they converge to. It is applied as a perturbation: on its own
it lengthens the tour, and the bet is that re-optimising recovers more than it lost often enough
to pay.

Three implementation choices make it affordable at these sizes:

* the kick is drawn inside a **bounded window**, so it costs O(window) and damages a region
  rather than the tour — the textbook uniform-cut-point version is O(n) per kick;
* re-optimisation is **seeded from the eight cities whose edges changed**, not restarted, which
  is why `lk_solve`'s queue loop was factored out as `lk_reopt` rather than copied;
* rejection **restores by copying the best tour**. An undo log through an arbitrary
  re-optimisation is complicated and easy to get subtly wrong; the copy is O(n) of memcpy
  against a much larger re-optimisation cost.

#### What it bought

| kicks | pr1002 | pr2392 | pcb3038 |
|---|---|---|---|
| 0 (plain local search) | 2.473% / 0.013s | 4.705% / 0.037s | 3.836% / 0.048s |
| 1 600 | 1.387% / 0.28s | 1.612% / 0.36s | 1.802% / 0.38s |
| 25 600 | 0.665% / 3.87s | 1.203% / 4.06s | 0.967% / 4.68s |
| 102 400 | 0.580% / 14.3s | 1.161% / 16.0s | **0.726% / 17.2s** |

A factor of three to four in quality. Every other section of this document moves the number by
hundredths of a percent; this moves it by whole points, and it is the only change that extends
the frontier *rightwards* into the region where a serious solver operates rather than shuffling
position along the existing one.

#### Against LKH: a crossover, not a win

| instance | LKH's cheapest possible point | our best strictly below that budget |
|---|---|---|
| pr1002 (n=1002) | 0.000% at **3.81s** | 1.387% at 0.28s |
| pr2392 (n=2392) | 0.000% at **54.57s** | 1.161% at 16.0s |

Three separate things are true and they should not be blurred together.

**Above LKH's floor, LKH wins outright and not narrowly.** It reaches the published optimum
exactly, at every run count tried, on both instances. Our curve asymptotes around 0.58% on
pr1002 and 1.16% on pr2392 and does not get closer with more kicks. There is no budget above
3.81s on pr1002 at which we are competitive.

**Below LKH's floor we are the only solver present, and that window widens steeply with n** —
3.81s at n=1002 against 54.57s at n=2392, so n grew 2.4x while the floor grew 14x. Because
`elkai` accepts a run count rather than a time limit, LKH returns nothing at all until its first
run completes. Every point of ours inside that window is non-dominated by construction.

**How much credit that deserves is limited, and the limit is worth stating.** It is partly a
property of `elkai`'s interface rather than of LKH: the real LKH accepts a time limit, and a
properly configured LKH would return *something* in a second. What survives that objection is
narrower but still real — our solver is genuinely anytime, and its quality in the 0.3–17s band
is respectable in absolute terms (0.7–1.6% over optimum) rather than merely unopposed.

#### The asymptote is the next binding constraint, and it is not the scheduling

More kicking stops helping. On pr2392 quadrupling the budget past 25 600 kicks buys 0.04 points.
But this is **not uniform across instances** — pcb3038 was still descending over the same step,
0.967% to 0.726% — so it is not a fixed property of the repertoire either. Something binds on the
structured `pr` instance that does not bind on `pcb` at the same budget.

Two candidate causes call for opposite investments, so `kick.py` carries a switch for each rather
than a guess: `accept_equal` (drift sideways across a plateau) and `patience` (accept a worsening
tour after N rejections). If neither moves the asymptote, the cause is the repertoire, and the
next step is not perturbation scheduling but **alpha-nearness candidate sets** — LKH builds its
candidates from a minimum spanning tree rather than by plain k-nearest, which improves every move
rather than adding one. That measurement is the sharpest open question in this document.

### 6.2 On the instances LKH finds easy, we lose

Everything above compares the FIS against a swept LK. The comparison against LKH was, until
now, run with the fuzzy engine switched off: `frontier_vs_lkh.py` measured the plain iterated
solver with `use_chain=False` and uniform kicks, and its `targeted` parameter was accepted and
never used. Two of its three default instances — `pr1002` and `d1291` — are instances
`tune_opt.py` **fits on**. So the headline question had no measurement behind it, and the
measurement it did have was partly on training data.

`lkh_compare.py` now runs the FIS against LKH on held-out instances only, and refuses at
import to measure on anything from the fitting pools. The FIS enters twice: as the fitted
`EFFORT`+`CHAIN` local search, and — since a perturbation has to be aimed somewhere — as a
**2x2 factorial** over the two places inference can act once kicking is in play. `EFFORT` can
choose where kicks land; `CHAIN` can set how deep each seeded re-optimisation goes.

#### First, a retraction

An earlier version of this section reported LKH's cheapest available point as 1.9 s at n = 783
rising to 160 s at n = 5915, and claimed a speed advantage of up to 1208x against it. **Those
were `elkai`'s defaults, not LKH's limits, and the claim was wrong.**

`elkai.Coordinates2D.solve_tsp` exposes only `runs` and hard-codes the parameter file to
`RUNS = n`, leaving `MAX_TRIALS` at LKH's default — the problem *dimension*. One "run" on
rl5915 is therefore 5915 improvement trials. The underlying `_elkai.solve_problem(params,
problem)` takes the parameter file as a string, so `MAX_TRIALS`, `TIME_LIMIT` and
`CANDIDATE_SET_TYPE` are all reachable. Capping trials and switching to nearest-neighbour
candidate sets moves LKH's cheap end by two to three orders of magnitude:

| n | reported before | actually |
|---|---|---|
| 783 | 1.9 s | **0.017 s** |
| 2392 | 22.7 s | **0.15 s** |
| 5915 | 160.2 s | **0.58 s** |

Every speedup multiple in the old anytime table — 186x, 361x, 911x — was measuring a default.
The corrected ratios are 0–4x. `lkh_compare.py` now sweeps 11 LKH configurations by default,
and the ones that make the comparison unflattering are in the default sweep rather than behind
a flag, because a flag is how such a thing quietly stops being run.

#### Where we stand, with LKH configured properly

Seven test instances, `small` scale, wall clock. "LKH under our budget" is the best tour any
measured LKH configuration reaches within the time our own best arm spent:

| instance | n | our best | at | LKH under that budget | non-dominated window |
|---|---|---|---|---|---|
| rat783 | 783 | 0.477% | 3.8 s | **0.000% @ 0.07 s** | 0.01–0.01 s |
| pcb1173 | 1173 | 0.336% | 4.6 s | **0.002% @ 3.02 s** | 0.01–0.35 s |
| rl1323 | 1323 | 0.526% | 4.0 s | **0.048% @ 1.08 s** | 0.01–0.54 s |
| pr2392 | 2392 | 0.988% | 6.0 s | **0.008% @ 4.27 s** | 0.02–0.15 s |
| pcb3038 | 3038 | 0.726% | 6.3 s | **0.058% @ 2.53 s** | 0.02–0.25 s |
| fnl4461 | 4461 | 0.468% | 6.6 s | **0.043% @ 5.18 s** | 0.03–0.28 s |
| **rl5915** | 5915 | **0.835%** | 11.2 s | 8.499% @ 4.19 s | **0.09–11.20 s** |

On six of seven, LKH reaches ten to a hundred times better quality inside our own time budget.
The non-dominated windows there are slivers a hundredth of a second wide — real, since no LKH
configuration matches those points, but not worth a sentence in an abstract.

**Note the measure.** "Non-dominated" means *no* measured LKH configuration is at least as good
on both axes. The earlier "beats LKH" column asked whether we beat *some* LKH point, which once
LKH is swept over parameters is nearly free — it can be satisfied by beating a badly configured
one. That is not a result and it is no longer reported.

#### rl5915 is the exception, and it survives its controls

There is a **17-second hole in LKH's own frontier** on rl5915:

| LKH frontier | gap | configuration |
|---|---|---|
| 4.19 s | 8.499% | `nn5/trials=100` |
| **21.30 s** | 0.446% | `alpha/trials=1` |

Nothing LKH offers between those budgets improves on 8.5%. Our `iterated_aim` reaches **0.835%
at 11.20 s** — an order of magnitude better than LKH at any budget under 21 s — and is
non-dominated from 0.09 s to 11.20 s.

**Why.** The `rl*` instances have grid-like, tie-heavy coordinates where 5-nearest candidate
lists are ambiguous and badly wrong; LKH at `nn5` sits at 8–18% there. Alpha-nearness on a
Held-Karp 1-tree fixes it and is what LKH is designed around, but that 1-tree is 21 s at this
size and cannot be made cheaper. Our solver uses 32-nearest lists — broad enough that ties do
not mislead it — plus perturbation to escape what breadth alone leaves.

**Two controls, both passed.** Giving LKH our 32-candidate budget reaches 0.343% but takes
**338 s**, thirty times our budget, because LKH's 5-opt basic move makes candidate breadth far
costlier than our sequential 2-opt chain — so "matched candidate budget" handicaps LKH rather
than levelling, and each solver at its own best configuration is the right comparison. And the
window is not an artefact of the arm chosen: `iterated_aim` is the best of the four at 0.835%
against the control's 1.344%, so the aiming is contributing rather than incidental.

**One instance.** Whether this is a property of tie-heavy instances or of rl5915 is exactly
what the hard ladder (§6.3) is for.

**LKH's cost does not follow a clean power law.** The local exponent between adjacent instances
swings from 0.76 to 4.93. `pcb3038` costs LKH 73.7 s at `runs=1` where the *larger* `fnl4461`
costs 98.5 s, so instance structure matters as much as size and any single fitted exponent will
mis-price some instance badly. `--dry-run` fits one anyway and should be read as an order of
magnitude.

### 6.3 On the instances LKH cannot solve, aimed perturbation wins

The seven instances above are drawn almost entirely from what Zheng et al. (VSR-LKH, AAAI 2021,
Table 3) call the **easy 74**: of 111 TSPLIB instances, the 74 on which LKH reaches the
published optimum in all ten runs, which they set aside as uninformative. We had been
benchmarking on the half of the benchmark the field discards.

Their **37 hard** instances contain a sharper subset — success rate **0/10**, LKH never reaches
the optimum in any run: `fl1577 rl1889 d2103 fl3795 rl5915 rl5934 brd14051 d15112 d18512
pla33810 pla85900`. Of those, `rl1889` and `rl5934` are in this project's fitting pools and
`pla33810`/`pla85900` are CEIL_2D which elkai mis-scores. **The remaining seven were already in
`benchmark.TEST`** — held out, EUC_2D, and never used in the LKH comparison.

All seven measured (`--ladder-hard`), at the matched budget of §5's 2x2. The **headroom**
column is what the uniform-kick control still gained over the final quadrupling of its own
budget: 0% means four times the kicks bought it the *same tour*, and it has plateaued.

| instance | n | set | headroom | control | **+ aimed kicks** | + `CHAIN` | + both |
|---|---|---|---|---|---|---|---|
| **fl1577** | 1577 | **hard** | **0.0%** | 4.459% | **0.840%** | 4.499% | 4.535% |
| **fl3795** | 3795 | **hard** | **0.0%** | 3.552% | **3.052%** | 3.552% | 3.469% |
| pr2392 | 2392 | easy | 3.5% | **1.161%** | 1.180% | 1.118% | 1.104% |
| rat783 | 783 | easy | 12.5% | **0.477%** | 0.488% | 0.693% | 0.500% |
| rl1323 | 1323 | easy | 14.4% | 0.573% | 0.811% | **0.526%** | 0.618% |
| d18512 | 18512 | hard | 18.5% | **0.536%** | 0.659% | 0.633% | 0.601% |
| brd14051 | 14051 | hard | 20.1% | **0.918%** | 0.932% | 1.061% | 1.058% |
| d15112 | 15112 | hard | 22.2% | **0.469%** | 0.544% | 0.541% | 0.624% |
| pcb3038 | 3038 | easy | 24.9% | **0.726%** | 0.941% | 0.901% | 0.812% |
| pcb1173 | 1173 | easy | 27.4% | **0.336%** | 0.809% | 0.603% | 0.810% |
| fnl4461 | 4461 | easy | 29.4% | 0.478% | **0.468%** | 0.567% | 0.743% |
| rl5915 | 5915 | hard | 40.8% | **0.796%** | 0.835% | 0.950% | 0.853% |
| **d2103** | 2103 | **hard** | 44.6% | 0.153% | **0.087%** | 0.595% | 0.600% |

**Aiming wins on both instances where the control has plateaued, and on two of the eleven where
it has not.** That is the criterion: not instance size, not kick density, but whether uniform
perturbation has stopped making progress. `figures_aim.py` draws it.

The two plateaued instances are the two `fl` instances — clustered drilling problems whose
difficulty lives in a few regions. `d2103` is the one substantial win outside the criterion, and
`fnl4461` is a 2% margin that should be read as a tie.

### Three wrong readings of this table, and why each looked convincing

This claim was reported wrongly three times before the seventh instance and the plot together
made the criterion visible. The pattern is worth more than the conclusion:

1. **"Aiming works above n ≈ 4000."** Written when `fnl4461` and `rl5915` were the only large
   instances measured. The ladder's large instances happened to include the only hard one, so
   size and hardness were perfectly confounded.
2. **"Aiming wins on four hard instances of four."** Written before `brd14051`, `d15112` and
   `d18512` existed. All three are losses.
3. **"Aiming needs a dense kick budget — ≥69 kicks per city."** This one had a *clean*
   separation and was still wrong. The density figure for `rl5915` was computed assuming it had
   been run at the large ladder's 409 600-kick budget; it had actually been run at 102 400, so
   its true density was 17 kicks per city — a win in the middle of the losing band. Re-running it
   at 409 600 to remove the confound then **flipped it to a loss** (control 0.796% against aiming
   0.835%), because the extra budget let uniform kicking catch up. Density does not enable
   aiming; it erodes it.

The common failure in all three is fitting an explanation to a covariate that happened to
correlate on the instances measured so far. The plateau criterion is the first one with a
mechanism behind it rather than a correlation, and it is the one the next section demonstrates
directly.

**The mechanism, visible directly on fl1577.** Uniform kicking plateaus and stays there:

| kicks | control | aimed |
|---|---|---|
| 6 400 | 4.504% | 2.184% |
| 25 600 | 4.463% | **0.840%** |
| 102 400 | 4.459% | 0.840% |
| 409 600 | 4.459% | 0.787% |

The control returns *the same tour* at 102 400 and 409 600 kicks — sixteen times the budget of
the 25 600 point and not one improvement. `fl1577` is a clustered drilling instance whose
difficulty lives in a few specific regions, so a uniformly-sited kick lands almost every time on
a part of the tour that is already as good as this neighbourhood allows. The `EFFORT` base
scores those regions low and spends the perturbation where the tour is weak. **5.3x better tour
at matched budget** — the largest effect any rule base has produced in this document, and the
one §6.1 predicted when it said the plateau might be the accept rule rather than the repertoire.

**Where that leaves us against LKH.** On the hard instances there is a real, wide
non-dominated window rather than a hundredth-of-a-second sliver:

| instance | n | headroom | LKH's best | our best | non-dominated window |
|---|---|---|---|---|---|
| fl1577 | 1577 | 260 | 0.058% | 0.787% | 0.01–0.84 s |
| **d2103** | 2103 | 195 | 0.004% | **0.087%** | **0.02–31.68 s** |
| fl3795 | 3795 | 108 | 0.142% | 3.052% | 0.03–8.57 s |
| rl5915 | 5915 | 69 | 0.006% | 0.835% | 0.06–11.20 s |
| brd14051 | 14051 | 29 | 0.015% | 0.918% | 0.13–1.88 s |
| d15112 | 15112 | 27 | 0.034% | 0.469% | 0.16–2.19 s |
| d18512 | 18512 | 22 | 0.021% | 0.531% | 0.21–2.33 s |

**`d2103` is the result and the rest is context.** There we are non-dominated over
**0.02–31.68 s** and reach 0.087%, a shorter tour than anything LKH produces below 39 s — a
genuine win against a properly configured LKH, on a held-out instance the literature classifies
as one LKH cannot solve, with a rule base fitted elsewhere. `rl5915` and `fl3795` have windows
of seconds for the same reason: LKH needs alpha-nearness there and the 1-tree is expensive.

**The three largest are not that.** Their windows are about two seconds wide and LKH reaches
0.015–0.034% — twenty times better than our best — shortly after. Whatever advantage exists on
this set does not survive to n ≥ 14 000 at these budgets.

**What it is not.** Seven instances, one seed, one machine. LKH's *failures* here are
0.004–0.142% given enough time, still far better than our best on six of seven. The claim is
bounded: in a window of seconds to tens of seconds, on instances where LKH's alpha-nearness
preprocessing is expensive *and* the kick budget is dense, an `EFFORT`-aimed perturbation reaches
tours LKH has not reached yet.

#### The 2x2 on the easy instances: aiming does not help there

The same factorial on the six easy instances, at each instance's matched budget, is in §6.3's
table. Aimed kicks win on **one of six** — fnl4461, by 0.010 — and are worst or near-worst on
three. On instances LKH solves outright, the perturbation has little to aim *at*: the tour is
already near-optimal almost everywhere, so a rule base that ranks cities by how much there is
to find is ranking noise, and it charges one `EFFORT` evaluation per city for the privilege.

Two earlier readings of this table were wrong and are worth recording, because each was a
plausible story fitted to too little data:

* **"Aiming does not work."** Written from the four instances below n = 2400, where it is
  exactly what the data says. The mechanism I proposed for it — that a city with improving
  moves still available is one the local search has not finished with, so the wrong place to
  kick — is contradicted by §6.3 and should not be carried forward.
* **"Aiming works above n ≈ 4000."** Written when fnl4461 and rl5915 were the only large
  instances measured. `fl1577` (n = 1577) and `d2103` (n = 2103) are the two largest aiming
  wins in the study, so size was never the variable. Instance hardness was, and the two were
  confounded because the ladder's large instances happened to include the only hard one.

`CHAIN` is roughly neutral throughout: one outright win (rl1323), never the worst arm on the
easy set, and consistently the *cheapest* — at rl5915 it reaches 1.111% in 5.78 s against the
control's 1.344% in 11.04 s, which is the §3.3 amortisation argument showing up in the
perturbation loop. On the hard instances it does not help, and combining it with aiming is
worse than aiming alone on all four.

**What the factorial cost to learn.** The first version of this measurement bundled aiming and
chain control into one arm. It was faster and worse than the control, with no way to say which
half did which — and on the hard set it would have hidden the result entirely, since
`iterated_fis` is not the best arm on any of the four. The factorial doubles the runtime of our
own arms, which is a rounding error next to LKH's, and it is the difference between a result
and an anecdote.

**Scale.** Ten instances, n = 783…5915, one machine, one seed. `lkh_compare.py --ladder` and
`--ladder-hard` reproduce the two halves.

---

### 6.4 Absolute standing: LKH as a yardstick

LKH (via `elkai`) completed 9 of the 10 test instances it was given (it timed out on
fl1577): **exactly optimal on 7, and within 0.03% on the other two** — but taking up to 80 s
on d2103, where every arm here finishes that instance in under 0.03 s. LKH is a yardstick, not a competitor this work
claims to beat — it implements a genuine 5-opt sequential search with alpha-nearness
candidate sets and backbone-guided restarts. The relevant comparison here is against the
*same* LK step under fixed parameters, which is what §5 reports.

---

## 7. What does not work

Negatives are worth as much as the positives here and are easier to lose track of, so this is
the index. Two are detailed below; the rest live where they were measured.

| | verdict | where |
|---|---|---|
| **fuzzy construction** (`CONSTRUCT`) | fails — ties nearest-neighbour, loses to greedy-edge, costs 7x NN's runtime. No longer fitted, in no reported arm | below |
| **deferred verification** | works at n ≥ 5000 and is the worst arm overall; cannot amortise a second pass below n = 1000 | below |
| **aiming kicks on easy instances** | best arm on 1 of 6; the payoff signal has nothing to aim at when the tour is already near-optimal everywhere | §6.3 |
| **combining aiming with `CHAIN`** | worse than aiming alone on all 4 hard instances; `iterated_fis` is the best arm on none of them | §6.3 |
| **the larger rule base** | worse overall than the small one, better only above n ≈ 5000, and under-fitted at a matched budget | §4.3 |
| **more fitting budget** | training score improves monotonically while validation degrades: 380 evals → 0.9997, 4000 evals → 1.069 | §4.1 |
| **family-specific fitting** | costs −0.0006 against noise ±0.0006; on 2 of 4 families the family's own fit is worse on it than a foreign one | §4.2 |
| **Cython instead of numba** | 0.87x on the hottest kernel, with identical output | §8 |
| **thread parallelism in the tuner** | numba holds the GIL; `n_jobs=4` measured 18.0 s against 17.2 s for `n_jobs=1` | §4.1 |

**Fuzzy construction fails.** The `CONSTRUCT` ranker reaches 22.60% over optimum against
nearest-neighbour's 22.69% and greedy-edge's 16.95% — it ties the trivial construction and
loses clearly to the good one, while costing seven times nearest-neighbour's runtime. Feeding
it to an unmodified LK gives q = 1.0084 against 1.0029 for the best fixed configuration from a
greedy-edge start. Local search erases most construction differences, so a better construction
has little to win while a worse one still costs its own runtime.

(An earlier run reported 48% here. That figure was the *GA-fitted* construction consequents,
which were far worse than the hand-written ones; `CONSTRUCT` is no longer fitted, so the number
above is the hand-written design being measured on its own. The conclusion did not change —
it was already losing to greedy-edge.)

**Deferred verification is the interesting one.** Running the cheap fuzzy schedule first and
then a full-effort pass over the cities it touched is the arm that reaches q = 0.9994 at
n ≥ 5000 (§5), the only sub-1.0 figure in the study — and it is also the *worst* fuzzy arm on
the full test set (1.0057), because below n = 1000 it pays for a second pass it cannot amortise
(1.0163). It is the sharpest illustration of the size dependence in the whole set of results.
Its guarantee is real and worth keeping: the verification pass can only find additional
improvements, so its tour is never longer, which `test_invariants.py` asserts.

---

## 8. Where the time goes, and why Cython would not take any of it back

An ordinary profiler is useless on this solver, and the reason is worth stating before any
number below is trusted. A whole solve is **one** `@njit` call: `lk_solve` and `iterated_lk`
enter nopython mode once and do not return to the interpreter until the tour is finished.
`cProfile` therefore reports a single entry taking 100% of the time and `line_profiler` sees
nothing at all. `experiments/profile_kernels.py` attributes the time three ways instead, each
checking the others.

### The profile

The fitted cost model (§3.4) applied to a real solve of pr2392 — the only method here that
measures the kernels *in situ*, contending for the same caches as everything else:

| | baseline LK | FIS `EFFORT`+`CHAIN` |
|---|---|---|
| reversal element swaps | **44.8%** | **29.7%** |
| candidate evaluations | 26.7% | 12.6% |
| chain levels entered | 20.5% | 9.9% |
| city scans | 6.1% | 16.2% |
| `CHAIN` decisions | — | 14.7% |
| `EFFORT` decisions | — | 11.4% |
| accepted moves | 1.9% | 5.4% |
| measured wall clock | 17.6 ms | **7.4 ms** |

Two things fall out of it.

**Segment reversal is the largest single cost in both arms**, at 45% and 30% — larger than the
inference the rule bases were built to make cheap, and larger than the candidate evaluation
that the search's whole structure is organised around. It is also the cost §9's third bug
already flagged as invisible to move counts. Any further work on speed belongs here, not in
the rule bases: a doubly-linked-list or two-level-list tour representation removes it more or
less entirely, and that is a bigger lever than anything in §4.

**The rule bases together account for 26% of the fuzzy arm's time and buy a 2.4x speedup.**
That is the honest framing of §3.3's overhead: it is real, it is the mechanism behind the size
dependence, and it is paid for several times over by the work it avoids. The ablation agrees
independently — `EFFORT` alone runs at 0.76x the baseline's time and `EFFORT`+`CHAIN` at
0.43x.

**A calibration note.** The shipped coefficients over-predict this machine by 1.58–1.69x, and
by close to the *same* factor on both arms. A near-uniform scale error is what a cost model
fitted on other hardware looks like, and it is harmless for the job it has: the tuner ranks
candidates rather than reading absolute times off it, which is what the fit's 0.9995 rank
correlation measures. The shares above are unaffected; the totals should be recalibrated by
re-running `costmodel.py` before anyone quotes them as absolute.

### Cython: measured, and it loses — so numba stays

The question was never whether Cython beats Python; nothing in the hot path is Python. It is
whether Cython beats **numba**, which already emits LLVM-optimised machine code with bounds
checking off. A faithful C transcription of `fis_eval1` — the smallest and hottest kernel in the
system, the chain cut-off — with typed memoryviews, `boundscheck=False`, `wraparound=False`,
`cdivision=True`, `-O3 -ffast-math -march=native`, and the benchmark loop *inside* the C
function so neither side paid a per-call boundary cost:

| | ns per call |
|---|---|
| numba `@njit` | **54.0** |
| Cython + gcc `-O3 -march=native` | 61.9 (0.87x) |

Outputs agreed exactly. **Cython was 13% slower**, on the kernel most favourable to it, with the
algorithm held fixed so the only variable was code generation. Porting the solver would have
cost a build toolchain, a compilation step and platform-specific artifacts, in exchange for a
loss — so the decision is numba, and the Cython harness has been deleted rather than left to
rot. This section is the record; `experiments/profile_kernels.py` still reports the two
measurements that explain the result.

Those two make it clear there was never much room. Membership evaluation — the part that was an
exponential before §3.3 tabulated it — is now **17.6 ns of the 54**, so a third of the kernel is a
table lerp neither compiler can improve on. And one Python-level call into nopython mode costs
**~375 ns**, seven times the kernel itself; the only place Cython could have won was on boundary
crossings, and there is one per *solve*, not one per call.

The measurement that would change this answer is not a different compiler. It is removing the
reversal cost, which is 45% of the baseline and belongs to the tour representation rather than
to any kernel's code generation.

---

## 9. Four bugs worth recording

Each produced plausible results rather than a crash, which is why `test_invariants.py` now
holds them.

1. **Or-opt gain accounting.** Relocating a segment is three reversals composed via
   `(AB)^R = B^R A^R`; getting one wrong yields a valid tour of the wrong length. Every
   accepted move's claimed gain is now checked against the real change in tour length, and
   `pos` is re-derived and compared — a stale position index corrupts every *later* move
   rather than this one.

2. **Candidate-list tie order depended on k.** TSPLIB instances are full of exact ties, and
   integer rounding creates many more. Three things were needed, each found by the previous
   fix failing: a deterministic key (the k-d tree's own order is arbitrary); enough queried
   neighbours that a tie group straddling the k-th place is fully seen, since otherwise the
   returned *set* is k-dependent even when the key is not; and *the same key the exact-scan
   fallback uses*, or the two paths disagree whenever the fallback fires. Nearest-neighbour
   tour length is now exactly k-invariant, and the lists are verified against brute force.

3. **The fix for (2) cost a factor of 90, and no correctness check could see it.** Widening
   the query until no row had a straddling tie is almost never satisfiable — 933 of rl5915's
   5915 rows have one, the coordinates being grid-like — so it widened until it was fetching
   every city. Candidate building went from 60 ms to seconds and, since both arms are charged
   for it, rl5915's LK arm read 17.4 s against its true 0.20 s. The lists it produced were
   *correct*, merely ruinous to produce. Contested rows are now settled individually with one
   ball query each, and the tests carry a loose wall-clock ceiling, because that is the shape
   of regression correctness checks are blind to.

4. **numba does not bounds-check.** Adding a fifth input to `CHAIN` left the 4-wide scratch
   buffers in place, so `xc[4]` was a silent out-of-bounds write rather than an exception.
   Buffers are now sized from each rule base's own antecedent array, with an invariant
   asserting the relationship — adding an antecedent is exactly the change that triggers it.

---

## 10. Where this sits in the literature

Full assessment in `references/PRIOR_ART.md`. The summary is that **the substrate is new and the
idea is not.**

An arXiv full-text search for `all:"fuzzy" AND all:"Lin-Kernighan"` returns **zero results** —
no fuzzy rule base has been put inside an LK inner loop. But the conceptual slot, a learned
per-decision controller replacing LK's fixed traversal rules, is occupied by strong work.
**VSR-LKH** (Zheng et al., AAAI 2021) learns which candidate edge to take by Q-learning, over 111
TSPLIB instances up to n = 85 900. **NeuroLKH** (Xin et al., NeurIPS 2021) generates LKH's
candidate sets and node penalties with a graph network. Both beat LKH itself.

Three positioning points follow, and they are the honest ones:

* **`EFFORT` is don't-look bits generalised.** Applegate, Cook & Rohe's chained LK already does
  per-city effort allocation with a boolean rule base, and already restricts kicks to the region
  around breakpoints. What is new is graded, learned and multi-output — not the idea of per-city
  effort, and not the idea of aiming a kick.
* **`CHAIN` is the least-occupied contribution.** VSR-LKH learns *which edge*; nothing found
  learns *whether to go deeper*. Variable-depth termination in LK is a fixed positive-gain rule
  plus a depth cap, and §3.1 finds depth is the parameter worth being clever about.
* **The defensible axis is method cost, not tour quality.** NeuroLKH needs ≈780 000
  Concorde-labelled instances and about four GPU-days, and fine-tunes its node decoder per size
  because the penalties do not generalise. VSR-LKH's own appendix concedes RL's warm-up: "in the
  beginning of the iterations, LKH can yield better solutions than the reinforced algorithms."
  A hand-written rule base with scale-free antecedents needs neither, and §4.2 shows it
  transfers.

**One prior-art item was chased down and does not block §6.3.** "XKICK, an intelligent kick
perturbation for the TSP" (Alfredo Garcia W, 2017) aims perturbations too, but derives its regions
from *comparing multiple tours* and replaces the double bridge with "a more complex change in the
tour". This work scores a *single* tour per city and keeps the double bridge, changing only where
it lands. XKICK is also unrefereed grey literature with no venue and no resolvable citations. The
full text could not be retrieved; the mechanism should be confirmed from the PDF before
publication.

---
## 11. Worth doing next

* **Settle the n ≥ 5000 crossing.** q = 0.9994 on four of seven instances is a suggestion, not
  a result. TSPLIB has few instances that large, but `synth.py` can generate them freely and
  the frontier-relative metric needs no optimum — so a held-out synthetic test set at
  n = 5000…50000, sized for the margin being measured, would settle it. This is the single
  highest-value next step and it is now cheap.
* **Give `large` a budget scaled to its dimension count** (§4.3). At a matched 380 evaluations
  it fits its own training set worse than `small` does, so its n ≥ 5000 advantage is achieved
  while under-fitted. This is the experiment that would decide whether the middling-AUC
  features are worth their runtime, and it is cheap.
* **Ablate the five kept antecedents individually.** §3.2 measured their predictive power in
  isolation but not their marginal contribution to the fitted system; `probe_frac` and `probe`
  are likely to be partly redundant with each other.
* **Replace the tour representation.** §8 makes this the highest-value performance work by a
  wide margin: segment reversal is 45% of the baseline solve and 30% of the fuzzy one, and a
  two-level list removes essentially all of it. Every other speed item on this list is a few
  percent of a cost that reversal dominates.
* **Cut the effort decision below 349 ns.** It is the entire mechanism of the size dependence
  and still ~70% of a city scan. Evaluating `EFFORT` once per region rather than per city, or
  caching it while a city's features have not moved, would lower the crossover — which is worth
  more than any further tuning, because it moves where the engine works rather than how well.
  Not by changing compilers, though: §8 measured Cython at 0.87x numba on the hottest kernel.
* **Settle the aimed-kick crossover, and then push on it.** §6.2's 2x2 gives the largest effect
  in this document — 0.881% in 3.10 s against the control's 1.344% in 11.04 s at n = 5915 — on
  two instances above n = 4000. `synth.py` can generate as many instances in that range as
  patience allows, and the metric needs no optimum, so this is cheap to settle and it is the
  highest-value open question here. If it holds, the aim is currently recomputed once per solve
  from the starting tour; recomputing it periodically as the tour changes is the obvious next
  move, and the same feature screen of §3.2 applies to "how stuck is this region", which is a
  different label from the per-city payoff the aim currently reuses.
* **Stop selecting on validation** (§4.1). A nested split, or a budget fixed a priori with no
  accept/reject gate, would make the reported validation numbers unbiased instead of
  optimistic.
* **Try the payoff prediction directly.** §3.2's reframing was used only to screen features, but
  it suggests a different architecture: predict payoff with the rule base, then map predicted
  payoff to effort with a single monotone curve. That is far fewer parameters than a rule base
  per LK parameter, and the screening data is already the training set for it.

---

## 12. Reproducing

```bash
pip install numpy scipy matplotlib numba elkai
pip install -e ../tribble-opt

python run_all.py            # every stage in dependency order, into results/
python run_all.py --list     # what those stages are and what each one writes
python run_all.py --dry-run  # the same, plus what the LKH stage would cost
```

`run_all.py --ladder` adds the full LKH size ladder, which is hours rather than minutes and is
off by default; `--dry-run` prices it first. Every stage is also a standalone script with
`--help`, and each writes into `results/`.

`costmodel.py` must run before `tune_opt.py`, and again after any change to the solver's hot
path — its fitted coefficients are what the objective spends. `test_invariants.py` runs first,
because four of the bugs in §9 produced plausible numbers rather than crashes.
