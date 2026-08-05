# A fuzzy inference system as a TSP strategy engine — findings

A Lin-Kernighan solver spends its time unevenly and its parameters uniformly. This work
replaces three of LK's fixed rules with fuzzy inference systems and measures what that
buys:

* **`CONSTRUCT`** — which city to visit next, instead of "the nearest unvisited".
* **`EFFORT`** — how hard to search each city, instead of one setting everywhere.
* **`CHAIN`** — whether to deepen or cut each gain chain, instead of stopping at a fixed depth.

Both arms call the same `lk.py`. The baseline *is* that module with `use_chain` off and
constant parameters, so the two cannot differ in move repertoire or in the speed of their
arithmetic — only in strategy. Every reported tour is checked to be a permutation and
re-scored from the coordinates under TSPLIB rounding, independently of the solver's own
bookkeeping.

**Where this stands.** Adaptive effort's standing improves monotonically with instance size,
and at the top of the range it crosses the frontier: at n ≥ 5000 the best arm reaches
**q = 0.9991** — a shorter tour than every LK configuration spending the same wall clock —
below 1.0 on five of seven instances. Over the whole test set, half of which is small
instances, no fuzzy arm beats the best fixed LK. The size trend is the reportable result; the
margin at the top is nine parts in ten thousand, which this measurement cannot establish as
real.

Fitting now transfers to unseen instances, which it previously did not. What did it: selecting
antecedents by measured predictive power rather than by argument, halving the rule count, and
generating synthetic instances to quadruple the training pool.

**Rule-base size trades against instance size.** The larger rule base (8 inputs, 55 rules, 157
parameters) is *worse overall* than the small one (5 inputs, 30 rules, 87) and *better above
n = 5000*. Both halves of that follow from the same mechanism, and neither is a tuning
accident. Fuzzy construction still fails.

**Against LKH the answer is no, and it is not close.** §9b puts the FIS itself on LKH's axes
for the first time — the earlier comparison ran with the fuzzy engine switched off, on two
instances the rule base is fitted on. On four held-out instances LKH reaches the published
optimum *exactly* at its cheapest available setting, and no arm here beats it at any budget.
What we have is a speed window below LKH's floor, which is partly an artefact of `elkai`'s
interface. Aiming perturbations with `EFFORT` — the obvious next use of the rule base — is
measured as a 2x2 factorial and **does not work**.

**Where the time actually goes is not where this work spent its attention.** Segment reversal
is 45% of the baseline solve and 30% of the fuzzy one (§11), larger than the inference §4 was
written to make cheap. Cython was measured against numba on the hottest kernel and is 13%
*slower*.

---

## 1. How this is measured

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

## 2. The result

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
instances are below n = 1000, where §4 explains why the fuzzy arms cannot win.

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
instances, at the **`large`** scale (§3b — the better choice in this band):

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

## 3. What the rule bases do

Averaged over city searches on the test instances, `EFFORT` and `CHAIN` run the solver at
**mean chain depth 6.4** where the baseline is pinned at 10, and **mean first-level breadth
25.0** where the baseline is 32.

The rule base spends more than the previous one did (depth 4.4, breadth 16.5) and gets a
better result, which is the depth-1 probe of §7 doing its job: with a look-ahead saying
whether gain is visible at all, the base can afford to go deep where it is rather than
staying globally timid.

That *depth* is the parameter worth being clever about was not the obvious guess. Sweeping
first-level breadth from 2 to 32 barely moves the clock: the sequential positive-gain
criterion truncates most candidate scans long before the breadth cap bites. Sweeping chain
depth from 4 to 10 costs 2.6×. So the rule bases earn their keep by deciding *which cities
deserve a deep chain*, and `CHAIN` — which cuts a chain mid-flight from its own gain
trajectory rather than at a fixed depth — is the most valuable of the three.

## 3b. Does a bigger rule base help? Only at the top of the size range

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
exactly the same amortisation curve as §4's inference overhead — it hurts wherever the work per
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

## 4. Making inference cheap enough to be worth consulting

A rule base consulted once per city scan must cost far less than the scan. It did not.
`costmodel.py` (§5) priced the pieces: a chain-continuation decision cost **494 ns** against
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
§2's size dependence, and the depth-1 probe of §7 adds a little more on top of it. It is a fixed cost per city, so it is amortised only once each city's
search does enough work to hide it — which is why the fuzzy arm loses below n ≈ 1000 and
leads above n ≈ 2000, and it says where the crossover has to be rather than leaving it as an
empirical curiosity.

---

## 5. A deterministic cost proxy, so the search can be reproducible

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

## 6. Fitting the rules: GA, then derivative-free descent

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

**3. Antecedents chosen by measured predictive power** rather than by argument (§7).

**The result:**

| | validation q | test q, n ≥ 5000 |
|---|---|---|
| hand-written | 1.0042 | 1.0048 |
| GA-fitted | **0.9997** | **1.0014** |

Fitting is now better than the hand-written rules on unseen data, which reverses the earlier
finding. It is better in the bands it was fitted for (n ≥ 1000) and clearly *worse* below
n = 1000 — 1.0182 against 1.0032 — which is the expected consequence of restricting fitting to
n ≥ 1000 and of the probe features costing per-city time that only amortises at scale.

### The second stage: derivative-free stepwise refinement

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

### More search still makes generalisation worse

| GA evaluations | training q | validation q |
|---|---|---|
| 140 | — | 0.990 |
| 380 (current) | 1.0033 | **0.9997** |
| 1440 | 0.9938 | 1.057 |
| 4000 | 0.9520 | 1.069 |

Training score improves monotonically while the answer does not. The default budget is small
deliberately, and the polish is gated on validation rather than trusted.

### What remains unsound

Selection still runs through validation — the best of 24 pooled candidates is *chosen* on it,
and so is the accept/reject decision on the polish. That makes validation part of the fitting
procedure and its numbers optimistic. Only the test set is untouched, and §2 is what it says.

### Practical notes

* **Thread parallelism buys nothing.** numba holds the GIL: `n_jobs=4` measured 18.0 s against
  17.2 s for `n_jobs=1`. The deterministic cost proxy of §5 was built partly to enable
  parallel evaluation, which turned out to be unavailable for an unrelated reason.
* **Evaluation cost depends on the candidate.** A vector telling the rule bases to use full
  depth and breadth everywhere makes the solver several times slower than the baseline, and
  bound-seeking optimisers walk into it: one PSO generation cost 99 s against the GA's 12.5 s.
  Evaluation now abandons a candidate once it exceeds twice the dearest baseline
  configuration's cost, ranked by how far over so the search keeps a gradient. PSO then
  completed 720 evaluations in 73 s.
* GA is the reported optimiser; PSO and ACO drivers remain selectable but are not part of the
  comparison.

## 7. Choosing antecedents by measurement, not by argument

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

### The master table

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
feature earns its runtime — §2 and §3 are where that is tested, and the depth/breadth increase
in §3 is the evidence that the probe changed behaviour rather than just adding cost.

## 8. Two components that do not work

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
n ≥ 5000 (§2), the only sub-1.0 figure in the study — and it is also the *worst* fuzzy arm on
the full test set (1.0057), because below n = 1000 it pays for a second pass it cannot amortise
(1.0163). It is the sharpest illustration of the size dependence in the whole set of results.
Its guarantee is real and worth keeping: the verification pass can only find additional
improvements, so its tour is never longer, which `test_invariants.py` asserts.

## 9. The move repertoire was the binding constraint all along

Every section above tunes *how much* search to spend and *where*. None of it changes **what
moves exist**, and that turns out to have been the ceiling the whole time.

The repertoire was sequential 2-opt chains plus Or-opt relocation. Both are improving moves, so
the search converges to a local optimum of that neighbourhood and stops. The rule bases can
decide how fast to get there and how cheaply; they cannot decide to leave. The evidence that
this bound everything is in the sweep itself: on pr1002 **every** fixed configuration tops out
at 2.473%, and spending more time — `k32/d10/b32`, `k48/d10/b32` — buys nothing or loses. That
is not a tuning plateau, it is the neighbourhood being exhausted.

### The double bridge

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

### What it bought

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

### Against LKH: a crossover, not a win

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

### The asymptote is the next binding constraint, and it is not the scheduling

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

### 9b. Putting the FIS itself against LKH — and the answer is no

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

Test instances, `small` scale, wall clock, best of 3:

| instance | n | LKH's floor | LKH's best | our best | at | beats LKH? |
|---|---|---|---|---|---|---|
| rat783 | 783 | 1.9 s | **0.000%** | 0.477% | 3.9 s | no arm |
| pcb1173 | 1173 | 3.1 s | **0.000%** | 0.336% | 4.7 s | no arm |
| rl1323 | 1323 | 4.8 s | **0.000%** | 0.526% | 4.0 s | no arm |
| pr2392 | 2392 | 23.5 s | **0.000%** | 0.988% | 6.3 s | no arm |

**No arm beats LKH anywhere, on any instance, at any budget.** LKH reaches the published
optimum exactly at its cheapest available setting on all four. There is no quality at which
we are competitive and no budget above its floor at which we are preferable. The only thing
we have is a *speed* window below that floor — 1.9 s to 23.5 s here — and that window is
partly an artefact of `elkai` taking a run count rather than a time limit. Being the only
solver present is not the same as being better, and §9's table should be read that way too.

### The 2x2: aiming the kicks with `EFFORT` does not work

Each arm read at one common budget per instance — the control's dearest point — because
comparing each at its own best budget compares different amounts of spending.

| instance | n | budget | control | + aimed kicks | + `CHAIN` depth | + both |
|---|---|---|---|---|---|---|
| rat783 | 783 | 3.86 s | **0.477%** | 0.488% | 0.693% | 0.500% |
| pcb1173 | 1173 | 4.66 s | **0.336%** | 0.809% | 0.603% | 0.810% |
| rl1323 | 1323 | 7.43 s | 0.573% | 0.811% | **0.526%** | 0.618% |
| pr2392 | 2392 | 6.15 s | 1.161% | 1.180% | 1.118% | **1.104%** |

**Aiming is never best and is worst or near-worst on three of four.** That is the clearest
negative result in this document, and it is a specific one: the `EFFORT` base predicts where a
*local search* will find an improving move, and §7 says it does that well (AUC 0.858 for the
best antecedent). It does not follow that those are the cities worth *perturbing*, and this
says they are not. The two questions differ in a way that now looks obvious — a city where
improving moves are still available is one the local search has not finished with, whereas a
kick is worth spending where the search has converged to something that needs breaking.

`CHAIN` is roughly neutral and its two wins are both at the larger end, which is consistent
with the amortisation argument of §4 and is not established by four instances.

**What this cost to learn.** The first version of this measurement bundled aiming and chain
control into one arm. It was faster and worse than the control, and there was no way to say
which half did which — on rat783 alone it looked like a win. The factorial doubles the runtime
of our own arms, which is a rounding error next to LKH's, and it is the difference between a
result and an anecdote.

**Scale.** Four instances spanning n = 783…2392, one machine. `lkh_compare.py --ladder` extends
to n = 5915; `--dry-run` prices it first, fitting LKH's cost law to the floors measured on the
machine it is about to run on. On these four that fit is n^2.3 and the full ladder costs about
50 minutes — the n^3.5 recorded in `results/legacy/` was measured on other hardware, which is
why the estimate is derived rather than stored.

---

## 9c. Absolute standing: LKH

LKH (via `elkai`) completed 9 of the 10 test instances it was given (it timed out on
fl1577): **exactly optimal on 7, and within 0.03% on the other two** — but taking up to 80 s
on d2103, where every arm here finishes that instance in under 0.03 s. LKH is a yardstick, not a competitor this work
claims to beat — it implements a genuine 5-opt sequential search with alpha-nearness
candidate sets and backbone-guided restarts. The relevant comparison here is against the
*same* LK step under fixed parameters, which is what §2 reports.

---

## 10. Four bugs worth recording

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

## 11. Where the time goes, and why Cython would not take any of it back

An ordinary profiler is useless on this solver, and the reason is worth stating before any
number below is trusted. A whole solve is **one** `@njit` call: `lk_solve` and `iterated_lk`
enter nopython mode once and do not return to the interpreter until the tour is finished.
`cProfile` therefore reports a single entry taking 100% of the time and `line_profiler` sees
nothing at all. `experiments/profile_kernels.py` attributes the time three ways instead, each
checking the others.

### The profile

The fitted cost model (§5) applied to a real solve of pr2392 — the only method here that
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
that the search's whole structure is organised around. It is also the cost §10's third bug
already flagged as invisible to move counts. Any further work on speed belongs here, not in
the rule bases: a doubly-linked-list or two-level-list tour representation removes it more or
less entirely, and that is a bigger lever than anything in §4.

**The rule bases together account for 26% of the fuzzy arm's time and buy a 2.4x speedup.**
That is the honest framing of §4's overhead: it is real, it is the mechanism behind the size
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
exponential before §4 tabulated it — is now **17.6 ns of the 54**, so a third of the kernel is a
table lerp neither compiler can improve on. And one Python-level call into nopython mode costs
**~375 ns**, seven times the kernel itself; the only place Cython could have won was on boundary
crossings, and there is one per *solve*, not one per call.

The measurement that would change this answer is not a different compiler. It is removing the
reversal cost, which is 45% of the baseline and belongs to the tour representation rather than
to any kernel's code generation.

## 12. Worth doing next

* **Settle the n ≥ 5000 crossing.** q = 0.9994 on four of seven instances is a suggestion, not
  a result. TSPLIB has few instances that large, but `synth.py` can generate them freely and
  the frontier-relative metric needs no optimum — so a held-out synthetic test set at
  n = 5000…50000, sized for the margin being measured, would settle it. This is the single
  highest-value next step and it is now cheap.
* **Give `large` a budget scaled to its dimension count** (§3b). At a matched 380 evaluations
  it fits its own training set worse than `small` does, so its n ≥ 5000 advantage is achieved
  while under-fitted. This is the experiment that would decide whether the middling-AUC
  features are worth their runtime, and it is cheap.
* **Ablate the five kept antecedents individually.** §7 measured their predictive power in
  isolation but not their marginal contribution to the fitted system; `probe_frac` and `probe`
  are likely to be partly redundant with each other.
* **Replace the tour representation.** §11 makes this the highest-value performance work by a
  wide margin: segment reversal is 45% of the baseline solve and 30% of the fuzzy one, and a
  two-level list removes essentially all of it. Every other speed item on this list is a few
  percent of a cost that reversal dominates.
* **Cut the effort decision below 349 ns.** It is the entire mechanism of the size dependence
  and still ~70% of a city scan. Evaluating `EFFORT` once per region rather than per city, or
  caching it while a city's features have not moved, would lower the crossover — which is worth
  more than any further tuning, because it moves where the engine works rather than how well.
  Not by changing compilers, though: §11 measured Cython at 0.87x numba on the hottest kernel.
* **Aim perturbations by something other than `EFFORT`.** §9b's 2x2 says the per-city payoff
  prediction is the wrong signal for where to *kick* — plausibly because a city with improving
  moves still available is one the local search has not finished with, whereas a kick is worth
  spending where it has converged. A "how stuck is this region" signal is a different feature
  screen, and §7's methodology applies to it directly.
* **Stop selecting on validation** (§6). A nested split, or a budget fixed a priori with no
  accept/reject gate, would make the reported validation numbers unbiased instead of
  optimistic.
* **Try the payoff prediction directly.** §7's reframing was used only to screen features, but
  it suggests a different architecture: predict payoff with the rule base, then map predicted
  payoff to effort with a single monotone curve. That is far fewer parameters than a rule base
  per LK parameter, and the screening data is already the training set for it.

## 13. Reproducing

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
because four of the bugs in §10 produced plausible numbers rather than crashes.
