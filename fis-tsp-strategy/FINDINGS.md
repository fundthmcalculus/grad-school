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

**Where this stands.** Adaptive effort is competitive with a swept LK, and its standing
improves monotonically with instance size: over the whole test set no fuzzy arm beats the
best fixed LK configuration, but at n ≥ 2000 the best fuzzy arm is the best arm measured,
and below n ≈ 1000 it is plainly worse. The margins at large n are smaller than this
measurement can resolve; the size trend is much larger and is the real result. Fitting the
rule bases with a genetic algorithm improves them on validation and makes them worse on
test. Fuzzy construction fails.

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
| `lk_32_6_32` — best fixed LK | 3.485% | 2.99 | **1.0035** |
| `lk_16_4_8` | 4.598% | 3.54 | 1.0040 |
| `lk_32_2_4` | 4.048% | 1.86 | 1.0045 |
| **FIS effort, hand-written** | 3.649% | 2.82 | **1.0049** |
| **FIS effort + chain, hand-written** | 3.711% | 2.33 | **1.0050** |
| `lk_32_3_8` | 3.876% | 2.16 | 1.0051 |
| `lk_48_10_32` | 3.305% | 4.59 | 1.0055 |
| FIS effort, GA-fitted | 3.791% | 3.27 | 1.0067 |
| FIS + deferred verification | 3.925% | 2.41 | 1.0075 |
| FIS effort + chain, GA-fitted | 4.415% | 1.92 | 1.0096 |
| FIS + fuzzy construction | 6.346% | 2.47 | 1.0299 |

Over the whole test set **no fuzzy arm beats the best fixed LK configuration.** The
hand-written effort arm is better than 8 of the 11 fixed configurations and worse than 3 —
a good operating point chosen without being told which one to pick, but not a new frontier.

### The size dependence is the finding

| arm | n < 1000 (5) | 1000–5000 (8) | n ≥ 5000 (7) |
|---|---|---|---|
| FIS + deferred verification | 1.0191 | 1.0065 | **1.0002** |
| FIS effort + chain, hand-written | 1.0090 | 1.0058 | **1.0013** |
| FIS effort, hand-written | 1.0098 | 1.0035 | 1.0029 |
| `lk_32_2_4` | 1.0084 | 1.0058 | 1.0002 |
| `lk_48_10_32` | 1.0169 | 1.0028 | 1.0005 |
| `lk_32_6_32` | **1.0042** | 1.0026 | 1.0040 |
| `lk_32_3_8` | 1.0143 | 1.0021 | 1.0019 |

Ranked over the 10 instances with n ≥ 2000, the fuzzy arm with a deferred verification pass
is the best arm measured:

| arm (n ≥ 2000) | mean gap | total s | q |
|---|---|---|---|
| **FIS + deferred verification** | 3.440% | 2.32 | **1.0009** |
| `lk_32_2_4` | 3.864% | 1.79 | 1.0011 |
| `lk_32_3_8` | 3.600% | 2.09 | 1.0014 |
| `lk_32_6_8` | 3.433% | 2.83 | 1.0015 |
| `lk_32_4_8` | 3.598% | 2.32 | 1.0018 |
| `lk_48_10_32` | 3.190% | 4.46 | 1.0022 |
| FIS effort + chain, hand-written | 3.645% | 2.25 | 1.0027 |

**These margins are not resolvable.** 1.0009 against 1.0011 is two parts in ten thousand of
tour length over ten instances, with timing taken as the best of three runs. The defensible
statement is that at n ≥ 2000 the fuzzy arm is *among the best*, and that this is a reversal
of its standing below n = 1000, where it is clearly worse. The size trend is an order of
magnitude larger than the between-arm differences, and §4 gives its mechanism.

---

## 3. What the rule bases do

Averaged over city searches on the test instances, `EFFORT` and `CHAIN` run the solver at
**mean chain depth 4.41** where the baseline is pinned at 10, and **mean first-level breadth
16.5** where the baseline is 32 — reaching a tour within ~0.2 points of a fixed LK that
spends twice as long.

That *depth* is the parameter worth being clever about was not the obvious guess. Sweeping
first-level breadth from 2 to 32 barely moves the clock: the sequential positive-gain
criterion truncates most candidate scans long before the breadth cap bites. Sweeping chain
depth from 4 to 10 costs 2.6×. So the rule bases earn their keep by deciding *which cities
deserve a deep chain*, and `CHAIN` — which cuts a chain mid-flight from its own gain
trajectory rather than at a fixed depth — is the most valuable of the three.

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
§2's size dependence. It is a fixed cost per city, so it is amortised only once each city's
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

## 6. Fitting the rules: what the GA bought, and what it cost

`tune_opt.py` fits 170 parameters — 152 rule consequents plus membership centres and widths —
with the `optimizers` library's genetic algorithm, over 9 training and 7 validation
instances, all n ≥ 1000.

**Triangular membership functions beat gaussian on every run that compared them** (validation
q 1.017 against 1.069 at matched budget) and are cheaper to tabulate. They are the default.

**The GA improves the rules where it can see them, and not where it cannot:**

| | validation q | test q |
|---|---|---|
| hand-written | 1.2026 | **1.0049** |
| GA-fitted | **1.0571** | 1.0067 |

It closes most of the validation gap and then loses on test. The reason is not mysterious:
selecting the best of 24 pooled candidates *on* validation makes validation part of the
fitting procedure, so its score stops being unbiased. Only the test set is untouched, and it
says the fit did not transfer.

### More search makes generalisation worse

| GA evaluations | training q | validation q |
|---|---|---|
| 140 | — | **0.990** |
| 1440 | 0.9938 | 1.057 |
| 2720 | 0.9188 | 1.017 |
| 4000 | 0.9520 | 1.069 |

Training score improves throughout while the answer degrades. With 170 parameters and 9
training instances that is the expected outcome, and the default budget is deliberately
modest because of it. (The 140-evaluation run's 0.990 is the only sub-1.0 validation figure
seen and it is a single short run, not a reproducible setting.)

Four defences were added, each becoming visible only after the previous was fixed:

1. **More and larger instances.** An early split with 8 of 20 instances under n = 200 made
   the optimiser chase per-city overhead — the one cost that only matters where this engine
   is the wrong tool. It produced better tours at 1.26× the cost. Fitting is now n ≥ 1000.
2. **A shrinkage prior toward the hand-written rules**, which encode what §3 measured. Better
   motivated than a plain norm penalty, and the reason the fitted vectors stay interpretable.
3. **Selection from a pool, on validation.** Scoring validation only at successive
   training-bests fails: once the two decouple, every training improvement lies further along
   the overfitting path, so that is the only region validation judges. A 684-evaluation run
   selected that way returned 4.21% where a 252-evaluation run of the same optimiser found
   3.75%.
4. **A smaller parameter space.** One shared set of three terms per rule base (18 membership
   parameters) rather than per-input terms (72); the expressive version overfits harder.

All four help. None is sufficient.

### Practical notes on the optimiser

* **Thread parallelism buys nothing.** numba holds the GIL: `n_jobs=4` measured 18.0 s
  against 17.2 s for `n_jobs=1`. The deterministic proxy of §5 was built partly to enable
  parallel evaluation, which turned out to be unavailable for an unrelated reason.
* **Evaluation cost depends on the candidate.** A vector telling the rule bases to use full
  depth and breadth everywhere makes the solver several times slower than the baseline, and
  bound-seeking optimisers walk into that region: one PSO generation cost 99 s against the
  GA's 12.5 s. Evaluation now walks instances in increasing size and abandons a candidate
  once it exceeds twice the dearest baseline configuration's cost, ranked by how far over it
  ran so the search retains a gradient. PSO then completed 720 evaluations in 73 s.
* GA is the reported optimiser; PSO and ACO drivers remain selectable but are not part of the
  comparison.

---

## 7. Three antecedents added

| input | what it reads | cost |
|---|---|---|
| `E_RANK` | how many strictly nearer neighbours the city's worse tour edge is ignoring | short scan of an ascending list |
| `E_PEAK` | nearest-neighbour distance ÷ mean candidate distance | two loads and a divide, both precomputed |
| `CH_REVCOST` | how much array this chain level's reversal moved | free — `reverse` already returns its swap count |

`E_RANK` is the most direct cheap statement of "is there anything here to find", and unlike
the existing excess feature it is scale-free without dividing by anything: it counts how many
strictly better neighbours exist rather than measuring how much longer the edge is.
`CH_REVCOST` is the only input taken from the cost model rather than from the search's own
logic — reversal traffic is separately priced at 4.4 ns/element, and a chain working where
every level shuffles half the tour is expensive in a way no gain number reveals.

`EFFORT` is now 6 inputs over 27 rules, `CHAIN` 5 over 24. **These have not been ablated
individually.** They are present in every arm in §2, but which of the three earns its keep is
unmeasured.

---

## 8. Two components that do not work

**Fuzzy construction fails.** The `CONSTRUCT` ranker reaches 48.2% over optimum against
nearest-neighbour's 22.7% and greedy-edge's 17.0%, and feeding it to an unmodified LK is
worse than feeding greedy-edge (q 1.0125 against 1.0035). Local search erases most
construction differences, so a better construction has little to win while a worse one still
costs its own runtime.

One caveat on that figure: those consequents were hand-written against gaussian terms, and
switching the default to triangular roughly doubled the construction gap (25% → 48%) while
barely moving `EFFORT` and `CHAIN`. That is itself evidence the construction consequents were
fitted to the old term shapes, so 48.2% overstates how bad the design is. The conclusion is
unchanged: it was already losing to greedy-edge under gaussian terms.

**Deferred verification is an interesting half-success.** Running the cheap fuzzy schedule
first and then a full-effort pass over the cities it touched is the arm that leads at n ≥ 2000
(§2) — but it leads by *spending more*, and it costs quality against the plain fuzzy arm on
small instances (q 1.0191 below n = 1000). Its guarantee is real and worth keeping: the
verification pass can only find additional improvements, so its tour is never longer, which
`test_invariants.py` asserts.

---

## 9. Absolute standing: LKH

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

## 11. Worth doing next

* **Ablate the three antecedents in §7.** They are in every measured arm; their individual
  contributions are unknown.
* **Settle whether the n ≥ 2000 lead is real.** Two parts in ten thousand over ten instances
  does not support a stronger claim than "among the best". More instances in that band and
  more timing repetitions would decide it.
* **Cut the 349 ns effort decision further.** It is the entire mechanism of the size
  dependence and still ~70% of a city scan. Evaluating `EFFORT` once per region rather than
  per city, or caching it while a city's features have not moved, would lower the crossover.
* **Stop selecting on validation** (§6). A nested split, or fitting on all 16 instances with
  the budget fixed a priori, would give an unbiased estimate rather than an optimistic one.

---

## 12. Reproducing

```bash
pip install numpy scipy matplotlib numba elkai
pip install -e ../tribble-opt

python test_invariants.py                              # correctness first
python costmodel.py                                    # writes costmodel.npz
python tune_opt.py                                     # writes tuned_opt.npz
python benchmark.py --reps 3 --tuned tuned_opt.npz     # writes results.json
python lkh_reference.py --max-n 2500 --timeout 90
python figures.py && python figures_tuning.py
```

`costmodel.py` must run before `tune_opt.py`, and again after any change to the solver's hot
path — its fitted coefficients are what the objective spends.
