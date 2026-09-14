# What the comparable literature measures, and what it would take to be comparable

Extracted from the papers themselves, not from abstracts or secondary summaries. PDFs are in
this directory (gitignored — they are published papers, re-downloadable from the arXiv IDs).

The short version: **the two papers occupying this project's conceptual slot both beat LKH
itself, on a much larger benchmark, under a time-matched protocol.** Neither the benchmark nor
the protocol resembles what `lkh_compare.py` currently does, and the gap between them is not a
matter of running more instances.

---

## 1. NeuroLKH — Xin, Song, Cao & Zhang, NeurIPS 2021 (arXiv:2110.07983)

### Benchmark

| | |
|---|---|
| **Primary data** | Uniform random points in the unit square. **1000 test instances per size.** |
| **Training sizes** | n = 100, 200, 500 |
| **Generalization sizes** | n = 1000, 2000, 5000 |
| **Training set** | n = 101…500, `500000/n` instances per size, **≈780 000 instances total**; Concorde used to label optimal edges |
| **Secondary data** | all **72 TSPLIB** instances with Euclidean distances and n < 10 000, **10 runs each** |
| **Hardware** | Intel i9-10940X CPU; one RTX-2080Ti for the network |
| **Training cost** | 16 epochs, **≈4 days** on the 2080Ti |

### Protocol — this is the part that matters most

> "For each testing problem size, we run the original LKH for 1, 10, 100, and 1000 trials, and
> record the total amounts of time in solving the 1000 instances. Then we impose the same
> amounts of time as time limits to NeuroLKH and VSR-LKH for solving the same 1000 instances."

So the comparison is **time-matched against LKH at four effort levels**, and NeuroLKH's time
includes GPU inference. Gaps are reported in **‰ (per mille), not per cent** — a detail worth
triple-checking before quoting any of these numbers, since it is a factor of ten.

### The numbers to beat

Gap in ‰ against Concorde; time is the **total for 1000 instances**, so divide by 1000 for
per-instance cost.

| size | LKH 1 trial | LKH 1000 trials | NeuroLKH (matched to LKH 1 trial) |
|---|---|---|---|
| n=100 | 2.353‰ / 33 s | 0.010‰ / 938 s | **0.111‰** |
| n=200 | 5.364‰ / 80 s | 0.031‰ / 2805 s | **0.533‰** |
| n=500 | 9.009‰ / 338 s | 0.063‰ / 7527 s | **0.826‰** |
| n=1000 | 10.593‰ / 1183 s | 0.347‰ / 12884 s | **0.899‰** |
| n=2000 | 11.264‰ / 4843 s | 0.509‰ / 25613 s | **0.755‰** |
| n=5000 | 12.284‰ / 40048 s | 0.455‰ / 103885 s | **0.484‰** |

**Per instance, LKH at one trial is 1.06% in 1.18 s at n=1000 and 1.23% in 40 s at n=5000.**

### The single most useful fact in the paper for us

> "the subgradient optimization in LKH and VSR-LKH needs 20 s, 51 s, 266 s, 1028 s, 4501 s and
> 38970 s" — for 1000 instances at n = 100, 200, 500, 1000, 2000, 5000.

Per instance: **0.02 s, 0.05 s, 0.27 s, 1.03 s, 4.5 s, 39.0 s.** That is the alpha-nearness
preprocessing, and at one trial it is essentially the *entire* cost — 39 s of the 40 s at
n = 5000. It confirms independently what `lkh_compare.py` found by sweeping
`CANDIDATE_SET_TYPE`: **LKH's floor is its preprocessing, and the preprocessing is optional.**

NeuroLKH's own SGN inference is 0.003–0.208 s per instance over the same sizes — i.e. it
*replaces* a superlinear preprocessing step with a near-linear one, which is a large part of
where its advantage at short time limits comes from. The paper says so directly: "the
improvement of NeuroLKH over baselines is particularly substantial for time-critical
applications."

---

## 2. VSR-LKH — Zheng, He, Zhou, Jin & Li, AAAI 2021 (arXiv:2012.04461)

### Benchmark

| | |
|---|---|
| **Data** | **all 111 symmetric TSPLIB instances**, up to n = 85 900 |
| **Runs** | **10 per instance** (3 for pla33810 and pla85900, capped at 100 000 s per run) |
| **Split** | **74 "easy"** — both LKH and VSR-LKH reach the optimum with no runtime difference; **37 "hard"** |
| **Metric** | gap to the optimum over time, single run; cumulative gap across instances |

### What it claims

On the 37 hard instances VSR-LKH "greatly promotes the performance of the state-of-the-art
algorithm LKH". On the 74 easy ones there is nothing to report — both are optimal.

One remark in the appendix is directly relevant to us:

> "In the beginning of the iterations, LKH can yield better solutions than the reinforced
> algorithms. This is because the trial and error characteristics of reinforcement learning."

A rule base fitted offline has no such warm-up cost. That is a real, narrow, defensible
advantage over the RL line — and it lives exactly in the short-time-limit regime this project
has been measuring in.

---

## 3. What this means for this project

### The benchmark we use is the wrong one, and is the *easy* half

Our 20-instance test set is a subset of VSR-LKH's 111, and almost all of it falls in their
**74 "easy"** bucket — the instances where LKH is simply optimal. That is exactly what
`lkh_compare.py` measures: LKH exactly optimal on six of seven, 0.006% on the seventh. We have
been benchmarking on the instances the field set aside as uninformative.

### The protocol is not comparable

| | this project | NeuroLKH / VSR-LKH |
|---|---|---|
| instances | 7–20 TSPLIB | 1000 generated per size, plus 72–111 TSPLIB |
| repetitions | best of 2–3 | 10 runs per instance |
| comparison | our budget vs LKH's budget, read off two curves | **time-matched**: LKH's measured time imposed as a limit on the challenger |
| gap units | % | **‰** |
| baseline LKH | `elkai` defaults, then a parameter sweep | LKH 3.0.6 with the authors' own scripts |

### Where we actually stand against their numbers

Only comparable loosely — different instances, different machine — but the order of magnitude
is informative, and it is not all bad:

* **LKH at one trial** is 10.6‰ (1.06%) at n = 1000 and 12.3‰ (1.23%) at n = 5000, *including*
  its preprocessing. Our iterated arms reach 0.34% at n = 1173 and 0.88% at n = 5915. On gap
  alone we are in the same band as their LKH baseline, not obviously behind it.
* **But** their LKH baseline is paying 1.0 s (n=1000) and 39 s (n=5000) for alpha-nearness
  preprocessing it does not have to pay. Our own sweep shows LKH with `nn/trials=100` reaching
  0.058% in 2.53 s on pcb3038, which dominates everything this project produces on that
  instance. The moment LKH is allowed to skip preprocessing, the comparison is not close.
* **NeuroLKH is 0.48–0.90‰ (0.05–0.09%)** across n = 1000…5000. That is roughly **ten times
  better** than anything here, and it is the number a paper in this space is measured against.

### What would have to change to make a comparable claim

1. **Move to the hard instances.** The 37 VSR-LKH lists are where there is anything to win.
2. **Adopt the time-matched protocol.** Measure LKH's time, impose it as our limit, report the
   gap. This removes the "who is on which part of the time axis" framing entirely — which is
   the framing that has been doing the work in `FINDINGS §6.2`, and it does not survive contact
   with a properly configured baseline.
3. **Report in ‰ at n ≥ 1000**, on 1000 generated instances per size, if the generated-instance
   comparison is wanted. `synth.py` already produces the instance families; only the volume and
   the reference solver (Concorde) are missing.
4. **Fix the baseline.** Any LKH comparison must sweep `MAX_TRIALS` and `CANDIDATE_SET_TYPE`.
   The `runs`-only baseline overstates LKH's cost by one to two orders of magnitude at the
   cheap end, and every claim built on it is a claim about `elkai`'s defaults.

### The narrow thing that is still ours

The RL line pays a warm-up cost (VSR-LKH's own appendix says so) and the neural line pays
training cost — ≈780 000 labelled instances and four GPU-days for NeuroLKH — plus per-instance
inference. A fitted fuzzy rule base costs neither: a handful of arithmetic operations in a
JIT'd inner loop, fitted once, with scale-free antecedents that transfer across two orders of
magnitude in n. That is a genuine and narrow claim, and it is about *cost of the method*, not
about tour quality. It should not be dressed up as beating LKH.

---

## Files

| file | paper |
|---|---|
| `2110.07983.pdf` | NeuroLKH (NeurIPS 2021) |
| `2012.04461.pdf` | VSR-LKH (AAAI 2021) |
| `2207.03876.pdf` | Reinforced LKH — the extended journal version, *Knowledge-Based Systems* 2023 |
| `2006.07054.pdf` | Joshi et al., "Learning TSP Requires Rethinking Generalization" (CP 2021) |
| `2501.04072.pdf` | Wang et al., "Multi-armed Bandit and Backbone boost LKH" (2025) |
