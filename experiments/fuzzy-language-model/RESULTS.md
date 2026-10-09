# Results: laboratory record

Hypotheses are written down **before** the run that tests them, with the date, and
are never edited afterwards. Outcomes are appended below each one. "Δ" is BPC (arm)
− BPC (reference): negative means the arm is better.

## Registered 2026-10-08, before the `tune` and `scaling` sweeps

Seen before registering: the 20K-param pilot curves for `softmax-mlp` (1.744 BPC at
60M chars) and `gru` (1.761), one seed each, at a single untuned lr = 3e-3. Nothing
else.

* **H1 (linear vs quadratic).** At equal parameters and equal characters,
  `softmax-mlp` beats `linear-mlp` (elu+1, fixed decay) at every width ≥ 32, by more
  than the seed std. The gap shrinks in the order linear > gla > delta. *Basis:* the
  literature (`literature/02`) at 40M–125M; not established below 1M.
* **H2 (FLM mixer ≡ softmax).** The Gaussian-TSK mixer differs from softmax attention
  only by dropping the rule weight exp(‖k‖²/2τ) and learning widths. With the FFN held
  fixed (`gauss-mlp` vs `softmax-mlp`), |Δ| < 0.02 BPC at d = 32 and d = 64.
* **H3 (TSK FFN cost).** With the mixer held fixed (`softmax-tsk` vs `softmax-mlp`), the
  TSK FFN is worse by more than 0.03 BPC at matched width. *Basis:* normalized-RBF
  layers with softmax competition suffer dead rules; the smoke run showed 6/64 live
  rules.
* **H4 (data-driven rule init).** With everything else fixed, data-initialized rule
  centers improve every TSK arm by more than the seed std and raise the live-rule
  count. *Manipulation check, already passed on a 0.5M-char smoke run:* layer-0 FFN live
  rules 6 → 38 / 64.
* **H5 (FRLM forms).** `frlm-delta` beats `frlm-acc` at every width (delta beats
  accumulation in the literature).
* **H6 (smallest model).** `softmax-mlp` reaches mean val BPC ≤ 2.0 at fewer than 15K
  total params within 30M chars; the best linear-cost arm needs at least as many.
* **H7 (fuzzy cost, the headline).** The best FRLM is within 0.10 BPC of the best
  neural linear-cost arm at matched parameters for d ≥ 32. *This is a guess, not a
  derivation. Its failure would be a result, not a bug.*

## Outcomes

*(appended as sweeps complete)*

### Protocol amendment, 2026-10-08, during `tune`

`softmax-mlp`'s best tune cell was lr = 1e-2, the top of the grid (1.830 BPC with
short-conv, against 2.020 at 3e-3). Its search was therefore truncated. lr = 3e-2 was
added to the grid **for every arm** before any other arm's tune result had been
seen. The tune budget is 10M chars, a third of the scaling budget, and that may favour
high learning rates; the bias is the same for every arm.

Also seen at this point, and outside the registered protocol: the 60M-char pilots at
lr = 3e-3, one seed, untuned. `linear-mlp` scored 1.699, `softmax-mlp` 1.744 and `gru`
1.761. **This runs against H1**, but it is one seed at an LR now known to be
sub-optimal for softmax, so H1 is decided by `scaling`, not by this.

Pilot addendum, same footing (one seed, lr = 3e-3, 60M chars, outside the protocol):
the fuzzy recurrent mixer (`fuzzy`, accumulating, fixed decay, **MLP** FFN, 21.7K
params) scored **1.703**, level with `linear-mlp`'s 1.699. Its curve: 1.978 → 1.856 →
1.800 → 1.763 → 1.734 → 1.715 → 1.706 → 1.703 at every 8.2M chars. This is the first
sign that a fuzzy-partition feature map costs nothing against elu+1 at this size. It
says nothing yet about the TSK FFN.

### Protocol amendment 2, 2026-10-09, after `tune` with lr ≤ 3e-2

The pre-registered edge check stopped the chain before `scaling`. lr = 3e-2 was best
for 7 of 8 arms; only `flm` peaked inside the grid, at 1e-2. lr ∈ {6e-2, 1e-1} were
added for **every** arm. A diverged cell records NaN and cannot be selected.

Best cells at lr ≤ 3e-2, 10M chars, seed 0, **provisional** (this is not the tuned
result):

| arm | params | best lr / sc | val BPC |
|---|---|---|---|
| delta-mlp | 19,944 | 3e-2 / 4 | 1.707 |
| gla-mlp | 19,812 | 3e-2 / 4 | 1.729 |
| linear-mlp | 19,684 | 3e-2 / 4 | 1.742 |
| gru | 15,840 | 3e-2 / – | 1.743 |
| softmax-mlp | 19,680 | 3e-2 / 4 | 1.763 |
| frlm-delta | 28,392 | 3e-2 / 4 | 1.862 |
| frlm-acc | 28,004 | 3e-2 / 4 | 1.864 |
| flm | 25,856 | 1e-2 / 4 | 1.924 |

Short-conv = 4 won for every arm it applies to (it is a no-op for `gru`).

### `tune` final (2026-10-09): 96 cells, 0 failures, edge check passed

The selection rule (lowest val BPC, 10M chars, d = 32, seed 0) picked:

| arm | lr | short-conv | val BPC | 1e-2/sc4 | 3e-2/sc4 | 6e-2/sc4 | 1e-1/sc4 |
|---|---|---|---|---|---|---|---|
| delta-mlp | 6e-2 | 4 | **1.699** | 1.750 | 1.707 | 1.699 | 1.716 |
| gla-mlp | 3e-2 | 4 | 1.729 | 1.777 | 1.729 | 1.732 | 1.744 |
| gru | 6e-2 | – | 1.733 | 1.848 | 1.743 | 1.733 | 1.867 |
| linear-mlp | 3e-2 | 4 | 1.742 | 1.793 | 1.742 | 1.745 | 1.742 |
| softmax-mlp | 3e-2 | 4 | 1.763 | 1.830 | 1.763 | 1.772 | 1.791 |
| frlm-delta | 3e-2 | 4 | 1.862 | 1.866 | 1.862 | 1.890 | 1.978 |
| frlm-acc | 3e-2 | 4 | 1.864 | 1.908 | 1.864 | 1.904 | 2.022 |
| flm | 1e-2 | 4 | 1.924 | 1.924 | 1.945 | 2.049 | 2.174 |

Caveats, stated before `scaling` reports:
* The tune is **one seed at one width**. Its gaps of ≤ 0.03 between neural arms are not
  established; `scaling` seeds 0–2 and the ten-seed `headline` decide them.
* The LR picked at d = 32 is applied at every width. The optimal LR usually falls with
  width, so the largest widths may be over-stepped. That applies to every arm alike, but
  it is not neutral if arms differ in LR sensitivity.
* The fuzzy arms prefer lower LRs and degrade faster above their optimum. That is
  consistent with a less well-conditioned parameterisation (Gaussian widths in exp).

## Design iteration: rule saturation in the fuzzy layers (registered 2026-10-09, before `fuzzyfix` results)

**This is not part of the head-to-head.** It is development work on the FLM/FRLM and is
reported as such. A variant that wins here enters the main comparison only after
getting the same tune search and seeds as every other arm.

Diagnosis: Cui, Wu & Xu (2021) show that a product-t-norm Gaussian TSK's normalized
firing is a softmax over exponents that grow linearly with the antecedent dimension D,
so one rule takes all the mass. Our TSK FFN has D = 32 and the mixer D = 16, and the
smoke rulebook showed 6–12 of 64 FFN rules ever winning. Their fix, HTSK
(exp_norm = mean), over-flattened in a 0.5M-char smoke run: FFN crispness 0.02 against a
uniform 0.016. sqrt (1/√D) is the midpoint.

* **H8.** At least one of {sqrt, mean} × {random, data} beats the existing `sum`/`random`
  tune cell for each of the three fuzzy arms at 10M chars. *Prediction:* `sqrt` with data
  init is best, by more than 0.03 BPC.
* **H9.** The gain, if any, comes mostly from the TSK FFN. Measured by the number of live
  FFN rules in the rulebook of the best variant against `sum`/`random`.

### `fuzzyfix` outcome (2026-10-09): 15 cells + 3 existing tune cells, seed 0, d = 32, 10M chars

| arm | sum/random | sum/data | sqrt/random | sqrt/data | mean/random | mean/data |
|---|---|---|---|---|---|---|
| flm | **1.924** | 1.973 | 1.954 | 1.925 | 1.973 | 1.940 |
| frlm-acc | 1.864 | 1.950 | 1.785 | 1.820 | **1.756** | 1.792 |
| frlm-delta | 1.862 | 2.002 | 1.768 | 1.763 | **1.753** | 1.776 |

* **H8: supported for the FRLMs, refuted for the FLM. The predicted winner was wrong.**
  HTSK (`mean`) with random init is best for both FRLMs, by −0.108 (acc) and −0.109
  (delta). The prediction was `sqrt` with data init. The FLM gains nothing from any
  variant.
* **H9: refuted, by attribution across arms.** `exp_norm` acts on the TSK FFN *and* the
  fuzzy mixer in the FRLMs, but only on the TSK FFN in the FLM (its Gaussian attention
  has no fixed rule base). The FLM is unchanged, so the FFN-side change is worth ≈ 0. The
  ~0.11 gain is therefore in the **recurrent rule base**. *Caveat:* this is an inference
  across arms, not a direct one-variable test, and it rests on one seed. A direct test
  (exp_norm on the mixer only) is still owed.
* **Data-driven rule init hurt in 5 of 6 comparable pairs.** The exception is
  frlm-delta sqrt, +0.005, which is within noise. This contradicts H4's direction;
  `ablate` tests H4 at 3 seeds.
* The 1/D exponent makes each rule's firing the *geometric mean* of its per-dimension
  memberships, not their product. It is a different aggregation operator: an averaging
  operator rather than a t-norm. The rule reading of the FRLM must say so.

**Search-budget disclosure.** These six variants are extra search that the neural arms
did not receive. The fixed arms therefore enter the comparison as separately labelled
arms (`frlm-acc-htsk`, `frlm-delta-htsk`) with the *same* 12-cell lr × short-conv tune
as every other arm. Any comparison against them must quote this disclosure.

### Registered 2026-10-09, before `h9`: direct test of where the HTSK gain lives

The `fuzzyfix` attribution (H9 refuted) was inferred *across arms*. `h9` tests it directly:
a 2×2 of exponent normalization on the recurrent mixer × on the TSK FFN (`--exp-norm`,
`--ffn-exp-norm`). It runs at each arm's tuned lr/short-conv, d = 32, 10M chars.
`frlm-acc` runs seeds 0–2, `frlm-delta` seed 0.

* **H9b.** Mixer-only HTSK (mean, sum) recovers ≥ 80% of the full (mean, mean) gain over
  (sum, sum), and FFN-only HTSK (sum, mean) recovers ≤ 20%.
* **Determinism check, a pass/fail precondition.** The seed-0 (sum, sum) and (mean, mean)
  cells repeat the existing `tune` and `fuzzyfix` cells and must reproduce their val BPC
  exactly (1.864 / 1.756 for frlm-acc; 1.862 / 1.753 for frlm-delta). If they do not,
  every single-seed comparison above is suspect.
* The seed spread of the `frlm-acc` cells gives the first noise estimate for the
  single-seed `fuzzyfix` gains.

### v2 tune (2026-10-09): the HTSK arms on the shared 12-cell grid

| arm | best lr / sc | val BPC | 1e-2/4 | 3e-2/4 | 6e-2/4 | 1e-1/4 |
|---|---|---|---|---|---|---|
| frlm-delta-htsk | 6e-2 / 4 | **1.747** | 1.880 | 1.753 | 1.747 | 1.791 |
| frlm-acc-htsk | 3e-2 / 4 | **1.756** | 1.823 | 1.756 | 1.761 | 1.801 |

* Both optima are interior; the edge check passed and `scaling2` started 11:25.
* **Determinism, observed:** each lr = 3e-2 / sc4 cell has exactly the configuration of
  the corresponding `fuzzyfix` mean/random cell, run independently. They reproduce:
  1.7555 against 1.756 for acc (the same number at different rounding), and 1.753
  against 1.753 for delta.
* On the provisional d = 32, one-seed tune table, the tuned FRLMs now sit between the
  neural linear-cost arms (linear 1.742, gla 1.729, delta 1.699) and softmax (1.763).
  That would put H7 (FRLM within 0.10 of the best neural linear-cost arm) inside its
  margin. **Not scored here**; `scaling2` and `headline2` decide it.

### ⚠ CORRECTION, 2026-10-09: the `fuzzyfix` attribution was wrong (`h9`, partial)

The `fuzzyfix` section above concluded **"H9 refuted, the gain is in the recurrent rule
base"**. That inference was made across arms (the FLM did not improve). **The direct
test overturns it.** Seed 0, frlm-acc, tuned lr/sc, d = 32, 10M chars:

| mixer exp-norm | FFN exp-norm | val BPC |
|---|---|---|
| sum | sum | 1.864 |
| mean | sum | 1.882 |
| sum | **mean** | **1.755** |
| mean | mean | 1.756 |

* **The whole HTSK gain is in the TSK FFN.** HTSK on the mixer alone does nothing (+0.018).
  **H9 as registered ("the gain comes mostly from the TSK FFN") is supported.** The
  earlier "refuted" verdict is withdrawn.
* Why the FLM did not improve with an FFN-side fix is now an open question. The
  cross-arm inference assumed that the FFN behaves the same whatever the mixer, and it
  does not.
* **Determinism check passed:** the (sum, sum) and (mean, mean) seed-0 cells reproduce
  the tune and fuzzyfix values (1.86395 against 1.864, 1.75555 against 1.756).
* First noise estimate: (sum, sum) at seed 1 gives 1.853, against 1.864 at seed 0. The
  ~0.11 gain is about 10× that spread. The frlm-acc 3-seed cells and the frlm-delta row
  are still running, and H9b is scored when they finish.
* **Lesson, the one AGENTS.md already states:** attribute changes to one variable at a
  time. Inference across arms is a hypothesis, not an attribution.
