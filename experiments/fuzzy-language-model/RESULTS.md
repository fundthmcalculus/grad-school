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

### Protocol amendment 3, 2026-10-09: ten-seed headline offloaded to a GPU host

The user asked for this to speed up the run. The planned `headline`/`headline2` grids
(CPU seeds 3–9, to merge with the CPU `scaling` seeds 0–2) are **replaced** by
`headline10`/`headline10v2`: all ten seeds 0–9 at d = 32, run on the GPU host as
complete single-platform grids. The laptop's chains skip the old grids via
`outputs/<grid>/SKIP` markers.

* **No table cell mixes platforms.** Every run JSON now records `device`,
  `device_name` and `host`, and `flm.analyze` keys cells on device. Hyperparameters
  still come from the CPU `tune`; that is a choice, not a measurement, so it is
  platform-free.
* The CPU path of the device change was verified **bit-identical end to end** before it
  was swapped in: same val BPC, every eval, the train curve and the sample text, on a
  fuzzydelta + HTSK + data-init run.
* **Free reproducibility check:** GPU seeds 0–2 at d = 32 repeat CPU `scaling` cells.
  CPU and GPU floating point differ, so per-seed values will not match exactly. The
  registered expectation is that the 3-seed means agree within the larger of the two
  seed stds. A larger gap is a platform effect, and it must be reported before any
  GPU number is quoted beside a CPU one.
* GPU wall-clock and chars/s are not comparable to the CPU timing columns and are not
  reported against them.

### `h9` outcome (2026-10-09): 16 cells, H9b scored

Mean val BPC; frlm-acc over seeds 0–2, frlm-delta seed 0. "Recovers" is the fraction of
the (sum, sum) → (mean, mean) gain that each single-layer change achieves.

| arm | sum/sum | mean/sum (mixer) | sum/mean (FFN) | mean/mean | full gain | mixer-only | FFN-only |
|---|---|---|---|---|---|---|---|
| frlm-acc (3 seeds) | 1.858 ± 0.006 | 1.860 ± 0.023 | **1.752 ± 0.014** | 1.759 ± 0.003 | 0.099 | −2% | **107%** |
| frlm-delta (1 seed) | 1.862 | 1.840 | **1.738** | 1.753 | 0.109 | 20% | **114%** |

* **H9b refuted** in both arms. It predicted that the gain lives in the mixer; it lives
  in the TSK FFN. This confirms the correction above.
* The (sum, sum) and (mean, mean) seed-0 cells reproduced exactly (determinism check).
* **Unregistered observation, not acted on:** FFN-only HTSK is slightly better than HTSK
  on both layers in both arms (−0.007 acc, −0.015 delta). The `-htsk` arms in
  `scaling2` use HTSK on both layers, so they may be marginally off the best
  configuration. Selecting FFN-only now would be a further round of fuzzy-only search.
  It is left as a candidate for a later, disclosed iteration.
* Mechanism, as a reading rather than a test: the FFN's antecedent dimension is 32, the
  mixer's per-head dimension 16. Saturation grows with dimension (Cui, Wu & Xu 2021,
  Fig. 1), so the wider FFN rule base is where saturation bites.

## `scaling` + `scaling2` outcome (2026-10-10): 180 runs, 0 failures, 3 seeds per cell

Full table: [`outputs/scaling-all/summary.md`](outputs/scaling-all/summary.md) (CSV beside it).
Plot: `outputs/scaling-all/bpc_vs_params.png`. CPU (i7-1185G7), 30M chars per run,
each arm at its d = 32 tune setting applied at every width.

**Smallest model reaching a mean val BPC threshold** (total params, log-interpolated;
`≤` means the smallest grid size already reaches it, so the number is an upper bound):

| arm | ≤ 2.0 | ≤ 1.8 | ≤ 1.6 | ≤ 1.4 |
|---|---|---|---|---|
| softmax-mlp | 7,435 | 12,496 | 26,701 | 70,210 |
| linear-mlp | 6,547 | 11,714 | 26,013 | 70,048 |
| gla-mlp | 6,369 | 11,493 | 25,211 | 66,560 |
| delta-mlp | ≤ 6,008 | 10,961 | 23,677 | 68,432 |
| gru | 5,850 | 10,268 | not reached | not reached |
| flm | 11,413 | 22,027 | 125,627 | not reached |
| frlm-acc / frlm-delta | ≤ 8,500 | ≈18,200 | not reached | not reached |
| frlm-acc-htsk † | ≤ 8,500 | 15,705 | 35,406 | 167,725 |
| frlm-delta-htsk † | ≤ 8,568 | 14,931 | 34,846 | 140,947 |

† post-hoc arms that received extra search (see the `fuzzyfix` disclosure).

### Scored hypotheses

* **H1: refuted.** Softmax does *not* beat linear attention at every width ≥ 32.
  - Mean ± std: d32 1.6726 ± .0022 vs 1.6599 ± .0018 (linear better); d48 1.4863 ± .0041
    vs 1.4911 ± .0029 (softmax better by about 1 std); d64 1.3942 vs 1.3935 (tie); d96
    1.2916 vs 1.2886 (linear better).
  - The ordering delta < gla < linear < softmax holds clearly only at d ≤ 32. From
    d = 48 up, all four are within about 0.01.
  - **At these sizes, quadratic attention buys nothing per parameter.** It does cost a KV
    cache larger than the whole model after about 150 characters (`flm/state.py`).
* **H5: refuted as stated (every width).** Delta beats accumulation from d = 32 (sum
  forms) or d = 48 (HTSK forms) upward. At d = 16 and 24 they are within seed noise or
  reversed.
* **H6: half supported.** Softmax reaches 2.0 BPC at 7,435 params (< 15K, as predicted).
  The linear-cost arms need **fewer**, not as many: DeltaNet ≤ 6,008 and GRU 5,850.
* **H7: refuted for the registered FRLMs; post-hoc HTSK arms inside the margin.** Gap to
  the best neural linear-cost arm at matched params, interpolated on the neural curve
  with no extrapolation past 158K:

  | params ≈ | 17K | 28K | 59K | 102K |
  |---|---|---|---|---|
  | frlm-acc (registered) | +0.13 | +0.17 | +0.27 | +0.32 |
  | frlm-delta (registered) | +0.14 | +0.16 | +0.25 | +0.30 |
  | frlm-acc-htsk † | +0.10 | +0.08 | +0.09 | +0.11 |
  | frlm-delta-htsk † | +0.08 | +0.08 | +0.07 | +0.08 |

  The registered arms fail H7. The HTSK FRLM (delta) stays 0.07–0.08 behind across the
  range, which is inside H7's 0.10 margin. Because of the extra search, that is a
  **post-hoc** observation, not a pass of H7.

### Not hypotheses, but must be read before quoting anything above

* **The LR does not transfer for gru, flm or the sum-form FRLMs.** Each was tuned at
  d = 32 and gets *worse* at large widths (gru 1.68 @15K → 1.81 @121K; frlm-acc 1.665
  @102K → 1.756 @220K). That is an optimisation failure, not a capacity measurement, so
  their large-width points do not mean "this architecture doesn't scale". The HTSK
  FRLMs scale cleanly, which suggests HTSK also conditions training, not only the rule
  saturation. **Owed:** a per-width LR check for every arm alike before any
  large-width claim about these arms.
* Timing columns come from 4–7 concurrent jobs on 4 cores. They show relative cost
  only (delta-rule arms ≈ 3–5× softmax per char at this T = 256), not throughput.

### Registered 2026-10-10, before `lrwidth`: does the d = 32 LR transfer to d = 96?

Every arm (all 10) at d = 96, seed 0, 30M chars, with its tuned LR ÷ 3 and ÷ 10. The
×1 cell is the existing `scaling` seed-0 run. The check is identical for every arm.

* **H10a.** For gru, flm, frlm-acc and frlm-delta (the arms that got worse with width),
  a divided LR improves d = 96 by more than 0.05 BPC and restores a monotone curve, i.e.
  d = 96 beats their d = 64 mean.
* **H10b.** For the softmax / linear / gla / delta / HTSK-FRLM arms (which scaled
  cleanly), the best divided LR moves d = 96 by less than 0.03 BPC.
* **Consequence, decided before the data:** if H10a holds, the large-width rows of
  those four arms are reported as LR-limited and are re-run at the transferred LR
  before any architecture conclusion is drawn from them.

## The ~2 BPC FRLM (registered 2026-10-10, before `minsize` and `lean`)

The user asked to focus on the smallest fuzzy recurrent model. At d = 16 (8,568 params)
`frlm-delta-htsk` scores 1.961 ± 0.012, already below 2.0, so its real crossing is
below the grid. Its parameters at d = 16: Gaussian widths 18%, rule centers 18%,
embedding 18%, qkv 18%, consequents 12%.

* **H11 (`minsize`, widths 8/10/12/14, 3 seeds, with delta-mlp as the reference).**
  `frlm-delta-htsk` crosses 2.0 BPC between 5K and 8.5K params. delta-mlp crosses it
  at fewer params (its 6K cell is at 1.996), so the fuzzy model's
  parameters-to-reach-2.0 ratio stays at or above the ~1.3× seen at 1.8 / 1.6 BPC.
* **H12 (`lean`: one-variable screen at d = 16, seeds 0–1, frlm-delta-htsk's tuned
  lr/sc; reference = its 3 `scaling2` seeds). This is design iteration, disclosed.**
  * `ffnonly` (HTSK on the FFN only; h9's best): ≤ base BPC.
  * `wdim` / `wrule` (widths shared across rules / one width per rule; −17% params):
    within +0.03 of base, i.e. better BPC *per parameter* than base.
  * `share2/3/4` (one block applied 2/3/4×, 5,076 params, −41%): `share2` is worse
    than base, and BPC improves monotonically with the number of applications.
  * **What counts as a winner:** a variant that is better than base *at matched
    params*. That means below the log-param interpolation of the base FRLM scaling
    curve, not merely below base's raw BPC. Winners get the standard 12-cell tune
    before they enter any comparison.

### Partial outcomes, 2026-10-10 15:30: `ablate` 24/36, `lrwidth` 6/20

**`lrwidth` (d = 96, seed 0, compared with the existing ×1 `scaling` seed-0 cell)**, attention arms only so far:

| arm | ×1 (tuned) | ÷3 | ÷10 |
|---|---|---|---|
| softmax-mlp | 1.3000 | 1.3068 (+0.007) | 1.3326 (+0.033) |
| linear-mlp | 1.2921 | 1.2828 (−0.009) | 1.3084 (+0.016) |
| gla-mlp | 1.2925 | 1.2841 (−0.008) | 1.3153 (+0.023) |

* **H10b holds so far:** for these arms, the d = 32 learning rate is still within 0.01 of the
  best of {×1, ÷3, ÷10} at three times the width. A lower LR does not rescue anything here.
* **H10a is still open.** The arms that degraded with width (gru, flm, frlm-acc,
  frlm-delta) are queued.

**`ablate` (3 seeds, compared with the `scaling` cells at the same width):**

| comparison | d = 32 | d = 64 | hypothesis |
|---|---|---|---|
| gauss-mlp − softmax-mlp (mixer only) | **−0.013** (1.6595 ± .0013 vs 1.6726 ± .0022) | **−0.008** (1.3858 vs 1.3942) | **H2 supported** (\|Δ\| < 0.02). The Gaussian-TSK mixer is slightly *better* |
| softmax-tsk − softmax-mlp (FFN only) | **+0.118** (1.7910 vs 1.6726) | **+0.315** (1.7093 vs 1.3942) | **H3 supported**, strongly |
| softmax-tsk+datainit − softmax-tsk | +0.132 | −0.017 (within std ≈ 0.02) | H4 |
| flm+datainit − flm | +0.078 | +0.023 | H4 |

* **H4 refuted so far.** Data-driven rule init hurts or is neutral in every completed pair.
  The frlm-* datainit cells are still running.
* **The FLM's poor scaling is the TSK FFN, not the attention.** Fuzzy attention matches
  softmax (H2). The classic (sum-exponent) TSK FFN costs +0.12 at d = 32 and +0.32 at
  d = 64, and the cost grows with width, which is the saturation pattern Cui, Wu & Xu
  predict. Combined with `h9` (HTSK on the FFN carries the whole FRLM gain), the obvious
  next arm is **FLM with an HTSK FFN**. It is not run yet.

## CPU phase stopped, 2026-10-10 15:30 (user decision; repeat on the GPU host)

All sweeps were stopped at the user's request. Complete: `tune` 120/120, `fuzzyfix` 15/15,
`h9` 16/16, `scaling` 144/144, `scaling2` 36/36. Incomplete: `ablate` 24/36, `lrwidth` 6/20,
`minsize` 0/24, `lean` 0/12. In-flight runs were killed before writing a record, so no
partial records exist. **H10a, H11 and H12 are unscored.** The write-up is
[`WRITEUP.md`](WRITEUP.md).

## Goal track, 2026-10-10: the most capable fuzzy LM under 100K parameters (design iteration)

**Not part of the registered head-to-head (H1–H9).** This is development work with a different
objective (best val BPC at < 100,000 total parameters, embedding included), run on CUDA. All
screens are d ≥ 64, L = 2, 10M chars, 1–3 seeds, and each stage was chosen after seeing the last;
nothing here was pre-registered, so none of it is a hypothesis test. Reproduce with
`python -m flm.sweep <v3-grid> --workers 4`; read with `python -m flm.screen <grid> [--seeds]`.
Do not run more than ~4 workers: 6 concurrent 16-head delta cells overflowed 12 GB VRAM
and stalled 40+ minutes with no output (killed; the 16-head cells were dropped, **not tested**).

**1. Where the deficit is (`v3-attrib`, d = 64, seed 0, lr 1e-2/3e-2 best).** Every mixer × FFN pairing:

| mixer | MLP FFN | TSK FFN (original) |
|---|---|---|
| softmax | 1.512 (72.6K) | 1.826 (97.2K) |
| gauss (FLM mixer) | 1.492 | 1.781 |
| linear | 1.503 | 1.720 |
| fuzzy (FRLM-acc) | 1.507 | ~1.715 (101.6K) |
| fuzzydelta | 1.514 | — |

The fuzzy mixers cost nothing against softmax/linear. The whole ~0.3 BPC fuzzy deficit, and
~25K of the parameters, was the zero-order, softmax-normalized TSK FFN.

**2. FFN redesign (`v3-ffn`, `v3-ffn2`, `v3-ffn3`; fuzzy-acc mixer, d = 64).** What helped, in order of effect:
tied per-rule Gaussian widths (saves params); fewer antecedent dims (`tsk_dim` 16); **first-order
consequents** (rule output `(v_r·x) o_r`, a GLU whose gate is the fuzzy membership);
**un-normalized firing** (no softmax across rules); HTSK exponent (`mean`). Raw firing with the
`sum` exponent is unstable (3 seeds: 1.517/1.588/1.557, std 0.029); with the HTSK exponent it is
stable (std 0.0003–0.002). 3 seeds, 10M chars: `o1r-da16-htsk-wdim` **1.4675 ± 0.0003**
(87.5K) vs the same mixer with a GELU MLP **1.4957 ± 0.0086** (77.0K) — the fuzzy FFN now beats the MLP.
Rejected: raw firing at `tsk_dim` 32 (1.685), lr 6e-2 (worse).

**3. Mixer and shape (`v3-mix`, `v3-heads`, `v3-spend`, `v3-fill`; 2 seeds).** The gain is
mostly **head count**: 1 head 1.534, 2 heads 1.464, 4 heads 1.441, 8 heads 1.420 (acc form).
The delta form with the HTSK mixer exponent overtakes accumulation once heads ≥ 4
(delta-mean h8 1.4065 vs acc-mean h8 1.4249). 16 rules/head beat 8 (1.4138 vs 1.4065)
and 32 is no better than 16 (1.4029). Short-conv 8 > 4 (1.3986 vs 1.4065). Depth: 3 layers
at d = 48 and 4 layers at d = 40 were no better than 2 layers at d = 64; 1 layer at d = 80 was
much worse (1.585). lr 3e-2 ≈ 4e-2 > 2e-2. Filling the budget gave 1.386 ± 0.004
(`D`: d = 72, 8 heads, 16 rules, FFN 1.5× rules, shortconv 8, 97,520 params); the
differences among the filled configs (1.386–1.402) are within ~3 seed-stds.
Not tried: 16 heads (see above), d = 64 with 3 layers under the budget, rule-count > 32.

**4. Final comparison (`v3-final`, `v3-ref`; CUDA, 30M characters, 3 seeds).**

| model | params | val BPC |
|---|---|---|
| fuzzy `D` | 97,520 | **1.2735 ± 0.009** (1.286 / 1.268 / 1.267) |
| neural delta-mlp, d 72, 8 heads | 93,272 | 1.2929 ± 0.004 |
| neural delta-mlp, d 72, 4 heads | 92,104 | 1.3108 ± 0.006 |
| neural delta-mlp, d 74, 2 heads | 96,430 | 1.3426 ± 0.007 |
| original FLM, d 64 (`scaling-gpu`) | 97,280 | 1.6231 ± 0.008 |

At 100M characters (seed 0, one run each): fuzzy `D` **1.2080**; neural delta-mlp d 64 (73K params, *smaller*, so not
matched) 1.3270. The fuzzy model leads the best matched neural reference by 0.019 BPC (about two fuzzy-model seed
stds), at 3× lower throughput (34K vs 114K chars/s). Disclosure: the fuzzy design got a search over heads, rules,
antecedent size, short-conv and lr that the neural reference got only for heads (and lr 3e-2 rather than its d = 32
tune pick of 6e-2); the LR study below was meant to close that and is **not finished**.

**5. Learning-rate study (`v3-lr`, `lrwidth`): INCOMPLETE at merge time.** Registered before running: fuzzy `D` and the
neural 8-head reference over lr ∈ {1e-2, 2e-2, 3e-2, 4e-2, 6e-2}, 30M chars, seeds 0–1 (3e-2 exists), plus the 14
unfinished upstream `lrwidth` cells (H10a) on CUDA. Running both at once with 8 concurrent jobs made each job ~4×
slower; at merge time `v3-lr` had 0/16 cells and `lrwidth` 13/20. **No LR conclusion is drawn here; H10a stays open.**
Resume with `python -m flm.sweep v3-lr --device cuda --workers 4` (and `lrwidth`); the sweeps skip finished cells.
Note `flm.sweep` appends `--device` after each job's argv, so `--device cuda` must be passed on the command line.

**Code changes in this track.** `ffn_order`, `ffn_width`, `ffn_norm` on the TSK FFN (defaults reproduce the previous
FFN exactly; `ffn_width` defaults to following upstream's `width_share`); grids in `flm/grids_v3.py`; `flm/screen.py`
(per-sweep summaries); `flm/report_data.py` + `report/` (the results page). `train.py` also now reports a diverged run
as NaN instead of its last finite eval and times CUDA steps with a synchronize.
## Architecture screen: n-gram rule initialisation (registered 2026-10-10, before any `screen` run)

**This is a fast screen, not a regression test.** It runs at d = 16 (~8.6K params, the
~2 BPC regime) at the 10M-char tune budget, seed 0 (the baseline also seed 1, for a noise
estimate), at `frlm-delta-htsk`'s tuned lr 6e-2 for every variant. Nothing here is citable;
a variant that looks promising is promoted to 3 seeds and its own lr check before it counts
for anything. The ~0.012 seed std at d = 16 (3 seeds, `scaling2`) is the yardstick.

### What was built, and what was seen while building it

A count-based n-gram LM is a crisp zero-order TSK system ("IF the last k characters are g
THEN p(next | g)"). `flm/ngram.py` initialises layer 0's TSK FFN from those statistics:
rule r ↔ one of the R − 1 most frequent length-k contexts (plus one wide background rule
that falls back to the unigram), antecedent centre = mean of the rule's own input over the
sample positions with that context, consequent decoded through the tied embedding so that
its logit lens reads log p(next | g). `--ante-taps 2` lets layer 0's antecedent read the
previous position too, which trigram rules need.

Seen **before** registering (manipulation checks and ceilings, none of it a training result):
* Reference count models on this data (1M valid chars): unigram 4.42, bigram **3.30**,
  trigram **2.39**, four-gram **1.82** BPC. The 8.6K-param FRLM (1.96) sits between trigram
  and four-gram.
* **A random d = 16 embedding cannot represent n-gram distributions:** the rank-16 readout
  of the bigram log-probs scores 7.5 BPC (worse than uniform, 6.6). With the embedding
  aligned to the top-16 principal directions of the conditional log-probs it scores 3.57
  (bigram) / 3.37 (trigram). A first init attempt without the aligned embedding failed
  (6.62 → 6.32) for exactly this reason, and the embedding alignment was added.
* **A 32-rule bank is the limit for trigrams:** the top-31 trigram contexts cover 43% of
  positions (ceiling 4.04 BPC) against 98% for bigrams (ceiling 3.59). Clustering contexts
  into 32 rules does not beat top-k (4.02 / 3.59).
* Init BPC with the full init, held out: bigram **3.70**, trigram+1 tap 4.21, trigram+2
  taps **4.16** (random init 6.62). Rule accuracy (the winning rule is the position's own
  context): bigram 0.77, trigram 0.53 with 1 tap, **0.87 with 2 taps**. Zeroing layer 0's
  consequents sends BPC to 12–40, so the rules carry the readout. The calibrated output
  gain (final-norm weight) reaches 16 on real data.

### Variants (one change from the base unless noted; all at d = 16, 10M chars)

| id | what | flags beyond the base |
|---|---|---|
| A | `frlm-delta-htsk` (both-layer HTSK) | — (seeds 0 and 1) |
| B | aligned embedding only | `--ngram-init 1 --ngram-mode emb` |
| C | + bigram rule placement | `--ngram-init 1 --ngram-mode placement` |
| D | + decoded consequents and output gain | `--ngram-init 1 --ngram-mode full` |
| E | 2-tap antecedent, random init | `--ante-taps 2` |
| F | trigram rules, 2 taps, full | `--ante-taps 2 --ngram-init 2 --ngram-mode full` |
| G | trigram placement only (F's control) | `--ante-taps 2 --ngram-init 2 --ngram-mode placement` |
| H | HTSK on the FFN only (h9's best) | `--exp-norm sum --ffn-exp-norm mean` |
| I | one block applied 3× (−41% params) | `--layers 3 --n-unique 1` |
| J | widths shared per dimension (−17% params) | `--width-share dim` |

B → C → D is the attribution chain for the n-gram init (embedding, then rule placement,
then consequents). E is the architecture control for F.

### Predictions (final val BPC at 10M chars unless stated; Δ = variant − A, seed 0)

* **P1. D is not better than A by more than 0.03.** The n-gram information is learned in
  the first few hundred steps anyway; the init removes that part of training, not the
  capacity limit. (My own probability of a > 0.03 gain: ~25%.)
* **P2. D reaches 3.0 BPC in at most half of A's steps** (curve evaluated every 100 steps,
  plus step 0). This is the effect I expect.
* **P3. B is within ±0.03 of A.** An aligned embedding basis is learned quickly.
* **P4. C is not better than A by more than 0.03,** and D − C is the consequent decoding's
  effect.
* **P5. E improves on A by ≥ 0.03,** more than the +3% parameter effect (≈ 0.009 from the
  `scaling2` slope of ~0.29 BPC per e-fold of params). Reading the previous position inside
  the FFN antecedent should help. (~50%.)
* **P6. F is not better than D.** At 32 rules the trigram rule bank starts worse (4.16 vs
  3.70) and covers less.
* **P7. H beats or matches A** by ≥ 0.01 (`h9` saw −0.007 / −0.015 at d = 32).
* **P8. I is worse than A by > 0.05** (41% fewer parameters). Exploratory.
* **P9. J is within +0.03 of A** with 17% fewer parameters.

**Promising** = better than A by more than max(0.03, 2·|A₀ − A₁|), or a faster early curve
with no worse final BPC. Promising variants go to seeds 1–2 and a 3-point lr check.
Caveats stated up front: one seed per variant; every variant uses A's lr, and the n-gram
variants start with a large output gain, so an lr effect could mask or fake a result;
the `placement` and `full` modes calibrate (kappa, alpha, gain) on 64 *training* windows.

### Screen stage 1 outcome (2026-10-10, 11 runs, 0 failures)

Final val BPC at 10M chars, d = 16, seed 0 (A also seed 1). Table: `outputs/screen/summary.md`.

| id | variant | params | final BPC | Δ vs A (seed 0) | steps to 3.0 / 2.5 / 2.2 |
|---|---|---|---|---|---|
| A | base | 8,568 | 2.1025 (seed 1: 2.0526) | | 200 / 300 / 800 |
| B | aligned embedding only | 8,568 | 2.0816 | −0.021 | 200 / 300 / 800 |
| C | + bigram placement | 8,568 | 2.0669 | −0.036 | 200 / 300 / 700 |
| D | + decoded consequents (bigram, full) | 8,568 | 2.0523 | −0.050 | 200 / 400 / 700 |
| E | 2-tap antecedent | 8,824 | 2.0600 | −0.043 | 200 / 300 / 800 |
| F | trigram, 2 taps, full | 8,824 | 2.0960 | −0.007 | 100 / 400 / 800 |
| G | trigram placement only | 8,824 | 2.1366 | +0.034 | 200 / 300 / 1000 |
| H | HTSK on the FFN only | 8,568 | 2.0361 | −0.066 | 200 / 300 / 700 |
| I | one block ×3 (−41% params) | 5,076 | 2.2050 | +0.102 | 200 / 400 / never |
| J | widths shared per dimension (−17%) | 7,096 | 2.1641 | +0.062 | 200 / 300 / 1000 |

**The noise.** The two base seeds differ by **0.050** (std ≈ 0.035), about three times the
0.012 I had taken from the 30M-char `scaling2` runs and used to set the 0.03 thresholds in
the registration above. **That was my error:** the noise at the 10M-char screen budget and
lr 6e-2 is much larger than at 30M, and I should have measured it before registering. It
makes the screen under-powered for the effects the predictions were about.

**The learning curves** (evaluated every 100 steps) show the n-gram init's head start is
gone almost at once: every variant, the base included, reaches 3.0–3.2 BPC by step 100.
D starts at 3.70 and is ahead at step 100 (3.01 against 3.18), **behind** at steps 200–300
(2.77 against 2.68, 2.52 against 2.47), and indistinguishable from the base from step ~500.
The rest of the base's early curve (6.6 → 3.2 in 100 steps) means n-gram statistics are
learned within the first ~100 of 1221 steps.

**Predictions, scored literally against the registered Δ (seed 0):**

| | prediction | outcome |
|---|---|---|
| P1 | D not better than A by > 0.03 | **not met as written** (−0.050), but the base's own seeds differ by 0.050, so it cannot be distinguished from noise |
| P2 | D reaches 3.0 in ≤ half of A's steps | **refuted** (200 vs 200; head start ≈ 100 steps at most) |
| P3 | B within ±0.03 of A | met (−0.021) |
| P4 | C not better than A by > 0.03 | **not met as written** (−0.036), within noise |
| P5 | E better than A by ≥ 0.03 | met as written (−0.043; −0.034 after the +3% parameter effect), within noise |
| P6 | F not better than D | met (F − D = +0.044) |
| P7 | H beats or matches A by ≥ 0.01 | met (−0.066, the largest) |
| P8 | I worse than A by > 0.05 | met (+0.102) |
| P9 | J within +0.03 of A | **refuted** (+0.062), but that is what its 17% parameter loss predicts from the `scaling2` slope (≈ +0.055): no free lunch from sharing widths |

Readings that are safe at this power: **I and J are worse, in line with (I: slightly
better than) what their parameter counts predict; the trigram variants are not better than
the bigram ones; the n-gram init does not speed up training beyond its first ~100 steps.**
Everything among A, B, C, D, E, H is within the noise of one another. Under the registered
promotion rule (> max(0.03, 2 × noise) = 0.10) **nothing was promoted.**

### Registered 2026-10-10, before `screen2` (stage 2): resolve what stage 1 could not

Same protocol. New runs: A seeds 2, 3; D, E, H seeds 1, 2. After it: A has 4 seeds; D, E,
H have 3 each. The verdict is computed by `flm.screen_report` (se = pooled base std ·
√(1/n_variant + 1/n_base); "real" if |Δ of means| > 2 se).

* **P10.** D's mean is within the noise of A's. *(My expectation; the n-gram knowledge is
  learned in the first ~100 steps either way.)* If D comes out "real, better", the init is
  worth keeping, since the curves already say it is not a speed-up.
* **P11.** E (2-tap antecedent) is better than A by ≥ 0.03 and "real". *(~40%.)*
* **P12.** H (FFN-only HTSK) is better than A by ≥ 0.02 and "real". *(~50%; `h9` saw 0.007
  and 0.015 at d = 32.)*
* **A finding is only called promising if it is "real, better" by this rule and the
  improvement is at least 0.03.** Anything else is reported as a null at this power.
