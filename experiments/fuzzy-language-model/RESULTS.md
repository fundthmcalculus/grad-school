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
stds), at lower throughput (34K vs 50K chars/s for the size-matched d = 72 / 8-head reference, ≈ 1.5×; **corrected 2026-10-10: the 114K chars/s originally quoted here was the smaller 73K delta-mlp d = 64 at 100M chars, not the matched reference**; speeds are from a shared GPU and noisy). Disclosure: the fuzzy design got a search over heads, rules,
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

## Registered 2026-10-10, before `v3-big` and `v3-post`: bigger fuzzy models and post-training

**Scope change (user, 2026-10-10):** "look at post training and maybe a slightly bigger model". The 100K cap is
relaxed to ≤ 165K parameters for this track; the 97.5K model `D` (fuzzy delta mixer, first-order raw-firing HTSK
FFN, 8 heads) stays the reference. Interpretation of "post-training": stages that run *after* pretraining from
a saved checkpoint, each compared against the control of continuing the same training for the same extra characters.
Seen before registering: `D` at 30M chars = 1.274 ± 0.009 (3 seeds), 100M = 1.208 (1 seed); nothing from any
bigger width or any post-training stage.

* **H13 (bigger).** `D` at d = 80 / 88 / 96 (117K / 139K / 162K params), 30M chars, lr 3e-2, 2 seeds: val BPC falls
  monotonically with params, by ≥ 0.03 BPC per +40K params, with no seed diverging. *Basis:* the neural curves
  fall ~0.1 per doubling at this size; the fuzzy arms' large-width failures were the old FFN, not the width.
* **H14 (continuation control).** Resuming `D` (30M) for +10M chars at lr 3e-3 (warmup 50, cosine to 0.1×)
  improves val BPC by ≥ 0.02. This is the control for H15–H16, not a post-training method.
* **H15 (EMA).** Weight EMA (decay 0.999) during the +10M continuation improves on the continuation's raw weights
  by < 0.01 BPC (the cosine decay already averages).
* **H16 (distillation).** Distilling from a larger neural teacher (delta-mlp d = 128, 280K params, 60M chars;
  logit KL, weight 0.5, T = 1) during the +10M continuation beats the continuation control by ≥ 0.02 BPC.
  *Risk:* at char level with ~1.2 BPC the teacher's soft targets carry little beyond the data, so failure is plausible.
* **Not tested here:** fine-tuning on a curated subset (no quality labels exist), RL, quantization, and
  dynamic evaluation (it changes the evaluation protocol and would need its own label).
* **Protocol.** Base models are trained with `--save-model`; continuation uses a different data-order seed
  (100 + base seed). Raw `val_bpc` is always the headline; EMA is reported as `val_bpc_ema`. Seeds: 2 per cell.

**Incident, 2026-10-10 17:15 (LR study).** The four `v3-lr` `lr1e-2` cells each trained ~1 h on a contended GPU and then
crashed at the final sample generation (`index on cuda, other tensors on cpu`): they had been launched before the
`train.py` merge-seam fix and Python had already loaded the old file. No record was written, so the cells are lost and
rerun when `v3-lr` is resumed. The `lr2e-2` cells (launched after the fix) are running. The `v3-lr` launcher was stopped
after that batch so the GPU serves `v3-big` / `v3-post`; `v3-lr` is **incomplete** (resume with `--workers 2`).
Lesson: a train script should write its numeric record *before* any optional post-step (sampling) that can crash.

### Outcomes, 2026-10-10 (`v3-big`, `v3-post`; CUDA, 2 seeds unless noted)

| model | params | 30M val BPC | +10M continuation | + EMA 0.999 | + distillation (w 0.5) |
|---|---|---|---|---|---|
| D d = 72 | 97,520 | 1.2753 (1.282 / 1.269) | **1.2486** (1.2548 / 1.2423) | 1.2481 | 1.2496 (1.2556 / 1.2436) |
| D d = 80 | 117,312 | 1.2485 | – | – | – |
| D d = 88 | 138,896 | 1.2351 | – | – | – |
| D d = 96 (1 seed post) | 162,272 | 1.2170 (1.222 / 1.212) | 1.1951 | 1.1948 | – |
| teacher: delta-mlp d = 128, 60M chars | 280,480 | 1.1153 (1 seed) | | | |

* **H13 supported.** BPC falls monotonically with size, no divergence: −0.040 from 97K to 139K (+41K), −0.032 from
  117K to 162K (+45K), but only −0.018 for the last +23K (139K → 162K): the gain per parameter is tapering.
* **H14 supported.** +10M chars at lr 3e-3 gives −0.027 / −0.026 at 97K and −0.027 at 162K. This is the control.
* **H15 supported** (gain < 0.01): EMA adds −0.0005 at 97K and −0.0003 at 162K. It is not worth carrying.
* **H16 refuted.** Distillation from a 280K-parameter teacher that is 0.15 BPC better does **not** beat plain
  continuation (+0.001 on both seeds, i.e. no effect). The teacher's samples are no more coherent than the student's.
* **Everything that "helped" post-training is more training.** The 0.027 gain equals what the 30M → 100M pretraining
  curve already predicts for +10M chars. No post-training method here adds anything beyond compute.
* **Generated text** (`flm/gen_samples.py`, `report/samples.json`): at 1.1–1.28 BPC all models write locally fluent
  TinyStories-style prose with real words (non-word rate 0.4–0.9% at T 0.7 / top-k 20, 2.4–3.7% at T 1.0) but with
  grammar slips, drifting characters and no narrative consistency. The failure is syntax and semantics, not spelling.
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

## Registered 2026-10-10 (evening), before `v5-bpe-probe`: do subwords help the fuzzy LM at equal parameters?

User goal: coherent text generation. At ~1.2 BPC every model writes real words but loses characters and plot
(see "Generated text" above). A 512-token byte-level BPE (`flm/bpe.py`; 2.40 chars/token; BPC is still bits per
character of the same validation text: the per-token character lengths sum exactly to the stream length) lets a
256-token window cover ~600 characters instead of 256. Probe: 3 cells, seed 0, 10M chars, lr 3e-2, D architecture.

* **H17.** The BPE d = 80 cell (≈150K params, embedding 41K) beats the character d = 96 cell (162K) by ≥ 0.03 BPC.
  *(~50%.)* *Basis:* characters spend capacity on spelling, but a 512×80 embedding is 27% of the parameters, and
  the ~2.4× fewer positions make the delta-rule state's job easier. A null or a loss is plausible and would say
  the embedding tax eats the gain at this size.
* Not decided by this probe: ten seeds, the 100K-parameter regime, or any claim about coherence (that needs the
  samples and a grader, not BPC).

### Outcome `v5-bpe-probe` (seed 0, 10M chars, lr 3e-2; one seed each, a screen)

| cell | params | val BPC | steps | chars/s |
|---|---|---|---|---|
| char D d = 96 | 162,272 | 1.3380 | 1221 | 90K |
| BPE512 D d = 72 | 127,328 | 1.2671 | 510 | 53K |
| BPE512 D d = 80 | 150,432 | **1.2479** | 510 | 53K |

**H17 supported (single seed).** BPE d = 80 beats character d = 96 by 0.090 BPC (predicted ≥ 0.03) with 7% fewer
parameters; even BPE d = 72 (127K) beats the 162K character model by 0.071. It does so in 510 optimizer steps against
1221, with the warm-up (200 steps) a large part of that budget, so the recipe is not tuned for it. The 10M-char sample is
syntactically better but contains invented words ("feop", "nide"): BPE pieces recombine into non-words, so the
non-word rate must be measured, not assumed. Registered next: **H18** (`v5-bpe30`, 2 seeds, 30M chars, d = 80, saved):
val BPC ≤ 1.15 (char D d = 96 at 30M is 1.217); and the non-word rate of its samples at T 0.7 / top-k 20 is ≤ 2%.

### Outcome `v5-bpe30` (H18; D d = 80, BPE512, 30M chars, 2 seeds, lr 3e-2)

| model | params | val BPC |
|---|---|---|
| **BPE512 D d = 80** | 150,432 | **1.1028 ± 0.0054** (1.108 / 1.097) |
| char D d = 96 (`v3-big`) | 162,272 | 1.2170 ± 0.0054 |
| neural delta-mlp d = 128, char, 60M chars (teacher) | 280,480 | 1.1153 (1 seed) |

**H18 supported on both parts.** BPC 1.103 ≤ 1.15, and the non-word rate of seed-0 samples at T 0.7 / top-k 20 is 1.5%
(≤ 2%; the character models are at 0.4–0.9%, so the subword model trades more invented words for better structure;
at T 1.0 it is 5.9% against 2.4–3.7%). The subword model beats a 280K character model that trained twice as long.
Samples: more story structure (dialogue, "something unexpected happened") but still semantic errors (a girl named Lily
"wants to play with Lily"), so BPC is not coherence. **Caveats:** two seeds; one tokenizer size (512) and one model
size; the comparison is at equal *characters*, not equal steps or wall-clock (BPE sees 2.4× more characters per step,
runs at 53K chars/s against the character model's 34K for D); the 150K model is over the original 100K cap.
Not yet tried: BPE vocabularies 256 / 1024, d = 72 at 30M (127K), longer training, any coherence grader.

## Registered 2026-10-10 (night), before `v5-sub100k`: the best subword model that fits under 100K parameters

The 1.103-BPC subword model (150K) is over the original cap. Two subword configurations fit: BPE384 d = 64 (97,824
params, 1.97 chars/token) and BPE512 d = 56 (86,496 params, 2.40 chars/token). (BPE256 is the byte alphabet with no
merges, 1.00 chars/token, so it is the character model; dropped.) D architecture, 8 heads, 30M chars, lr 3e-2, seeds 0–1, saved.
Seen before registering: the 150K result only; nothing at these sizes.

* **H19.** The better of the two beats the sub-100K *character* D (97,520 params, 1.2753 at 30M, 2 seeds) by ≥ 0.08 BPC.
  *(~65%.)* *Basis:* the gain at 150K was 0.114 over the 162K character model; shrinking the width lowers the benefit, and
  the embedding is a larger share of a small model.
* **H20.** The 384-vocabulary d = 64 model beats the 512-vocabulary d = 56 model (the wider model wins over the
  richer vocabulary at equal budget). *(~55%.)*
* Not covered: ten seeds; BPE vocabularies other than these; any claim about coherence.

### Outcome `v5-sub100k` (30M chars, 2 seeds)

| model | params | chars/token | val BPC |
|---|---|---|---|
| **BPE512 D d = 56** | 86,496 | 2.40 | **1.2054 ± 0.0090** (1.196 / 1.214) |
| BPE384 D d = 64 | 97,824 | 1.97 | 1.2260 ± 0.0024 |
| char D d = 72 (previous best under 100K) | 97,520 | 1.00 | 1.2753 |

**H19 refuted, narrowly:** the gain is 0.070, not ≥ 0.08. **H20 refuted:** the richer vocabulary beats the wider model
(1.205 vs 1.226). The practical result stands: an 86.5K-parameter subword model at 30M characters (1.205) matches the
97.5K character model after 100M characters (1.208, one seed), and 13.5K parameters remain unspent.

## Registered 2026-10-10 (night), before `v5-spend`: spending the last of the 100K budget on subwords (screen)

10M chars, seeds 0–1, lr 3e-2, **warm-up 100 steps for every arm** (at 2.4–3.2 chars/token, 10M chars is only 380–510
steps, so the default 200-step warm-up would dominate; the baseline gets the same). All arms < 100K params:
(a) BPE512 d = 56 (86.5K, baseline), (b) same with FFN rules 2.0× d (94.6K), (c) same with 24-dim antecedent (90.1K),
(d) BPE768 d = 48 (81.1K), (e) BPE1024 d = 48 (93.3K).
* **H21.** At least one of (b)–(e) beats (a) by ≥ 0.02 BPC (seed noise at this stage is ~0.01). *(~60%.)* My lean is (e):
  a bigger vocabulary shortens the sequence further and 94K params still fits. A winner is re-run at 30M chars, 2 seeds,
  before it is quoted.

### Outcome `v5-spend` (10M chars, 2 seeds, warm-up 100)

| arm | params | val BPC |
|---|---|---|
| (e) BPE1024 d = 48 | 93,344 | **1.2604 ± 0.0050** |
| (d) BPE768 d = 48 | 81,056 | 1.3080 |
| (b) BPE512 d = 56, FFN 2.0× | 94,616 | 1.3271 |
| (a) BPE512 d = 56 (baseline) | 86,496 | 1.3335 |
| (c) BPE512 d = 56, 24-dim antecedent | 90,080 | 1.3413 |

**H21 supported:** (e) beats the baseline by 0.073 (both seeds). More FFN rules or a wider antecedent buy ≈ 0 (b: −0.006,
c: +0.008, inside the 0.01 noise); a larger vocabulary buys the most, and the ordering 512 < 768 < 1024 is monotone
in vocabulary at roughly constant parameters. The trend is untested above 1024 (a 2048 vocabulary at d = 40 would
exceed the cap).

## Registered 2026-10-10 (night), before `v5-final100k`: confirm the best sub-100K subword model

BPE1024 d = 48 (93,344 params), lr 3e-2, warm-up 100, saved. Cells: 30M chars seeds 0–1; 100M chars seed 0.
* **H22.** At 30M chars it scores ≤ 1.15 BPC (the 87K BPE512 model scored 1.205; the 10M gain was 0.073 and shrinks
  with training). *(~55%.)*
* **H23.** At 100M chars it scores ≤ 1.08. *(~50%.)* For scale: the 280K character teacher scored 1.115 at 60M chars.
* Samples from the 30M checkpoint are scored for non-words (≤ 2% at T 0.7 / top-k 20 as before) before any claim about text.

### Outcome `v5-final100k`, 30M part (BPE1024 d = 48, 93,344 params, 2 seeds)

**H22: BPC part supported, non-word part not met.** 30M-char val BPC **1.1292 ± 0.0031** (1.126 / 1.132) at 93,344 params,
against 1.2753 for the previous best under 100K (character D) and 1.2054 for BPE512 d = 56. The non-word rate of
its samples is **2.2%** at T 0.7 / top-k 20, above the registered ≤ 2%: a larger vocabulary invents more words
("pursess", "cractor"). Sampling settings fix it without changing the model (seed-0 checkpoint, 12 samples × 300 tokens):

| setting | non-word rate | repeated-sentence share |
|---|---|---|
| T 0.7, top-k 20 | 2.15% | 0.0% |
| T 0.7, top-k 5 | 1.33% | 0.9% |
| T 0.6, top-k 10 | 0.94% | 1.1% |
| **T 0.5, top-k 10** | **0.65%** | 2.9% |
| T 0.4, no top-k | 0.66% | 3.7% |

At T 0.5 / top-k 10 the subword model's non-word rate is below the character models' (0.4–0.9% at T 0.7), with a
small repetition cost. Text is still not coherent (the 30M sample at T 0.7 shows a story arc of sorts, "Mia found a big
red ball … put the ball in the park", then incoherent speech).

### Outcome `v5-final100k`, 100M part (H23)

BPE1024 d = 48 (93,344 params), 100M chars, seed 0: **val BPC 1.0548** (H23 ≤ 1.08: met, one seed). For scale: the 97.5K
character model scored 1.208 at 100M, and the 280K character teacher 1.115 at 60M. At T 0.5 / top-k 10 the text reads
as simple TinyStories prose but loops ("They did not like the box. They did not like the box.") and loses the plot, so
low temperature trades invented words for repetition (repeated-sentence share 2.9%, measured on 12 samples).

## Registered 2026-10-10 (late), before `v5-long`: does more data keep paying for the 93K subword model?

BPE1024 d = 48 (93,344 params). 100M chars scored 1.0548 (seed 0, lr 3e-2); 30M scored 1.129 (2 seeds): −0.074 per
3.3× data. Cells: 300M chars, seed 0, lr ∈ {3e-2, 2e-2}, warm-up 100, saved. Training throughput is ~112K chars/s alone,
so each cell is ~45 min of GPU; two run concurrently. Not decided: ten seeds, or any coherence claim.
* **H24.** 300M chars scores ≤ 1.00 BPC at the better lr. *(~50%.)* *Basis:* the 30M → 100M step was −0.074;
  a 3× step again at a decaying return gives ~−0.04 to −0.06, i.e. 0.995–1.015, so this is close to a coin flip.
* **H25.** The better lr at 300M is 2e-2, not 3e-2. *(~55%.)* *Basis:* optimal lr usually falls with token count.
* **Stopping rule:** if 300M is ≥ 1.03, further data is not the lever and the next step is the architecture (rules, heads),
  not more compute.

**Limitation, added 2026-10-10 (night; raised by the CPU session).** "The best model under 100K is a fuzzy subword model" is a
statement about **subwords**, not yet about **fuzziness**: no neural model on the same BPE1024 tokenizer at 93,344
parameters has been run, and D's learning rate on subwords was only ever 3e-2 (2e-2 is running in `v5-long`). The earlier
fuzzy-vs-neural comparisons are character-level, 2–3 seeds, and below the ten-seed protocol. The neural BPE control is
defined in `flm/grids_v6.py` (`v6-d-lr`, `v6-ref-tune`, `v6-final10`, ~2–3 GPU-hours; needs the user's go-ahead).

### Outcome `v5-long` (BPE1024 D d = 48, 93,344 params, seed 0, 300M chars, ~19 min each, two cells concurrent)

| lr | val BPC at 25 / 50 / 75 / 100% | final |
|---|---|---|
| 3e-2 | 1.1206 / 1.0773 / 1.0420 / 1.0212 | **1.0212** |
| 2e-2 | 1.1178 / 1.0730 / 1.0401 / 1.0227 | 1.0227 |

**H24 refuted (narrowly):** 1.021 is not ≤ 1.00. **H25 refuted:** the two learning rates differ by 0.0015, inside seed noise
(the lower lr leads until the last quarter and loses at the end of the cosine). Returns on data are shrinking: 30M → 100M gave
−0.074, 100M → 300M gave −0.034 (one seed each), so 1B characters would be expected to give about −0.02 (≈ 1.00) for ~1 h of GPU.
The registered stopping rule (≥ 1.03 means data is not the lever) did not trigger: more data still helps, with diminishing return.
Single seed, so differences under ~0.01 are not established.
### Screen stage 2 outcome (CPU session, 2026-10-10 evening; 8 more runs, 0 failures)

Merged with stage 1 (`outputs/screen-all/summary.md`; `python -m flm.screen_report screen screen2`).
d = 16, 10M chars, final val BPC. The base now has **4 seeds: mean 2.0767, std 0.0258** (the two-seed
spread of 0.050 in stage 1 overstated it). "Real" = |Δ of means| > 2 standard errors.

| variant | seeds | final BPC (mean ± std) | Δ vs base | verdict |
|---|---|---|---|---|
| A base | 4 | 2.0767 ± 0.026 | | |
| **D** bigram n-gram init, full | 3 | 2.1041 ± 0.045 | **+0.027** | within noise (2 se = 0.039) |
| **E** 2-tap antecedent | 3 | 2.0466 ± 0.012 | −0.030 | within noise (just under the bar) |
| **H** HTSK on the FFN only | 3 | 2.0520 ± 0.017 | −0.025 | within noise |

**P10 (D within noise of A): supported.** D's stage-1 "−0.050" was one lucky seed; its three-seed mean
is nominally *worse* than the base. **P11 (E real and ≥ 0.03): not met** (−0.030, but not real). **P12 (H real
and ≥ 0.02): not met** (−0.025, not real). Under the registered promotion rule, **nothing at d = 16 on the
zero-order FFN is promoted.** The n-gram initialisation is a null here: its starting-BPC advantage (3.70 vs
6.62) is gone within ~100 steps and the final BPC is no better; E and H are consistent with small gains
(≈ 0.025–0.03) that this design cannot resolve (se of a 3-vs-4-seed difference ≈ 0.02).

What this does and does not say: it is a result for the **old zero-order, softmax-normalised FFN** at 8.6K
parameters. D's FFN (first-order, raw firing) has far lower seed noise (0.0003–0.002 at 10M chars), so the
same ideas are testable there at 0.01 (`v4-taps`, `PLAN_GPU.md` §7); this stage-2 result is a prior for the
sign and size of E, not evidence about D. Lesson recorded: with 2 base seeds the noise estimate was
off by 35%, and a 1-seed variant ranking is not usable at this design's noise.

## 2026-10-10 (late): running the neural BPE control's tuning grids (`v6-d-lr`, `v6-ref-tune`)

Smoke cells (D lr 2e-2 and delta h4 sc4 lr 2e-2, 10M chars, seed 0) ran end to end: 93,344 vs 93,376 params (0.03% apart),
1.2765 vs 1.2854 BPC, ~30 s per cell. At this speed the two tuning grids (8 + 24 cells) are ~15 min of GPU, so they are
run now. Predictions (set by the CPU session's plan, not changed): none scored here until the grids finish. `v6-final10`
(30 cells × 30M chars, ten seeds; about an hour) is **not** started and waits for the user's go-ahead.

### Outcome `v6-d-lr` and `v6-ref-tune` (BPE1024, 93.3K params, 10M chars; D seeds 0–1, reference seed 0; warm-up 100)

| arm | best cell | val BPC |
|---|---|---|
| D (fuzzy) | lr **6e-2** (4e-2: 1.2542, 3e-2: 1.2604, 2e-2: 1.2809) | **1.2502 ± 0.0013** |
| neural, tuned over 24 cells (delta/gla × heads × short-conv × lr) | delta, 8 heads, short-conv 4, lr 4e-2 (93,392 params) | **1.2666** |
| neural, 2nd best | delta, 4 heads, short-conv 4, lr 6e-2 | 1.2694 |

At 10M chars the tuned fuzzy model leads the tuned neural model on the same tokenizer by **0.016 BPC** (D's two seeds differ by
0.002, so this is well outside D's seed noise; the reference is one seed). **Two caveats before anyone quotes it:** (1) D's best
lr (6e-2) is the **edge** of this grid, so D is not yet tuned; the grid is extended to 8e-2 and 1e-1 (`v6-d-lr-edge`) before
this is read; (2) it is 10M chars, not the 30M / ten-seed protocol, so `v6-final10` (30 cells, ~1 h, needs the user) is still the
citable comparison. Also noted: the neural reference's lr optimum (4e-2) is inside its grid.

**Edge check `v6-d-lr-edge` (2 seeds, 10M chars):** lr 8e-2 → 1.2519 ± 0.0013, lr 1e-1 → 1.2628 ± 0.0055. Both are worse than 6e-2 (1.2502),
so D's optimum is interior and D is tuned at 10M chars. The 0.016 lead over the tuned neural reference stands at 10M chars.
## CPU-session notes, 2026-10-11 (rulebook of the best sub-100K model; a cross-platform check)

**Cross-platform check (not a hypothesis).** The GPU session's BPE1024 D checkpoint (`outputs/v5-final100k/
D-d48-bpe1024_100M_s0.pt`, 93,344 params) was loaded on the CPU laptop with the committed tokenizer
(`data/tinystories/bpe1024.json`) and a validation token stream rebuilt from it (`flm.bpe.encode_stream`; token character
lengths sum to the 22,115,781-character stream, 3.162 chars/token). `flm.train.evaluate` returns **1.0548**, identical to
the recorded GPU value to four decimals. So that record reproduces across platforms, and the tokenizer and stream are the
same as the GPU session's.

**Rulebook** (`flm/rules.py`, `outputs/v5-final100k/D-d48-bpe1024_100M_s0.rules.md`; 400 validation windows). In layer 0, 63
of the 72 first-order feed-forward rules win at more than 0.1% of positions, so the rule bank is not collapsed. A
rule's output is gated by a signed gain, and the sign flips what it pushes. Readable examples: rule 30 fires after
sentence-ending punctuation (`.`, `!`, `?`) and, at positive gain (+3.97 ± 1.46), raises sentence starters (`·Th`, `↵`,
`·He`, `·Lucy`); rule 49 fires after `·very`, `'m`, `·up` and, at negative gain, raises `·surprised` and `·not`.
Limits, stated plainly: the logit lens reads only the direct path (final norm, tied embedding), many rules are word-piece
completions (`gener` → `ous`) whose reading is partial, and not every rule is interpretable. This shows the model *can* be
read rule by rule; it does not show the readings are faithful to the full computation.

**Correction to a transcribed number.** `RESULTS.md` (goal-track section 4) says D trains "at 3× lower throughput (34K vs
114K chars/s)". The 114K figure is the *smaller* 73K delta-mlp at 100M chars. Against the size-near reference (d72, 8 heads)
the recorded speeds are 50K / 56K / 85K chars/s on the shared GPU, so the gap is roughly 1.5–2.5× and noisy. The dashboards
and `WRITEUP.md` show the range. This file is the GPU session's, so the sentence there is left for it to fix.

## Registered 2026-10-11, before `v6-final10`: the ten-seed comparison on the same tokenizer

Arms (30M chars, seeds 0–9, BPE1024, ~93.3K params, warm-up 100): D at lr 6e-2; ref1 = delta h8 sc4 lr 4e-2 (93,392); ref2 =
delta h4 sc4 lr 6e-2 (93,376). Both references are the two best of the 24-cell tune (10M chars, seed 0, so the selection has a
winner's-curse bias *against* nothing here: it favours the references). Decision rule from `flm.compare`: a lead is **real**
if |Δ of means| > 2 standard errors. Cost: 30 cells ≈ 1 h with 4 workers; run under the user's standing "free reign to design and
run experiments" instruction, one sweep at a time.
* **H26.** D's ten-seed mean is lower than both references' by a *real* margin. Predicted lead **0.01–0.02 BPC** over the better
  reference. *(~60% it stays real: the 10M lead was 0.016 with a one-seed reference and the lead usually shrinks with training.)*
  A null ("within noise") would be reported as the result, with the 1.5× throughput cost.

### Outcome `v6-final10` (H26; BPE1024, ~93.3K params, 30M chars, seeds 0–9, 30/30 finite, `python -m flm.compare v6-final10 --ref D`)

| arm | params | val BPC (mean ± std, n = 10) | Δ vs D | 2 × s.e. | verdict |
|---|---|---|---|---|---|
| **D (fuzzy), lr 6e-2** | 93,344 | **1.1291 ± 0.0039** | | | |
| ref2: delta h4, sc4, lr 6e-2 | 93,376 | 1.1564 ± 0.0044 | +0.0274 | 0.0037 | D better, real |
| ref1: delta h8, sc4, lr 4e-2 | 93,392 | 1.1593 ± 0.0038 | +0.0303 | 0.0034 | D better, real |

**H26 supported on direction and reality, missed on size.** The fuzzy model beats both tuned neural references on the same
tokenizer at the same parameter count by 0.027–0.030 BPC, about 8× the registered 2-standard-error bar; the predicted lead was 0.01–0.02,
so the lead is larger than predicted. The references were selected at 10M chars, seed 0, over 24 cells (winner's curse favours
them, not D). Two incidental findings: (1) D at lr 6e-2 (1.1291, ten seeds) equals D at lr 3e-2 (1.1292, two seeds, `v5-final100k`), so
at 30M chars the learning rate between 3e-2 and 6e-2 does not matter; the 10M-char optimum of 6e-2 does not carry any advantage at 30M;
(2) D's seed std (0.0039) is not smaller than the references' (0.0038–0.0044), so D is not more fragile.
**Standing caveats:** one tokenizer (BPE1024), one size (93K), one data budget (30M chars), no 100M/300M ten-seed run, and D trains ≈ 1.5× slower
per character. The comparison is at equal parameters and equal characters; at equal wall-clock the neural models would see 1.5× more text.
The character-level claim is a separate, older comparison (2–3 seeds, `v3-final`/`v3-ref`).

## Registered 2026-10-11, before `v7-vocab`: does a bigger vocabulary still help under 100K parameters?

The vocabulary ladder at roughly constant parameters was monotone: 512 (d 56, 86K) 1.334 < 768 (d 48, 81K) 1.308 < 1024 (d 48, 93K) 1.260
at 10M chars. Same recipe (D architecture, lr 3e-2, warm-up 100, 10M chars, seeds 0–1, 8 heads unless noted) for the sizes that fit:
BPE1536 d = 40 (93,792), BPE2048 d = 32 (87,840), BPE1536 d = 36 with 4 heads (81,808). Baseline: BPE1024 d = 48, 1.2604.
* **H27.** The best of the three beats 1.2604 by ≥ 0.02 BPC. *(~35%.)* *Basis:* each step up so far gained 0.03–0.07, but the width falls
  from 48 to 40 / 32, where the mixer and FFN get thin (dh = 5 or 4 per head) and the embedding is 66–74% of the parameters, so the
  trend is more likely to turn over here than to continue. A null or a loss would place the optimum at ≈ 1024 under this cap.

### Outcome `v7-vocab` (10M chars, 2 seeds, lr 3e-2, warm-up 100)

| arm | params | val BPC |
|---|---|---|
| BPE1536 d = 40, 8 heads | 93,792 | **1.2336 ± 0.0032** |
| BPE2048 d = 32, 8 heads | 87,840 | 1.2378 ± 0.0012 |
| BPE1536 d = 36, 4 heads | 81,808 | 1.2715 ± 0.0074 |
| BPE1024 d = 48 (baseline) | 93,344 | 1.2604 ± 0.0050 |

**H27 supported (predicted ~35%):** the best arm beats the 1024 baseline by 0.027. The trend continued to 1536 and 2048 even though the width fell
to 40 and 32. 4 heads cost 0.038 (so heads, not just width, matter, as before). 2048 is within 0.004 of 1536 at 6K fewer params, so the optimum
vocabulary under this cap is somewhere in 1536–2048+; not found.

## Registered 2026-10-11, before `v7-confirm`: do the vocabulary gains hold at 30M chars?

BPE1536 d = 40 and BPE2048 d = 32 (8 heads, lr 3e-2, warm-up 100), 30M chars, seeds 0–1. Reference: BPE1024 d = 48 at 30M, lr 3e-2:
1.1292 (2 seeds); ten-seed lr 6e-2: 1.1291.
* **H28.** The better of the two scores ≤ 1.115 at 30M (a gain ≥ 0.014; the 10M gain was 0.027 and gains usually shrink with training).
  *(~50%.)* If it does, the best sub-100K model changes (and the neural control on the same tokenizer is no longer matched: a neural 1536 reference
  would be owed before a fuzzy-vs-neural claim at that vocabulary).

### Outcome `v7-confirm` (30M chars, 2 seeds, lr 3e-2, warm-up 100)

| arm | params | val BPC | vs BPE1024 d = 48 (1.1291, ten seeds) |
|---|---|---|---|
| BPE1536 d = 40 | 93,792 | **1.1205 ± 0.0012** (1.122 / 1.119) | −0.0086 |
| BPE2048 d = 32 | 87,840 | 1.1316 ± 0.0026 | +0.0025 |

**H28 missed, narrowly (predicted ~50%):** 1.1205 is not ≤ 1.115. The gain from 1024 → 1536 shrank from 0.027 (10M chars) to 0.009 (30M), and
2048 lost its 10M edge entirely. D's ten-seed std is 0.0039, so a two-seed 0.009 gap is only about 2 standard errors and is **not** a
confirmed improvement. Reading: under a 100K cap the vocabulary optimum is ≈ 1024–1536 at 30M chars, and the 10M-char screens overstated
larger vocabularies (short runs favour tokens that cover more text per step). **The ten-seed-validated headline stays BPE1024 d = 48
(1.1291 ± 0.0039, beats tuned neural references by 0.027–0.030).** BPE1536 d = 40 is a candidate that would need ten seeds
and its own neural control to replace it. 100M / 300M numbers for 1536 were not run.

## Registered and run 2026-10-11: a model-graded coherence measure (`flm/coherence_set.py`, `report/coherence_*.json`)

**Why:** every claim above is about bits per character; the user's goal is coherent text, and until now it was judged by reading. **Method:** 7 checkpoints ×
8 (prompt, seed) pairs, all at T 0.5 / top-k 10, each cut to exactly 600 characters (so character and subword models are compared on equal
text length), shuffled and anonymised (`coherence_blind.json`). **Grader: Claude, this session, one rater, blind** (scores written to
`coherence_scores.json` *before* the key was opened). Rubric, 1–5 each: **G** grammar; **C** consistency of characters and objects (names kept, no
swapping, no vanishing); **P** sense / plot (does it go anywhere). Total 3–15. A 600-character window can span the end of one story and the start of another;
all of it was graded.
* **GC1 (predicted before scoring).** The ranking of total score follows the ranking of validation BPC. *(~80%.)*
* **GC2.** At matched BPC, a subword model scores no lower than a character model. *(~50%; I had no basis for a direction.)*

| model | params | val BPC | G | C | P | total (3–15) ± s.e. |
|---|---|---|---|---|---|---|
| BPE1024 D, 300M chars | 93,344 | 1.021 | 4.00 | 3.38 | 2.50 | **9.88 ± 0.28** |
| BPE1024 D, 100M chars | 93,344 | 1.055 | 3.38 | 2.38 | 1.88 | 7.62 ± 0.50 |
| BPE1536 D d = 40, 30M | 93,792 | 1.119 | 3.50 | 2.12 | 1.75 | 7.38 ± 0.47 |
| BPE512 D d = 80, 30M | 150,432 | 1.108 | 3.25 | 2.25 | 1.88 | 7.38 ± 0.39 |
| neural delta-mlp d = 128, char, 60M | 280,480 | 1.115 | 2.62 | 2.00 | 1.38 | 6.00 ± 0.53 |
| char D d = 96, 30M | 162,272 | 1.222 | 2.38 | 1.50 | 1.00 | 4.88 ± 0.28 |
| char D d = 72, 30M | 97,520 | 1.282 | 1.75 | 1.12 | 1.00 | 3.88 ± 0.21 |

**GC1 supported:** the order of total score is the order of BPC for 6 of 7 models (the exception: BPE512 150K, BPC 1.108, ties the 1536 model, 1.119, within error).
**GC2 supported but weak:** the subword 1536 model (BPC 1.119) scores 7.4 against the character teacher (BPC 1.115) at 6.0, a gap of ~1.4 totals (~2 s.e.).
**What it says:** the 300M-character model is the first to write mostly-grammatical, mostly-consistent short stories (G 4.0, C 3.4), and its P score
(2.5/5) shows that plot sense is still the weak axis: stories stay on one topic and loop rather than develop. Rough slope: ~2.4 total points per 0.1 BPC
between 1.28 and 1.02 (extrapolating it to a "good" score of 12–13 would need ≈ 0.9 BPC; a guess, not a measurement).
**Caveats:** one rater, who is also the author's model (no inter-rater check, preference for repetition-free text); n = 8 per model, so differences below ~1.5
totals are not established; one sampling setting; the 600-character window; BPC for the BPE1536 row is the 2-seed 30M mean (1.1205), for the graded seed-0 checkpoint 1.122.

> Naming note: the GPU session's batch-32 context test below is `v8a-ctx` (hypotheses G29 / G29b). The CPU session's `v8-ctx` grid and its H29 (`PLAN_CONTEXT.md`) are a different experiment.

## Registered 2026-10-11, before `v8a-ctx`: does a longer context help (plot needs longer memory)?

Coherence grading (above) found plot sense the weakest axis. The BPE1024 model reads 256 tokens ≈ 900 characters; TinyStories are often longer. Cells:
BPE1024 D d = 48, 30M chars, lr 3e-2, warm-up 100, seeds 0–1, context ∈ {128, 512} tokens (baseline: 256 tokens, 1.1292 at lr 3e-2 two seeds; 1.1291 ten seeds
at lr 6e-2). The validation metric is the same bits per character on the same text, but its windows are `ctx`-long (non-overlapping), so a longer-context model is also evaluated
with a longer window; the comparison is "best model you can build with this context", not "same model, more context at test time".
* **G29.** ctx 512 beats ctx 256 by ≥ 0.01 BPC, and ctx 128 is worse than ctx 256 by ≥ 0.02. *(~65% for both.)* *Basis:* in-context learning on a short
  story gets better with more context; the delta-rule state decays (data-dependent) so the gain may flatten.
* A coherence re-grade of the 512 model is run only if G29's first part holds.

### Outcome `v8a-ctx` as registered (30M chars, 2 seeds, batch 32)

| context | val BPC | steps |
|---|---|---|
| 128 tokens | 1.1357 ± 0.0015 | ≈ 2300 |
| 256 tokens (baseline) | 1.1292 (2 seeds) | ≈ 1160 |
| 512 tokens | 1.1570 ± 0.0020 | ≈ 580 |

**G29 refuted as registered** (512 is worse by 0.028, not better by 0.01; 128 is worse by only 0.007, not by ≥ 0.02). **But the design is confounded:** at a fixed
character budget and batch size, doubling the context halves the number of optimizer steps (≈ 580 vs 1160), and the warm-up/cosine schedule is the same. So this
cannot separate "longer context hurts" from "fewer updates hurt". Registered amendment, **G29b**: hold tokens per step constant (ctx 512 with batch 16, ≈ 1160 steps, the
same as the 256 baseline), 2 seeds; if ctx 512 is then ≥ 0.01 better than 1.1292, the first run's loss was the step count, not the context. *(~45%.)*

### Outcome `v8-ctx-b16` (G29b: ctx 512, batch 16, 1160 steps = the ctx-256 baseline's step count; 30M chars, 2 seeds)

ctx 512: **1.1226 ± 0.0034** (1.126 / 1.119) against ctx 256: 1.1292, a gain of 0.0066. **G29b missed (needed ≥ 0.01; predicted ~45%).** It does settle the confound:
the batch-32 ctx-512 run's 1.1570 was a step-count effect (580 steps), not a context effect; with steps matched the longer context is slightly *better*, by about
1.5–2 seed standard errors (not established). ctx 128 at batch 32 (2× the steps) was only 0.007 worse than 256, so context length is a weak lever for BPE1024 D
at this size; the validation windows also lengthen with ctx, which flatters the longer-context models somewhat.

**Operational finding (2026-10-11):** two concurrent `ctx 512 / batch 16` jobs ran > 40 min with no first eval and ~11 GB VRAM, while the same cell alone takes
**86 s**. On this laptop GPU, running heavy cells concurrently can stall everything (VRAM spill); run long-context or large-batch cells **one at a time**.
(Earlier 4-way concurrent batch-32 ctx-512 jobs did finish, so the failure is not deterministic.)
### Second rater for the blinded coherence set (CPU session, 2026-10-11)

A second Claude instance (the CPU session, which had not seen the key or the first rater's scores) scored all 56 texts in `report/coherence_blind.json` on grammar, consistency and plot, 1-5 each, before opening `coherence_key.json`; the scores are in `report/coherence_scores_rater2.json`. Both raters are the same model family, so this checks rater noise, not rater bias.

- Agreement: total-score correlation 0.83 (G 0.79, C 0.75, P 0.59); 42 of 56 totals within 1 point. Rater 2 is harsher by 0.75 points on average (5.96 against 6.71), mostly on plot.
- Rater 2 per-model totals (/15, mean ± se, n = 8): BPE1024 300M 8.62 ± 0.37; BPE1024 100M 7.25 ± 0.53; BPE1536 d40 30M 7.12 ± 0.40; BPE512 30M 6.62 ± 0.37; neural char teacher 4.38 ± 0.26; char D d96 3.88 ± 0.12; char D d72 3.88 ± 0.12.
- The ranking by bits per character is reproduced (the two lowest character-D models tie). The subword-vs-character-teacher gap at similar BPC is larger under rater 2 (BPE1024 100M against the teacher: 2.9 points, against 1.6 for rater 1), so the weak-evidence reading holds in direction and is not smaller than first reported. Plot is the weakest axis for both raters (rater 2: 1.88/5 at 300M).
- Caveats unchanged: same model family, n = 8 per model, one sampling setting. Neither rater is a human.

## Registered 2026-10-11, before `v8-ctx`: does more context help D? (CPU session; plan in PLAN_CONTEXT.md)

D is the fuzzy recurrent mixer (`fuzzydelta`), so context scaling is the recurrent-model question.
Probes: `flm/context_probe.py` (bits by position, bits by distance from story start, name recall).

* **H29 (baseline, 100M BPE1024 D checkpoint, ctx 256), registered before running the probe:** name-recall gain
  above 2 bits at k = 0 and near 0 by k = 24; bits drop from the 0-255 bucket to the 256-511 bucket and do
  not improve past 256. **Outcome (24 windows of 1,024 tokens, CPU):** gain 2.01 bits at k = 0 (borderline), 1.75 (k=1),
  1.26 (k=3), 1.35 (k=6), 1.07 (k=12), 0.79 (k=24, 280 tokens): decays smoothly, not to 0, so "near 0 by 24" is
  **refuted**. Bits by position 1.072 / 1.080 / 1.061 for 0-255 / 256-511 / 512-1023: no drop at the second
  bucket (**refuted**) and a small improvement past 256 (**refuted**, but 24 windows, so within noise);
  the model run at 4x its training length does not break down. Bits by story distance: 0.900 (0-63 tokens),
  1.11-1.12 afterwards: the story opening is cheap, the body is flat. The prior margin is 1.4-2.3 bits, so the
  gain is measured against a name preference of similar size; treat k >= 12 as weak. Not yet seen at ten seeds
  or on other checkpoints.
* **H30:** name-recall gain at k = 24 (280 tokens) rises with training ctx (256 < 512 < 1024) by more than 0.3 bits. ~50%.
* **H31:** `val_bpc` at fixed position buckets improves by less than 0.01 from ctx 256 to 1024 (stories are about one
  window long). ~65%.
* **H32:** 32 rules per head at ctx 1024 beats 16 rules by more than 0.01 bits per character. ~30%.

## Registered 2026-10-11, before `v9-lr`/`v9-size`: where does coherence appear? (CPU session; PLAN_COHERENCE.md)

Success criteria fixed in advance: blind total >= 12/15 and plot >= 3.5/5 from both raters; explainability
requires a causal rulebook (rule-ablation of the most-cited FFN rules costs more than random rules).
H33 plot > 3/5 by the 3.3M model (~40%); H34 nothing under 2M reaches 12/15 (~85%); H35 ablation of cited rules
exceeds random at every size (~70%); H36 the best lr falls with size (~60%).

## Registered 2026-10-11, before `v10-fib`: Fibonacci attention at story size (CPU session; PLAN_FIBONACCI.md)

User direction: keep to story size, try local-plus-sparse-far memory. Predictions H37-H41 are in PLAN_FIBONACCI.md:
fibgauss-L8 within 0.01 of dense (~55%); L4 worse / L16 no better (~50%); fibsoftmax-L8 within 0.01 of fibgauss-L8 (~55%);
name recall at k = 12 close to dense but lower at non-Fibonacci lags (~40%); best attention arm does not beat D (~60%).
v9 (size ladder) is on hold; v8-ctx trimmed to ctx 256/512.

## Outcome `v10-fib` (GPU session, 2026-10-11): Fibonacci-lag attention vs dense Gaussian attention vs D (CPU session's grid, PLAN_FIBONACCI.md H37–H41)

BPE1024, 10M chars, seed 0, ctx 512 / batch 16 for the attention arms (D at ctx 256), all ~93–95K params, warm-up 100; **run one cell at a time** (each takes 6–36 s alone).
Single seed per cell, so differences < ~0.01 are not established.

| arm | params | lr 2e-2 | lr 6e-2 | best |
|---|---|---|---|---|
| **D (fuzzy delta)** | 93,344 | 1.2772 | 1.2534 | **1.2534** |
| fibgauss L = 16 | 94,944 | 1.3321 | 1.3209 | 1.3209 |
| fibgauss L = 8 | 94,944 | 1.3406 | 1.3227 | 1.3227 |
| fibgauss L = 4 | 94,944 | 1.3385 | 1.3242 | 1.3242 |
| fibsoftmax L = 8 | 94,896 | 1.3450 | 1.3344 | 1.3344 |
| dense Gaussian attention | 94,944 | 1.3602 | 1.3536 | 1.3536 |

* **H37 refuted** (predicted fibgauss-L8 within 0.01 of dense): Fibonacci is *better* than dense by **0.031** BPC at the best lr (and by 0.020 at lr 2e-2).
* **H38 refuted:** L4 is worse than L8 by only 0.0015 (needed > 0.01); L16 is better than L8 by 0.0018 (within the 0.01 band, as predicted for that half).
* **H39 refuted, marginally:** fibsoftmax-L8 is 0.0117 worse than fibgauss-L8 (needed ≤ 0.01).
* **H41 supported:** the best attention arm (1.3209) is 0.068 behind D (1.2534).
* **H40 not informative:** name-recall gain at k = 12 (bits; `flm.context_probe`, 56 pairs): dense 0.011, fibgauss-L8 0.000 → within 0.3 of each other as predicted, but
  **both are ≈ 0 at every k** (k = 0: 0.107 / 0.133; k ≥ 6: ≤ 0.04). **D: 2.10 (k = 0), 1.36 (k = 3), 0.85 (k = 12), 0.47 (k = 24).** The 10M-char attention models have not learned
  to retrieve a name at all; the delta-rule D has, at 10M chars and ctx 256.
* **Caveats:** 10M characters is short (attention induction behaviour is known to appear later than loss differences); the attention arms use 4 heads, ctx 512 and FFN mult 2.0 per
  the CPU session's grid, not a tuned configuration; one seed. So the safe reading is "at this budget and size, Fibonacci attention beats dense Gaussian attention by ~0.03 BPC,
  and neither approaches D on loss or on name recall"; not "attention cannot do it". A 30M-char two-seed run of the best attention arm (per the plan's rule) was not run.

## Outcome `v8-ctx` and `v8-rules` (GPU session ran the CPU session's grids, 2026-10-11; BPE1024 D d = 48, lr 6e-2, 30M chars, seeds 0–1, constant tokens per step)

| cell | params | val BPC |
|---|---|---|
| ctx 256, 16 rules/head | 93,344 | 1.1324 ± 0.0031 (1.129 / 1.135) |
| ctx 512, 16 rules/head | 93,344 | 1.1232 ± 0.0019 (1.125 / 1.121) |
| ctx 512, **32 rules/head** | 96,416 | **1.1125 ± 0.0004** (1.112 / 1.113) |

Context 512 vs 256: **−0.0092** (matches my `v8-ctx-b16` result, −0.0066 at lr 3e-2: both small, same sign). Doubling the rules per head at ctx 512: **−0.0107** for +3.1K params,
with a very small seed spread (0.0004). Both gains are ≈ 3 seed standard errors of the ctx-256 cell but only two seeds each; the 32-rule cell is the best BPE1024 D at 30M chars so far
(1.1125, under 100K parameters: 96,416) and the first improvement over the ten-seed 1.1291 headline that is not within seed noise. **Not ten-seed validated, and its own neural control
was not rerun at ctx 512.** (Predictions H30–H32 for these grids are the CPU session's in `RESULTS.md`; scoring is theirs.)
## Registered 2026-10-11, before `v11-hybrid`: D plus Fibonacci attention (CPU session; PLAN_FIBONACCI.md stage 2)

H42 the best hybrid beats D by > 0.01 bpc (~35%); H43 its name-recall gain at k = 24 exceeds D's 0.47 by > 0.3 bits (~35%);
H44 layer order matters, |hybDF - hybFD| > 0.01 (~55%); H45 the hybrid's cited FFN rules stay causal (~70%).

## Outcome `v11-hybrid` (GPU session, 2026-10-11; BPE1024, 94,144 params, 10M chars, seed 0, one cell at a time; hybDF = D layer then Fibonacci-attention layer, hybFD = attention first)

| cell | lr 3e-2 | lr 6e-2 | name-recall gain, bits, k = 0 / 3 / 12 / 24 (lr 6e-2) |
|---|---|---|---|
| hybFD (attention → D) | 1.2808 | 1.2811 | 1.13 / 0.74 / 0.32 / 0.11 |
| hybDF (D → attention) | 1.2866 | 1.2857 | 0.12 / 0.05 / 0.003 / 0.00 |
| pure D (`v10-fib`, lr 6e-2) | | 1.2534 | 2.10 / 1.36 / 0.85 / 0.47 |

* **H42 refuted:** the best hybrid (1.2808) is 0.027 *worse* than D (needed > 0.01 better; predicted ~35%).
* **H43 refuted:** hybFD's k = 24 recall gain is 0.106, far below D's 0.473 (needed > +0.3 above it).
* **H44 refuted on loss** (|hybDF − hybFD| = 0.005 < 0.01) **but order matters hugely for retrieval:** with the attention layer second (hybDF) the name recall is ≈ 0 (0.12 bits at k = 0),
  with D second (hybFD) it is 1.13 bits. D keeps the memory, and only when its state is read by the last layer; an attention layer after it destroys the memory path (my reading;
  not tested directly).
* **H45 not run** (`flm.rule_ablation` on the hybrid checkpoints was not run).
* **Reading:** swapping one of D's two recurrent layers for Fibonacci attention costs 0.027 BPC and about half the name recall, at the same parameter budget; the hybrid is not a gain at
  10M chars. One seed per cell, so differences < 0.01 are not established; the 0.027 loss is about 5× that.

## Registered 2026-10-11, before `v12-long512`: does the best 30M configuration (ctx 512, 32 rules per head) improve long-run loss and coherence?

BPE1024 D d = 48, 32 rules per head, ctx 512 / batch 16, lr 6e-2, warm-up 100, 300M chars, seed 0, saved (96,416 params). Reference: the same recipe at ctx 256 / 16 rules (lr 3e-2): 1.0212 at 300M.
* **G30.** val BPC ≤ 1.00 (a gain ≥ 0.02 over 1.0212; the 30M gain from ctx 512 plus 32 rules was ≈ 0.02 over ctx 256 / 16 rules). *(~55%.)*
* **G31.** Blind coherence total (same rubric, same 600-char protocol, new set mixing this model with the previous 300M model and the 100M model, labels removed) exceeds the previous 300M model's
  9.88 by ≥ 1.0. *(~40%.)* The plot score (P) is the one to watch.

### Outcome `v12-long512` (G30; BPE1024 D d = 48, 32 rules/head, ctx 512 / batch 16, lr 6e-2, 300M chars, seed 0)

**val BPC 0.9976** at **96,416 parameters** (curve at 25 / 50 / 75 / 100%: 1.1372 / 1.0804 / 1.0275 / 0.9976; 119K chars/s alone, ~25 min). **G30 supported (≤ 1.00), by 0.002**, which is
inside what one seed can move, so "below 1.0" is a one-seed statement; the margin over the previous 300M model (1.0212, ctx 256 / 16 rules, lr 3e-2) is 0.024. It is the first model under 100K
parameters below one bit per character. Single seed; no neural control at this configuration; the lr was 6e-2 here against 3e-2 in the reference run (D's lr barely mattered at 30M chars,
but this is a second difference besides ctx and rules).

### Outcome of coherence set 2 (G31; blind, fresh texts: seeds 2 and 3, same 4 prompts, T 0.5 / top-k 10, 600 chars; rater 1 = Claude GPU session; `report/coherence2_blind.json`, scores `coherence2_scores_rater1.json`, written before the key was opened; the key is withheld until rater 2 has scored)

| model | val BPC | G | C | P | total (3–15) ± s.e., n = 8 |
|---|---|---|---|---|---|
| BPE1024 D, ctx 512, 32 rules, 300M (new) | 0.9976 | 3.25 | 2.50 | 1.88 | **7.62 ± 0.35** |
| BPE1024 D, ctx 256, 300M | 1.0212 | 3.50 | 2.50 | 2.12 | 8.12 ± 0.51 |
| BPE1024 D, ctx 256, 100M | 1.0548 | 3.38 | 2.62 | 2.12 | 8.12 ± 0.65 |

(An earlier attempt reusing seeds 0–1 was discarded unscored-for-this-purpose: its texts were deterministic repeats of the first set, so rater 1 could tell which were the new model's. Seeds 2–3 fix that.)

**G31 refuted:** the new model does not exceed the previous 300M model's 9.88 by 1.0; on fresh texts it scores 7.62 against 8.12 for both older models, i.e. **no difference at this sample size, and the sign is the
wrong way**. **This also weakens the first coherence table:** the same 300M checkpoint scored 9.88 ± 0.28 on seeds 0–1 and 8.12 ± 0.51 on seeds 2–3 (a 1.8-point swing, ≈ 3 s.e.), so one 8-story sample
per model is a noisy estimate (the run-to-run noise of this measure is ~±1.5 totals). Pooling both rounds (n = 16): 300M ctx-256 **9.00**, 100M **7.87**; the new model has n = 8 only (7.62). **What survives:**
the large separations in round 1 (subword vs character models, 3.9 → 7–9) are far outside this noise; the *fine* ordering among the three best (BPC 0.998–1.055) is not established, and a 0.024 BPC gain
does not show up in the coherence score. Plot sense (P ≈ 2/5) did not move at all across 1.055 → 0.998 BPC.
## Registered 2026-10-12, before `v13-state`: does more recurrent state keep helping? (CPU session)

Verified from the records: v8-ctx ctx256 1.1324, ctx512 1.1232, ctx512 r32 1.1125 (30M characters, 2 seeds each).
`v13-state`: 64 and 128 rules per head at ctx 512 (102,560 and 114,848 parameters).
* **H46:** r64 beats r32 (1.1125) by more than 0.005 bits per character. ~50%.
* **H47:** r128 beats r64 by less than r64 beats r32 (diminishing returns). ~65%.
* **H48:** more rules make a rule's winning share more concentrated and the rulebook shorter to read, not longer: the
  number of rules needed to cover 80% of winning positions per head grows by less than 2x from r32 to r128. ~50%.

## Outcome `v13-state` (GPU session, 2026-10-11; BPE1024 D d = 48, ctx 512 / batch 16, lr 6e-2, 30M chars, seeds 0–1; one cell at a time, ~100 s each)

| rules per head | params | val BPC | Δ vs previous row |
|---|---|---|---|
| 16 (`v8-ctx`) | 93,344 | 1.1232 ± 0.0019 | |
| 32 (`v8-rules`) | 96,416 | 1.1125 ± 0.0004 | −0.0107 |
| 64 | 102,560 | 1.1113 ± 0.0030 (1.108 / 1.114) | −0.0012 |
| 128 | 114,848 | **1.0979 ± 0.0062** (1.104 / 1.092) | −0.0134 |

* **H46 refuted:** r64 beats r32 by 0.0012, not > 0.005.
* **H47 refuted:** the r64 → r128 step (−0.0134) is *larger* than the r32 → r64 step (−0.0012), so there is no diminishing return across this range; the curve is not smooth (16 → 32 and 64 → 128 gain, 32 → 64 does not).
  With two seeds per cell and r128's own spread 0.012 (seeds 1.104 / 1.092), the non-monotone shape could be seed noise on r64; a clean reading needs more seeds.
* **H48 (rulebook concentration) not tested.**
* **Cap:** r64 and r128 exceed the original 100K cap (102.6K / 114.8K); the user relaxed the cap to ≤ 165K for the bigger-model track. Under 100K the best remains r32 (96,416; 1.1125 at 30M, 0.9976 at 300M).
### Rule-ablation of the best model (CPU session, 2026-10-12)

`flm.rule_ablation` on `outputs/v12-long512/D-d48-bpe1024-ctx512-r32_300M_s0.pt` (0.9976 BPC on the full validation slice; 1.0484 on the
120 x 256-token windows used here), 8 rules silenced per layer, 20 random draws. Registered earlier for the 100M model only; this
is the same test on the new best model. Layer 0: cited rules +0.387 bits per character (they win 52.7% of positions), random 8 rules
+0.173 mean, 95th percentile +0.254. Layer 1: cited +0.167 (56% of positions), random mean +0.048, p95 +0.067. In both layers the cited
rules cost more than the random 95th percentile: the rulebook is causal for the model that gets 0.9976 BPC, as it was for the 100M model
(+0.226 vs p95 +0.157; +0.096 vs +0.052). One checkpoint, one seed; silencing is not renormalised (this FFN is un-normalised, `--ffn-norm none`).
Windows: the first 120 x 256 tokens from validation position 900,000, not the whole slice.

### Coherence set 2 with both raters (key published after rater 2 scored; rater 2 = CPU session, blind, wrote `coherence2_scores_rater2.json` before seeing the key)

Rater agreement on totals: r = 0.83 (rater 2 scores ~1.2 lower on average). Mean of the two raters per model (totals out of 15, n = 8 texts each, ± s.e. over texts):

| model | val BPC | rater 1 | rater 2 | mean of raters | plot (mean of raters, /5) |
|---|---|---|---|---|---|
| BPE1024 D, ctx 512, 32 rules, 300M | 0.9976 | 7.62 | 6.25 | **6.94 ± 0.30** | 1.69 |
| BPE1024 D, ctx 256, 300M | 1.0212 | 8.12 | 6.88 | 7.50 ± 0.57 | 1.81 |
| BPE1024 D, ctx 256, 100M | 1.0548 | 8.12 | 7.12 | 7.62 ± 0.60 | 1.88 |

**G31 refuted by both raters.** The 0.024-BPC gain from ctx 512 / 32 rules does not appear as a coherence gain (the sign is slightly negative, inside noise); plot sense is ≈ 1.7–1.9 / 5 for all three
at 0.998–1.055 BPC. The CPU session's criterion for "coherent" (a blind total ≥ 12/15 **and** plot ≥ 3.5/5 from both raters, with set-to-set noise ≈ ±1.5) is **not met by any model here**; the best measured
total across both rounds and both raters is ≈ 9 (rater 1, set 1, a sample that did not replicate on fresh seeds). Rule ablation (CPU session) says the FFN rules the model cites are causal, so the model is
not "decorating"; it is simply not producing plot at this size.
### Coherence set 2, both raters, and v13-state verified (CPU session, 2026-10-12)

Set 2 (24 texts, fresh seeds, n = 8 per model), totals /15 and plot /5, rater 1 / rater 2 (rater 2 scored blind; agreement r = 0.83 on totals):
0.9976-BPC ctx512 r32 300M model 7.62 / 6.25 (plot 1.88 / 1.50); previous 300M model 8.12 / 6.88 (2.12 / 1.50); 100M model 8.12 / 7.12 (2.12 / 1.62).
Both raters find no gain from the 1.05 -> 1.00 BPC improvement, and every model sits at 6-8/15 with plot 1.5-2.1/5, against the criterion of
12/15 and 3.5/5. Lower validation loss through context, rules, vocabulary and budget has not bought coherence at ~95K parameters.

`v13-state` verified from the records (ctx 512, 30M chars, 2 seeds): r64 1.1113 (1.1083, 1.1143), r128 1.0979 (1.1040, 1.0917), against r32 1.1125.
H46 (r64 beats r32 by > 0.005) refuted: 0.0012. H47 (diminishing returns) refuted in the other direction: r128 gains 0.0134 over r64, so more
state keeps helping once past 64 (102,560 and 114,848 parameters; non-monotone, 2 seeds, so r128 is the thing to confirm before believing the curve).

## Registered 2026-10-11, before `v14-rule-seeds`: are the rule-count steps real? (GPU session)

`v13-state` found r32 → r64 −0.0012 and r64 → r128 −0.0134 on two seeds, a non-smooth curve. Same config as `v8-rules` / `v13-state` (BPE1024 D d = 48, ctx 512 / batch 16, lr 6e-2, 30M chars), seeds 2–4 added for r32, r64 and r128
(9 cells, one at a time, ~100 s each). Pooled with seeds 0–1 this gives five seeds per rule count.
* **G49.** With five seeds, r128 beats r32 by ≥ 0.005 BPC (a real gain of the 2-s.e. kind, using `flm.compare`'s rule). *(~65%.)*
* **G50.** With five seeds, r64 is within 0.005 of r32 (the smooth curve was a two-seed artefact, or r64 is truly no better). *(~55%.)*
* The cap: r64 (102.6K) and r128 (114.8K) exceed 100K; this is a question about the relaxed-cap track (≤ 165K) and about the shape of the rule-count curve, not a new sub-100K model.

### Outcome `v14-rule-seeds` (G49 / G50; BPE1024 D d = 48, ctx 512 / batch 16, lr 6e-2, 30M chars, seeds 0–4 pooled from `v8-rules`, `v13-state`, `v14-rule-seeds`)

| rules per head | params | val BPC, mean ± s.e. (sd), n = 5 | per-seed |
|---|---|---|---|
| 32 | 96,416 | 1.1095 ± 0.0048 (0.0107) | 1.112 1.113 1.108 1.093 1.122 |
| 64 | 102,560 | 1.1082 ± 0.0034 (0.0077) | 1.108 1.114 1.106 1.097 1.116 |
| 128 | 114,848 | 1.1002 ± 0.0031 (0.0068) | 1.104 1.092 1.101 1.095 1.109 |

Differences (flm.compare-style rule, real if |Δ| > 2 s.e.): r128 − r32 = −0.0093 (2 s.e. 0.0114) within noise; r64 − r32 = −0.0013 (0.0118) within noise; r128 − r64 = −0.0080 (0.0092) within noise.
**G49 refuted** (the r128 gain is a point estimate of 0.009 but not a real effect at five seeds). **G50 supported** (r64 is within 0.005 of r32).
**Correction to my own earlier claim:** the "r32 seed spread of 0.0004" and "gain outside seed noise" from the two-seed `v8-rules` cell were a two-seed fluke; r32's five-seed sd is 0.0107 (seed 3 at 1.093, seed 4 at 1.122).
What does survive: ctx 512 + 32 rules (mean 1.1095, n = 5) vs the ten-seed ctx-256 / 16-rule baseline (1.1291 at lr 6e-2): −0.0196, ≈ 4 standard errors, so that combination does beat the baseline (the two changes are not separated).
The rule count from 32 to 128 is a trend of about −0.009 that five seeds cannot confirm.

### Outcome `v14-state2`, new cells only (r192 and r256; the r64/r128 seeds 2–3 cells of that grid duplicate `v14-rule-seeds` and were not rerun; BPE1024 D d = 48, ctx 512 / batch 16, lr 6e-2, 30M chars, seeds 0–1; ~2 min each, one at a time)

| rules per head | params | val BPC (seeds) | mean |
|---|---|---|---|
| 192 | 127,136 | 1.0945 / 1.1154 | 1.1050 |
| 256 | 139,424 | 1.1025 / 1.1118 | 1.1072 |
| (128, five seeds, for reference) | 114,848 | 1.104 1.092 1.101 1.095 1.109 | 1.1002 ± 0.0031 |

r192 and r256 are **not better than r128**: both means sit inside r128's five-seed noise (seed sd ≈ 0.007–0.011; r192's two seeds differ by 0.021). **The rule-count curve from 32 to 256 is flat within noise** (1.1095, 1.1082, 1.1002, 1.1050, 1.1072);
the one trend the five-seed r32/r64/r128 comparison could not confirm (−0.009) is not continued by r192/r256. At 30M chars, adding rules beyond ~32 per head within the ≤ 165K cap does not reliably help. Throughput falls with rules
(119K → 91K → 84K chars/s for r32 → r192 → r256). This closes the "more recurrent state" lever for loss at this budget; coherence was not re-graded (the loss gain was < 0.02, the registered condition for re-grading).

## Registered 2026-10-11, before `coherence3` (decoding): does the decoding strategy move coherence at fixed weights?

Weights fixed: the 0.9976-BPC checkpoint (BPE1024 D, ctx 512, 32 rules, 300M chars). Four strategies, each 4 prompts × seeds 4 and 5 (fresh texts; 8 per strategy; 600 chars; same rubric, blind, shuffled, key withheld until scored):
**A** T 0.5 / top-k 10 (the baseline used in sets 1–2); **B** T 0.7 / top-p 0.9; **C** A plus a repetition penalty (logits of tokens seen in the last 64 tokens divided by 1.3 when positive, multiplied when negative);
**D** A plus a 4-gram block (a token is forbidden if it would complete a 4-token sequence already in the text). Rater 1 = Claude (GPU session).
* **DC1.** C and D each beat A on the total score by ≥ 1.0 (repetition is the visible failure; the earlier samples loop). *(~55% for each.)*
* **DC2.** B is no better than A (higher temperature costs grammar more than it gives variety). *(~65%.)*
* **DC3.** Plot sense (P) rises by ≥ 0.5 for the best of C / D. *(~30%: loop removal could make stories wander instead of go anywhere.)*
Noise reminder: ±1.5 totals set to set; n = 8 per cell, so only differences of ≥ ~2 are readable.

### Outcome `coherence3` (decoding at fixed weights; rater 1 = Claude GPU session, blind, scores `coherence3_scores_rater1.json` written before the key was opened; the 0.9976-BPC checkpoint; 8 texts per strategy, seeds 4–5)

| strategy | G | C | P | total (3–15) ± s.e. |
|---|---|---|---|---|
| **A** T 0.5 / top-k 10 (baseline) | 3.50 | 2.62 | 2.12 | **8.25 ± 0.77** |
| D = A + 4-gram block | 3.12 | 2.12 | 1.75 | 7.00 ± 0.53 |
| B T 0.7 / top-p 0.9 | 2.88 | 1.88 | 1.50 | 6.25 ± 0.34 |
| C = A + repetition penalty 1.3 | 2.75 | 1.75 | 1.50 | 6.00 ± 0.61 |

**DC1 refuted:** neither C (−2.25) nor D (−1.25) beats A; C is the worst arm. **DC2 supported:** B is worse than A (−2.0). **DC3 refuted:** plot sense falls, not rises (2.12 → 1.50–1.75).
Sanity check: S023 (A) and S029 (D) are character-identical texts (the 4-gram block never triggered) and received identical scores. The repetition penalty also produced a degenerate "deep deep deep …" story and artifacts (`???`),
i.e. it damages grammar more than it removes loops. **Reading:** the baseline decoding is already the best of the four; coherence at this model size is not limited by the sampler. Noise ±1.5; n = 8 per arm, so only the
C/B-versus-A gaps (≈ 2 totals) are readable; one rater; not independently graded (the CPU session may re-grade `coherence3_blind.json`; the key is withheld until then).

## Registered 2026-10-11, before `v12-long512` seed 1: is the 0.9976-BPC headline a one-seed fluke?

Same config as `v12-long512` (BPE1024 D d = 48, 32 rules per head, ctx 512 / batch 16, lr 6e-2, 300M chars), seed 1, saved. The seed-0 run scored 0.9976; the five-seed 30M-char spread of this configuration is sd 0.011.
* **G32.** Seed 1 scores ≤ 1.010 (within about 1 sd of seed 0, as the 30M seed spread suggests), so "under 1.01 bits per character at 96K parameters" holds on two seeds. *(~70%.)* A score above 1.02 would mean the headline was a lucky seed and the honest figure
  is the previous 1.0212 (ctx 256, 16 rules) plus noise.

### `coherence3` with both raters (key published after rater 2 scored; rater 2 = CPU session, blind, `coherence3_scores_rater2.json`)

| strategy | rater 1 | rater 2 | mean of raters ± s.e. (n = 8) | plot (mean of raters, /5) |
|---|---|---|---|---|
| A T 0.5 / top-k 10 (baseline) | 8.25 | 7.00 | **7.62 ± 0.65** | 1.88 |
| D A + 4-gram block | 7.00 | 7.25 | 7.12 ± 0.56 | 1.62 |
| B T 0.7 / top-p 0.9 | 6.25 | 6.62 | 6.44 ± 0.50 | 1.44 |
| C A + repetition penalty 1.3 | 6.00 | 6.25 | 6.12 ± 0.55 | 1.38 |

Inter-rater agreement on totals r = 0.63 (lower than sets 1–2: 0.83), rater means 6.88 and 6.78. **Correction to my one-rater reading above:** with two raters the ordering A > D > B > C survives for the mean, but the gaps shrink
(A − C = 1.5 with s.e. ≈ 0.85; A − D = 0.5) and rater 2 put D marginally *above* A (7.25 vs 7.00). So "the baseline sampler is best" is a ranking that is **not established**; what two raters do support is that none of the three alternatives is better than
the baseline, and that the repetition penalty and the higher temperature are, if anything, worse (≈ 1.2–1.5 totals, ~2 s.e.). **DC1 stays refuted** (neither C nor D beats A by ≥ 1 in the mean); DC2 stays supported; DC3 stays refuted (plot A 1.88 vs 1.38–1.62).

### Outcome `v12-seed1` (G32; BPE1024 D d = 48, 32 rules/head, ctx 512 / batch 16, lr 6e-2, 300M chars, seed 1; 96,416 params)

Seed 1: **1.00528** (curve 25 / 50 / 75 / 100%: 1.1367 / 1.0850 / 1.0352 / 1.0053). **G32 supported** (≤ 1.010). Two seeds: 0.9976 and 1.0053, **mean 1.0015 ± 0.0039 (sd 0.0054)**.
**Correction to the headline:** "below one bit per character" (0.9976) was the better of two seeds; the honest two-seed figure is **≈ 1.00 BPC (1.0015)**, i.e. "about one bit per character at 96,416 parameters". The earlier ctx-256 / 16-rule 300M run
(1.0212, one seed, lr 3e-2) is 0.02 behind the mean, about 4 standard deviations of the seed spread of this configuration, so the improvement over it is probably real but is a two-vs-one-seed comparison.

## Registered 2026-10-11, before `v15-share`: does weight-shared depth buy capability at a fixed parameter count?

Same parameters, more compute: BPE1024 D d = 48, ctx 512 / batch 16, 32 rules per head, lr 6e-2, 30M chars, seeds 0–1, with **2 unique blocks applied 2× (baseline, the five-seed mean is 1.1095), 4× or 6×** (`--layers 4|6 --n-unique 2`). Parameter count stays 96,416
(only the block weights are counted once). The CPU-phase result on the old zero-order FFN at d = 16 was that BPC improved monotonically with the number of applications.
* **G33.** Four applications of 2 unique blocks beat the baseline by ≥ 0.01 BPC at 30M chars (the five-seed baseline sd is 0.011, so this means ≈ 2 s.e. of a two-seed mean). *(~40%.)* *Basis:* depth helps in-context mechanisms (a second recurrent
  layer cost D 0.07 against a single layer), but at d = 48 the extra applications reuse weights tuned for the two-layer stack.
* **G34.** Six applications are no better than four by more than 0.01 (returns flatten). *(~60%.)*
* If G33 holds, a 300M run follows and the coherence set is regraded (blind, two raters), since this would be the first in-cap change with a plausible link to multi-step structure rather than just loss.

### Outcome `v15-share` (G33 / G34; BPE1024 D d = 48, ctx 512 / batch 16, 32 rules, lr 6e-2, 30M chars, seeds 0–1; 96,416 params in every cell)

| layer applications of 2 unique blocks | val BPC (seeds) | mean ± s.e. | chars/s |
|---|---|---|---|
| 2 (baseline, five seeds, `v14-rule-seeds`) | | 1.1095 ± 0.0048 | ~119K |
| 4 | 1.121 / 1.122 | 1.1217 ± 0.0004 | ~60K |
| 6 | 1.119 / 1.129 | 1.1241 ± 0.0052 | ~40K (est.) |

**G33 refuted** (needed ≥ 0.01 better; weight-shared depth is 0.012 *worse* at 4 applications, ≈ 2.5 s.e. of the baseline mean; a modest effect). **G34 supported** (6 is not better than 4; it is 0.002 worse).
Reusing two blocks more times does not buy capability for D at this size and costs 2–3× the compute; the CPU-phase observation that BPC improved with applications was for the old zero-order FFN at d = 16 and does not carry over.
No 300M run and no coherence re-grade (registered condition not met).

## Conclusion of the sub-100K track (GPU session, 2026-10-11)

**Declared finished by the GPU session; reopen on the user's instruction.** The goal was "make the fuzzy language model as capable as possible under 100K parameters". Where it ended:

* **Best model under 100K:** BPE1024 delta-form fuzzy LM (D), d = 48, 2 layers, 8 heads, 32 rules per head, first-order raw-firing HTSK TSK FFN, ctx 512 / batch 16, lr 6e-2: **96,416 parameters**.
  1.0015 ± 0.0039 BPC at 300M chars (two seeds: 0.9976, 1.0053); 1.1095 ± 0.0048 at 30M (five seeds). At 93,344 parameters (ctx 256, 16 rules) it beats the two best tuned neural references on the same tokenizer by 0.027–0.030 BPC over ten seeds at 30M chars.
  From the original character-level fuzzy design (≈ 1.62 BPC at ≈ 97K, 30M chars) to 1.11 at the same budget and parameter count; the TSK feed-forward layer (not the mixer) was the whole original deficit, subword tokens were the largest later gain.
* **What did not matter at this size (each moved loss by ≤ ~0.03 and none moved coherence):** longer context, more rules per head beyond 32, Fibonacci-lag attention (better than dense attention, far behind D), attention–recurrent hybrids (worse than D), distillation from a 280K teacher, weight averaging, continued training beyond the data curve, decoding strategy, and weight-shared depth.
* **Coherence is not achieved.** Two blind raters score the best models' stories 6.9–7.6 / 15 (plot sense ≈ 1.7–1.9 / 5); the criterion the CPU session proposed (≥ 12/15 and plot ≥ 3.5 from both raters) is far away. The measure is noisy (≈ ±1.5 per set); the large separations (character vs subword models) are real, the fine ordering among the top models is not.
* **The one untested lever is parameter count** (the CPU session's `v9` ladder, 251K–7M). It is outside the stated cap and was not run on the GPU host; the CPU session ran a 251K model on its own host under its user's approval and will report it separately.
* **Open items not closed:** PR #272 was not merged (the permission classifier blocked it); the `v3-lr` study and the character-level ten-seed plan (`v4-*`) were not run to completion; D's learning-rate sensitivity was checked at 10M and 30M chars only.
