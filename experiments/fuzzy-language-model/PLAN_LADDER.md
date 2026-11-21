# Size ladder between 96K and 1.32M (`v32-ladder`)

User, 2026-10-11: "Raise the cap and keep moving" (after the 1.32M x 1B result: BPC 0.636, blind total ~10/15, plot ~3, vs ~5.4/1.4 for 96K). Question: where between 96,416 and 1,323,904 parameters does blind coherence start to move? Code: `flm/run_ladder.py` (sizes from `flm.grids_v9.SIZES`, lr 2e-2 = the d192 optimum assumed, warm-up 300, seed 0, saves checkpoints). The cap exception now covers this ladder (sizes <= 1.32M); anything larger (2.2M-7M) still needs a separate yes.

Speed at 30M characters (GPU): d96-L2 (251,168) 81 s, BPC 0.9593; d128-L4 (653,248) 161 s, BPC 0.8630. Reference at 30M: 96K 1.1095, 1.32M 0.8214. Extrapolated cost at 1B characters: ~45 min and ~90 min.

## Registered before the 1B runs
* BPC at 1B: d96-L2 0.78-0.84; d128-L4 0.68-0.74 (the 1.32M gained 0.19 from 30M to 1B).
* Blind coherence (new set, fresh seeds, two blind raters, same protocol as set 10, models: 251K, 653K, 1.32M x 1B, 96K): totals rise monotonically with size; 251K lands within 1.5 points of the 96K (about 5.5-7), 653K lands between 7 and 9.5, only 1.32M reaches ~10. So the threshold for "plot >= 2.5" sits between 653K and 1.32M (60%).
* Decision rule: if the 653K model already reaches total >= 9.5 and plot >= 2.5 from both raters, coherence is reachable well below 1M; run a further rung (400K) and stop. If 251K and 653K both stay <= 7.5, the step between 653K and 1.32M is where it happens and the ladder needs d160/d192 rungs at 1B (not run without asking).

## Results: 1B-char runs and blind coherence set 12 (recorded 2026-10-11)

BPC at 1B chars (seed 0, lr 2e-2, warm-up 300): **251K (d96-L2) 0.8172** (prediction 0.78-0.84, in range; 2,654 s); **653K (d128-L4) 0.6943** (prediction 0.68-0.74, in range; 5,302 s). 1.32M 0.6355, 96K 0.982.

Blind coherence set 12 (`report/coherence12_*`, fresh seeds 30-31, 4 models x 8 texts, two Claude sub-agent raters that read only the blind file; scores written before the key was opened; rubric G/C/P 1-5, total 3-15):
| model (1B chars) | rater A total (P) | rater B total (P) |
|---|---|---|
| 96K | 4.75 (1.25) | 4.62 (1.12) |
| 251K | 6.38 (1.88) | 6.25 (1.62) |
| 653K | 8.88 (2.38) | 9.00 (2.38) |
| 1.32M | 10.00 (3.12) | 10.12 (2.88) |

Predictions: monotone in size - **right** (both raters, every step). 251K within 1.5 pts of 96K (~5.5-7) - 251K is +1.6/+1.6 above the 96K arm, but inside the 5.5-7 absolute band: **borderline / partly wrong** on the relative form. 653K between 7 and 9.5 - **right** (8.9/9.0). Only 1.32M ~10 - **right**; this set reproduces set 10 (9.6/10.4 there).
Decision rule: 653K did **not** reach total >= 9.5 and plot >= 2.5 on both raters (plot 2.38), and 653K is not <= 7.5, so neither branch fires cleanly. Reading: coherence rises smoothly with size (~+1.6 per ~2.6x params at this data), with no sharp threshold; the 10/15, plot >= 2.5 bar sits at about 1M-1.3M at 1B chars. Nothing under 100K gets past ~5-5.5 (this rater pair's scale). I am not running a 400K rung (the rule's "stop" case does not apply, and the gain is a smooth lever, not a cheap one); 2.2M+ rungs are not approved.
Caveats: one seed per size, 8 texts per arm, same-family raters, one rater scale (96K arm 4.7 here vs 5.3 in set 10: set-to-set noise ~0.5).
