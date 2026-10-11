# Size ladder between 96K and 1.32M (`v32-ladder`)

User, 2026-10-11: "Raise the cap and keep moving" (after the 1.32M x 1B result: BPC 0.636, blind total ~10/15, plot ~3, vs ~5.4/1.4 for 96K). Question: where between 96,416 and 1,323,904 parameters does blind coherence start to move? Code: `flm/run_ladder.py` (sizes from `flm.grids_v9.SIZES`, lr 2e-2 = the d192 optimum assumed, warm-up 300, seed 0, saves checkpoints). The cap exception now covers this ladder (sizes <= 1.32M); anything larger (2.2M-7M) still needs a separate yes.

Speed at 30M characters (GPU): d96-L2 (251,168) 81 s, BPC 0.9593; d128-L4 (653,248) 161 s, BPC 0.8630. Reference at 30M: 96K 1.1095, 1.32M 0.8214. Extrapolated cost at 1B characters: ~45 min and ~90 min.

## Registered before the 1B runs
* BPC at 1B: d96-L2 0.78-0.84; d128-L4 0.68-0.74 (the 1.32M gained 0.19 from 30M to 1B).
* Blind coherence (new set, fresh seeds, two blind raters, same protocol as set 10, models: 251K, 653K, 1.32M x 1B, 96K): totals rise monotonically with size; 251K lands within 1.5 points of the 96K (about 5.5-7), 653K lands between 7 and 9.5, only 1.32M reaches ~10. So the threshold for "plot >= 2.5" sits between 653K and 1.32M (60%).
* Decision rule: if the 653K model already reaches total >= 9.5 and plot >= 2.5 from both raters, coherence is reachable well below 1M; run a further rung (400K) and stop. If 251K and 653K both stay <= 7.5, the step between 653K and 1.32M is where it happens and the ladder needs d160/d192 rungs at 1B (not run without asking).
