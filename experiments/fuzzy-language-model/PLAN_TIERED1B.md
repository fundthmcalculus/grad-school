# v35-tiered1b: the tiered 98,280-parameter recipe at 1B characters (registered 2026-10-12; user: "Tiered 1B run is good to go")

Run: `python -m flm.run_tiered_1b 0` (seed 0, warm-up 100, lr 6e-2, `_1b` stream, `--save-model`; ~1 h on the RTX 4080). References at 1B chars (plain embedding, same recipe): v17-1b seed 0 0.97958, seed 1 see `outputs/v17-1b-s1`; at 3B chars 0.97767 (the current sub-100K best). Tiered at 300M: 0.98099 / 0.98209.
Predictions: val BPC 0.958-0.972 (60%); beats the 96K 3B-char best 0.97767 with a third of the data (80%); below 0.955 (10%); above 0.975 (10%). Coherence is not re-rated (set 11 showed the 300M BPC gain did not move it).

## Result
Seed 0, 98,280 params, 1B chars, 2,698 s: **val BPC 0.96796** (log: 1.114 at 5k steps, 0.9716 at 35k). Prediction 0.958-0.972 (60%): **right**; beats the 96K 3B-char best 0.97767 with a third of the data (80%): **right** (-0.0097); below 0.955 (10%) and above 0.975 (10%): neither. Against the 96K reference at the same 1B chars: 0.97958 / 0.98446 (seeds 0/1), so -0.0116 / -0.0165 at one seed. **New sub-100K best.** One seed; not a ten-seed claim. Checkpoint `outputs/v35-tiered1b/t768e16m225_1000M_s0.pt` loads only via `flm.run_tiered`.

### Seed 1 (2,682 s)
Seed 1: **0.96129** vs the 96K reference's 0.98446 at the same seed and budget (**-0.0232**); seed 0 was 0.96796 vs 0.97958 (-0.0116). Mean over the two seeds **0.9646** vs reference mean **0.9820** (-0.0174), both seeds better, and both better than the 96K 3B-char best 0.97767. Prediction 0.958-0.972 held for both seeds. Two seeds is a confirmation, not a ten-seed claim. The tiered recipe now stands at 30M (five seeds, -0.013), 300M (two seeds, -0.017/-0.023) and 1B (two seeds, -0.012/-0.023).

### Blind coherence set 13, registered before rating (`report/coherence13_*`; fresh seeds 40-41; models: tiered 98K 1B seed 0, tiered 98K 1B seed 1, 96K 1B seed 0, 96K 3B; 4 models x 8 texts; two Claude sub-agent raters reading only the blind file; scores before the key)
Predictions: each tiered 1B model within 1 point of the 96K 3B model on both raters (70%); a tiered 1B model at 7/15 or more (10%); the best sub-100K BPC (0.961) does not move coherence beyond the ~5-6/15 band seen in sets 10-12 (85%).

#### Set 13 scores (two raters; written before the key was opened; total of G+C+P means, 8 texts per model)
| model | rater A (plot) | rater B (plot) |
|---|---|---|
| tiered 98K 1B seed 0 (0.968) | 8.00 (2.25) | 7.38 (1.88) |
| tiered 98K 1B seed 1 (0.961) | 6.12 (1.62) | 6.12 (1.62) |
| 96K 1B (0.980) | 7.00 (1.88) | 7.00 (2.00) |
| 96K 3B (0.978) | 6.75 (2.00) | 6.38 (1.62) |
Predictions: each tiered 1B model within 1 point of the 96K 3B model on both raters (70%): **partly wrong** (seed 1 yes; seed 0 is +1.25 / +1.00); a tiered model at 7/15 or more (10%): **happened for seed 0** (8.00 / 7.38) but not for seed 1; no movement beyond the earlier band (85%): **right in substance**: the two tiered seeds differ from each other by 1.9 / 1.3 points, as much as any model difference in the set, and the 96K 1B model sits between them. Two-seed tiered means 7.06 / 6.75 vs 96K 1B 7.00 / 7.00 and 96K 3B 6.75 / 6.38: indistinguishable. Rater scale shifted up against set 12 (the 96K 1B model scored 4.7 there, 7.0 here), so only within-set comparisons mean anything. **The 0.961-0.968 BPC record does not buy coherence; seed-to-seed spread in rated coherence is as large as the effect being sought.** Caveats: 8 texts per model, same-family raters, one checkpoint per seed.
