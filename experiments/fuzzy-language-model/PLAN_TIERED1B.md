# v35-tiered1b: the tiered 98,280-parameter recipe at 1B characters (registered 2026-10-12; user: "Tiered 1B run is good to go")

Run: `python -m flm.run_tiered_1b 0` (seed 0, warm-up 100, lr 6e-2, `_1b` stream, `--save-model`; ~1 h on the RTX 4080). References at 1B chars (plain embedding, same recipe): v17-1b seed 0 0.97958, seed 1 see `outputs/v17-1b-s1`; at 3B chars 0.97767 (the current sub-100K best). Tiered at 300M: 0.98099 / 0.98209.
Predictions: val BPC 0.958-0.972 (60%); beats the 96K 3B-char best 0.97767 with a third of the data (80%); below 0.955 (10%); above 0.975 (10%). Coherence is not re-rated (set 11 showed the 300M BPC gain did not move it).

## Result
Seed 0, 98,280 params, 1B chars, 2,698 s: **val BPC 0.96796** (log: 1.114 at 5k steps, 0.9716 at 35k). Prediction 0.958-0.972 (60%): **right**; beats the 96K 3B-char best 0.97767 with a third of the data (80%): **right** (-0.0097); below 0.955 (10%) and above 0.975 (10%): neither. Against the 96K reference at the same 1B chars: 0.97958 / 0.98446 (seeds 0/1), so -0.0116 / -0.0165 at one seed. **New sub-100K best.** One seed; not a ten-seed claim. Checkpoint `outputs/v35-tiered1b/t768e16m225_1000M_s0.pt` loads only via `flm.run_tiered`.
