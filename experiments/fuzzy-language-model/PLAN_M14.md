# PLAN_M14: one test run of an all-fuzzy model scaled to 14M parameters on TinyStories (registered 2026-10-23, before the run)

Owner's request: before the Pile / Pythia-14M work, do **one** test run on an FLM scaled to 14M parameters, on TinyStories as before, to learn how long things take. Not a tuned result, not a claim about quality: a timing and sanity run. One training job at a time.

## The model
`M14_v6144_h1024_d384_l4` (`report/m14_arms.json`): the F7 design scaled up, not re-searched. BPE6144 tokenizer (existing TinyStories stream and frequency file), 1,024 crisp head tokens, tail width 8, d 384, 4 blocks (F7: d32, 2 blocks), 8 mixer heads, mixer antecedent dimension 96 (F7: 16), 32 mixer rules per head, feed-forward rules 2,149 in layer 1 and 1,440 in layer 0 (the F7 split 0.67), **14,067,715 parameters** (Pythia-14M: 14,067,712). Fitted with `flm.shape_budget.count` by bisection on the rule count. The rule counts are large (11x d for F7's 1.9x d) because the feed-forward rules carry most of the parameters in this design; it is the scale-up that matches the parameter count, not a recommendation. Compare: F7 holds 65% of its parameters in the embedding; here the embedding is 2% (1,024 x 384 head plus a tail of 5,120 x 8), so this model is almost all block, unlike Pythia-14M (about 90% embedding).

## Probes done before the run (no result claimed from them)
- CPU smoke (13 steps, 0.4M characters): builds, 14,067,715 parameters, loss falls (3.92 -> 1.83 bits per character), no NaN; 346 s on the CPU.
- GPU smoke (same): 33,000 characters per second including start-up (F7 reaches 97,000), so about 15 minutes per 30M characters and about 2.5 hours per 300M characters at this batch; not yet measured at steady state.
- A learning-rate probe (below), because the recipe's lr 6e-2 was tuned at 100K parameters and a 140x larger model could diverge.

## Run
1. **Learning-rate probe**: 3M characters, warm-up 20, seed 0, lr 6e-2 / 2e-2 / 6e-3; pick the best final validation bits per character that does not diverge. A guard against a wasted run, not a tuning claim (a short run favours higher rates).
2. **The test run**: 30M characters, seed 0, warm-up 600 (about 900 steps, as F7), floor 0, batch 16, context 512, weight decay 0.01, the chosen lr; evaluation every 100 steps, checkpoint saved. Output in `outputs/v81-m14/`.
3. Report: wall time and characters per second (steady state), peak GPU memory, the validation curve, bits per character against F7 at 30M (1.058) and against Pythia-14M zero-shot on TinyStories (1.248), and the extrapolated cost of a 300M run.

## Predictions
Characters per second at steady state is between 30K and 70K (70%); 30M characters take at most 20 minutes (70%); no divergence at the chosen lr (90%); lr 6e-2 diverges or is worse than 2e-2 on the probe (55%); the 30M-character bits per character is below F7's 1.058 by at least 0.05 (60%) and below 0.98 (35%); it is below Pythia-14M's zero-shot 1.248 (97%); peak GPU memory is below 8 GB (75%).

## Not in scope
No 300M run, no Pile data, no comparison claim; those come after this run is read and the owner decides.

## Results (recorded 2026-10-23; `outputs/v81-m14/`)
**Learning-rate probe** (3M characters, 91 steps, warm-up 20, seed 0): lr 6e-2 2.0966, lr 2e-2 2.2181, lr 6e-3 **1.3978**; none diverged. 6e-3 is the lowest rate tried, so the best rate may be lower; the recipe's 6e-2 is the wrong regime at 14M.
**Test run** (30M characters, seed 0, lr 6e-3, warm-up 600, floor 0, batch 16, context 512, 905 steps, 14,067,715 parameters): validation **0.9764 bits per character**; curve 3.92 (step 0), 1.31 (200), 1.20 (400), 1.10 (600), 1.00 (800), 0.976 (905); no divergence. Training time 208 s (wall 251 s including evaluation and start-up).
| | 14M, 30M characters | F7 (100K), 30M characters | F7, 300M characters | Pythia-14M zero-shot |
|---|---|---|---|---|
| TinyStories bits per character | **0.976** | 1.058 (five-seed mean) | 0.956 (floor 0, seed 0) | 1.248 |
Units note: the result file's `chars_per_sec` field (35,560 here, 96,957 for F7) is **tokens** per second (905 steps x 16 x 512 tokens / 208 s); in characters the 14M run is 30.0M / 208 s = **144,000 characters per second**, against about 333,000 for F7 (30M in 90 s). So the 14M model costs about 2.3x F7 per character, not the 3x the GPU smoke suggested (that smoke included start-up).
Cost extrapolation (same batch, one job at a time): 30M characters 3.5 minutes; **300M characters about 35 minutes**; 1B about 2 hours (a 1B run is outside the standing 300M ceiling). Peak GPU memory was not recorded (no sampling was running); nvidia-smi showed 7.2 GB during an F7 job on the same card, and this run did not spill (no hang).
Sample at 30M characters: fluent short sentences with name repetition ("Tom ... Tom and his friend ... Lucy"), the usual 100K-model drift but more coherent clause structure; not rated.
Prediction scoring: characters per second between 30K and 70K (70%): **right as logged** (35.6K, but that is tokens per second; true characters per second 144K, outside the range); 30M characters at most 20 minutes (70%): **right** (3.5 minutes, far under); no divergence (90%): **right**; lr 6e-2 worse than 2e-2 on the probe (55%): **wrong** (6e-2 was better, 2.10 against 2.22; both far worse than 6e-3); below F7 by at least 0.05 (60%): **right** (-0.082); below 0.98 (35%): **right** (0.976); below Pythia-14M zero-shot 1.248 (97%): **right**; peak memory below 8 GB (75%): **not measured**.
Reading: at 140x the parameters and the same 30M characters, the scaled-up fuzzy model reaches 0.976, 0.082 better than F7 at the same budget and 0.020 away from F7 trained for ten times as long. Time is cheap: a 300M run is about 35 minutes. This is one seed, one untuned learning rate, and the F7 design scaled by hand, so it says "the design scales and is not slow", nothing more. Pythia-14M's 1.248 on TinyStories is zero-shot (it never saw the corpus) and is not a like-for-like comparison.
Next (needs the owner): a learning-rate screen at 14M (lower than 6e-3; 3M-character probes are 40 seconds each), then a 300M run (about 35 minutes), then the Pile sample on the other machine.

## Learning-rate screen and the 300M run (owner: "let's do some LR screens for now, then the 300M run", 2026-10-23; registered before it runs)
Screen: the same 14M model, 30M characters, seed 0, warm-up 600, floor 0, batch 16, at lr 1.5e-3, 3e-3 and 1e-2, with the existing 6e-3 run (0.9764) as the fourth point; each takes about 3.5 minutes. Seed 0 only: differences below about 0.005 are noise (the 100K models' seed standard deviation is 0.003). The 300M run then uses the best lr of the four (warm-up 1000, floor 0, seed 0, checkpoint saved, evaluation every 5,000 steps; about 9,000 steps, about 35 minutes), the one job running. Caveat: a 30M screen of 905 steps may favour a higher rate than a 9,000-step run needs (the stage-2 lesson was about step count, and the 100K recipe tuned lr at 30M and kept it at 300M); if the best rate is the smallest tried, it is not bracketed and I say so.
Predictions: the best of the four is 3e-3 or 6e-3 (65%), 1e-2 (15%), 1.5e-3 (20%); the best improves on 6e-3 by at least 0.01 (35%); 1e-2 is worse than 6e-3 (65%); no run diverges (85%); the 300M run ends below 0.93 (55%), below F7's 300M 0.956 by at least 0.02 (65%); it takes 25-50 minutes (75%) with at least 130,000 characters per second at steady state (70%).

## Learning-rate screen, result and the upper side (recorded 2026-10-23)
30M characters, seed 0, 14M model: lr 1.5e-3 1.0957, 3e-3 1.0368, 6e-3 0.9764, **1e-2 0.9449**; none diverged; the best of the four is the highest rate tried, so the optimum is not bracketed from above (the only evidence above 1e-2 is the 3M-character probe, where 2e-2 and 6e-2 lost to 6e-3, a weak guide at 91 steps). Prediction scoring: best is 3e-3 or 6e-3 (65%): **wrong**; best is 1e-2 (15%): **happened**; the best beats 6e-3 by 0.01 (35%): **right** (0.0316); 1e-2 worse than 6e-3 (65%): **wrong**; no divergence (85%): **right**.
The 300M run started at 1e-2 (the chain picks the best of the four). Registered now, before they run (queued behind the 300M run, one job at a time; seed 0, 30M characters, warm-up 600): lr 1.5e-2 and 2e-2. Predictions: 1.5e-2 beats 1e-2 by at least 0.005 (40%); 2e-2 beats 1e-2 (25%); 2e-2 diverges or is worse than 1e-2 by at least 0.02 (45%); the best of the six is 1e-2 or 1.5e-2 (65%). If 1.5e-2 or 2e-2 wins by more than 0.01 the 300M result is for a slightly low rate and I say so.

## Estimate: how long would a full Pile training take (preliminary, 2026-10-23, from the 30M run; refined when the 300M run gives a steady-state speed)
Measured: the 14M model trains at **35,560 tokens per second** on the RTX 4080 Laptop (208 s for 905 steps of 16 x 512 tokens), in FP32 with the 100K recipe's code, batch 16.
Characters per token on the Pile sample (2 shards, 5.7M characters, so +/-15%): a BPE of 6,144 trained on it gives **2.96** characters per token on held-out text (BPE8192: 2.94-3.27; Pythia's own tokenizer on the folded text: 3.73). So our speed is about 35,560 x 3.0 = **107,000 Pile characters per second**. Pythia's 300B tokens are about 300e9 x 3.73 = **1.1 trillion characters** of the folded text (the deduplicated Pile at 207B Pythia tokens: 0.77 trillion).
| budget | characters | time on this GPU (RTX 4080 Laptop) |
|---|---|---|
| the standing ceiling (300M characters) | 3.0e8 | 47 minutes |
| Chinchilla-style, 20 tokens per parameter (281M tokens at 3.0 characters per token) | 0.84e9 | 2.2 hours |
| 10x that | 8.4e9 | 22 hours |
| one epoch of the deduplicated Pile (207B Pythia tokens) | 0.77e12 | about 83 days |
| Pythia's 300B tokens | 1.1e12 | **about 121 days** (4 months) |
Other hardware (estimates from specifications, not measured): a Colab T4 is 3-4x slower (about a year for Pythia's budget; 3 hours for the Chinchilla-style one); an A100 with TF32 enabled perhaps 3x faster (about 40 days); data-parallel across eight A100s, which this trainer does not support, about 5-6 days.
Efficiency note: 6 x 14.07M parameters x 35,560 tokens per second is 3.0 teraflops of useful work, against about 25-34 teraflops of FP32 peak on this GPU, so about 10%; the many small kernels and batch 16 are probably the limit, and a larger batch, kernel fusion or `torch.compile` could plausibly give 2-3x. A bigger batch changes the optimisation (the batch-8 lesson cuts both ways) and needs its own learning-rate screen.
Reading: a full Pythia-budget Pile run is not feasible on one consumer GPU (months); budgets up to about 10 billion characters (a day) are, and a Chinchilla-style 14M run costs a couple of hours. For the Pythia comparison the sensible choice is a matched small budget (the 300M-character ceiling, 47 minutes, or the 0.84-billion-character Chinchilla-style budget, 2.2 hours), reported with the data difference stated, not a reproduction of Pythia's 300B tokens.
Prediction scoring for this estimate waits for the 300M run's measured speed: steady-state tokens per second within 10% of 35,560 (65%).
