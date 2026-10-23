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
