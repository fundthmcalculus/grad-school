# Decision log (CPU session, written for the user to read in the morning)

The user said on 2026-10-12, going to bed: "Go ahead with the run. I am going to bed, so I want you to take every
decision, just log it so I can see the plans." Entries are newest last. Each says what was decided, why, and what
would change the decision.

## D1 (2026-10-12): run the 251K-parameter CPU model
* **Decision:** train D at d 96, 2 layers (251,168 parameters, 2.7x the 93K model), BPE1024, ctx 512, batch 16,
  30M characters, lr 3e-2, seed 0, on this host's CPU (8 threads, about 70 minutes). Output
  `outputs/v9-cpu/d96-L2_30M_s0.{json,pt}`, log `outputs/v9-cpu/d96.log`.
* **Why:** the one test I can run myself that bears on the cap. Nothing at 93-139K parameters moved blind coherence
  (6-8 of 15, plot 1.5-1.9 of 5). If 2.7x the parameters at the same data moves plot, size is the lever; if not, the
  size ladder is less promising than I thought.
* **Honest limit:** 251K parameters will not by itself reach 12/15. This is a trend test. lr 3e-2 was not tuned at this
  size (the 93K model's optimum was 6e-2 at 10M, flat between 3e-2 and 6e-2 at 30M), so a null is mildly weaker.

## D2: how the run will be judged (fixed before it finishes)
* Compare against the same-data control already on disk: `outputs/v5-final100k/D-d48-bpe1024_30M_s0.pt` (93K, ctx 256,
  30M characters) and the best model `outputs/v12-long512/...300M_s0.pt`. Three models x 8 texts, fresh seeds (4,5),
  `flm.coherence_set`, shuffled; I score G/C/P blind before opening the key. Two-rater agreement cannot be repeated overnight
  (the other rater is the GPU session), so this is a single-rater read and is reported as such.
* **Prediction (registered now):** the 251K model's plot score is within 0.5 of the 93K control (65%); its validation
  BPC at 30M is at least 0.04 lower than the 93K model's 1.1232 (ctx 512) (70%).
* **If plot rises by 0.5 or more:** size is the lever; recommend the size ladder to the GPU session's user with this as the
  evidence, and queue d128-L4 (653K) on CPU at 10M characters as the next point. **If not:** the next CPU step is the
  rule-ablation and rulebook on the 251K model (explainability at a larger size), and the recommendation to the user is
  that coherence needs a scale this project's cap does not allow.

## D3: what I will not do without the user
* Run the size ladder on the GPU host: that is the GPU session's user's decision. The user's message today covers my own
  host and my own choices. I relayed it to the GPU session as information, not as an approval.

## D4 (2026-10-12, after the run): stop scaling on CPU; what the evidence says
* **Result:** the 251K model scored 0.9700 BPC at 30M characters, a 0.15 BPC gain over the 93K model at equal data, and blind coherence did not move (6.75 vs 7.00 vs 8.25 /15,
  plot 1.3-1.8 /5, one rater, noise about +/-1.5). By the rule fixed in D2, size at this scale is not the lever. Loss and coherence have come apart: every model from
  1.13 down to 0.97 BPC sits at 6-8 /15.
* **Decision:** do not queue a larger CPU model. The next point on the ladder (653K) would take about 2 hours at 10M characters and would be undertrained, so it could not separate size from data; and
  the coherence-relevant scale (millions of parameters, 100M+ characters) is beyond this host. I am not asking the GPU host to run the size ladder; that is its user's call (D3).
* **Recommendation for the morning:** (1) the criterion (12/15, plot 3.5/5) has not been approached by any variation inside or just above the 100-250K cap; (2) if coherence is the goal, the
  untested levers are a much larger model on much more data, or a different mechanism for plot (explicit story or character state), not a bigger version of this one;
  (3) explainability holds but thins as rules multiply (8 cited rules explain 23-35% of positions at 144 rules, 53-56% at 72).
* **What would change this:** a blind two-rater grading that puts the 251K model at plot 2.5 or above (my single-rater scores can be wrong), or the GPU session's second-seed 300M run
  moving coherence.
