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
