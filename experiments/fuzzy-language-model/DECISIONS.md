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

## D5 (2026-10-12, overnight): one more CPU point, 1.32M parameters
* **Decision:** train D at d 192, 4 layers (1,323,904 parameters, 5.3x the 251K model, 14x the headline), BPE1024, ctx 512, batch 16, lr 2e-2 (a guess: the 93K model's optimum
  was 6e-2 and larger models usually want less; untuned), 30M characters, seed 0, on this CPU: measured 2,900 characters per second, so about 3 hours.
  Output `outputs/v9-cpu/d192-L4_30M_s0.{json,pt}`, log `outputs/v9-cpu/d192.log`.
* **Why:** the 251K result left a question the cap cannot answer: is the flat coherence a property of tiny models or of this architecture? One 14x point separates them at no
  GPU cost. You said to take the decisions and log them; this is the only experiment left that fits on this host overnight, and it ties up the CPU for about 3 hours, which I judged acceptable while you sleep.
* **Predictions (registered before it finishes):** val BPC at most 0.93 (60%); blind plot (single rater, same protocol, fresh seeds) 2.0 or above (35%); total 10/15 or above (25%).
* **After it finishes:** grade a blind set (1.32M, 251K, 93K at 300M; seeds 8-9), run rule-ablation with 8 and with 32 cited rules, log D6. If plot is still under 2.0 the conclusion in D4 stands for this architecture at 30M characters.
* **I am not changing anything else while it runs** (jobs import code from disk; I do no code merges until it ends).

## D6 (2026-10-11): result of D5 and what I am recommending
* **Result:** 1.32M parameters at 30M characters: 0.8300 BPC, blind coherence 7.50 /15 (plot 1.75), indistinguishable from the 96K models (5.75-8.25 /15). The registered plot prediction (2.0 or above) failed.
  The conclusion in D4 stands, now out to 14x the headline parameters: **lower loss and more parameters have not produced coherent stories in this architecture at 30M characters**.
* **What it does show:** explainability degrades with size. The rulebook stays causal, but 8 rules out of 288 explain only 0.2-3% of the loss in each layer.
* **Decision:** no further CPU models. The 1.32M model saw about 7 training tokens per parameter (30M characters is 9.5M BPE tokens), against about 3,300 for the 96K model on 1B characters, so the
  size question is still not cleanly closed: a 1.3M model on 1B characters is the test, and it takes GPU hours. That is for you and the GPU session's user (see D3), not me.
* **Recommended morning decision:** (a) accept that, inside the current cap and this architecture, coherence is not reachable and stop; (b) approve one GPU run of the 1.32M configuration (v9 `d192-L4`) on the 1B-character stream
  (the GPU session's own 1B-character runs of the 96K model are the scale reference; a 14x larger model costs more per character, and I have not measured it on a GPU) to close the size-times-data question; or (c) change the mechanism (an explicit story state).
  I would do (b) once, then (c). I have not asked the GPU session to start (b).

## D7 (2026-10-11): what I decided about (b) and (c) from D6, and why I stopped
You asked me to take every decision, so these are decided rather than left open.
* **(b) the 1.32M model on the 1B-character stream: not started by me.** It needs the GPU host, whose session and user set a 100-165K parameter cap; my instruction covers my own host and my own
  choices, and a peer message cannot approve it for them. It stays registered (`flm/grids_v9.py`, `d192-L4`) and is my first recommendation if the cap is lifted.
* **(c) an explicit story-state mechanism: not started.** I judged it a poor use of an unsupervised night: with no model above 0.5M parameters that I can train to convergence here, any gain from a new mechanism at
  93-250K parameters would be read against a coherence measure whose noise (about 2 points per set) is larger than the effects every lever so far has produced, so I could not tell a working mechanism from noise. The
  measurement problem (a second human or stronger rater, more texts per model) should come first.
* **Stop condition reached:** every in-cap lever and every size I can run here has been tried, none moved blind coherence, and the criterion (12/15, plot 3.5/5, both raters) is not met. I am recording
  that as the result, not continuing to generate runs. The result stands as: the fuzzy recurrent model learns fluent, locally consistent children's text and its rules are causal but diffuse; plot is not reached at
  93K-1.32M parameters and up to 1B characters of data.
