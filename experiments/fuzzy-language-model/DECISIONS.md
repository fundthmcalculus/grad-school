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

## D8 (2026-10-11): second rater on set 7, and the request I sent
* The GPU session rated set 7 blind (r = 0.75 with me): 1.32M model 7.81 +/- 0.51 (plot 1.88 / 1.75), 96K at 1B characters 7.00 +/- 0.41, 96K at 300M characters 6.12 +/- 0.19. With two raters the order follows size and
  data (1.32M > 1B > 300M) but each gap is only 1-2 standard errors, and the best mean total of any model across seven sets is about 8.2/15 (plot 1.4-2.4 /5). A 14x parameter increase bought about 0.8 points against about 2 points
  of set-to-set noise. The conclusion in D6/D7 stands: the criterion is not met and the effect of size is not separated from data.
* **Delegation:** I sent the GPU session the request to put the 1.32M x 1B-character run to its user (cell, flags, and the fact that it is above its cap), and said it should run it only if approved. The GPU session had already
  said it will not run it without that approval, so it is waiting on you (and on whoever sets that session's cap). If it is declined, the record reads "not tested above 165K parameters."

## D9 (2026-10-11): correction and new direction from the user
* **Correction:** the GPU run the user started is **3B characters on the 96K model** (`v18-3b`, about 2.2 hours; sizing run `v18-3b-size`: 1.1052 BPC at 30M characters in 82 s), not the 1.32M configuration on 1B characters. I had assumed the latter;
  my progress page said so and is corrected. The 1.32M x 1B request to the GPU session is moot.
* **User instruction relayed by the GPU session:** focus on truly tiny models and make nothing larger yet. I am following it: no further CPU runs above 96K parameters, and the 251K and 1.32M CPU results stay on record as evidence only.
* **Next, inside that scope:** when `v18-3b` finishes, rate its coherence set blind as second rater, refresh the progress page (https://claude.ai/artifact/SEDQDg6ZPvBeHZZth515kP), and record whether 3B characters moves blind coherence.

## D10 (2026-10-12): the 3B run finished; I graded it
* `v18-3b` finished at 0.9777 BPC (flattened: 1B to 3B is inside seed noise). I generated and rated a blind set myself (set 8, one rater): 3B 6.88 /15 (plot 1.38), 1B 7.50, 300M 5.75. No coherence gain from 3B characters.
* **New from the GPU session, bearing on explainability** (`PLAN_RULEDIST.md`, `flm/rule_dist.py`): the output distribution is an exact sum of per-rule terms, but it is a superposition: it takes about 32 of 72 active rules per position and layer to stay
  within 0.03 BPC, a dominant rule's stated top-10 tokens contain the true next token only 15-19% of the time, and low-rank or sparse students degrade fast. So the single-winner rulebook I used to claim explainability is not a faithful reading of the model.
  My ablation result (cited rules are causal) stands but is weaker than "the rulebook explains the model". I am recording that the explainability half of the goal is **less established than I reported**, not more.

## D11 (2026-10-12): a partial rehabilitation of explainability
The GPU session's `rule_dist` showed a single winning rule is an unfaithful reading. I built the calibrated reader it listed as unrun: 8 rules per position and layer plus the embedding and mixer terms recover 94% of the model's top-10 hit rate (0.767 vs 0.814) and 70% of its top-10 set;
16 rules recover 98%. Rules alone do not (0.679 even with all 72). So the honest explainability claim is: **the fuzzy rules supply most of the prediction, a readable 8-16 rules per position and layer plus named non-rule terms reproduce it; the rule book alone does not.** Two of my three predictions were wrong.
This is still a reading of next-token prediction, not an explanation of why the stories are not coherent.

## D12 (2026-10-12): rule-view analysis done (user chose it); what I took from it
* Rule roles are stable and readable (90-95% of the top rules keep their label across data halves): that is a real explainability result for the rules themselves.
* Rule-set and token repetition within a text track the raters at about -0.3 (3-4 null sd); the number of active rules and usage entropy do not. So the rule view can *flag* the repetitive stories, and I added a repeat marker to the demo page.
* The reading is not less faithful in loops (0.80 vs 0.71), so repetition is explained by the rules, not hidden from them.
* It still does not explain plot; the r is about -0.3 on 144 pooled texts, and the criterion (12/15, plot 3.5/5) is untouched.

## D13 (2026-10-11, scheduled check-in): the repetition-aware loss did not help coherence
At the 13:06 check-in I pulled the GPU session's work (v23 warm-up sweep, cfwide, v24-diag, v25-rep, v26-hybrid, v27-cover and v28-emb queued), found item 4 (the repetition-aware loss) marked "coherence not rated", and rated it myself (set 9, one rater): baseline 7.25, weighted 6.62, unlikelihood 5.62 /15. The loss cut repeats 20% and the stories did not improve. The only measured correlate of low coherence is therefore not a usable training target by this route.
