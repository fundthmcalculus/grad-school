# PLAN_SURGERY: "brain surgery" on a trained all-fuzzy model (CPU only; run 2026-10-23; result written up 2026-10-23)

## The question (registered before the run)
On the shipped 100K model (`flm-allfuzzy-f7`, weights `allfuzzy/F7re_300M_s0.npz`: V6144, head 640, tail 8, d32, 99,882 parameters), at a position where the model picked the wrong next piece, can a user name the piece that *should* have followed and have the model corrected by editing the **rules that fired**, with the edit small, nameable, undoable, and with little damage elsewhere? The motivating claim: because every block component is a fuzzy system, the surgeon can say which rules fired and by how much each was changed, which a dense model does not allow.

## Method
- **Where to cut.** The last block's feed-forward bank: the output scores are an exact function of the residual `x` before the final gain controller, `x = x_mid + sum_r s_r c_r` (`s_r` the rule's strength, `c_r` its consequent vector). Nothing after that block reads the bank's output, so editing `c_r` changes a position's scores only through `s_r`; it cannot change the memory or any later layer's computation.
- **The edit.** Find the smallest residual change `dx` that makes the wanted piece the top-1 by a margin of 0.5 logits: central-difference gradient of (wanted minus current rival) through the final controller, then a line search along that direction (doubling, then 30 bisections; capped at 50x |x|). Turn `dx` into consequent changes: `Delta c_r = s_r dx / sum s^2` over the rules edited (the minimum-norm change; with all 61 rules it is one gradient step on the consequents).
- **Variants:** `rule1`, `rule3` (the strongest 1 or 3 rules), `rule_all` (all 61), and two dense baselines that need no rules: `bias` (a bias on the wanted piece's output score, everywhere) and `unembed` (a rank-one change of the wanted piece's output embedding along the event's final hidden state; the usual locate-and-edit move on a dense head).
- **Events** (`flm.surgery_exp`, seed 0): from 6,000 validation positions in the first half of the 481 cached windows (one million validation characters), 100 *near misses* (the wanted piece ranked 2nd-4th) and 100 *far misses* (11th-100th). **Bystanders:** 30,000 positions from the second half (different stories, never an event); after each edit they are re-scored for top-1 flips, the change in bits per character, and the mean KL. **Twins:** held-out positions with the same last two input pieces and the same true piece where the model was also wrong.
- Cost: collecting the cache took 146 s on 3 cores; the experiment took 7,106 s on 3 cores (it re-scores 30,000 positions per event and method).

## Results (medians over events that succeeded; p90 = the 90th percentile)
| stratum, method | events fixed | bystander top-1 flips median / p90 | change in bits per character median / p90 | KL median / p90 | edit size (relative norm of the consequent change) median | wrong twins fixed (same prev+current piece) / (same current piece) |
|---|---|---|---|---|---|---|
| near, rule1 | 97/100 | 2.19% / 5.08% | +0.00116 / +0.00583 | 0.0035 / 0.0160 | 19.5% | 82% / 49% |
| near, rule3 | 97/100 | 2.31% / 5.06% | +0.00103 / +0.00452 | 0.0027 / 0.0128 | 7.5% | 82% / 48% |
| near, rule_all | 97/100 | 2.40% / 4.71% | +0.00103 / +0.00383 | 0.0028 / 0.0112 | 1.3% | 82% / 48% |
| near, bias | 97/100 | 1.00% / 5.29% | +0.00187 / +0.01103 | 0.0051 / 0.0303 | - | 88% / 70% |
| near, unembed | 97/100 | 0.50% / 2.39% | +0.00061 / +0.00360 | 0.0015 / 0.0095 | - | 87% / 63% |
| far, rule1 | 56/100 | 8.18% / 19.48% | +0.0203 / +0.1453 | 0.053 / 0.410 | 77.8% | 78% / 53% |
| far, rule3 | 56/100 | 10.49% / 20.11% | +0.0188 / +0.1132 | 0.052 / 0.310 | 31.8% | 79% / 62% |
| far, rule_all | 56/100 | 9.51% / 19.55% | +0.0133 / +0.1121 | 0.038 / 0.309 | 5.1% | 80% / 65% |
| far, bias | 56/100 | 3.36% / 15.35% | +0.0081 / +0.0666 | 0.023 / 0.187 | - | 87% / 81% |
| far, unembed | 56/100 | 1.23% / 3.95% | +0.0024 / +0.0130 | 0.0066 / 0.0366 | - | 78% / 74% |
Base (30,000 bystanders): 0.9605 bits per character, top-1 accuracy 44.3%. Twin counts: near 58 events with same-prev twins, 88 with same-current-piece twins; far 22 and 40. Fixed events are the same for every method (they share the line search), so the variants differ only in how the change is spread.

**Failures, and why far misses succeed exactly 56 times in 100.** Near: 3 of 100 failed (all three reachable in principle, below). Far: 44 of 100 failed, and the cause is two things, checked by a linear program on the output table (can any output vector make the wanted piece the top-1? The scores are `E h`, so the wanted piece can win only if its embedding is a vertex of the table's convex hull):
- 7 failures are **provably impossible**: all seven want a *tail* piece (a fuzzy set: its embedding is a weighted average of the 8 prototypes, so it lies inside the hull and no hidden state can make it the argmax). Tail targets: 27 of the 100 far events, only 2 succeed; head targets: 54 of 73 succeed.
- The other 37 failures (19 head, 18 tail) **are reachable in principle** (the linear program finds a direction with positive margin) but not by the planner here: it follows one straight line, the gradient of (wanted minus the *current* rival), and along that line another piece overtakes before the wanted one wins, up to the 50x |x| cap. A planner that re-targets the rival or follows the linear program's direction would fix some of them; it was not built.
So 56 = 100 - 7 (impossible tail pieces) - 37 (planner limit). The near-miss success rate is 97 because the wanted piece is already close.

**Does a correction generalise to the same situation in another story?** Cheap, and run (`flm.surgery_twins`, 97 near-miss events with a plan, the 3-rule edit). Two readings. (1) *Same preceding pieces*, in the table above: of the held-out positions where the model was wrong with the same last two pieces and the same true piece, 82% (near) are right after the edit for rule edits, 87-88% for the dense baselines. (2) *Same rule signature*: held-out positions (other stories) whose three strongest last-block rules are the event's three, median 85 per event: among those whose true piece is the wanted one and where the model was wrong, **71% (921 of 1,297) are fixed**; among those whose true piece is something else and where the model was right, **30% (3,074 of 10,301) are broken**. So the corrected rule edit does carry over to the same situation, but the same three rules fire in situations whose right answers differ, and the edit pushes the wanted piece in all of them: on that class the edit breaks about three times as many positions as it fixes.

## Reading (honest)
- The rule edits are **not more local than the rank-one unembedding edit on this metric.** Unembed beats every rule variant on flips (near 0.5% against 2.2-2.4%; far 1.2% against 8-10%) and on bits-per-character damage (near +0.0006 against +0.0010-0.0012; far +0.0024 against +0.013-0.020). Only against the context-free `bias` baseline do rule edits do better on the *tail* (near p90 +0.0038 against +0.0110 in bits per character), not on the median.
- Edits are not free: the far-miss median damage (+0.013 to +0.020 bits per character, p90 +0.11) is 13-18 times the near-miss damage, and the two worst examples below show a strong edit turning generation into loops ("back back back", "on the slide on the slide").
- What interpretability does buy here (this is the finding that stands): **you can name the rules and the edit.** The surgeon can show which feed-forward rules fired (their strengths at that position), the antecedent of each in words (`AllFuzzyLM.rule_text`), the size of the change per rule (the all-rules edit changes the consequents by 1.3% at the median for near misses, 5.1% for far), apply it, regenerate, and **undo it exactly** (the browser test restores the scores to 1e-6). The edit is auditable and architecturally bounded to the output side of the last block (it cannot alter the memory or the other layers), which a rank-one change to a dense embedding is not described by. It is *not* evidence that the rule edit damages less.
- The honest framing for the demo: "you can see exactly what you changed and take it back", not "you can change the model without side effects".

## Limits
One checkpoint (the shipped F7, seed 0); one run; bystanders come from the same corpus and the second half of the same windows as the events (no other corpus, no other model); the margin (0.5 logits) is one setting; near misses are an easy stratum (rank 2-4); no held-out evaluation of the corrected *sentence* beyond the twins above (the continuations below are anecdotes, not a measure); the planner's failures (37 reachable cases) understate what a better planner could fix; the dense baselines are applied to the event position's scores only, as a surgeon would apply them, without re-optimisation.

## Examples (selected by a fixed rule, not by quality)
Rule: near-miss events 0, 25, 50, 75 and far-miss events 0, 25, 50, 75 in event order (successes and failures alike); then the first two far-miss events on which the all-rules edit failed and the two on which it did the most collateral damage. The planner's all-rules edit; probabilities at temperature 1 for the event position; continuations are 30 pieces at temperature 0.5, top-k 10, the same seed before and after. Full data: `report/surgery_examples.json`.

**Example 1: fixed_rule_near (event 0 of the near list)**  
Context: `... there. It was a mail bird! He watched the mail bird deliver a`  
Model chose `' new'`; the surgeon wanted `' special'` (a head piece), rank 3 before.  
Top 3 before: `' new'` 0.081; `' big'` 0.069; `' special'` 0.032  
Top 3 after the all-rules edit (61 rules, size 2.8% of the consequents): `' special'` 0.052; `' big'` 0.032; `' new'` 0.013  
Top 3 after the unembedding edit: `' special'` 0.121; `' new'` 0.073; `' big'` 0.063  
Continuation before: `' big, shiny, red rock.\nJohn was so excited and started to laugh. He started to dance, but he was too scared. He was'`  
Continuation after: `' big, strong, red apple.\nHis mommy was so excited. She asked her mommy to help him. "Let\'s make a surprise," said'`
- bystander flipped: `... house, there was a little girl named Lily.` true `' She'` (p 0.570 -> 0.368), old top-1 `' She'`, new top-1 `' Lily'`
- bystander flipped: `... town, there lived a little girl named Lily.` true `' She'` (p 0.537 -> 0.351), old top-1 `' She'`, new top-1 `' Lily'`
- bystander flipped: `... time, there was a big, colorful tent.` true `' It'` (p 0.289 -> 0.130), old top-1 `' It'`, new top-1 `' The'`

**Example 2: fixed_rule_near (event 25 of the near list)**  
Context: `... to talk. "Sharing is when you give something to someone else,"`  
Model chose `' said'`; the surgeon wanted `' he'` (a head piece), rank 3 before.  
Top 3 before: `' said'` 0.427; `' she'` 0.193; `' he'` 0.143  
Top 3 after the all-rules edit (61 rules, size 3.2% of the consequents): `' he'` 0.481; `' she'` 0.292; `' they'` 0.093  
Top 3 after the unembedding edit: `' he'` 0.451; `' said'` 0.274; `' she'` 0.124  
Continuation before: `' said the owl.\nThe wise owl was happy. He said, "Thank you, little mouse!" The little mouse was happy. They played and'`  
Continuation after: `' he said.\nThe wise owl looked at the bird and said, "I can help you find your way to make it very fast." The little'`

**Example 3: fixed_rule_near (event 50 of the near list)**  
Context: `... strawberries. It looked yummy and colorful. They shared it with her mom and`  
Model chose `' dad'`; the surgeon wanted `' her'` (a head piece), rank 2 before.  
Top 3 before: `' dad'` 0.453; `' her'` 0.082; `' mom'` 0.077  
Top 3 after the all-rules edit (61 rules, size 1.9% of the consequents): `' her'` 0.280; `' dad'` 0.170; `' mom'` 0.065  
Top 3 after the unembedding edit: `' her'` 0.449; `' dad'` 0.272; `' mom'` 0.046  
Continuation before: `' dad. They were happy that their dad was not happy.\nThey were happy to have a big party with their friends. They played all day and'`  
Continuation after: `' her mom and dad. She hugged her and said, "Thank you, Lily, for you, Mr. Bear. I\'m sorry and I was'`

**Example 4: fixed_rule_near (event 75 of the near list)**  
Context: `... a time, a little girl was playing outside. She was walking up a`  
Model chose `' big'`; the surgeon wanted `' hill'` (a head piece), rank 4 before.  
Top 3 before: `' big'` 0.165; `' tree'` 0.044; `' small'` 0.034  
Top 3 after the all-rules edit (61 rules, size 1.7% of the consequents): `' hill'` 0.105; `' tree'` 0.064; `' rock'` 0.041  
Top 3 after the unembedding edit: `' hill'` 0.220; `' big'` 0.133; `' tree'` 0.036  
Continuation before: `' big tree and saw a little bug. The bug was very scared. It was very special. \nThe bug was so excited and she said,'`  
Continuation after: `' hill and saw a big tree. She was very excited to see what it was.\nThe girl saw her friend, Sam. "What is this'`

**Example 5: fixed_rule_far (event 0 of the far list)**  
Context: `... Whip! Can I keep it???? His mom replied, ???No,`  
Model chose `' I'`; the surgeon wanted `' J'` (a tail piece, a fuzzy set over the 8 prototypes), rank 31 before.  
Top 3 before: `' I'` 0.282; `' you'` 0.191; `' it'` 0.049  
**Edit failed**: no step up to 50x |x| along the planned direction reaches the margin: the edit failed  
Continuation, unedited: `' I am so happy. Can I help you?"\nSo, Lco and his mom went outside to play. They looked around and found a'`

**Example 6: fixed_rule_far (event 25 of the far list)**  
Context: `... not fall into the deep hole.
The family was very happy that the`  
Model chose `' squirrel'`; the surgeon wanted `' animal'` (a tail piece, a fuzzy set over the 8 prototypes), rank 75 before.  
Top 3 before: `' squirrel'` 0.108; `' bird'` 0.048; `' wind'` 0.031  
**Edit failed**: no step up to 50x |x| along the planned direction reaches the margin: the edit failed  
Continuation, unedited: `' sun was full of water. They both had a great time at the park, and they were all so happy. The family was happy and happy.'`

**Example 7: fixed_rule_far (event 50 of the far list)**  
Context: `... all! It was a turtle! The turtle woke up and said, "`  
Model chose `'Thank'`; the surgeon wanted `'Hey'` (a tail piece, a fuzzy set over the 8 prototypes), rank 11 before.  
Top 3 before: `'Thank'` 0.207; `'I'` 0.129; `'Hello'` 0.083  
Top 3 after the all-rules edit (61 rules, size 3.3% of the consequents): `'Hey'` 0.112; `'I'` 0.068; `'Help'` 0.059  
Top 3 after the unembedding edit: `'Hey'` 0.257; `'Thank'` 0.156; `'I'` 0.098  
Continuation before: `'Thank you, little boy!" His friends were happy and played with the little boy all day.\n<|eot|>\nOnce upon a time, there was a big,'`  
Continuation after: `'Hey, can you help me fix your toy?" The little boy said, "Yes, it\'s okay. Let\'s play together." The little girl'`

**Example 8: fixed_rule_far (event 75 of the far list)**  
Context: `... Lily and her dad took the dog home and gave it a nice bath.`  
Model chose `' They'`; the surgeon wanted `' Now'` (a head piece), rank 12 before.  
Top 3 before: `' They'` 0.477; `' The'` 0.198; `' Lily'` 0.128  
Top 3 after the all-rules edit (61 rules, size 10.2% of the consequents): `' Now'` 0.280; `' Lily'` 0.170; `' So'` 0.103  
Top 3 after the unembedding edit: `' Now'` 0.442; `' They'` 0.268; `' The'` 0.111  
Continuation before: `' They all became friends and played together all day.\n<|eot|>\nOne day, a little boy named Tim went to the park. He saw a big, shiny'`  
Continuation after: `' Now, they could play together every day, and they all lived happily ever after.\n<|eot|>\nOne day, a little girl named Mia went to the park'`

**Example 9: failed_rule_all (event 2 of the far list)**  
Context: `... sorry for the spider. She said she had a band-aid and a`  
Model chose `' big'`; the surgeon wanted `' kiss'` (a tail piece, a fuzzy set over the 8 prototypes), rank 38 before.  
Top 3 before: `' big'` 0.031; `' lot'` 0.027; `' plan'` 0.021  
**Edit failed**: no step up to 50x |x| along the planned direction reaches the margin: the edit failed  
Continuation, unedited: `' plan. She said, "I want to share the sock. I will be careful. You are a good job. You should be careful."\n'`

**Example 10: failed_rule_all (event 3 of the far list)**  
Context: `... Lily. She said they were very kind and fun. She said she would`  
Model chose `' be'`; the surgeon wanted `' miss'` (a tail piece, a fuzzy set over the 8 prototypes), rank 94 before.  
Top 3 before: `' be'` 0.115; `' always'` 0.114; `' not'` 0.089  
**Edit failed**: no step up to 50x |x| along the planned direction reaches the margin: the edit failed  
Continuation, unedited: `' always be friends and always be kind and not like to be kind.\n<|eot|>\nOnce upon a time, there was a little boy named Tim. Tim was'`

**Example 11: max_collateral (event 17 of the far list)**  
Context: `... and pushed himself down. He felt the wind in his hair and the sun`  
Model chose `'.'`; the surgeon wanted `' on'` (a head piece), rank 17 before.  
Top 3 before: `'.'` 0.411; `' was'` 0.151; `' were'` 0.030  
Top 3 after the all-rules edit (61 rules, size 16.4% of the consequents): `' on'` 0.440; `' was'` 0.267; `' went'` 0.078  
Top 3 after the unembedding edit: `' on'` 0.405; `'.'` 0.246; `' was'` 0.090  
Continuation before: `'. They were not happy.\nBut then, something unexpected happened. A big wind came and blew the slide on the slide. The slide broke into'`  
Continuation after: `' on his head on the slide on his head on the slide on the swings on the slide on the slide on the slide on the slide on his head'`
- bystander flipped: `... boy named Tim found a big drum in his room` true `'.'` (p 0.997 -> 0.000), old top-1 `'.'`, new top-1 `' on'`
- bystander flipped: `... a time, there was a little girl named Lucy` true `'.'` (p 0.997 -> 0.000), old top-1 `'.'`, new top-1 `' on'`
- bystander flipped: `... a time, there was a helpful dog named Spot` true `'.'` (p 0.996 -> 0.000), old top-1 `'.'`, new top-1 `' on'`

**Example 12: max_collateral (event 18 of the far list)**  
Context: `... grabbed his little hands and pushed on the table. The table started to move`  
Model chose `' and'`; the surgeon wanted `' back'` (a head piece), rank 49 before.  
Top 3 before: `' and'` 0.165; `'.'` 0.116; `'!'` 0.108  
Top 3 after the all-rules edit (61 rules, size 14.0% of the consequents): `' back'` 0.425; `' away'` 0.258; `' off'` 0.185  
Top 3 after the unembedding edit: `' back'` 0.215; `' and'` 0.130; `'.'` 0.091  
Continuation before: `'! It was a big, red ball! The ball was too high and it was very big and warm.\nThe ball flew down and started to'`  
Continuation after: `' back back back back back back back back back back back back down down down back back back back away away back back back back away back back back back'`
- bystander flipped: `... much fun. When it was time to go home` true `','` (p 0.994 -> 0.003), old top-1 `','`, new top-1 `' back'`
- bystander flipped: `... It makes me think of a dream," Tom says` true `'.'` (p 0.992 -> 0.016), old top-1 `'.'`, new top-1 `' back'`
- bystander flipped: `.... It will make your knee better," Lily says` true `'.'` (p 0.991 -> 0.016), old top-1 `'.'`, new top-1 `' back'`

## The demo (gh-pages): design and what is built
Not republished, `demo/dist` untouched. **Built and tested:** `demo/web/surgery.js` (about 75 lines) plus one line in `demo/web/flm.js` (`step(..., {explain:true})` also returns the residual `xPre`), and `demo/web/test_surgery.mjs` (parity against the numpy edit: three events, relative edit size identical to four decimals, margin -0.94 -> 0.50, -4.64 -> 0.50, -1.08 -> 0.50, the wanted piece is the top-1 after applying, and undo restores the scores to 1e-6; planning takes 6-9 ms; run with `node demo/web/test_surgery.mjs`, it needs `python -m flm.surgery_check` for `surgery.check.json`).
**The computation.** For the all-rules minimum-norm edit the second step is closed form: `Delta c_r = s_r dx / sum_r s_r^2`. The first step (`dx`) needs a one-dimensional search because the final gain controller makes the scores a nonlinear function of `x`; a first-order closed form (`dx = (margin - m) g / |g|^2` with `g` the gradient of wanted minus rival) is a good start but can miss the margin, so `surgery.js` runs the same gradient plus bisection as the Python (about 40 evaluations of the 6,144 x 32 score table, 6-9 ms). No closed form is needed or claimed for `dx`.
**Minimal UI.** The existing explanation view already lists the rules that fired for each generated piece. Add: (1) click a generated piece; the panel shows its top-5 alternatives with probabilities; (2) pick (or type) the piece that should have followed; (3) the page shows the plan: the rules to change (id, strength, one-line IF-THEN text from `ruleShape`), the edit's relative size, the margin before and after, and a warning when the plan failed (a tail piece, or no step reaches the margin) or when the relative size exceeds 10% ("large edit"); (4) *Apply* edits the in-memory last-block consequents (`Surgery.apply`) and regenerates from the clicked position with the same seed; (5) *Undo* (`Surgery.undo`) restores the weights exactly; a list of applied edits is kept and can be exported as JSON (`toJSON`/`fromJSON`) and re-applied. Three small safeguards worth building with it: a counter of "other positions that changed" measured on the current generated text, a limit of a few active edits, and a clear note that an edit is applied to *every* later position (the strongest-collateral examples above show why).
Not built: the page UI itself and any re-planning for tail pieces (those cannot be reached; the page must say so).
