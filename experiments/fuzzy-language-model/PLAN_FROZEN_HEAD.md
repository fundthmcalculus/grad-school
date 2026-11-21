# Plan: the frozen named head: testing at 100K and 50K, and a TRIBBLE hot start for the layers below it (written 2026-10-29)

Status: a plan, nothing in it has run. Predictions are written into `PLAN_BIJECTIVE.md` (registered, dated, before the runs) one stage at a time, as before. Numbers below come from `PLAN_BIJECTIVE.md` and `PLAN_BACKWARD.md`; seed counts are given because several effects are small.

Owner's two requests, 2026-10-29:
1. Because the frozen, explainable head costs less at larger sizes, write a plan for testing 100K and 50K models later. There is still room to improve.
2. The frozen head should allow the TRIBBLE approach of hot-starting the layers *below* the invertible embedding / de-embedding layer.

"Head" here means the invertible fuzzy embedding F, the invertible gain G and the named character dictionary C, which together are both the embedding and the de-embedding. Everything between F and F inverse is "the layers below" (the blocks: fuzzy-delta mixer plus order-1 TSK feed-forward).

## 1. What we know

All at 300M characters, `oh48` (width 48), fp32, lr 0.01. "Frozen" means the dictionary C is frozen; F and G still train with the model. Cost = bits per character against the learned dictionary of the same shape; negative is better.

| corpus | size | frozen C | trainable named start | pull-back 0.0001 per step | seeds |
|---|---|---|---|---|---|
| TinyStories | 100K | +0.0057 | -0.0079 | not run | 3 |
| Cosmopedia v2 | 100K | +0.0179 | -0.0117 | -0.0021 | 2 |
| Cosmopedia v2 | 200K | +0.0057 | -0.0150 | -0.0069 | 2 (frozen, learned); 1 (start, pull; seed 1 running) |
| the Pile | 100K | +0.059 | -0.0005 | +0.0133 | 2 |

Reading: freezing costs less as the model grows (Cosmopedia 0.0179 to 0.0057 on doubling) and more as the text gets more varied (0.006, 0.018, 0.059). The tied head on Cosmopedia 100K is 1.8496 (2 seeds) against 1.8288 for learned invertible and 1.8467 frozen.

Dictionary design (30M characters, 2 seeds, about 100K parameters): a flag per letter with every other coordinate named (`oh48`, 48 dims) costs +0.005 frozen; `oh40` (a 6-bit code for the marks) +0.045; `squeeze32` (a 6-bit code for every non-letter) +0.196; letters as a 5-bit code +0.26 to +0.51. Rule: a frozen dictionary needs equal-distance letters, and the same logic appears to hold for the other frequent classes.

Width at about 100K parameters (learned dictionary, 30M characters, 2 seeds): 64/124 rules 1.3978; 56/156 1.3934; 32/328 1.3700; 24/456 1.3815; 16/706 1.4251; 8/1443 1.6297. Earlier size ladder (learned, 30M): the tied head wins up to about 40K parameters, the invertible model from about 70K, so **50K sits just above the crossover**.

Where the Pile cost sits (evaluation only, `flm/diagnose_dictionary.py`, Pile 300M seed 0 checkpoints, 512 windows, frozen +0.052 in this sample): marks 7% of positions, +0.022 (42% of the cost; 4.47 to 4.78 bits each); lower-case letters 69%, +0.017 (32%; 2.29 to 2.31 bits); upper-case 4%, +0.008 (14%; 5.75 to 5.92); digits 1.6%, +0.004 (8%; 4.65 to 4.92); space and newline almost nothing. So the cost is concentrated in the characters whose coordinates are not one flag each (marks: group flags plus a binary code; digits: one linear coordinate; upper case: one flag), and the learned dictionary beats the frozen one most per position where the named structure is weakest.

Budget note: the dictionary is 98 x 48 = 4,704 parameters, 4.7% of a 100K model and 9.4% of a 50K model, counted in every total above even when frozen. If one frozen dictionary is shared by many models it is not a per-model cost; report both ways.

Shapes at about 50K parameters (computed with the same builder; rules per system / second system): width 48: 68/48 (49,689); width 40: 98/69 (50,116); width 32: 140/98 (50,211); tied head width 56: 78/55 (50,385). The named dictionary leaves little for rules at width 48, so narrower widths are the natural 50K candidates.

## 2. Room to improve (ranked by what the evidence says; each is a test, not a claim)

1. **Equal-distance structure for the frequent non-letters.** The Pile cost is mostly marks, upper case and digits. Candidates: flags (not a binary code) for the frequent marks, an upper-case design that does not rely on one flag, digits as flags plus a value. This costs dimensions, so it competes with item 2. Evidence: `oh40` and `squeeze32` results, plus the per-class diagnostic above.
2. **Shape search with the named dictionary** (width against rules at fixed parameters). Done only for the learned dictionary; at 50K the choice between width 32, 40 and 48 matters most.
3. **Pull-back strength scaled to the step count.** 0.001 per step was right at 3,660 steps and 0.0001 at 36,600; tune per size and corpus (0.0001 and 0.00003 both worked on Cosmopedia).
4. **F's own capacity and the frozen-F question.** F's depth (2) and rules (4) and the scale limit (0.75) were set before the named dictionary existed and never re-tuned. Also the cost of freezing F and G, not only C (needed for item B0 below).
5. **Learning rate and warm-up for the frozen and pulled arms.** lr 0.01 and warm-up 1000 are inherited from the learned-dictionary study; several earlier best rates sat at the grid edge.
6. **Budget.** The owner's rule is 300M characters per run; a longer run would test whether the frozen cost keeps shrinking, but that is the owner's call.
7. **Greedy looping.** All models repeat under greedy decoding on Cosmopedia and the Pile (distinct 8-gram share 0.0 to 0.26). Not a dictionary problem; a separate track if text quality matters.

## 3. Plan A: testing at 100K and 50K

Corpora: TinyStories for screens (30M characters), Cosmopedia v2 for confirmations (300M), the Pile for the hard case (300M). Arms: learned, frozen, trainable start, pull-back (scaled), and the tied head. Two seeds at every cell, three where an effect is under 0.01. Settings: fp32, lr 0.01 unless item 5 changes it.

| stage | what | runs | estimate |
|---|---|---|---|
| A1 | 50K screen, TinyStories 30M: widths 32, 40, 48 (140/98, 98/69, 68/48 rules) x learned, frozen, start, pull x 2 seeds, plus tied x 2 seeds | 26 | about 45 min |
| A2 | 100K shape screen with the named dictionary, TinyStories 30M: widths 32, 40, 48 x the same 4 arms x 2 seeds (item 2) | 24 | about 40 min |
| A3 | lr retune for the frozen and learned arms at the best 100K and 50K shapes (3 rates x 2 arms x 2 seeds) | 12 | about 20 min |
| A4 | 50K confirmation, Cosmopedia 300M, best shape: 4 arms + tied x 2 seeds | 10 | about 2.8 h |
| A5 | 50K and 100K on the Pile 300M, best shape: learned, frozen, pull x 2 seeds | 6 | about 1.7 h |
| A6 | dictionary redesign (item 1) at the best 100K shape, 30M, frozen only, 2 seeds per design, 3 designs | 6 | about 20 min |

Total about 6.5 h of GPU time, plus short setup. Estimates come from measured speeds at 100K with two runs sharing the GPU: about 200 s per 30M-character run and 33 to 36 minutes per 300M-character run; 50K models are assumed to run at least as fast.

Expectations to register before each stage (expectations, not results): the frozen cost on Cosmopedia at 50K lands between +0.03 and +0.06 (the 100K and 200K points suggest about +0.04 by extrapolation); a narrower width (32 or 40) beats width 48 at 50K by at least 0.02, because 68 rules is too few; the tied head is within 0.01 of the invertible learned model at 50K (the old ladder put the crossover near 40K to 70K); a redesigned dictionary (item 1) cuts the frozen Pile cost by at least a third.

Decision rules: the frozen head is "good enough" at a size when its Cosmopedia cost is at most 0.03 and its Pile cost at most 0.05; the pull-back is the fallback where the frozen cost is above that. If at 50K the tied head beats the invertible learned model, say so and report the size below which the invertible model is not the better choice.

## 4. Plan B: a TRIBBLE hot start for the layers below the frozen head

### 4.1 What TRIBBLE means here (my reading, from `PLAN_BACKWARD.md`; correct me)
Consequent-first construction: commit to the answers (here the next characters) first, build each rule's antecedent backward from the positions that share an answer, fit the consequents in closed form, and only then let gradient descent refine. No gradients in the construction, a fraction of a second, then trained on as a hot start.

### 4.2 What happened before, and why it is not a reason to skip this
On the 96K BPE1024 model (`PLAN_BACKWARD.md`, `v22-tribble` to `v27-cover`, 30M characters, five seeds) every consequent-first variant was worse than random initialisation at the end: +0.045 to +0.064 bits in 5 of 5 seeds (centroid, least-squares, association clusters, aligned PPMI embedding, wide rules); hybrids and a coverage placement +0.025 to +0.042; only antecedent-first data placement was level (-0.0066, within noise). The line was closed on 2026-10-11. Measured mechanisms: the layer-1 feed-forward was dead after training (3.2 effective rules against 43.4, mean gate 0.000); the construction's output gain pegged at 1,024; the better the starting loss, the worse the final one.

What the frozen head changes, mapped to each failure:

| earlier failure | under a frozen head |
|---|---|
| the answers were arbitrary: the token embedding was random at construction time, so "answer clusters" were arbitrary token groups | the answers are the 98 fixed, named codewords, and F makes the target state exact: `z_y = G(F(C_y))` for next character y. The groups can follow the named coordinates (letters, space, marks, digits, upper case). Addressed in principle. |
| the output scale was swallowed by the output RMSNorm (gain 1,024 in every layer) | the invertible model has no lossy norm; the scale is set by the fixed gain G and the dictionary. Needs checking, but the cause is removed. |
| layer-1 rules died (raw-firing Gaussians off the data get no gradient) | not fixed by the head. Build with global widths, avoid D-squared (outlier) centre placement, and measure effective rules after training. |
| a better start loss did not mean a better end | not fixed; the gates below guard against it. |

The honest prior is negative (zero of the earlier constructions helped). The new test is worth running because two of the four failure causes are removed by the head.

### 4.3 Prerequisite B0: the head must be fixed before the blocks are built
The targets live in meaning space, so F and G (not only C) must exist before the construction. Today the frozen-dictionary arms train F and G with the model. Options: (i) take C, F and G from a trained donor model (`FIXED_FG=<checkpoint>`, existing; the earlier cost with a learned dictionary at width 56 was about +0.045); (ii) train a small donor first, then freeze; (iii) later: construct F itself. B0 measures the cost of freezing C, F and G together with the named dictionary on Cosmopedia 300M (seeds 0 and 1, 100K, donors taken from the finished `final100k` learned and trainable-start checkpoints) and the cross-donor case (a donor trained on TinyStories used on Cosmopedia). About 70 minutes. This is also the test for sharing one meaning layer across models (the block-swap idea in `PLAN_BIJECTIVE.md`).

### 4.4 Construction (what to build)
For each block in order, with the frozen head giving exact next-character targets:
1. group positions by the named class of the next character (a few classes at first, then finer);
2. antecedent of rule r: Gaussian over the block's normalised input, centre at the group mean, width the global input spread;
3. consequent: closed-form ridge least squares of the residual target (`z_y` minus the block's input state, since blocks are residual) on the rules' firing;
4. no output gain search (nothing to peg); mixer rules in stage 2 only if stage 1 passes.
Code: adapt `flm/conseq_first.py` (built for the BPE FFN) to the invertible model's blocks, hook in through `--rule-init`-style switch without touching `train.py`'s other paths.

### 4.5 Stages and gates
| stage | what | gate to continue |
|---|---|---|
| B1 | construction only, no training (forward passes, CPU): initial validation bits per character against random initialisation and against the character unigram and bigram entropies (to be computed first) | beats random by at least 1 bit and beats the unigram by at least 0.3 bits; the earlier constructions never reached the unigram |
| B2 | 30M TinyStories, 100K, frozen head from a donor: random init against constructed (and constructed plus warm-up 300), 5 seeds, curves every 100 steps, effective rules in both layers, mutual information between winning rule and next character | paired final difference at most +0.005 and at least 25% fewer steps to 1.5 bits; or at most +0.01 with the explainability gain below |
| B3 | the same at 50K, plus a short-budget test (3M characters) where a hot start should matter most | the hot start helps at the short budget even if not at 30M |
| B4 | only if B2 passes: mixer construction, then a 300M confirmation | |

Kill criteria (a third closure, recorded as before): B1 fails; or B2 is worse than random by at least 0.02 in 4 of 5 seeds; or layer-1 effective rules after training are below 10.

### 4.6 Why it is worth it even if the loss does not move
With a fixed, named head, a constructed rule has a meaning by construction (rule r answers "next character is in group g"). If the rules stay labelled after training (measured by mutual information between the winning rule and the next character, and by the group the rule's consequent points to), that is an explainability gain the random start cannot give, and the current rules are diffuse (about half are active; a single rule tells less about the next character than the previous character does). A hot start also helps most at small sizes and short budgets, where 50K models have little capacity to waste.

## 5. Order of work and decisions needed

1. Now, no GPU: write the unigram and bigram baselines; build `conseq_ci` and run B1 (CPU, forward passes); redesign candidates for item 1.
2. A1 and A2 screens (about 1.5 h) to fix the 50K and 100K shapes; A6 beside them.
3. B0 (about 70 min), then B2 and B3 (about 1 h) if B1 passed.
4. A3, then the 300M confirmations A4 and A5 (about 4.5 h) at the settled shapes.
Total GPU: about 6.5 h (plan A) plus about 2.5 h (plan B), about 9 h; the screens (steps 1 to 3) are about 3.5 h.

Owner decisions: whether the dictionary counts toward the parameter budget at 50K (report both, default); whether to waive the 300M cap for one longer run; which donor model to freeze F and G from (default: the finished 100K Cosmopedia learned model, plus the TinyStories one for the cross-donor test).

## 6. Open questions
- Is the frozen-F cost small enough that B2 is a fair test of the hot start (B0)?
- Do the named group classes beat finer classes as rule answers, or is the useful granularity the single character?
- At 50K, does the named dictionary move the tied-head crossover (now near 40K to 70K for the learned dictionary)?
- Does a redesigned dictionary (item 1) make the pull-back unnecessary on the Pile?
