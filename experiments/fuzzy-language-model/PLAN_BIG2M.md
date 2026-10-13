# PLAN_BIG2M: raise the size to ~2M, then the Cosmopedia sequence (registered 2026-10-12, before any run)

User: "Let's raise the model size a bit, say 2M or so, try that, then try the sequence of cosmopedia. Send me an explicit notification when they are done." This is outside the 100K cap by the user's explicit instruction (beyond the earlier 1.32M exception).

**Model.** d256 x 4 layers, 8 heads, 16 rules/head, FFN 1.5x, tsk_dim 16, shortconv 8 (`SIZES["d256-L4"]`, 2,223,936 params), recipe of `flm.run_ladder` (lr 2e-2, warm-up 300, seed 0, ctx 512/batch 16). Runner `flm/run_big2m.py`, outputs `outputs/v37-big2m`. lr not re-tuned at this width (a transfer from the d192 optimum).

**Sequence** (one GPU cell at a time): (1) TinyStories 1B chars; (2) coherence set 14 (2.2M vs 1.32M vs 96K, seeds 50-51, blind, scores before key); (3) Cosmopedia 30M, then 300M, then 1B chars at 2.2M. A 3B-char Cosmopedia run at 2.2M (~10 h estimated) is not included.

**Predictions.** (1) TinyStories 1B BPC 0.58-0.62 (65%; 1.32M gave 0.636, the ladder 96K/251K/653K/1.32M gave 0.98/0.817/0.694/0.636, diminishing steps); below 0.636 (90%). (2) Coherence total above 1.32M's 10.0: 60%; at or above 12/15 (the session's bar): 25%. (3) Cosmopedia: 30M BPC 1.45-1.65 (50%); 300M 1.25-1.45 (50%); at 300M it beats the 300M-token trigram (1.622) (85%); 1B below 1.30 (50%).

## Outcome 1: TinyStories 1B chars, d256 x 4 (2,223,936 params), seed 0, lr 2e-2, warm-up 300
**val BPC 0.59891** (38,653 steps, 5,717 s, 55K chars/s). Ladder at 1B chars: 96K 0.980, 251K 0.817, 653K 0.694, 1.32M 0.636, **2.22M 0.599**. Prediction 0.58-0.62 (65%): **right**; below 0.636 (90%): **right**. Gain per size step keeps shrinking (-0.037 for 1.7x the parameters). One seed, lr not re-tuned: a ranking point, not a citable claim. Coherence set 14 waits until the Cosmopedia cells finish (one GPU cell at a time).

## Outcome 2: Cosmopedia at 2.22M (d256 x 4, seed 0, lr 2e-2, warm-up 300; Cosmopedia valid, first 1M chars)
| chars | 2.22M BPC | 96K ref | 96K tiered |
|---|---|---|---|
| 30M | **1.4727** (209 s) | 1.873 (5 seeds) | 1.856 |
| 300M | **1.2007** (2,089 s) | 1.761 | 1.744 |
Trigram baseline (300M tokens) 1.622: beaten at both budgets (prediction 85%: **right**). 30M prediction 1.45-1.65 (50%): **right** (1.473). 300M prediction 1.25-1.45 (50%): **wrong**, better than predicted (1.201). The 96K model was capacity-bound on this corpus (it never reached the trigram); 23x the parameters gains 0.40 BPC at 30M and 0.56 at 300M, so on broad text size, not data, is the lever. 1B-char cell running.

## Outcome 3: Cosmopedia 1B chars at 2.22M: val BPC **1.13354** (6,961 s; 47,108 steps). Prediction below 1.30 (50%): **right**. 30M/300M/1B: 1.473 / 1.201 / 1.134 (still falling, diminishing: -0.27 then -0.07).

## Coherence set 14 registered before rating (`report/coherence14_*`; fresh seeds 50-51; TinyStories 1B models: 2.22M (0.599), 1.32M (0.636), 96K (0.980); 3 models x 8 texts; two Claude sub-agent raters reading only the blind file; rubric G grammar / C within-text consistency / P plot, each 1-5, total 3-15; scores before the key)
Predictions: 2.22M total above 1.32M on both raters (60%); 2.22M >= 12 (25%); 96K between 4 and 7.5 (80%).

### Set 14 scores (two raters; both score files written before the key was read; total of G+C+P means, 8 texts per model; plot in parentheses)
| model | rater A | rater B |
|---|---|---|
| 2.22M, 1B (0.599) | 9.50 (2.75) | 9.88 (3.00) |
| 1.32M, 1B (0.636) | 9.88 (2.88) | 10.50 (3.12) |
| 96K, 1B (0.980) | 4.75 (1.25) | 5.25 (1.50) |
Predictions: 2.22M above 1.32M on both raters (60%): **wrong** (0.4-0.6 lower, within rater/seed noise); 2.22M >= 12 (25%): **wrong**; 96K between 4 and 7.5 (80%): right. Reading: the 96K-to-1.3M jump (~+5) is large and the 1.32M-to-2.2M step is zero within noise, even though BPC fell 0.037. Coherence on this rubric saturates near 10/15 for these models; the 12/15 bar was not reached at any size tried (one seed per size, 8 texts per model).
