# PLAN_BIG2M: raise the size to ~2M, then the Cosmopedia sequence (registered 2026-10-12, before any run)

User: "Let's raise the model size a bit, say 2M or so, try that, then try the sequence of cosmopedia. Send me an explicit notification when they are done." This is outside the 100K cap by the user's explicit instruction (beyond the earlier 1.32M exception).

**Model.** d256 x 4 layers, 8 heads, 16 rules/head, FFN 1.5x, tsk_dim 16, shortconv 8 (`SIZES["d256-L4"]`, 2,223,936 params), recipe of `flm.run_ladder` (lr 2e-2, warm-up 300, seed 0, ctx 512/batch 16). Runner `flm/run_big2m.py`, outputs `outputs/v37-big2m`. lr not re-tuned at this width (a transfer from the d192 optimum).

**Sequence** (one GPU cell at a time): (1) TinyStories 1B chars; (2) coherence set 14 (2.2M vs 1.32M vs 96K, seeds 50-51, blind, scores before key); (3) Cosmopedia 30M, then 300M, then 1B chars at 2.2M. A 3B-char Cosmopedia run at 2.2M (~10 h estimated) is not included.

**Predictions.** (1) TinyStories 1B BPC 0.58-0.62 (65%; 1.32M gave 0.636, the ladder 96K/251K/653K/1.32M gave 0.98/0.817/0.694/0.636, diminishing steps); below 0.636 (90%). (2) Coherence total above 1.32M's 10.0: 60%; at or above 12/15 (the session's bar): 25%. (3) Cosmopedia: 30M BPC 1.45-1.65 (50%); 300M 1.25-1.45 (50%); at 300M it beats the 300M-token trigram (1.622) (85%); 1B below 1.30 (50%).
