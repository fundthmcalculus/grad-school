# Plan: a strict sliding-window fuzzy LM, and the frozen invertible dictionary (2026-11-07)

**Question (user):** can the strict autoregressive dependence (token t depends only on the previous w tokens) turn the
dense T x T mixing into an N-diagonal (banded) system that is solved in O(T w) memory and O(T w^2) compute, instead of
O(T^2) memory and O(T^3) compute? Two arms, as the user asked: **(A)** the banded *solve* of the fuzzy delta mixer
(`bsolve`), and **(B)** explicit *window attention* with no solve (`wgauss`). A third strict variant (`wdelta`, a
per-position restart) is included because `bsolve` turned out not to be a strict window (see Findings below).
Second requirement: test every mixer against a **frozen dictionary of invertible embeddings** (PR #289, `oh48`).

**Code:** `flm/windowed.py` (mixers), `flm/test_windowed.py` (exactness and receptive-field tests),
`flm/bench_windowed.py` (wall clock and memory), `flm/run_window.py` (runner: bijective ci model with the windowed mixers,
optionally frozen to `outputs/embspace/oh48.npz`). Branch `claude/busy-ptolemy-hvelqb` (on top of PR #289's head).

## Step 1: exactness and receptive field (done before any training)

Registered before running (predictions in brackets, then the result):

| check | prediction | result |
|---|---|---|
| wgauss = dense Gaussian attention at w >= T (same parameters) | equal | equal (1e-10, float64) |
| bsolve and wdelta = dense fuzzy delta mixer at w >= T, decay none and data | equal | equal (1e-9) |
| bsolve core = dense masked solve at w < T (independent) | equal | equal (1e-10) |
| wdelta core = dense recurrence re-run on each window (independent) | equal | equal (1e-10) |
| wgauss and wdelta move zero beyond lag w-1 | exactly zero | exactly zero |
| bsolve moves beyond lag w-1 | **leaks** (inverse of a banded triangular matrix is dense) | leaks, and the leak decays geometrically: at w=4, \|dy\| beyond lag 3 falls from 1e-2 (lag 4) to 1e-5 (lag 13) and 1e-7 (lag 19) |

`pytest flm/test_windowed.py`: 7 passed.

## Step 2: cost (measured, CPU, one thread, batch 32, d 48, 2 heads, 6 rules, w 32; `flm/bench_windowed.py`)

| T | dense fuzzy delta | bsolve | wdelta | dense Gaussian attn | wgauss |
|---|---|---|---|---|---|
| 256 | 257 ms | 490 ms | 1176 ms | 73 ms | 377 ms |
| 512 | 1485 ms | 1173 ms | 2564 ms | 303 ms | 767 ms |
| 1024 | 7905 ms | **3719 ms** | 5893 ms | 1375 ms | 1614 ms |

Peak RSS from `ru_maxrss` is not usable here (it only grows across the run); memory is reported analytically instead:
per head and batch item, dense delta holds ~T^2 entries (G, Gs, KK, the solve), bsolve ~T*w (the u's and the gathered
window), wdelta ~T*w^2 (the batched local systems). wdelta therefore beats dense memory only for w < sqrt(T).
The wall-clock crossover for bsolve is T ~ 512 at w = 32; the Python loop over T is a large part of its constant.

## Step 3: pilot training (CPU, 4M characters, seeds 0 and 1)

Model: the bijective all-fuzzy model of PR #289 (`BIJ=ci`, the `oh48` schedule's arguments: d 48, 2 layers, 2 heads,
6 rules, FFN_R 196 / FFN_R0 137, decay data, ctx 256, batch 32), lr 0.01 with a 100-step warm-up (the sweep's
1000-step warm-up is longer than a 4M-character run). Arms:

| arm | mixer | window | embedding |
|---|---|---|---|
| A0 | fuzzydelta (dense) | 256 | learned codebook |
| A1 | bsolve | 32 | learned codebook |
| A2 | wdelta | 32 | learned codebook |
| A0f / A1f / A2f | as above | as above | **frozen `oh48`** (`FIXED_C=outputs/embspace/oh48.npz`) |
| B0 | gauss (dense) | 256 | learned (plain tied TinyLM, `flm.train`) |
| B1 | wgauss | 32 | learned |

**Predictions (registered now, scored below):**
1. Frozen `oh48` costs 0 to 0.03 BPC against learned at 4M characters, for each mixer (the 100K cost at 30M was +0.018, shrinking with size; at 4M it could be larger).
2. The window costs quality: wdelta (A2) and wgauss (B1) are worse than their dense references by at least 0.02 BPC (the dense arms see 8x more context).
3. bsolve (A1) is within 0.02 BPC of dense (A0), because its leak is small: a leak that is nearly nothing does not change the loss much. This is weak evidence and is not a claim of strictness.
4. Seed spread at 4M is larger than at 30M: two seeds cannot separate effects under 0.01. Any difference under 0.01 is reported as not resolved.

## Findings

**Exactness:** all 7 checks pass (Step 1). The mixers are what they claim to be at full window and at w < T.

**Receptive field (the main finding):** the banded solve does not give a strict AR window. Its band is w, but the
solution u_t = (I + band)^-1 (beta v) depends on the whole past through the chain u_{t-1}, u_{t-2}, ... The leak is
geometric in this random-init test (about 1e-5 at lag 13), so it is effectively short, but it is not zero, and the size
depends on the spectrum of beta K K^T (set by training). The N-diagonal solve is therefore a *cheaper approximation* of the
dense delta mixer, not a windowed model. The strict version is the restart (wdelta), which costs T w^2 instead of T w.

**Cost:** the banded solve is faster than dense from T ~ 512 on CPU; the restart is faster only at T = 1024 and its memory
(T w^2) only beats dense for w < sqrt(T). Dense Gaussian attention is faster than the explicit gather at T <= 1024
(the gather is memory-bound on CPU); the window attention's win is asymptotic, not at this context length.

## Results (pilot, complete)

Validation BPC on 500K characters, end of training (step 489 of 4M characters). Seeds 0 and 1. Restarts of the container killed runs twice; each run resumed from its last checkpoint (`--resume auto`), so the curves are continuous but the wall-clock times are not usable.

**Bijective track (PR #289 all-fuzzy model, d 48, lr 0.01):**

| arm | mixer | window | embedding | seed 0 | seed 1 | mean |
|---|---|---|---|---|---|---|
| A0 | dense fuzzy delta | 256 | learned | 1.8779 | 1.8802 | 1.8790 |
| A0f | dense fuzzy delta | 256 | frozen `oh48` | 1.8414 | 1.8383 | 1.8399 |
| A1 | bsolve | 32 | learned | 1.8860 | 1.8834 | 1.8847 |
| A1f | bsolve | 32 | frozen `oh48` | 1.8451 | 1.8428 | 1.8439 |
| A2 | wdelta | 32 | learned | 1.8816 | 1.8842 | 1.8829 |
| A2f | wdelta | 32 | frozen `oh48` | 1.8388 | 1.8389 | 1.8388 |

**Attention track (plain tied TinyLM, 41,856 params; not comparable with the bijective track):**

| arm | mixer | window | seed 0 | seed 1 | mean |
|---|---|---|---|---|---|
| B0 | dense Gaussian attention | 256 | 2.0053 | 1.9985 | 2.0019 |
| B1 | window Gaussian attention (`wgauss`) | 32 | 1.9816 | 1.9679 | 1.9748 |

**Paired differences (seed by seed):**

| comparison | seed 0 | seed 1 | mean |
|---|---|---|---|
| bsolve − dense, learned | +0.0081 | +0.0031 | +0.0056 |
| wdelta − dense, learned | +0.0037 | +0.0040 | +0.0039 |
| bsolve − dense, frozen | +0.0037 | +0.0045 | +0.0041 |
| wdelta − dense, frozen | −0.0027 | +0.0006 | −0.0010 |
| frozen − learned, dense | −0.0364 | −0.0419 | −0.0392 |
| frozen − learned, bsolve | −0.0409 | −0.0406 | −0.0408 |
| frozen − learned, wdelta | −0.0428 | −0.0453 | −0.0441 |
| window attention − dense attention | −0.0237 | −0.0306 | −0.0271 |

**Predictions, scored**

| # | prediction | outcome |
|---|---|---|
| 1 | frozen `oh48` costs 0 to 0.03 BPC against learned | **wrong**: frozen is better by 0.039 to 0.044 on every mixer, both seeds |
| 2 | window costs ≥ 0.02 BPC for wdelta and wgauss | **wrong**: wdelta is within 0.004 of dense; window attention is 0.027 better than dense |
| 3 | bsolve within 0.02 of dense | **right**: +0.004 to +0.006 |
| 4 | seed spread too large to resolve effects under 0.01 | **right**: the window and banded-solve gaps are within 0.01 of zero; the frozen effect is not |

**Checkpoints (codebook geometry, distance correlation with `oh48`):** frozen runs = +1.000 at every checkpoint (manipulation check passes). Learned runs stay unrelated to `oh48` (−0.01 to +0.04) from step 100 to 400.

**Reading (pilot only; two seeds, 4M characters):**
- Both strict windows are within about 0.005 BPC of the dense fuzzy delta mixer at w = 32, for both embeddings. The window costs little at this scale and context length. A plausible but untested reason is that at this size most of the useful signal is short-range, so 32 characters carry most of it.
- The banded solve (bsolve) is not a strict window (see Step 1), and it costs about the same as the strict restart. There is no quality reason to prefer the N-diagonal solve over `wdelta`, which gives a true window.
- The frozen dictionary is clearly better than the learned codebook here, by about 0.04 BPC. This is the opposite of the prediction and of the PR #289 result at 100K/30M (frozen costs +0.018). Candidate explanations to test next: codebook scale (frozen std 0.48 vs learned 0.6), and the learned table being under-trained at 4M characters. Not yet tested.
- Window attention is 0.027 better than dense attention at this size. A plausible cause is that a softmax over 256 keys learns more slowly; this is a hypothesis, untested.
- Strict windowing as a *speed* win needs T ≫ 256. The clean CPU benchmark (Step 2) puts the banded solve ahead of dense only from T ≈ 512.

**Not done:** the trained-model receptive-field measurement for `bsolve` (needs a loader for the installed mixer weights); a longer-context run (T = 1024+) where the window's cost advantage would matter; more seeds for any gap under 0.01.

## Handoff: the T = 2048 run (for a local compute machine)

The question this run answers: at a context where the window should matter, does the strict window (wdelta) keep up
with dense, and how much does the window size matter? The pilot was 256 characters of context, where it cannot.

**Setup on the machine**
```bash
cd experiments/fuzzy-language-model
pip install -r requirements.lock          # torch 2.14.1 (CPU wheel pinned; use a CUDA build on a GPU host)
export GRAD_SCHOOL_DATA=<repo>/data
python -m flm.data                        # fetch TinyStories and write data/tinystories/{train,valid}.bin (~2.2 GB)
python -m flm.canonical_dictionary_oh     # writes outputs/embspace/oh48.npz (frozen named dictionary)
python -m pytest flm/test_windowed.py -q  # 7 exactness and receptive-field checks; must pass before any run
```

**Arms** (bijective model as in the pilot, frozen `oh48`, d 48, lr 0.01, warm-up 100, 4M characters, seeds 0 and 1):

| arm | mixer | WIN | ctx | batch |
|---|---|---|---|---|
| T2048-dense | fuzzydelta | (unused) | 2048 | 8 |
| T2048-w32 | wdelta | 32 | 2048 | 8 |
| T2048-w128 | wdelta | 128 | 2048 | 8 (needs a GPU; est. ~10 GB on CPU) |

Batch 8 (not the pilot's 32) keeps the dense T² attention tensors in memory at T = 2048. Steps are then 4× as many per
character, so the lr schedule is per step at a different scale than the pilot. Record the batch in every JSON.

**Command, one arm** (repeat per arm and seed; `--ckpt-every 100 --resume auto` lets an interrupted run continue):
```bash
BIJ=ci WIN=32 FIXED_C=outputs/embspace/oh48.npz FUZZ_FFN_R=196 FUZZ_FFN_R0=137 \
python -m flm.run_window --mixer wdelta --decay data --exp-norm mean --ffn-exp-norm mean \
  --heads 2 --ffn tsk --ffn-order 1 --ffn-norm none --tsk-dim 8 --ffn-mult 1.5 \
  --d 48 --layers 2 --rules 6 --shortconv 0 --ctx 2048 --batch 8 --chars 4e6 \
  --device cpu --threads 4 --eval-chars 500000 --eval-every 0 --warmup 100 \
  --lr 0.01 --lr-floor 0.02 --seed 0 \
  --out outputs/window_t2048/T2048-w32_s0.json \
  --ckpt-dir outputs/window_t2048/ckpt/T2048-w32_s0 --ckpt-every 100 --resume auto
```
For the dense arm, use `--mixer fuzzydelta` and drop `WIN`. For w128, set `WIN=128`. The results land in
`outputs/window_t2048/*.json` (git-ignored); the `val_bpc` field is the number to compare.

**Predictions (registered now, before the run):**
1. Dense beats w32 at T = 2048 by at least 0.01 BPC on both seeds (longer context helps the dense mixer).
2. w128 closes at least half of that gap.
3. Per step, w32 is at least 2× faster than dense at T = 2048 (the T² term dominates dense; the benchmark at T = 1024 was 1.3×).

**Known limits:** the restarts on the sandbox killed runs repeatedly; checkpoints every 100 steps are the recovery
path. The bsolve arm is left out: at T = 2048 its per-token Python loop makes it impractical on CPU. The frozen-versus-
learned effect from the pilot (about 0.04 BPC) is not explained yet and is not part of this run's design.
