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

## Results (pilot)

_To be filled after the runs. Each cell: validation BPC at 4M characters, mean of seeds 0 and 1, and the paired difference
to its dense reference per seed. Two seeds cannot support a ranking under 0.01; the table says so._
