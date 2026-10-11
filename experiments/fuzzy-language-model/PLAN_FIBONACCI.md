# Plan: Fibonacci attention at story size (for the GPU session)

CPU session, 2026-10-11. User direction: keep to story size for now, and try an attention model that
remembers mostly local tokens but some much further back, at Fibonacci distances.

**The model.** `--mixer fibgauss` (Gaussian-TSK attention: one fuzzy rule per visible past token) or
`--mixer fibsoftmax` (ordinary softmax attention) with `--fib-local L`: a query at position t sees the keys
at lags 0..L-1 (dense local window) and then only the Fibonacci lags >= L (8, 13, 21, 34, 55, 89, 144, 233, 377
for L = 8). At ctx 512 that is at most 17 visible keys of 512 (3%): the O(T log T) pattern, with log-spaced
reach. Two layers compose lags, so a fact at lag 20 is reachable as 13 + 8 + ...; the probe measures it.
Readability is better than dense attention: each head's receptive field is a fixed, short list of lags, so the
attention pattern can be shown as a table. `flm/models.py` (`fib_lags`, `fib_mask`, tests in `test_models.py`:
`fib_local >= max_len` reproduces the dense model exactly, and `fib_local = 4` differs from it).
The mask form is exact but not fast (still a dense T x T product); a gather implementation is for later.

**Grid `v10-fib`** (`flm/grids_v10.py`, 12 cells, 10M characters, seed 0, lr 2e-2 and 6e-2, BPE1024, d 48,
2 layers, ctx 512; attention arms 4 heads, ffn-mult 2.0, 94,944 parameters vs D's 93,344):

| arm | what it tests |
|---|---|
| `D` | the recurrent fuzzy mixer, the current headline (baseline) |
| `dense-gauss` | full attention at story size, the upper reference for the sparse arms |
| `fibgauss-L4`, `-L8`, `-L16` | how much dense local window the Fibonacci tail needs |
| `fibsoftmax-L8` | is the fuzzy (Gaussian) form needed, or is the lag pattern doing the work? |

Run one cell first. Then, for every saved checkpoint, run `python -m flm.context_probe CKPT --len 512` (name
recall by filler length is the real test: k = 3 puts the name at about lag 50, k = 12 at about 150, k = 24 at about
280) and `python -m flm.rule_ablation CKPT` for the FFN rules. A winner goes to 30M characters, two seeds.

**Registered predictions (before running):**
* **H37:** `fibgauss-L8` is within 0.01 bits per character of `dense-gauss` at 10M characters (most of what a
  story needs is local plus a few anchors). ~55%.
* **H38:** `fibgauss-L4` is worse than `-L8` by more than 0.01, and `-L16` is no better than `-L8` by more than 0.01. ~50%.
* **H39:** `fibsoftmax-L8` is within 0.01 of `fibgauss-L8` (the lag pattern, not the Gaussian form, carries it). ~55%.
* **H40:** name-recall gain at k = 12 for `fibgauss-L8` is within 0.3 bits of `dense-gauss`, but at a lag that is not
  Fibonacci (check k = 5, about lag 70-80) it is lower; layer composition does not fully fill the gaps. ~40%.
* **H41:** the best attention arm does not beat D at 10M characters at this size (earlier CPU work: softmax and
  fuzzy mixers tie, and the recurrent form had the speed advantage). ~60%.

**Caveats.** One seed; 10M characters (10M screens overstated larger vocabularies at 30M, so a 30M check is
required before believing any ordering); attention arms differ from D by 1.7% in parameters; lr tuned at only two values.
