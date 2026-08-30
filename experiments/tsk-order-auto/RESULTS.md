# Results: does `tsk_order="auto"` remove the full-2nd foot-gun?

**Run of record:** 2026-08-30 · seeds 0–9 · `tribble-fis` pin `ae0ef13`
(contains #200, `147c8c4`) · `python experiments/tsk-order-auto/run.py` ·
raw per-seed values in [`results.json`](results.json).

Machine: NEX-210200, i9-14900HX, Windows 11, numpy 2.4.6 / sklearn 1.9.0,
scipy-openblas. Wall-clock ratios are portable; absolute seconds are not.

## Short version

`tsk_order="auto"` discharges grad-school issue #120's acceptance criterion, and
does slightly better than the issue asked for. On diabetes-scale data it never
produces a negative test R² and lands within 0.005 of the best order a caller
could have hand-picked with the labels in front of them. On the two larger
datasets it *beats* every hand-set order, because its candidate list reaches
`3rd`, which neither the issue nor any current table considers.

The finding that reframes the issue: **`full-2nd` is not the best order on any of
the three datasets.** #120 treats it as the accuracy lever that needs a guard.
Measured against the full candidate set, `3rd` dominates it wherever the data is
plentiful and `1st` dominates it wherever it is scarce. There is no regime here
in which the dense all-pairs basis is what you want.

## Test R² (mean ± sample std, ten paired seeds, 25% held out)

| Order | Diabetes (sklearn) | Concrete | Bike Sharing (n=4000) |
|---|---|---|---|
| `0th` | +0.3327 ± 0.1175 | +0.2341 ± 0.0810 | +0.3423 ± 0.0355 |
| `1st` | +0.4474 ± 0.0791 | +0.6377 ± 0.0434 | +0.5425 ± 0.0244 |
| `2nd` | +0.4178 ± 0.0830 | +0.7861 ± 0.0320 | +0.6258 ± 0.0142 |
| `full-2nd` | **+0.1573 ± 0.1705** | +0.8089 ± 0.0365 | +0.6077 ± 0.0684 |
| `3rd` | +0.4049 ± 0.0845 | +0.8504 ± 0.0266 | +0.6512 ± 0.0160 |
| **`auto`** | **+0.4428 ± 0.0807** | **+0.8504 ± 0.0266** | **+0.6512 ± 0.0160** |

Every order sees the identical train/test partition at a given seed, so the
per-seed differences are the model's and not the split's.

| Dataset | rows | features | full-2nd coeffs/rule | rows/coeff | `auto`'s pick |
|---|---|---|---|---|---|
| Diabetes | 442 | 10 | 66 | 6.7 | `1st` ×8, `2nd` ×2 |
| Concrete | 1030 | 8 | 45 | 22.9 | `3rd` ×10 |
| Bike Sharing | 4000 | 12 | 91 | 44.0 | `3rd` ×10 |

## Finding 1 — the acceptance criterion is met

#120 asks for an order *"usable as a default with no negative test R² on
diabetes-scale data, without hand-applying the rows/coeff check."*

`auto`'s worst diabetes seed is **+0.307**; its ten-seed mean is +0.4428 ± 0.0807.
`full-2nd` on the same paired splits reaches **−0.144** at seed 0 and averages
+0.1573 ± 0.1705 — a spread more than twice any other arm's, which is the
signature of a model whose test error is dominated by how the split happened to
fall rather than by what it learned.

`auto` beats `full-2nd` on **10/10** diabetes seeds, **10/10** concrete seeds and
**8/10** bikeshare seeds.

## Finding 2 — safety costs ~nothing, and usually pays

A guard that avoided the overfit by collapsing to `0th` would satisfy the letter
of the criterion and be useless. This one does not:

| Dataset | best fixed order | `auto` | Δ |
|---|---|---|---|
| Diabetes | `1st` +0.4474 | +0.4428 | **−0.0046** |
| Concrete | `3rd` +0.8504 | +0.8504 | 0.0000 |
| Bike Sharing | `3rd` +0.6512 | +0.6512 | 0.0000 |

The only cost is on diabetes, and it is 0.005 against a seed-to-seed spread of
0.081 — it is the CV occasionally choosing `2nd` (2 seeds of 10) where `1st` was
marginally better, and the two orders differ by 0.03 R². Against `full-2nd`,
which is what a caller reaching for the interaction basis would otherwise have
set, `auto` is **+0.29 / +0.04 / +0.04** across the three.

## Finding 3 — the cost is 1.5–2.3×, not 5×

`auto` runs k-fold CV over five candidate orders, so the naive expectation is
~5× (or 5×k). Measured against `1st`:

| Dataset | `1st` | `auto` | ratio |
|---|---|---|---|
| Diabetes | 0.15 s | 0.22 s | 1.53× |
| Concrete | 0.15 s | 0.24 s | 1.63× |
| Bike Sharing | 0.19 s | 0.45 s | 2.35× |

The premise fit — clustering and membership construction, which is the expensive
half — is done once and shared across the candidates; only the consequent solve
repeats. The ratio grows with n because the consequent solve is the part that
scales with rows.

## Tradeoffs, stated

**1. `auto` is available but is not the shipped default.** `TribbleRegressor`
still defaults to `tsk_order="1st"`. #120's phrasing — *"usable as a default"* —
is satisfied in the sense that a safe automatic order now exists and is one
keyword away; it is not satisfied in the sense that a caller who writes nothing
gets it. Changing the library default is a tribble-fis decision and would move
every number in this repo that does not pin an order explicitly.

**2. Nothing in `reproduce/` uses it, and one table would move if it did.**
`reproduce/tables/table_4_1_mog_baselines.py:165` is the single place in the
suite that requests `tsk_order="full-2nd"` (Table 4.5's full-2nd row). Its
datasets sit at 22.9 and 44.0 rows per coefficient, well clear of the danger
band, **so the foot-gun is not currently firing in any published table.** But
switching that row to `auto` would raise Concrete from +0.809 to +0.850 — a real
improvement that is also a changed published number, so it needs a re-derivation
and a prose re-quote rather than a one-line edit. Deliberately not done here.

**3. The pick is not stable at small n.** On diabetes `auto` chose `1st` eight
times and `2nd` twice across ten seeds. That instability is priced in above (it
is the whole of the −0.005), but it means `tsk_order_` should be read as "what
this fit chose", not as a property of the dataset. On concrete and bikeshare the
pick was unanimous.

**4. Three datasets, one model family.** These are the three regression problems
#120 named, measured on `TribbleRegressor` directly. The issue's original numbers
came from `FuzzySystemsExperiments/ruspini_first_fit.py`, a different first-fit
model with a refinement pass, which no longer exists in this repo's history (see
the issue's 2026-08-28 comment). Same phenomenon, different code path — the
absolute values are not comparable to the issue's table, only the shape is.

## Reproducing

```bash
source reproduce/hostenv.sh    # Windows hosts without MSVC; no-op elsewhere
uv run --project tribble-fis python experiments/tsk-order-auto/run.py
```

~36 s for the full 3 × 6 × 10 matrix. Concrete and Bike Sharing read from
`data/`; if those files are absent the datasets report as skipped and Diabetes
still runs, since it ships with scikit-learn.

`test_tsk_order_auto.py` is the CI-affordable slice: diabetes only, three seeds,
~5 s, asserting the relations above rather than absolute values.
