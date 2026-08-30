# Experiment: TSK consequent-order selection (`tsk_order="auto"`)

**Status:** complete · **Started:** 2026-08-30 · **Closes:** grad-school #120 ·
**Primary model:** `tribblefis.gaussian_regressor.TribbleRegressor`

Findings are in [`RESULTS.md`](RESULTS.md). Short version: `tsk_order="auto"`
(tribble-fis #200, in the `ae0ef13` pin) meets #120's acceptance criterion — no
negative test R² on diabetes-scale data with no hand-applied rows/coeff check —
at a cost of 1.5–2.3× a single fit and ≤0.005 R² against the best hand-picked
order. On the two larger datasets it *beats* every fixed order by reaching
`3rd`. `full-2nd`, the order #120 was written about, turns out not to be the best
choice on any of the three datasets.

## Question

A full-2nd TSK consequent fits `1 + 2·n_features + C(n_features, 2)` coefficients
per rule. Issue #120 measured that this overfits catastrophically when training
rows do not outnumber those coefficients by roughly 5×, and asked for an order
that is safe to leave unset. Does `auto` deliver that without giving up the
accuracy the higher orders exist for?

## Method

Three regression datasets — the ones #120 named — at ten paired seeds, 25% held
out, every candidate order plus `auto`:

| Dataset | rows | features | full-2nd coeffs/rule | rows/coeff |
|---|---|---|---|---|
| Diabetes (sklearn) | 442 | 10 | 66 | 6.7 |
| Concrete | 1030 | 8 | 45 | 22.9 |
| Bike Sharing (n=4000) | 4000 | 12 | 91 | 44.0 |

Diabetes is the case that matters: it sits below the band where the interaction
basis pays for itself, and it is where #120 measured the cliff. The other two are
the control — they are where `full-2nd` is supposed to win, so they test that the
guard is not simply throwing accuracy away.

## Files

| File | What it is |
|---|---|
| `run.py` | The full 3 × 6 × 10 study (~36 s). Writes `results.json`. |
| `test_tsk_order_auto.py` | CI-gated slice: diabetes, 3 seeds, ~5 s. Asserts relations between arms, not absolute R², so it cannot flake on a different BLAS. |
| `RESULTS.md` | Tables, the three findings, and the four tradeoffs. |
| `results.json` | Per-seed R², timings and `tsk_order_` picks from the run of record. |

## Running

```bash
source reproduce/hostenv.sh    # Windows hosts without MSVC; no-op elsewhere
uv run --project tribble-fis python experiments/tsk-order-auto/run.py
uv run --project tribble-fis --with pytest python -m pytest experiments/tsk-order-auto -q
```

`SEEDS`, `ORDERS` and `DATASETS` override the defaults, e.g.
`SEEDS=0 DATASETS=diabetes ORDERS=full-2nd,auto` for a quick look.
