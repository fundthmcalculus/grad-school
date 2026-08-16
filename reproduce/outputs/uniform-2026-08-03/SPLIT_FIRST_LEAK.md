# Does the Concrete reconciliation's transductive preprocessing inflate its numbers?

**Measured 2026-08-03. Answer: no, and every control behaves exactly as it should.**
The leak is real, the flat MoG arms are the only rows exposed to it, and its effect on
every one of them is inside the seed spread. Where it moves the closed-form arms at all,
removing it makes them *slightly better*, which is the opposite of the direction a
flattering artifact would take.

## What the defect is

`reproduce/tables/table_concrete_reconciliation.py`'s `prepare()` runs on the full
dataset and `mog_arm` splits afterwards. Two of those steps are data-dependent:

| step | what it does on all 1,030 rows | severity |
|---|---|---|
| `F.unit_scale(y_raw)` | min-max the target using the full target's min and max | **none for R²** — affine, and R² is invariant under an affine map of the target |
| `partition_output(N_BUCKETS, yt)` | `pd.qcut` quantile edges, plus per-bucket means | **a real leak** — a data-dependent *nonlinear* discretization whose `y_bucket_mean` is passed to `solve_tsk_consequents` and on into `predict_tsk`, so test-fold targets reach the prediction path |

The feature scaler is fit on the full frame in **both** code paths — `prepare()` and
`preprocess_for_others()` — so it is transductive for the tree, mixture, CART and Random
Forest arms too, and is therefore symmetric across arms. It does not bias the
*comparison*. The target partition is the asymmetric half: nothing but the flat MoG arms
touches it.

That asymmetry is why Table 6.1's caption could not honestly say "one protocol", and it
is why the CART/RF rank-invariance control does not cover this — a rank-invariant model
is precisely the model that cannot detect leakage through a monotone scaler or a
quantile partition.

## The variant

`REPRO_SPLIT_FIRST=1` selects `prepare_split_first()`, which fits every data-dependent
step on the training fold alone:

- the target's min and max come from the train rows, and both folds are scaled with them
  (`F.unit_scale_with`), so test targets may legitimately fall outside `[0, 1]`;
- `partition_output` runs on the train fold, so the quantile edges and every bucket mean
  are train-only; test rows are labelled with the train-derived edges via `np.digitize`,
  which is cosmetic here because `mog_arm` reads only `yte["y_value"]`;
- the feature scaler is fit on train (`F.fit_scaler`) and applied to both
  (`F.apply_scaler`).

It cannot be hoisted out of the seed loop the way `prepare()` is, because that hoist is
valid only *because* the preprocessing ignores the split — which is the property being
removed. So it pays one preprocessing per seed. `span`, the MPa rescale applied to RMSE
for reporting, still comes from the full raw target: it is a display unit rather than a
model input, and holding it fixed keeps the RMSE column on one scale across arms.

Opt-in, not the default. Switching the default would re-quote most of Chapters 4 and 6,
and that is an author's decision rather than a side effect of a bug fix.

## The measurement

Ten seeds, `tribble-fis 4b33a0d`. Left column `outputs/full-2026-08-03/`, right column
`outputs/splitfirst-2026-08-03/`.

| row | as shipped | train-fold only | Δ R² |
|---|---:|---:|---:|
| flat MoG-TSK 0th, closed-form | −0.434 ± 0.241 | −0.416 ± 0.230 | **+0.018** |
| flat MoG-TSK 1st, closed-form | 0.787 ± 0.026 | 0.792 ± 0.020 | **+0.005** |
| flat MoG-TSK 2nd, closed-form | 0.832 ± 0.027 | 0.837 ± 0.025 | **+0.005** |
| flat MoG-TSK 0th, refined | 0.517 ± 0.210 | 0.579 ± 0.068 | **+0.062** |
| flat MoG-TSK 1st, refined | 0.866 ± 0.029 | 0.834 ± 0.054 | **−0.032** |
| flat MoG-TSK 2nd, refined | 0.877 ± 0.037 | 0.870 ± 0.047 | **−0.007** |
| fuzzy tree, raw | 0.583 ± 0.067 | 0.583 ± 0.067 | +0.000 |
| mixture of experts, raw | 0.679 ± 0.062 | 0.679 ± 0.062 | +0.000 |
| CART, raw | 0.825 ± 0.047 | 0.825 ± 0.047 | +0.000 |
| Random Forest, raw | 0.909 ± 0.018 | 0.909 ± 0.018 | +0.000 |
| fuzzy tree, log+min-max | 0.689 ± 0.056 | 0.689 ± 0.056 | +0.000 |
| mixture of experts, log+min-max | 0.762 ± 0.061 | 0.762 ± 0.061 | +0.000 |
| CART, log+min-max | 0.826 ± 0.047 | 0.826 ± 0.047 | +0.000 |
| Random Forest, log+min-max | 0.909 ± 0.019 | 0.909 ± 0.019 | +0.000 |

## Reading it

**The eight control rows moving by exactly 0.000 is the check that makes the rest
meaningful.** Those arms never touch the output partition and their own preprocessing is
unchanged between the two variants, so a non-zero delta there would have meant the
variant was changing something it did not claim to. It is not.

**The three closed-form flat arms are within noise and point the wrong way for an
artifact.** +0.005, +0.005 and +0.018 against spreads of ±0.020–0.241, with removal of
the leak *raising* every one. A leak that inflated the headline would have to move them
down.

**The refined arms move more, and one of them matters for how Table 6.1 is read.** The
1st-order refined arm falls 0.866 → 0.834, a −0.032 move against spreads of ±0.029 and
±0.054. That is inside one standard deviation, so "within noise" is the correct reading —
but it is not zero, and it is the same order as several gaps Chapter 6 draws conclusions
from, so it belongs in the disclosure rather than in a footnote. The 2nd-order refined arm
moves −0.007, well inside its own spread.

**One incidental result worth keeping.** The 0th-order refined arm's spread collapses from
±0.210 to ±0.068 and its mean rises 0.062. Removing the leak makes that arm markedly more
*stable*, not less. The 0th-order model is the one the chapter already reports as worse
than predicting the mean, so nothing rests on it, but a preprocessing step that
destabilises the weakest arm is the opposite of what a flattering artifact looks like.

## What this settles and what it does not

**Settled.** The worry that Chapters 4 and 6's flat-model numbers are inflated by
test-fold leakage is not supported. No cell moves beyond its seed spread, and the
direction of the closed-form moves is against inflation.

**Not settled.** This measures Concrete at three output buckets and ten seeds. The
mechanism is `pd.qcut` on 1,030 rows with a 20% test fold; on a smaller dataset, more
buckets, or a heavier-tailed target the same defect could bite harder, because the whole
effect is how much a fold's own values shift a quantile edge. The generator now carries
both paths so that is one flag to check rather than an argument to have.

**Still owed.** `preprocess_for_others` fits the feature scaler on the full frame, which
this variant does not change. It is symmetric across arms so it cannot bias the
comparison, but it is transductive, and a deployment pipeline would not do it. Removing it
would move every row in the table rather than six of fourteen.

## Reproducing

```bash
# as shipped
uv run --project tribble-fis python reproduce/tables/table_concrete_reconciliation.py

# train-fold only, written somewhere of its own so the two can be diffed
REPRO_SPLIT_FIRST=1 \
REPRO_OUTPUT_DIR=reproduce/outputs/splitfirst-2026-08-03 \
  uv run --project tribble-fis python reproduce/tables/table_concrete_reconciliation.py
```
