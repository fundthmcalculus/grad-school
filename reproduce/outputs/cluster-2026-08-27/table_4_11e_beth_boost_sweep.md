**Table 4.11(e) — BETH: does each arm's knob beat a plain score threshold?**

| Knob | Setting | Realized test false-alarm | Detection (knob) | Detection (plain threshold, matched FA) | Δ detection | Youden's J |
|---|---|---|---|---|---|---|
| Tribble boost θ | 0 | 0.8366 ± 0.0702 | 0.997 ± 0.001 | 0.997 | -0.0001 ± 0.0004 | +0.160 |
| Tribble boost θ | 0.2 | 0.7466 ± 0.0542 | 0.996 ± 0.001 | 0.995 | +0.0005 ± 0.0007 | +0.249 |
| Tribble boost θ | 0.4 | 0.6797 ± 0.0432 | 0.995 ± 0.001 | 0.994 | +0.0006 ± 0.0006 | +0.315 |
| Tribble boost θ | 0.5 | 0.6496 ± 0.0418 | 0.995 ± 0.001 | 0.994 | +0.0007 ± 0.0007 | +0.345 |
| Tribble boost θ | 0.6 | 0.6226 ± 0.0456 | 0.995 ± 0.001 | 0.994 | +0.0007 ± 0.0006 | +0.372 |
| Tribble boost θ | 0.7 | 0.5811 ± 0.0370 | 0.995 ± 0.001 | 0.994 | +0.0007 ± 0.0007 | +0.413 |
| Tribble boost θ | 0.8 | 0.5361 ± 0.0433 | 0.995 ± 0.001 | 0.994 | +0.0010 ± 0.0006 | +0.458 |
| Tribble boost θ | 0.9 | 0.5148 ± 0.0484 | 0.995 ± 0.001 | 0.993 | +0.0011 ± 0.0005 | +0.480 |
| Tribble boost θ | 0.95 | 0.4359 ± 0.0533 | 0.995 ± 0.001 | 0.993 | +0.0012 ± 0.0006 | +0.559 |
| Tribble boost θ | 0.99 | 0.2818 ± 0.0516 | 0.994 ± 0.001 | 0.993 | +0.0003 ± 0.0006 | +0.712 |
| Tribble boost θ | 0.999 | 0.2242 ± 0.0560 | 0.993 ± 0.001 | 0.993 | +0.0001 ± 0.0007 | +0.769 |
| iForest contamination | 0.001 | 0.0135 ± 0.0113 | 0.464 ± 0.481 | 0.464 | -0.0000 ± 0.0000 | +0.451 |
| iForest contamination | 0.005 | 0.0846 ± 0.0257 | 0.949 ± 0.010 | 0.949 | -0.0003 ± 0.0004 | +0.864 |
| iForest contamination | 0.01 | 0.1528 ± 0.0143 | 0.983 ± 0.010 | 0.984 | -0.0003 ± 0.0006 | +0.831 |
| iForest contamination | 0.02 | 0.1741 ± 0.0034 | 0.993 ± 0.000 | 0.993 | -0.0000 ± 0.0000 | +0.819 |
| iForest contamination | 0.05 | 0.2383 ± 0.0066 | 0.993 | 0.993 | +0.0000 | +0.755 |
| iForest contamination | 0.1 | 0.3209 ± 0.0053 | 0.993 ± 0.000 | 0.993 | +0.0000 | +0.672 |
| iForest contamination | 0.2 | 0.5763 ± 0.0353 | 0.996 ± 0.001 | 0.996 | +0.0000 ± 0.0000 | +0.420 |
| iForest contamination | 0.3 | 0.7546 ± 0.0275 | 0.999 ± 0.001 | 0.999 | +0.0000 ± 0.0000 | +0.244 |
| OC-SVM ν | 0.001 | 0.1380 ± 0.0105 | 0.993 ± 0.000 | 0.993 | +0.0001 ± 0.0001 | +0.855 |
| OC-SVM ν | 0.005 | 0.1515 ± 0.0113 | 0.993 | 0.993 | +0.0000 ± 0.0001 | +0.842 |
| OC-SVM ν | 0.01 | 0.1515 ± 0.0086 | 0.993 | 0.993 | +0.0000 | +0.842 |
| OC-SVM ν | 0.02 | 0.2352 ± 0.0242 | 0.995 ± 0.001 | 0.995 | +0.0006 ± 0.0004 | +0.760 |
| OC-SVM ν | 0.05 | 0.3196 ± 0.0082 | 0.996 ± 0.000 | 0.996 | +0.0000 ± 0.0000 | +0.677 |
| OC-SVM ν | 0.1 | 0.3851 ± 0.0056 | 0.996 | 0.996 | -0.0000 ± 0.0000 | +0.611 |
| OC-SVM ν | 0.2 | 0.4594 ± 0.0065 | 0.996 | 0.996 | -0.0001 ± 0.0000 | +0.537 |
| OC-SVM ν | 0.3 | 0.4843 ± 0.0009 | 0.996 | 0.997 | -0.0001 ± 0.0001 | +0.512 |

> The question: at a **matched false-alarm rate**, does turning each arm's own knob detect more than simply moving the decision threshold on that same arm's continuous score? 'Δ detection' is knob minus plain threshold; **0.000 across a grid means the knob is a reparameterisation of the threshold**, which is a fine thing to be but not a claim to make about it. The control threshold is the (1 − realized FA) quantile of the arm's own benign *test* scores, so both numbers in a row sit at one operating point rather than on two curves. **Analytic prediction for θ, which this table checks rather than assumes:** `_anomaly_argmax` forms the anomaly column as `complement(conorm(clip(class_firing + θ, 0, 1)))`, and with a single known class there is exactly one class column — `t_conorm(x, None, …)` aggregates column-wise, so it is the identity. The anomaly label therefore wins exactly when `firing < (1 − θ)/2`, i.e. **in the one-class setting θ is a hard threshold on firing strength and the norm/conorm family is irrelevant because there is one column to aggregate.** Chapter 4 §4.3 argues a weaker version of this for the multi-class case (that θ=0.99 degenerates to a max-membership rejector); the one-class reduction makes it total at every θ. **Isolation Forest's `contamination` is the method's control**: it provably only sets `offset_` from a quantile of the training scores and never touches the trees, so its Δ must be 0.000 — that it comes back 0.000 is what licenses reading a non-zero Δ elsewhere as real, and one fit per seed is therefore correct rather than a shortcut. **`ν` is the one genuinely different knob**: it enters libsvm's QP objective, so every value is a different fitted model, and its control is one fixed (ν=0.01) model's score thresholded. Measured verdicts — **Tribble boost θ**: largest |Δ| 0.0012 against a detection seed-spread of 0.0006 — ABOVE the noise floor, so it does something a threshold cannot; **iForest contamination**: largest |Δ| 0.0003 against a detection seed-spread of 0.0007 — below the noise floor, so a threshold in disguise on this data; **OC-SVM ν**: largest |Δ| 0.0006 against a detection seed-spread of 0.0000 — ABOVE the noise floor, so it does something a threshold cannot. Every arm is fitted on the SAME 20,000-row benign subsample per seed, so the sample-count confound Table 4.11(d) corrects is not re-introduced; no wall-clock is reported because this table is about decision curves and timing belongs to (d). Scored on the full 188,967-row test split (158,432 anomalous); mean ± sample std across common.SEEDS ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).

> Generated by `reproduce/`; seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. `N/A` marks a cell whose method/dataset was unavailable.

> **Machine.** host: NEX-210200 · os: Windows-11 · cpu: Intel(R) Core(TM) i9-14900HX · cores: 32 physical, 32 logical · ram: 95.6 GiB · gpu: NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB · python: 3.13.7
>
> Wall-clock times are machine-dependent; ratios are not. Markdown tables report normalized ratios where a timing is involved, and the companion CSV carries the absolute seconds.
