# Screen screen: final val BPC and early-curve speed

Base `A-base`: 2 seeds, mean 2.0776, std 0.0353. Δ is the difference of means against the base; `real` means |Δ| > 2 standard errors (se = base std · √(1/n_variant + 1/n_base)). Nothing here is citable.

| variant | params | seeds | start BPC | to 3.0 | to 2.5 | to 2.2 | final BPC (mean ± std) | Δ vs base | verdict |
|---|---|---|---|---|---|---|---|---|---|
| A-base | 8,568 | 2 | 6.620 | 200 | 300 | 800 | 2.0776 ± 0.035 |  |  |
| B-embonly | 8,568 | 1 | 6.890 | 200 | 300 | 800 | 2.0816 ± n/a | +0.0040 | within noise (2se = 0.087) |
| C-biplace | 8,568 | 1 | 6.911 | 200 | 300 | 700 | 2.0669 ± n/a | -0.0106 | within noise (2se = 0.087) |
| D-bifull | 8,568 | 1 | 3.702 | 200 | 400 | 700 | 2.0523 ± n/a | -0.0253 | within noise (2se = 0.087) |
| E-taps2 | 8,824 | 1 | 6.792 | 200 | 300 | 800 | 2.0600 ± n/a | -0.0175 | within noise (2se = 0.087) |
| F-trifull | 8,824 | 1 | 4.171 | 100 | 400 | 800 | 2.0960 ± n/a | +0.0184 | within noise (2se = 0.087) |
| G-triplace | 8,824 | 1 | 7.028 | 200 | 300 | 1000 | 2.1366 ± n/a | +0.0590 | within noise (2se = 0.087) |
| H-ffnonly | 8,568 | 1 | 6.651 | 200 | 300 | 700 | 2.0361 ± n/a | -0.0415 | within noise (2se = 0.087) |
| I-share3 | 5,076 | 1 | 6.601 | 200 | 400 | never | 2.2050 ± n/a | +0.1274 | real, worse |
| J-wdim | 7,096 | 1 | 6.620 | 200 | 300 | 1000 | 2.1641 ± n/a | +0.0865 | real, worse |

## Init diagnostics (held-out calibration windows, lowest seed)

* **B-embonly**: k=1, mode=emb, init_bpc=6.8947
* **C-biplace**: k=1, kappa=0.2, alpha=4.0, gain=16.0, init_bpc=6.9146, calibrated_init_bpc=3.6942, coverage=0.9797, rule_acc=0.7653, rules_winning=32, mode=placement
* **D-bifull**: k=1, kappa=0.2, alpha=4.0, gain=16.0, init_bpc=3.6942, calibrated_init_bpc=3.6942, coverage=0.9797, rule_acc=0.7653, rules_winning=32, mode=full
* **F-trifull**: k=2, kappa=0.2, alpha=64.0, gain=16.0, init_bpc=4.1537, calibrated_init_bpc=4.1537, coverage=0.4212, rule_acc=0.8713, rules_winning=32, mode=full
* **G-triplace**: k=2, kappa=0.2, alpha=64.0, gain=16.0, init_bpc=7.0214, calibrated_init_bpc=4.1537, coverage=0.4212, rule_acc=0.8713, rules_winning=32, mode=placement
