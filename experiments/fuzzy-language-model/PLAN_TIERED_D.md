# v33-tiered-d: spend the tiered embedding's saved parameters on a wider d_model (registered before running)

Arms (30M chars, seeds 0-4, same recipe as v30-tiered; `flm/run_tiered_d_sweep.py`): d56 head 448 tail 16 ffn 1.5 (96,608 params); d56 head 384 tail 16 ffn 1.5 (94,048). Reference: t768e16m225 (d48, 98,280) 1.0965; baseline 1.1095.
Predictions: neither arm beats t768e16m225 by more than 0.005 (60%): fewer full-width head tokens costs more than d 48->56 buys. One of them beats it by >= 0.005 on 4/5 seeds (25%). Both worse than baseline (15%).

## Results (30M, seeds 0-4; paired vs v22-tribble baseline 1.1095)
| arm | params | BPC | paired | seeds better |
|---|---|---|---|---|
| d56 head 448 ffn 1.5 | 96,608 | 1.1021 | -0.0074 | 4/5 |
| d56 head 384 ffn 1.5 | 94,048 | 1.1107 | +0.0012 | 2/5 |
| (ref) d48 head 768 ffn 2.25 | 98,280 | 1.0965 | -0.0130 | 4/5 |
Prediction (neither beats t768e16m225 by >= 0.005): **right**; both are worse than it (+0.0056, +0.0142). Spending saved parameters on width loses to keeping head tokens full-width and spending them on FFN. The wider-d line is closed; t768e16m225 remains the best sub-100K recipe (0.981 at 300M).
