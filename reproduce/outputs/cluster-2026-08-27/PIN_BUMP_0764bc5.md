# Pin bump to tribble-fis `0764bc5` — what moved, and why

_Suite `bumped-0764bc5-2026-08-22`, 17/17 tables `ok`, 2026-08-22.
Compared against two archives: `full-2026-08-22` (the previous run, pinned
`141596e`) and `goal-8h-2026-08-11-fullsuite` (the archive the prose was written
from, pinned `80e98d7` / cluster `85b68a8`)._

## 1. Only one commit in this bump can change a number

`tribble-cluster` did not move: `635ed6e` in both this run and
`full-2026-08-22`. `tribble-fis` moved `141596e -> 0764bc5`, four commits:

| commit | PR | behaviour |
|---|---|---|
| `5253aa0` | #171 | `wasserstein_distance` weights CDF gaps by `dx` — **changes results** |
| `25da26a` | #172 | label -> category in the screen — perf only |
| `2109e9b` | #173 | scalar closeness in the dedup scan — perf only |
| `0764bc5` | #175 | label -> category in the membership dict — perf only |

**The perf claim was verified at table scale, not argued.** `tribble-fis` was
checked out at `5253aa0` (#171 alone) and `table_4_8_mf_dedup` re-run — the table
most sensitive to dedup, which is exactly what #173 rewrites:

```
table_4_8_mf_dedup        : IDENTICAL
table_4_8_mf_dedup_sweep  : IDENTICAL
table_4_9_correction_pass : IDENTICAL
```

So every number that moved in this run is attributable to **#171 alone**.

## 2. Against the prose's archive, Table 4.1 reproduces — except Bike Sharing

| row | `goal-8h` | now | |
|---|---|---|---|
| PhiUSIIL (classification) | — | — | reproduces |
| RT-IOT2022 (12-class) | acc=0.927 ± 0.002 | acc=0.923 ± 0.011 | within noise |
| Concrete (regression) | R2=0.795 ± 0.025 | R2=0.808 ± 0.030 | within noise |
| Concrete (full 2nd order) | R2=0.852 ± 0.030 | R2=0.867 ± 0.031 | within noise |
| **Bike Sharing (regression)** | **R2=0.939 ± 0.004** | **R2=0.960 ± 0.003** | **+0.021, changed** |

Bike Sharing was predicted before the run: upstream `69e0bab` (#102) changed
`pin_extremes`, which is §4.3.2/G5's own recommendation adopted upstream. The
predicted size was +0.025; the measured size is +0.021.

Training time is the other visible change, and it is the perf work landing:
RT-IOT2022 **37.42 s -> 2.88 s**, Bike Sharing 0.82 -> 0.17, Concrete 0.41 ->
0.06. `table_4_4_openset` fell **3h38m -> 1h17m** end to end.

## 3. Against `full-2026-08-22`, the classification screen is repaired

That run was pinned *before* #171, so it carries the broken screen:

| row | pre-#171 | now | Δ |
|---|---|---|---:|
| PhiUSIIL (classification) | acc=0.729 ± 0.023 | acc=0.997 ± 0.001 | **+0.268** |
| RT-IOT2022 (12-class) | acc=0.500 ± 0.244 | acc=0.923 ± 0.011 | **+0.423** |

## 4. Cluster side: nothing moved but clocks

Every `changed` flag on a `tribble-cluster` table is a wall clock, consistent
with the identical pin:

- `table_3_7_g2_dtw_nonmetric` — the cell is the string `600s matrix + 0.2s
  reorder` -> `593s matrix + 0.2s reorder`;
- `table_3_1_complexity_fit` — the three "fitted exponent" cells are *fits to a
  timing curve*, so they inherit timing noise (3.11 -> 3.05, 1.82 -> 1.77,
  1.77 -> 1.73);
- `table_3_4_gpu_speedups` — raw `CPU (s)` / `GPU (s)` cells.

**`compare_runs.py` does not recognise any of these as timing**, because its
`_TIME_WORDS` check reads the column name only. Timing embedded in a prose
string, and quantities derived from timings, are both missed. Filed below.

## 5. Two findings that need the document's attention

### 5a. `table_4_8`: the dedup reduction story weakens

| dataset | reduction @ 1x | @ max-lossless |
|---|---|---|
| BreastCancer | 23.1% -> **0.0%** | 47.2% -> **0.0%** |
| Glass | 37.2% -> 12.1% | 22.2% -> 24.0% |
| Digits | 18.9% -> 17.4% | 22.8% -> 14.3% |

On the corrected screen, BreastCancer dedups **nothing**. §4.3.1's reduction
claims were measured on the broken ranking and need re-reading.

Concrete's *raw* MF count doubled (33.9 -> 67.1) even though the regressor runs
`top_n=-1`. That is not order-dependence: `take_top_features` falls through to
`top_p=0.95` when `top_n <= 0`, keeping every feature scoring >= 0.05, so #171's
changed scores changed *how many* features clear the bar.

### 5b. `table_4_4b_theta_sweep`: all 18 cells moved

At the shipped operating point θ=0.99, detection − false alarm goes
**+0.472 -> +0.346** relative to `full-2026-08-22`. Against `goal-8h` the
open-set table moves far less (complement rule +0.394 -> +0.367), so the
document's own numbers are close to intact; it is the intermediate run that was
anomalous.

## 6. Owed

- **B17 (new).** `compare_runs.py` classifies timing by column name only. Add
  timing-in-prose and derived-from-timing, or these three cluster tables will
  report spurious "changed" on every run.
- **B18 (new).** `table_3_2_memory_precision` moved against `goal-8h`:
  `ordering vs float64 (N=2,000)` went `0.001 ± 0.001` -> `0.999 ± 0.002`
  (float32) and `1.000 (exact)` (float64). That is the cluster pin
  `85b68a8 -> 635ed6e`, not this bump, and it looks like a repair — but the
  prose was written against the 0.001 figures and has not been re-read.
- C16 (open) — the complement rule's order dependence, mechanism still unconfirmed.
- C17 (open) — hoist the θ-independent work out of the sweep.
