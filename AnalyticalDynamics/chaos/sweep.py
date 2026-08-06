"""Hyperparameter sweep for the FIS time-step operator.

Writes results/sweep.csv with one row per (dataset, config) and three metric
families per row (see fis_timestep for what pooled / trained / holdout mean).

The config list is not a grid: it is the grid the exploratory probe said was
worth spending time on, plus two configurations kept specifically to document
negative results.

  * ``harmonic`` encodings are in the list because they were the obvious first
    idea and they fail: a TSK consequent is affine in its inputs, so a rule
    cannot bend through a full oscillation, and adding sin/cos of k*sqrt(g/l)*t
    looked like the fix. It is not. At 8 harmonics (17 inputs) the score drops;
    at 24 (49 inputs) R^2 goes to -2.8. The product t-norm over 49 Gaussian
    memberships drives every firing strength toward zero, so the normalised
    firing weights become numerical noise and the ridge solve fits nothing. The
    lesson is that antecedent dimensionality, not consequent expressiveness, is
    the binding constraint in this library.
  * ``n_gaussians`` > 0 (fixed, coarse memberships) is kept because it is the
    only knob that improved held-out-initial-condition score on the
    *frictionless* problems while making every other number worse.

Run:
    python sweep.py                             # every chain length -> sweep.csv
    python sweep.py --quick                     # 4-config smoke subset
    python sweep.py --n 5 --out sweep_n5.csv    # just n=5, into its own file

`--n` and `--out` exist so a new chain length can be added without re-running the
~67 minutes of n=2 and n=3 that are already scored. run_all.py merges every
sweep CSV it finds, so a separate file is equivalent to appending to sweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import time

import pendulum_data as pdata
from fis_timestep import FisConfig, RESULT_DIR, load, run

#: Chain lengths swept by default; n=2 and n=3 are the paper's, n=5 extends it.
N_LINKS = (2, 3, 5)


def configs(quick=False):
    out = []
    buckets = (40, 120) if quick else (40, 80, 120, 200, 300)
    for nb in buckets:
        for order in ("1st", "full-2nd"):
            out.append(FisConfig(n_output_buckets=nb, tsk_order=order))
    if quick:
        return out

    # Capacity is the dominant knob; these probe everything else around it.
    out.append(FisConfig(n_output_buckets=300, tsk_order="2nd"))
    out.append(FisConfig(n_output_buckets=300, tsk_order="full-2nd", l2_reg=1e-4))
    out.append(FisConfig(n_output_buckets=300, tsk_order="full-2nd", l2_reg=1e-9))
    out.append(
        FisConfig(
            n_output_buckets=300, tsk_order="full-2nd", output_partition="quantile"
        )
    )
    # Low-capacity / coarse-membership region: the only place held-out-IC score
    # on the frictionless problems ever improved.
    for ng in (4, 8):
        out.append(FisConfig(n_output_buckets=40, n_gaussians=ng))
        out.append(
            FisConfig(n_output_buckets=120, n_gaussians=ng, tsk_order="full-2nd")
        )
    # Retained to document the failure, not because it is expected to win.
    out.append(FisConfig(n_output_buckets=120, encoding="harmonic", n_harmonics=8))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="4-config smoke subset")
    ap.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=list(N_LINKS),
        metavar="N",
        help=f"chain lengths to sweep (default {list(N_LINKS)})",
    )
    ap.add_argument(
        "--out",
        metavar="NAME",
        help="output CSV name under results/ (default sweep.csv)",
    )
    args = ap.parse_args()

    quick = args.quick
    datasets = [(n, f) for n in args.n for f in (False, True)]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    default = "sweep_quick.csv" if quick else "sweep.csv"
    path = RESULT_DIR / (args.out or default)

    rows = []
    cfgs = configs(quick)
    total = len(datasets) * len(cfgs)
    done = 0
    t_start = time.perf_counter()
    print(
        f"sweeping {[pdata.system_name(n) for n in args.n]} x "
        f"{{frictionless, friction}} x {len(cfgs)} configs -> {path.name}",
        flush=True,
    )

    for n_links, friction in datasets:
        split = load(n_links, friction)
        for cfg in cfgs:
            done += 1
            try:
                res, _ = run(split, cfg)
                rows.append(res.flat())
                print(
                    f"[{done:3d}/{total}] {split.label:20s} {cfg.key():48s} "
                    f"pooledR2={res.pooled['r2']:+.4f} "
                    f"trainR2={res.trained_ic['r2']:+.4f} trainRMSE={res.trained_ic['rmse']:.4f} "
                    f"holdR2={res.holdout_ic['r2']:+.4f} holdRMSE={res.holdout_ic['rmse']:.4f} "
                    f"({res.fit_seconds:.0f}s)",
                    flush=True,
                )
            except Exception as exc:  # a failed config is a result, not a crash
                print(
                    f"[{done:3d}/{total}] {split.label:20s} {cfg.key():48s} "
                    f"FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )
                rows.append(
                    {
                        "dataset": split.label,
                        "config": cfg.key(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {path} ({len(rows)} rows) in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
