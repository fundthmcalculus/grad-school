"""Hyperparameter sweep for the FIS time-step operator.

Scores one row per (dataset, config), three metric families per row (see
fis_timestep for what pooled / trained / holdout mean). Two grids share this
module because they share every line of scoring machinery and differ only in
which configs get tried:

  * the **main** grid (`configs()`) is not a hyperparameter search grid: it is
    the set the exploratory probe said was worth spending time on, plus two
    configurations kept specifically to document negative results.
      - ``harmonic`` encodings are in the list because they were the obvious
        first idea and they fail: a TSK consequent is affine in its inputs, so
        a rule cannot bend through a full oscillation, and adding
        sin/cos of k*sqrt(g/l)*t looked like the fix. It is not. At 8
        harmonics (17 inputs) the score drops; at 24 (49 inputs) R^2 goes to
        -2.8. The product t-norm over 49 Gaussian memberships drives every
        firing strength toward zero, so the normalised firing weights become
        numerical noise and the ridge solve fits nothing. The lesson is that
        antecedent dimensionality, not consequent expressiveness, is the
        binding constraint in this library.
      - ``n_gaussians`` > 0 (fixed, coarse memberships) is kept because it is
        the only knob that improved held-out-initial-condition score on the
        *frictionless* problems while making every other number worse.
  * the **lowcap** grid (`configs(lowcap=True)`) is a follow-up: the main grid
    found held-out-IC score climbs with rule count on the friction problems
    but saturates and drifts *down* on the frictionless ones (see
    bracket_diagnostic.py for why), so this scans below the main grid's floor.
    Nothing below 40 rules per output beat the main grid's frictionless
    optimum; recorded here so the bound is documented rather than assumed.

Run:
    python sweep.py                             # main grid, every chain length
    python sweep.py --quick                     # 4-config smoke subset
    python sweep.py --lowcap                    # lowcap grid instead
    python sweep.py --n 5 --out sweep_n5.csv    # just n=5, into its own file

`--n` and `--out` exist so a new chain length can be added without re-running
what is already scored; run_all.py's pipeline instead always requests the full
fixed grid via `run_sweep()` directly (see stage_sweep / stage_lowcap there).
"""

from __future__ import annotations

import argparse
import csv
import time

import pendulum_data as pdata
from fis_timestep import FisConfig, RESULT_DIR, load, run

#: Chain lengths swept by default; n=2 and n=3 are the paper's, n=5 extends it.
N_LINKS = (2, 3, 5)


def configs(quick=False, lowcap=False):
    if lowcap:
        out = []
        for nb in (2, 3, 5, 8, 12, 20):
            out.append(FisConfig(n_output_buckets=nb))
            out.append(FisConfig(n_output_buckets=nb, tsk_order="full-2nd"))
        for nb in (5, 12, 20, 40):
            for ng in (2, 3, 5):
                out.append(FisConfig(n_output_buckets=nb, n_gaussians=ng))
        return out

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


def run_sweep(cfgs, datasets, log=print):
    """Score every (dataset, config) pair. Pure: no file I/O.

    Returns a list of `Result.flat()` rows; a config that raises is recorded as
    an error row (`{"dataset", "config", "error"}`) rather than aborting the
    sweep, since a failed config is itself a result worth keeping.
    """
    rows = []
    total = len(datasets) * len(cfgs)
    done = 0
    for n_links, friction in datasets:
        split = load(n_links, friction)
        for cfg in cfgs:
            done += 1
            try:
                res, _ = run(split, cfg)
                rows.append(res.flat())
                log(
                    f"[{done:3d}/{total}] {split.label:20s} {cfg.key():48s} "
                    f"pooledR2={res.pooled['r2']:+.4f} "
                    f"trainR2={res.trained_ic['r2']:+.4f} trainRMSE={res.trained_ic['rmse']:.4f} "
                    f"holdR2={res.holdout_ic['r2']:+.4f} holdRMSE={res.holdout_ic['rmse']:.4f} "
                    f"({res.fit_seconds:.0f}s)"
                )
            except Exception as exc:  # a failed config is a result, not a crash
                log(
                    f"[{done:3d}/{total}] {split.label:20s} {cfg.key():48s} "
                    f"FAILED {type(exc).__name__}: {exc}"
                )
                rows.append(
                    {
                        "dataset": split.label,
                        "config": cfg.key(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="4-config smoke subset")
    ap.add_argument("--lowcap", action="store_true", help="low-capacity grid instead")
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
        help="output CSV name under results/ (default sweep.csv / sweep_lowcap.csv)",
    )
    args = ap.parse_args()

    datasets = [(n, f) for n in args.n for f in (False, True)]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.lowcap:
        default = "sweep_lowcap.csv"
    else:
        default = "sweep_quick.csv" if args.quick else "sweep.csv"
    path = RESULT_DIR / (args.out or default)

    cfgs = configs(quick=args.quick, lowcap=args.lowcap)
    kind = "low-capacity " if args.lowcap else ""
    print(
        f"{kind}sweep: {[pdata.system_name(n) for n in args.n]} x "
        f"{{frictionless, friction}} x {len(cfgs)} configs -> {path.name}",
        flush=True,
    )
    t_start = time.perf_counter()
    rows = run_sweep(cfgs, datasets, log=lambda m: print(m, flush=True))

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
