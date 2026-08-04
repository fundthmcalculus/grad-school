"""Follow-up sweep: the low-capacity region, for the frictionless holdout.

The main sweep found the two regimes behave differently in capacity. On the
friction problems, held-out-IC score climbs with rule count all the way to the
grid's ceiling (double_friction R^2 0.567 -> 0.994 from 2 to 300 rules per output).
On the frictionless problems it saturates almost immediately and then drifts
*down*: double_frictionless peaks near 5-20 rules at R^2 ~0.55 and falls to 0.43
by 300.
Once the bracketing training trajectories have diverged (see
bracket_diagnostic.py), fitting the training initial conditions more exactly moves
the prediction away from the conditional mean, which is the best available answer.

So the main sweep's grid was plausibly bounded on the wrong side for the two
frictionless datasets. This script scans below it, and also scans fixed coarse
membership counts, since `n_gaussians > 0` was the one knob in the main sweep that
improved frictionless held-out score.

Result: it did not help. Nothing below 40 rules per output beat the main sweep's
frictionless optimum, and the best held-out configuration overall remains
`nb40` with 8 fixed Gaussians per feature. Recorded here so the bound is
documented rather than assumed.

Run: python sweep_lowcap.py
"""

from __future__ import annotations

import csv
import time

from fis_timestep import FisConfig, RESULT_DIR, load, run

DATASETS = [(2, False), (3, False), (2, True), (3, True)]


def configs():
    out = []
    for nb in (2, 3, 5, 8, 12, 20):
        out.append(FisConfig(n_output_buckets=nb))
        out.append(FisConfig(n_output_buckets=nb, tsk_order="full-2nd"))
    for nb in (5, 12, 20, 40):
        for ng in (2, 3, 5):
            out.append(FisConfig(n_output_buckets=nb, n_gaussians=ng))
    return out


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    cfgs = configs()
    total = len(DATASETS) * len(cfgs)
    done = 0
    t0 = time.perf_counter()
    for n_links, friction in DATASETS:
        split = load(n_links, friction)
        for cfg in cfgs:
            done += 1
            try:
                res, _ = run(split, cfg)
                rows.append(res.flat())
                print(
                    f"[{done:3d}/{total}] {split.label:20s} {cfg.key():44s} "
                    f"holdR2={res.holdout_ic['r2']:+.4f} holdRMSE={res.holdout_ic['rmse']:.4f} "
                    f"trainR2={res.trained_ic['r2']:+.4f} ({res.fit_seconds:.0f}s)",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{done:3d}/{total}] {split.label:20s} {cfg.key():44s} "
                      f"FAILED {type(exc).__name__}: {exc}", flush=True)
                rows.append({"dataset": split.label, "config": cfg.key(),
                             "error": f"{type(exc).__name__}: {exc}"})

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path = RESULT_DIR / "sweep_lowcap.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path} ({len(rows)} rows) in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
