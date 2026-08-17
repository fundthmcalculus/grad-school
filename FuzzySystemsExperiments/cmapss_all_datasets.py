"""The DS02 best-case pipeline, run across every N-CMAPSS dataset.

`cmapss_ds02_rul.py` is the pipeline, explained; this is that exact pipeline
applied unchanged to each of the ten N-CMAPSS files in turn, to see how one
fixed recipe -- tuned only on DS02 -- generalizes across the other operating
regimes and fault modes. Each dataset is trained and scored on its own
published train/test split; nothing is re-tuned per dataset.

For each file it reports the per-sample RMSE (comparable to the published
DS02 baselines) and the recommended monotone RMSE, and writes a markdown table
to `cmapss_all_datasets_report.md`. Files are processed one at a time and freed
between, so peak memory stays near a single dataset's footprint.

Needs: h5py, numpy, pandas, scikit-learn, tribble-fis.  Run:

    python cmapss_all_datasets.py --h5-dir NASA-CMAPSS
"""

import argparse
import gc
import glob
import os
import time

from cmapss_ds02_rul import fit_and_score, load_dataframe

REPORT = "FuzzySystemsExperiments/cmapss_all_datasets_report.md"


def run_one(h5_path):
    """The full pipeline on one file; returns a result row (or a skip reason)."""
    name = os.path.basename(h5_path).replace("N-CMAPSS_", "").replace(".h5", "")
    t0 = time.perf_counter()
    try:
        dev, cond_cols, sensor_cols = load_dataframe(h5_path, "dev")
        test, _, _ = load_dataframe(h5_path, "test")
        r = fit_and_score(dev, test, cond_cols, sensor_cols, verbose=False)
        n_engines = r["per_cycle_monotone"]["unit"].nunique()
        row = dict(
            dataset=name,
            status="ok",
            engines=int(n_engines),
            per_sample_rmse=r["per_sample_rmse"],
            monotone_rmse=r["monotone_rmse"],
            seconds=time.perf_counter() - t0,
        )
        del dev, test, r
        gc.collect()
        return row
    except Exception as exc:  # a truncated/corrupt file shouldn't stop the run
        gc.collect()
        return dict(
            dataset=name,
            status=f"skipped: {type(exc).__name__}",
            engines=0,
            per_sample_rmse=float("nan"),
            monotone_rmse=float("nan"),
            seconds=time.perf_counter() - t0,
        )


def write_report(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [
        "# N-CMAPSS RUL across all datasets",
        "",
        "The DS02 best-case pipeline (`cmapss_ds02_rul.py`), applied unchanged to "
        "every N-CMAPSS file on its own train/test split -- a zero-shot "
        "generalization check, not a per-dataset best case. `per-sample RMSE` is "
        "the figure comparable to published baselines; `monotone RMSE` is the "
        "recommended output after the running-minimum clamp.",
        "",
        "| dataset | test engines | per-sample RMSE | monotone RMSE | seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in ok:
        lines.append(
            f"| {r['dataset']} | {r['engines']} | {r['per_sample_rmse']:.2f} | "
            f"{r['monotone_rmse']:.2f} | {r['seconds']:.1f} |"
        )
    if ok:
        mean_ps = sum(r["per_sample_rmse"] for r in ok) / len(ok)
        mean_mo = sum(r["monotone_rmse"] for r in ok) / len(ok)
        lines.append(
            f"| **mean of {len(ok)}** | | **{mean_ps:.2f}** | **{mean_mo:.2f}** | |"
        )
    skipped = [r for r in rows if r["status"] != "ok"]
    if skipped:
        lines += ["", "Skipped:"] + [
            f"- {r['dataset']}: {r['status']}" for r in skipped
        ]
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(h5_dir):
    files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    print(f"Found {len(files)} N-CMAPSS files in {h5_dir}\n")
    print(
        f"  {'dataset':10s} {'engines':>7s} {'per-sample':>10s} {'monotone':>9s} {'sec':>6s}"
    )
    rows = []
    for path in files:
        r = run_one(path)
        rows.append(r)
        if r["status"] == "ok":
            print(
                f"  {r['dataset']:10s} {r['engines']:7d} {r['per_sample_rmse']:10.2f} "
                f"{r['monotone_rmse']:9.2f} {r['seconds']:6.1f}"
            )
        else:
            print(f"  {r['dataset']:10s}  {r['status']}")
    write_report(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        print(
            f"\nmean per-sample RMSE {sum(r['per_sample_rmse'] for r in ok) / len(ok):.2f}"
            f" over {len(ok)} datasets"
        )
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", default="NASA-CMAPSS")
    main(parser.parse_args().h5_dir)
