#!/usr/bin/env python3
"""Check the upstream contracts the harness silently assumes, before it spends hours.

Every defect the 2026-08-22 pass found had the same shape: an upstream function
kept its name and changed its behaviour, or kept its behaviour and changed its
name, and the harness carried on. Nothing crashed. The generators emitted `N/A`
or a plausible wrong number, exited 0, and the orchestrator reported **ok**.

This is the guard for that *class*, not for any one instance. Each check states a
property the harness relies on and that a caller cannot see from a signature:

  W1-SCALE      `wasserstein_distance` is a distance in the data's units, so
                scaling the data scales it. The shipped implementation is
                scale-INVARIANT (checklist B14), which silently corrupts the
                feature-differentiation screen every gauss_math table runs on.
  VAT-MATRIXFREE `pvat.vat_prim_mst_seq` reproduces `compute_vat`'s ordering.
                It returned an ascending-index permutation for months (B6).
  SCALER-ALIAS  `UnitScalar is MinMaxScaler` and `StandardScalar is
                StandardScaler`. B13 checked this by hand once; if the aliases
                ever diverge, "log + min-max" silently becomes z-score.
  MODEL-NAMES   The estimator classes the generators construct still exist.
                Three files were left importing the pre-rename names, and two of
                them degraded to `N/A` rather than failing (B12(c)).
  DATASETS      Every in-repo dataset a generator names resolves to a file that
                exists. `glass.csv` moved into `data/` and three call sites kept
                looking at the repo root; two returned `None` into silence, so
                Table 4.8's Glass row and the whole of Table 4.9 went missing
                from every archive (B16(f)).
  PIN-MATCH     The revision actually imported is the revision `PROVENANCE.txt`
                records. `tribble-fis` resolves `tribble-clustering` and
                `optimizers` from its own lockfile, not from the sibling
                submodules, so the two drift apart in silence and the archive
                names code that did not run (B18).

**An unavailable import is reported SKIP, never PASS.** A check that passes
because it could not run is the bug this file exists to catch.

    uv run --project tribble-fis     python reproduce/preflight.py --suite fis
    uv run --project tribble-cluster python reproduce/preflight.py --suite cluster

Exit status is 0 when nothing FAILED (SKIPs are not failures), 1 otherwise.
`run_all_tables.sh` runs it before the numeric phase and, on failure, stamps the
archive NOT CITABLE rather than refusing to run -- the same treatment `--fast`
gets, and for the same reason: the risk is someone quoting the output later, not
someone producing it now.
"""

from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def _w1_scale():
    """W1 has the units of x, so scaling the data must scale the distance."""
    try:
        import numpy as np

        from tribblefis.stats_numba import wasserstein_distance
    except Exception as exc:  # noqa: BLE001
        return SKIP, f"tribblefis.stats_numba not importable ({type(exc).__name__})"

    rng = np.random.default_rng(0)
    u, v = rng.normal(0, 1, 400), rng.normal(1.5, 2, 400)
    base = float(wasserstein_distance(u, v))
    scaled = float(wasserstein_distance(u * 1000.0, v * 1000.0))
    if base == 0:
        return FAIL, "returned 0 on non-identical samples"
    ratio = scaled / base
    if abs(ratio - 1000.0) / 1000.0 < 0.01:
        return PASS, f"x1000 data -> x{ratio:.1f} distance"
    return FAIL, (
        f"x1000 data -> x{ratio:.3f} distance (expected x1000). "
        "Scale-invariant, so not W1: the dx weighting is missing. "
        "Corrupts the feature screen -- see checklist B14."
    )


def _vat_matrix_free():
    """The matrix-free reorder must reproduce the reference ordering."""
    try:
        import numpy as np

        from tribbleclustering import pvat
    except Exception as exc:  # noqa: BLE001
        return SKIP, f"tribbleclustering not importable ({type(exc).__name__})"
    if not hasattr(pvat, "vat_prim_mst_seq"):
        return SKIP, "vat_prim_mst_seq not exported at this pin"

    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.normal(size=(300, 4)))
    gram = x @ x.T
    sq = np.einsum("ij,ij->i", x, x)
    dm = sq[:, None] + sq[None, :] - 2.0 * gram
    np.maximum(dm, 0.0, out=dm)
    np.sqrt(dm, out=dm)

    ref = pvat.compute_vat(dm)
    ref = np.asarray(ref[1] if isinstance(ref, tuple) else ref).ravel()
    got = np.asarray(pvat.vat_prim_mst_seq(x)).ravel()
    if ref.shape != got.shape:
        return FAIL, f"shape mismatch {ref.shape} vs {got.shape}"
    agree = float((ref == got).mean())
    if agree == 1.0:
        return PASS, "ordering elementwise identical to compute_vat"
    ascending = bool(got.size > 2 and np.all(np.diff(got[1:]) == 1))
    hint = " (ascending-index permutation -- the B6 signature)" if ascending else ""
    return FAIL, f"agreement {agree:.4f}, chance ~{1/len(x):.4f}{hint}"


def _scaler_aliases():
    """`log + min-max` must not silently become z-score."""
    try:
        import tribblefis.scaling as s
    except Exception as exc:  # noqa: BLE001
        return SKIP, f"tribblefis.scaling not importable ({type(exc).__name__})"

    pairs = [("UnitScalar", "MinMaxScaler"), ("StandardScalar", "StandardScaler")]
    missing = [n for p in pairs for n in p if not hasattr(s, n)]
    if missing:
        return FAIL, f"missing: {', '.join(missing)}"
    bad = [f"{a} is not {b}" for a, b in pairs if getattr(s, a) is not getattr(s, b)]
    if bad:
        return FAIL, "; ".join(bad) + " -- every log+min-max number is in question"
    return PASS, "UnitScalar is MinMaxScaler; StandardScalar is StandardScaler"


def _model_names():
    """The estimator classes the generators construct must still exist."""
    wanted = {
        "tribblefis.gaussian_classifier": [
            "TribbleClassifier",
            "TribbleSequenceClassifier",
        ],
        "tribblefis.gaussian_regressor": ["TribbleRegressor"],
    }
    import importlib

    missing = []
    for mod_name, names in wanted.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            return SKIP, f"{mod_name} not importable ({type(exc).__name__})"
        missing += [f"{mod_name}.{n}" for n in names if not hasattr(mod, n)]
    if missing:
        return FAIL, "renamed or removed: " + ", ".join(missing)
    return PASS, "all three estimator classes present"


def _datasets():
    """Every in-repo dataset a generator names must resolve to a real file."""
    data = os.environ.get("GRAD_SCHOOL_DATA", os.path.join(ROOT, "data"))
    # Only the ones a generator loads from disk in this repository. Datasets that
    # ship with scikit-learn or download on demand are deliberately not listed.
    required = {
        "glass.csv": "Tables 4.6-4.9 (Glass)",
        # Concrete has a documented .xls fallback under AEEM6097/, so its absence
        # from data/ is not fatal; the loader says so and rebuilds the CSV.
        "Concrete_Data.csv": "Tables 4.1-4.3, 6.1 (Concrete)",
    }
    optional = {
        "PhiUSIIL_Phishing_URL_Dataset.csv": "Table 4.1 PhiUSIIL row",
        "RT_IOT2022.csv": "Tables 4.4/4.7b",
    }
    missing = [
        f"{n} ({why})"
        for n, why in required.items()
        if not os.path.exists(os.path.join(data, n))
    ]
    absent = [n for n in optional if not os.path.exists(os.path.join(data, n))]
    if missing:
        return FAIL, "not found in data/: " + "; ".join(missing)
    note = f"; optional absent: {', '.join(absent)}" if absent else ""
    return PASS, f"{len(required)} required dataset(s) present{note}"


def _install_fresh():
    """The imported package must match the submodule source it was built from.

    `UV_NO_EDITABLE=1` (which hostenv.sh sets, because the editable build path
    ignores DIST_EXTRA_CONFIG) installs the submodules as built wheels. uv can
    then serve a CACHED wheel after the submodule pin moves, so a pin bump
    silently does not take effect and the whole suite re-measures the old code
    under the new SHA -- which the archive's PROVENANCE would then record as the
    new one. Caught exactly that way on 2026-08-22: the tribble-fis pin moved to
    pick up the wasserstein fix and `W1-SCALE` still failed, because the venv was
    serving the pre-fix build.

    Remedy is `uv run --project <sub> --reinstall-package <name> ...` once.
    """
    import importlib
    import filecmp

    stale = []
    for pkg, sub in (
        ("tribblefis", "tribble-fis"),
        ("tribbleclustering", "tribble-cluster"),
    ):
        try:
            mod = importlib.import_module(pkg)
        except Exception:  # noqa: BLE001
            continue
        installed = os.path.dirname(getattr(mod, "__file__", "") or "")
        source = os.path.join(ROOT, sub, "src", pkg)
        if not installed or not os.path.isdir(source):
            continue
        if os.path.normcase(installed) == os.path.normcase(source):
            continue  # editable install: nothing can be stale
        for name in sorted(os.listdir(source)):
            if not name.endswith(".py"):
                continue
            a, b = os.path.join(source, name), os.path.join(installed, name)
            if os.path.exists(b) and not filecmp.cmp(a, b, shallow=False):
                stale.append(f"{pkg}/{name}")
                break
    if stale:
        return FAIL, (
            f"installed build is STALE against the submodule source ({', '.join(stale)}). "
            "Two different causes, and they need different fixes -- PIN-MATCH below "
            "tells you which: a cached wheel wants "
            "`uv run --project <submodule> --no-cache --reinstall-package <dist-name> ...` "
            "(the DIST name, hyphenated), while a lockfile pinning an older revision "
            "wants `uv lock --upgrade-package <dist-name>` in the project being run"
        )
    return PASS, "installed builds match the submodule sources"


def _pin_match():
    """The revision actually imported must be the revision the archive records.

    `run_all_tables.sh` stamps `git rev-parse HEAD` of each submodule into
    `PROVENANCE.txt`. That is only the truth for a package installed *from* the
    submodule directory. `tribble-fis/pyproject.toml` sources both
    `tribble-clustering` and `optimizers` from git URLs with no revision, so
    `uv run --project tribble-fis` resolves them from tribble-fis's own
    `uv.lock` and never looks at the sibling checkout at all.

    Nothing re-resolves those pins on their own and `uv lock --check` cannot see
    them drift (a revision-less git source is "in sync" with any locked commit),
    so they go stale quietly. Found 2026-08-27 with `tribble-clustering` five
    commits and `optimizers` eleven commits behind the submodules the archives
    were recording -- across seeding fixes, an FCM membership fix and a GA
    crossover fix.

    Caught one archive before it mattered: an audit of every existing
    `PROVENANCE.txt` shows none of them misnamed its code, and the next run
    would have been the first that did. The check exists because that margin was
    luck, not design -- nothing was watching this axis at all.

    Note the check is not "submodule SHA equals lock pin". Which record is
    correct depends on how the environment was built: a study run with
    `--with-editable tribble-opt` really does run the submodule checkout, and
    recording its SHA is right. Only the installed distribution knows, so that is
    what this reads.

    A local-directory install cannot drift, so it passes without comment.
    """
    import importlib.metadata as md
    import json
    import subprocess

    pairs = [
        ("tribble-clustering", "tribble-cluster"),
        ("optimizers", "tribble-opt"),
        ("tribble-fis", "tribble-fis"),
    ]
    bad, seen = [], []
    for dist_name, sub in pairs:
        try:
            direct = md.distribution(dist_name).read_text("direct_url.json")
        except Exception:  # noqa: BLE001 - not installed in this environment
            continue
        if not direct:
            continue
        try:
            info = json.loads(direct)
        except ValueError:
            continue
        commit = (info.get("vcs_info") or {}).get("commit_id")
        if not commit:
            continue  # installed from a directory: it is the checkout by definition
        sub_path = os.path.join(ROOT, sub)
        try:
            head = subprocess.run(
                ["git", "-C", sub_path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            continue
        seen.append(dist_name)
        if commit != head:
            bad.append(
                f"{dist_name} imports {commit[:7]}, submodule {sub} is {head[:7]}"
            )
    if bad:
        return FAIL, (
            "; ".join(bad)
            + " -- PROVENANCE records the submodule SHA, so the archive would name "
            "code that did not run. Fix with `uv lock --upgrade-package <dist-name>` "
            "in the project being run, then reinstall"
        )
    if not seen:
        return SKIP, "no git-sourced vendored dependency installed in this environment"
    return PASS, f"{', '.join(seen)} match their submodule HEADs"


CHECKS = [
    ("W1-SCALE", "fis", _w1_scale),
    ("SCALER-ALIAS", "fis", _scaler_aliases),
    ("MODEL-NAMES", "fis", _model_names),
    ("VAT-MATRIXFREE", "cluster", _vat_matrix_free),
    ("DATASETS", "any", _datasets),
    ("INSTALL-FRESH", "any", _install_fresh),
    ("PIN-MATCH", "any", _pin_match),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["all", "fis", "cluster"])
    args = ap.parse_args()

    print(f"preflight ({args.suite})")
    print("-" * 74)
    results = []
    for name, suite, fn in CHECKS:
        if args.suite != "all" and suite not in (args.suite, "any"):
            continue
        try:
            verdict, detail = fn()
        except Exception as exc:  # noqa: BLE001
            verdict, detail = FAIL, f"check itself raised {type(exc).__name__}: {exc}"
        results.append((name, verdict))
        mark = {PASS: "ok  ", FAIL: "FAIL", SKIP: "skip"}[verdict]
        print(f"  [{mark}] {name:<15} {detail}")

    failed = [n for n, v in results if v == FAIL]
    skipped = [n for n, v in results if v == SKIP]
    print("-" * 74)
    if failed:
        print(f"  {len(failed)} FAILED: {', '.join(failed)}")
        print("  Numbers produced under a failed invariant are NOT citable.")
    else:
        print(f"  {len(results) - len(skipped)} passed, {len(skipped)} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
