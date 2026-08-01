"""Run a tribble-cluster experiment with its output landing in grad-school.

The Chapter 3 experiments live in the `tribble-cluster` submodule and write
their figures next to their own source, into
`tribble-cluster/experiments/figures/`. That means reproducing a proposal figure
dirties a pinned submodule, and the evidence for a grad-school table ends up
stored in a library that knows nothing about the proposal. This runner inverts
that: the experiment code stays where it is, and only the *destination* moves
up into this repository.

It also fixes the invocation. Every one of these scripts does
`from experiments.blockwise_vat import ...`, which needs the submodule ROOT on
`sys.path`. Running them by path (`python experiments/adversarial_eval.py`)
puts `experiments/` there instead and they die with `ModuleNotFoundError: No
module named 'experiments'` before doing any work -- so this runner puts the
right directory on the path rather than leaving it to the caller.

Usage, from the repo root:

    uv run --project tribble-cluster --with scipy \\
        python reproduce/experiments/run_cluster_experiment.py adversarial_eval

    # everything at once
    uv run --project tribble-cluster --with scipy \\
        python reproduce/experiments/run_cluster_experiment.py --all

Override the destination with `REPRO_FIG_DIR`. Nothing is written inside the
submodule; check with `git -C tribble-cluster status` afterwards.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLUSTER = ROOT / "tribble-cluster"
DEFAULT_OUT = ROOT / "reproduce" / "outputs" / "figures" / "cluster"

# module name -> the entry points to call, in order. These scripts do their work
# under `if __name__ == "__main__":` rather than exposing a single main(), so the
# call sequence has to be named here; importing alone runs nothing.
EXPERIMENTS = {
    "adversarial_eval": ("run",),        # Table 3.4
    "principled_stitch": ("run",),       # Table 3.5
    "hardening_eval": ("part_a", "part_b"),  # Table 3.6
}


def run_one(name: str, out_dir: Path) -> int:
    """Import one experiment with FIG_DIR redirected, then call its entry points."""
    if name not in EXPERIMENTS:
        print(f"  [error] unknown experiment {name!r}; "
              f"known: {', '.join(sorted(EXPERIMENTS))}")
        return 1

    # The submodule root, not experiments/, is what the absolute imports need.
    if str(CLUSTER) not in sys.path:
        sys.path.insert(0, str(CLUSTER))

    mod = importlib.import_module(f"experiments.{name}")

    # The scripts call FIG_DIR.mkdir(exist_ok=True), which is NOT recursive, so
    # the parents have to exist before the rebind or they raise FileNotFoundError.
    out_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(mod, "FIG_DIR"):
        print(f"  [warn] {name} has no FIG_DIR; it may write inside the submodule")
    mod.FIG_DIR = out_dir

    print(f"=== {name} -> {out_dir} ===")
    for entry in EXPERIMENTS[name]:
        fn = getattr(mod, entry, None)
        if fn is None:
            print(f"  [error] {name}.{entry}() not found -- the API has drifted; "
                  f"update EXPERIMENTS in this file")
            return 1
        fn()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", nargs="?", help="experiment module name")
    ap.add_argument("--all", action="store_true", help="run every registered experiment")
    args = ap.parse_args()

    if not args.all and not args.experiment:
        ap.error("give an experiment name or --all")

    out_dir = Path(os.environ.get("REPRO_FIG_DIR", DEFAULT_OUT))
    names = sorted(EXPERIMENTS) if args.all else [args.experiment]

    failed = [n for n in names if run_one(n, out_dir) != 0]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(names)} experiment(s) finished. Figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
