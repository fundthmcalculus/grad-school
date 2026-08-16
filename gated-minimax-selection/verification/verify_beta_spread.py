"""
Verify NERFCM beta-spread activation on real non-Euclidean data.

This script loads real datasets, computes dissimilarity matrices, and runs
NERFCM to confirm that beta (the beta-spread parameter) activates when needed
for non-Euclidean data, not just staying 0 as on clean synthetic sets.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import project modules
# This file now lives at gated-minimax-selection/verification/, so the package
# it imports (ivat_mf, nerfcm, battery...) is one level up and the repo root is
# two. Resolved from __file__ rather than a developer's home directory.
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))


def _dataset(name, required=True):
    """Locate a dataset CSV, or fail with a message that says what to do.

    Checked in order: $GRAD_SCHOOL_DATA, the repo root, this script's directory.
    These files are gitignored, so a fresh clone genuinely will not have them --
    an explicit error naming the search path beats a stack trace from read_csv.
    """
    import os

    here = Path(__file__).resolve().parent
    roots = []
    if os.environ.get("GRAD_SCHOOL_DATA"):
        roots.append(Path(os.environ["GRAD_SCHOOL_DATA"]))
    # Datasets moved to <repo>/data/ when the experiments came out of the
    # submodules. Repo root and this directory stay as fallbacks so an older
    # working copy still resolves; nothing reaches outside the repo.
    repo_root = here.parent.parent
    roots += [repo_root / "data", repo_root, here.parent, here]
    for root in roots:
        candidate = root / name
        if candidate.exists():
            return str(candidate)
    if not required:
        return None
    raise FileNotFoundError(
        f"{name} not found. Looked in: {', '.join(str(r) for r in roots)}. "
        f"It is gitignored, so a fresh clone will not have it; place it at the "
        f"repo data/ directory, or set GRAD_SCHOOL_DATA to hold it."
    )


import ivat_mf as im
from nerfcm import nerfcm


def load_iris():
    """Load Iris: the local CSV if present, else UCI id 53 (same data)."""
    path = _dataset("IRIS.csv", required=False)
    if path:
        df = pd.read_csv(path)
        X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
        species_map = {s: i for i, s in enumerate(df["species"].unique())}
        y = np.array([species_map[s] for s in df["species"]])
    else:
        from ucimlrepo import fetch_ucirepo

        iris = fetch_ucirepo(id=53)
        X = iris.data.features.values
        y = iris.data.targets.values.ravel()
    return X, y, "Iris"


def load_glass():
    """Load Glass: the local CSV if present, else UCI id 42 (same data)."""
    path = _dataset("glass.csv", required=False)
    if path:
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        from ucimlrepo import fetch_ucirepo

        glass = fetch_ucirepo(id=42)
        X = glass.data.features.values
        y = glass.data.targets.values.ravel()
    return X, y, "Glass"


def load_heart():
    """Load a heart dataset, first 100 rows.

    CAUTION: the two sources here are NOT the same dataset, so a run that falls
    back is not comparable to one that does not -- the returned name says which
    was used, and that name should be carried into any reported result.

      local  heart_2020_cleaned.csv -- BRFSS 2020, ~320k rows, binary target
      UCI 45 Cleveland heart disease -- 303 rows, multiclass `num` target

    Local is preferred so existing numbers stay reproducible; the fetch is a
    fallback for a machine that has no data directory.
    """
    path = _dataset("heart_2020_cleaned.csv", required=False)
    if path:
        df = pd.read_csv(path).iloc[:100]
        X = pd.get_dummies(df.iloc[:, :-1], drop_first=True).values
        y = df.iloc[:, -1].values
        return X, y, "Heart BRFSS-2020 (n=100)"
    from ucimlrepo import fetch_ucirepo

    heart = fetch_ucirepo(id=45)
    X = heart.data.features.values[:100]
    y = heart.data.targets.values.ravel()[:100]
    return X, y, "Heart Cleveland/UCI-45 (n=100)"


def standardize(X):
    """Standardize features to zero mean, unit variance."""
    X = X.astype(float)
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)


def test_nerfcm_beta_spread(X, y, name, c, n_seeds=5):
    """
    Run NERFCM on dataset and track beta activation.

    Returns:
        dict with keys: name, n_samples, n_features, true_c, beta_values,
                       beta_mean, beta_max, any_activated (beta > 0)
    """
    X = standardize(X)

    # Compute dissimilarity matrix
    D = im.dissimilarity(X)

    # Run NERFCM multiple times with different random seeds
    betas = []
    for seed in range(n_seeds):
        U, beta, n_iter = nerfcm(D, c, seed=seed, verbose=False)
        betas.append(beta)

    beta_mean = np.mean(betas)
    beta_max = np.max(betas)
    any_activated = np.max(betas) > 0

    return {
        "name": name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "true_c": c,
        "beta_values": betas,
        "beta_mean": beta_mean,
        "beta_max": beta_max,
        "any_activated": any_activated,
    }


def main():
    print("NERFCM Beta-Spread Activation Verification on Real Data")
    print("=" * 70)
    print()

    # Define test datasets
    test_cases = [
        (load_iris(), 3),  # 3 iris species
        (load_glass(), 6),  # 6 glass types
        (load_heart(), 2),  # 2 heart conditions
    ]

    results = []

    for (X, y, name), c in test_cases:
        print(
            f"Testing {name} dataset (n={X.shape[0]}, features={X.shape[1]}, c={c})..."
        )
        result = test_nerfcm_beta_spread(X, y, name, c, n_seeds=5)
        results.append(result)

        # Print details
        print(
            f"  Beta values across 5 seeds: {[f'{b:.4f}' for b in result['beta_values']]}"
        )
        print(f"  Mean beta: {result['beta_mean']:.4f}")
        print(f"  Max beta:  {result['beta_max']:.4f}")
        print(f"  Activated: {'YES ✓' if result['any_activated'] else 'NO'}")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(
        f"{'Dataset':<20}{'n_samples':<12}{'beta_mean':<12}{'beta_max':<12}{'Activated'}"
    )
    print("-" * 70)

    for r in results:
        activated = "YES ✓" if r["any_activated"] else "NO"
        print(
            f"{r['name']:<20}{r['n_samples']:<12}{r['beta_mean']:<12.4f}{r['beta_max']:<12.4f}{activated}"
        )

    print()
    print("FINDINGS:")
    print("-" * 70)

    activated_count = sum(1 for r in results if r["any_activated"])

    if activated_count == len(results):
        print("✓ CONFIRMED: Beta-spread activates on ALL real datasets tested.")
        print("  This indicates that real-world data exhibits non-Euclidean properties")
        print("  that trigger the beta-spread transform, validating its importance.")
    elif activated_count > 0:
        print(
            f"✓ PARTIAL: Beta-spread activates on {activated_count}/{len(results)} datasets."
        )
        print("  The spread is needed for some but not all real datasets.")
    else:
        print("✗ NOT OBSERVED: Beta-spread did not activate on any dataset.")
        print("  This may indicate that Euclidean dissimilarity is sufficient for")
        print("  these particular real datasets, or that the data is limited in scope.")

    print()
    print("Interpretation:")
    print("-" * 70)
    if activated_count > 0:
        print("The beta-spread mechanism is actively needed on real data.")
        print(
            "On synthetic datasets, beta stayed at 0 (correct - clean, separable data)."
        )
        print(
            "On real data with measurement noise and complex structure, beta activates"
        )
        print(
            "to restore metric admissibility and enable proper relational clustering."
        )
    else:
        print(
            "Consider testing on additional datasets or those known to be non-Euclidean."
        )
        print(
            "Graph-based, string-based, or tree-structured data may show stronger effects."
        )


if __name__ == "__main__":
    main()
