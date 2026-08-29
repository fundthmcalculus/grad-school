"""
Darwin Handwriting Classification using Trapezoidal Membership Functions

This is a trapezoid variant of darwin.py that uses the new trapezoid membership
functions for the Darwin handwriting dataset. It demonstrates how trapezoids
perform on a real classification task and allows comparison with Gaussian MFs.
"""

import os
import time

from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.gauss_plot import report_figures_of_merit


def load_data():
    """DARWIN via the shared loader (``repro_data.load_darwin``), reading
    ``data/darwin.csv``. The old inline copy read a bare ``darwin.csv`` absent
    from the repo root; the dataset is gitignored (see ``data/README.md``).
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from repro_data import load_darwin

    return load_darwin()


def main():
    start_time = time.time()
    X, y = load_data()

    # Get the number of unique values in y
    n_unique = y.nunique()
    print(f"Number of unique values in y: {n_unique}")

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Initialize and fit the TRAPEZOID Mixture Classifier (using fast method by default)
    print("\nTraining trapezoid-based classifier (fast method)...")
    clf = MixtureOfGaussiansFuzzyClassifier(
        member_function="trap", trapz_method="em"
    )  # trapz_method="fast" is now default
    clf.fit(X_train, y_train)

    top_n_todo = clf.top_features_
    trapz_memberships = clf.model_

    # Create the actual fuzzy model and predict on test set
    print("\nEvaluating trapezoid classifier on test set...")
    report_figures_of_merit(
        X_test,
        y_test,
        trapz_memberships,
        n_unique,
        start_time,
        top_n_todo,
        label="test (trapezoid-fast)",
    )

    # Now, plot a set of distributions for the most-differentiating variables
    # plot_var_gauss_dist(X_train, y_train, top_n_todo, trapz_memberships)


if __name__ == "__main__":
    main()
