import time

import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gauss_plot import report_figures_of_merit

# tribble-fis PR #67 deleted gauss_math.{standard_transform,log_transform,
# detect_and_apply_log_transform}. UnitScalar replaces standard_transform's
# min-max-to-[0,1] behaviour (the name was always a misnomer -- it never
# z-scored).
from tribblefis.scaling import UnitScalar
from tribblefis.gaussian_classifier import TribbleClassifier


def load_data():
    """Statlog Shuttle via the shared loader (``repro_data.load_shuttle``),
    reading ``data/shuttle.csv``. The old copy network-fetched via
    ``ucimlrepo`` (id=148), which named the columns (``Rad Flow`` ...); the
    canonical file ships ``atrib1..atrib9``, so those sensor names are
    unavailable downstream -- see the scaling note in ``main()``. y is a Series
    so the ``.nunique()`` and ``stratify=y`` usage below is unchanged.
    """
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from repro_data import load_shuttle

    X, y = load_shuttle()
    return X, pd.Series(y, index=X.index)


def main():
    start_time = time.time()
    X, y = load_data()

    # Get the number of unique values in y
    n_unique = y.nunique()
    print(f"Number of unique values in y: {n_unique}")

    # SCALING, and why it changed with the loader.
    #
    # The old copy hand-picked seven sensor columns by name (Rad Flow, Fpv
    # Close, ...) -- names ucimlrepo attached to Statlog Shuttle -- and min-max
    # scaled that subset. The canonical data/shuttle.csv ships the same nine
    # attributes as atrib1..atrib9 with no sensor names and no in-repo map back
    # to them, so that named subset cannot be reselected. The scaling is
    # therefore applied canonically across all features via UnitScalar's own
    # log->normalize with dynamic-range auto-detection -- the custom
    # per-column transform this migration exists to remove.
    #
    # ==> THIS SAMPLE'S NUMBERS MOVE, and not by rounding: a different (and
    # larger) set of columns is scaled, in the canonical order. Rank-based
    # quantities are unaffected (the transform is monotone per feature) but a
    # Gaussian-membership fit sees a different distribution. Nothing here
    # reproduces the old output, and nothing pretends to.
    _scaler = UnitScalar().set_output(transform="pandas")
    X = _scaler.fit_transform(X)

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Initialize and fit the Gaussian Mixture Classifier
    clf = TribbleClassifier()
    clf.fit(X_train, y_train)

    top_n_todo = clf.top_features_
    gaussian_memberships = clf.model_

    cm_train, top_confusion_train, confused_data_train = report_figures_of_merit(
        X_train,
        y_train,
        gaussian_memberships,
        n_unique,
        start_time,
        top_n_todo,
        label="train",
    )

    # Update references after augmentation
    gaussian_memberships = clf.model_

    cm_test, top_confusion_test, confused_data_test = report_figures_of_merit(
        X_test,
        y_test,
        gaussian_memberships,
        n_unique,
        start_time,
        top_n_todo,
        label="test",
    )


if __name__ == "__main__":
    main()
