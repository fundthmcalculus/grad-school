import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.scaling import UnitScalar
from tribblefis.gauss_plot import report_figures_of_merit


def load_data():
    """BETH via the shared loader (``repro_data.load_beth``), reading data/beth/.

    UNIFIED TO CANONICAL. The old copy read a single ``beth_data/`` file (absent
    from this repo) and predicted ``sus`` -- BETH's SECOND label. This predicts
    the primary ``evil`` label on the canonical test split and drops ``sus`` and
    ``timestamp`` as leaks (the LEAKY_COLUMNS ``table_4_11`` uses; ``sus`` is a
    second label). That is a deliberate change of what this script measures, per
    the consolidation policy, so this script's numbers move. y is a Series so the
    ``.nunique()`` and ``stratify=y`` usage below is unchanged.
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from repro_data import load_beth

    splits = load_beth()
    X, y = splits["test"]
    X = X.drop(columns=["sus", "timestamp"], errors="ignore")
    y = pd.Series(y, index=X.index).map({0: "legit", 1: "evil"})
    return X, y


def main():
    start_time = time.time()
    X, y = load_data()

    # Get the number of unique values in y
    n_unique = y.nunique()
    print(f"Number of unique values in y: {n_unique}")

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # BEHAVIOUR CHANGE. This used to be
    #   log_transform(X, ["timestamp","processId","mountNamespace","eventId","userId"], 1)
    # -- an explicit column list, pre-split, and no normalization whatsoever.
    # UnitScalar auto-detects the logged set by dynamic range and min-max bounds
    # to [0,1] afterwards. Adding the normalization is deliberate: FIS membership
    # functions want bounded inputs. No `log_dynamic_range` is claimed to
    # reproduce the old list, so the library default (3.0) is used.
    #
    # Not a pipeline: `report_figures_of_merit` and `clf.augment` below are handed
    # X frames directly and must see the same scaled space the memberships were
    # built in, so the transform has to be materialized rather than hidden inside
    # an estimator. Fitted on train only.
    # set_output("pandas") keeps the column names and index the tribblefis helpers
    # below rely on, so no manual DataFrame re-wrapping is needed.
    _scaler = UnitScalar().set_output(transform="pandas").fit(X_train)
    if _scaler.log_features_:
        print(f"Auto-detected log transform for: {list(_scaler.log_features_)}")
    X_train, X_test = _scaler.transform(X_train), _scaler.transform(X_test)

    # Initialize and fit the Gaussian Mixture Classifier
    clf = TribbleClassifier(top_n=3)
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

    for (true_class, confused_class), confusion_data in confused_data_train.items():
        X_local_train, y_local_train = confusion_data["X"], confusion_data["y"]
        # Augment the existing classifier
        clf.augment(X_local_train, y_local_train)

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
