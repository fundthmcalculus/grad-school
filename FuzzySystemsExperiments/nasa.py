import time

import numpy as np
from sklearn.model_selection import train_test_split

from tribblefis.gauss_plot import report_figures_of_merit
# tribble-fis PR #67 deleted gauss_math.{standard_transform,log_transform,
# detect_and_apply_log_transform}. UnitScalar replaces standard_transform's
# min-max-to-[0,1] behaviour (the name was always a misnomer -- it never
# z-scored); the bare `log_transform` has no successor, but it was only ever
# np.log1p, applied HERE *after* the scaling rather than before it. UnitScalar's
# own log step runs before its min-max, so it cannot express this order and
# `log_dynamic_range=None` disables it; the log1p stays explicit below.
from tribblefis.scaling import UnitScalar
from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier


def load_data():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    statlog_shuttle = fetch_ucirepo(id=148)

    # data (as pandas dataframes)
    X = statlog_shuttle.data.features.astype(np.float32)
    y = statlog_shuttle.data.targets['class'].astype(np.str_)

    # metadata
    print(statlog_shuttle.metadata)

    # variable information
    print(statlog_shuttle.variables)
    return X, y


def main():
    start_time = time.time()
    X, y = load_data()

    # Get the number of unique values in y
    n_unique = y.nunique()
    print(f"Number of unique values in y: {n_unique}")

    scaled_cols = ["Rad Flow", "Fpv Close", "Fpv Open", "High", "Bypass",
                   "Bpv Close", "Bpv Open"]
    # min-max to [0,1] on this subset only, then log1p -- the original order.
    X[scaled_cols] = UnitScalar(log_dynamic_range=None).fit_transform(
        X[scaled_cols]).astype(X[scaled_cols].dtypes.iloc[0])
    X[scaled_cols] = np.log1p(X[scaled_cols])

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Initialize and fit the Gaussian Mixture Classifier
    clf = MixtureOfGaussiansFuzzyClassifier()
    clf.fit(X_train, y_train)

    top_n_todo = clf.top_features_
    gaussian_memberships = clf.model_

    cm_train, top_confusion_train, confused_data_train = report_figures_of_merit(
        X_train, y_train, gaussian_memberships, n_unique, start_time, top_n_todo, label="train"
    )

    # Update references after augmentation
    gaussian_memberships = clf.model_

    cm_test, top_confusion_test, confused_data_test = report_figures_of_merit(
        X_test, y_test, gaussian_memberships, n_unique, start_time, top_n_todo, label="test"
    )


if __name__ == "__main__":
    main()
