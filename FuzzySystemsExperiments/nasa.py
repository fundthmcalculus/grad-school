import time

import numpy as np
from sklearn.model_selection import train_test_split

from tribblefis.gauss_plot import report_figures_of_merit
# tribble-fis PR #67 deleted gauss_math.{standard_transform,log_transform,
# detect_and_apply_log_transform}. UnitScalar replaces standard_transform's
# min-max-to-[0,1] behaviour (the name was always a misnomer -- it never
# z-scored).
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
    # BEHAVIOUR CHANGE, and the reason is structural rather than cosmetic.
    #
    # This script was the one sample that normalized BEFORE logging: it min-max
    # scaled this column subset to [0,1] and then applied np.log1p on top of the
    # scaled values. UnitScalar's order is fixed at log -> normalize, so it simply
    # cannot express the old order; the previous code kept it only by disabling the
    # scaler's log step (`log_dynamic_range=None`) and doing the log1p by hand
    # afterwards. That is exactly the custom transformation code this migration is
    # meant to remove, so it is converted to the canonical order instead.
    #
    # ==> THIS SAMPLE'S NUMBERS WILL MOVE, and not by rounding. Both orders are
    # monotone per feature, so anything rank-based would be unaffected, but
    # MixtureOfGaussiansFuzzyClassifier fits Gaussian membership functions to
    # actual values and will see a different distribution. Nothing here reproduces
    # the old output, and nothing pretends to. Unverified: load_data() fetches
    # Statlog Shuttle over the network via ucimlrepo, so this cannot be run here.
    #
    # `log_dynamic_range=0` forces the log1p onto every column it is handed, which
    # is how the old code behaved (it logged all seven unconditionally). It is the
    # only way to say "log exactly these columns" through this API -- there is no
    # explicit log-column list -- and it is applied to the subset precisely so the
    # logged set stays the seven columns named above.
    X[scaled_cols] = UnitScalar(log_dynamic_range=0).fit_transform(
        X[scaled_cols]).astype(X[scaled_cols].dtypes.iloc[0])

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
