import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gauss_math import simple_gaussian_predict
from tribblefis.scaling import UnitScalar
from tribblefis.gauss_plot import (
    report_figures_of_merit,
    plot_confusion_matrix,
    plot_classification_report,
    plot_membership_functions,
)


def load_data():
    """PhiUSIIL via the shared loader (``repro_data.load_phiusiil``), which reads
    ``data/PhiUSIIL_Phishing_URL_Dataset.csv`` and applies the canonical
    preprocessing (drop label + the text columns -> 50 features, label mapped to
    "legit"/"phish"). The old inline copy read a ``phishing_data/`` path absent
    from the repo, so this script could not run from the repo root.

    ``sample_size=None`` keeps all rows; y is returned as a Series indexed like X
    so the ``.nunique()`` and ``stratify=y`` usage below is unchanged.
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from repro_data import load_phiusiil

    X, y = load_phiusiil(sample_size=None)
    return X, pd.Series(y, index=X.index)


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

    # BEHAVIOUR CHANGE. This used to be two explicit, pre-split `log_transform`
    # calls with no normalization at all: 17 count/length columns at offset 1,
    # then 4 ratio columns at offset 1e-4. Both are now one UnitScalar.
    #
    # What happened to `offset=1e-4`: nothing, and nothing is needed. All four
    # ratio columns (SpacialCharRatioInURL, DegitRatioInURL, ObfuscationRatio,
    # CharContinuationRate) have min == 0 -- measured on the real dataset, with
    # 184334 and 235310 exact zeros in two of them -- so the offset's only job was
    # keeping `np.log(0)` from returning -inf. UnitScalar applies
    # `log1p(x - col_min)`, which is 0 at x == col_min by construction, so there is
    # no -inf to dodge and no offset parameter to carry over. Note the consequence:
    # `log(x + 1e-4)` spread [0, 0.397] over [-9.21, -0.92], nine decades of
    # near-zero detail; `log1p` on the same range is [0, 0.334] and very nearly
    # linear. The old code magnified the near-zero ratio mass; this does not.
    #
    # The logged *set* also differs, measured on the real dataset: at the library
    # default 3.0 UnitScalar logs 13 columns, of which 2 (TLDLegitimateProb,
    # URLTitleMatchScore) were never in the explicit lists and 10 explicit ones --
    # including all four ratio columns -- fall below threshold. No
    # `log_dynamic_range` reproduces the old lists, so none is faked.
    #
    # Not a pipeline: `report_figures_of_merit` / `simple_gaussian_predict` below
    # take X frames directly and must see the same scaled space as the
    # memberships. Fitted on train only.
    # set_output("pandas") keeps the column names and index the tribblefis helpers
    # below rely on, so no manual DataFrame re-wrapping is needed.
    _scaler = UnitScalar().set_output(transform="pandas").fit(X_train)
    print(f"Auto-detected log transform for: {list(_scaler.log_features_)}")
    X_train, X_test = _scaler.transform(X_train), _scaler.transform(X_test)

    # Initialize and fit the Gaussian Mixture Classifier
    clf = TribbleClassifier(
        top_n=5,
        n_gaussians={
            "LineOfCode": 3,
            "NoOfExternalRef": 2,
        },
    )
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

    print("1-pass Total Model Stats:")
    print("=" * 80)
    print(f"N_rules={gaussian_memberships.n_rules}")
    print(f"N_memberships={gaussian_memberships.n_membership_functions}")
    print(f"Possible rules={gaussian_memberships.possible_rules}")

    # for (true_class, confused_class), confusion_data in confused_data_train.items():
    #     X_local_train, y_local_train = confusion_data["X"], confusion_data["y"]
    #     new_gaussian_memberships = create_gaussian_membership_dict(
    #         X_local_train, y_local_train, top_n_var_names=top_n_todo
    #     )
    #     # Now, we need to augment the existing gaussian memberships
    #     gaussian_memberships = gaussian_memberships.augment(new_gaussian_memberships)

    cm_test, top_confusion_test, confused_data_test = report_figures_of_merit(
        X_test,
        y_test,
        gaussian_memberships,
        n_unique,
        start_time,
        top_n_todo,
        label="test",
    )

    # print("2-pass Total Model Stats:")
    # print("=" * 80)
    # print(f"N_rules={gaussian_memberships.n_rules}")
    # print(f"N_memberships={gaussian_memberships.n_membership_functions}")
    # print(f"Possible rules={gaussian_memberships.possible_rules}")

    # Create simple gaussian model from GaussianMixtureModel
    simple_model = gaussian_memberships.to_simple_model(None)

    print("\nSimple Gaussian Classifier Model Stats:")
    print("=" * 80)
    print(f"N_rules={len(simple_model.rules)}")
    print(f"N_memberships={len(simple_model.input_mfs)}")

    # Compare results on test set
    y_pred_simple = simple_gaussian_predict(X_test[top_n_todo], simple_model)
    simple_accuracy = np.mean(y_pred_simple == y_test)
    print(f"Simple Model Accuracy (test): {simple_accuracy:.4f}")
    plot_confusion_matrix(
        y_test, y_pred_simple, title=f"TSK Model Confusion Matrix (Simple Set)"
    )
    plot_classification_report(
        y_test, y_pred_simple, title=f"TSK Model Classification Report (Simple Set)"
    )

    # Plot membership functions
    plot_membership_functions(gaussian_memberships)
    # Plot membership functions of the simple model
    plot_membership_functions(simple_model)


if __name__ == "__main__":
    main()
