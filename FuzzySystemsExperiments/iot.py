import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gauss_math import (
    generate_synthetic_data,
    simple_gaussian_predict,
)
from tribblefis.gauss_plot import (
    report_figures_of_merit,
    plot_membership_functions,
    plot_confusion_matrix,
    plot_classification_report,
)
from tribblefis.scaling import UnitScalar


def load_data():
    """RT-IOT2022 via the shared loader (``repro_data.load_rt_iot2022``), which
    reads ``data/RT_IOT2022.csv`` and drops the leaky unnamed per-class index
    column plus the ``proto``/``service`` strings -> 81 features. The old inline
    copy read an ``rt-iot2022/`` path absent from this repo and dropped a
    differently-named ``id`` column; per the consolidation policy this unifies
    onto the canonical leak-free loader. y is returned as a Series indexed like X
    so the ``.nunique()`` and ``stratify=y`` usage below is unchanged.
    """
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from repro_data import load_rt_iot2022

    X, y = load_rt_iot2022()
    return X, pd.Series(y, index=X.index)


def main(augment_data=False):
    X, y = load_data()
    start_time = time.time()

    # Get the number of unique values in y
    n_unique = y.nunique()
    print(f"Number of unique values in y: {n_unique}")

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # BEHAVIOUR CHANGE. This used to log an explicit 12-column list (the
    # *_pkts_per_sec / flow_iat.* / active.* / flow_duration rate and duration
    # features) at offset 1, pre-split, and normalize nothing. UnitScalar
    # auto-detects the logged set by dynamic range and min-max bounds to [0,1].
    # The normalization is deliberate -- FIS membership functions want bounded
    # inputs. Library default `log_dynamic_range=3.0`; nothing is claimed to
    # reproduce the old list. Unverified: rt-iot2022/ is not in this repo.
    #
    # Not a pipeline: report_figures_of_merit / simple_gaussian_predict /
    # generate_synthetic_data below all take X frames directly and must see the
    # same scaled space as the memberships. Fitted on train only.
    # set_output("pandas") keeps the column names and index the tribblefis helpers
    # below rely on, so no manual DataFrame re-wrapping is needed.
    _scaler = UnitScalar().set_output(transform="pandas").fit(X_train)
    print(f"Auto-detected log transform for: {list(_scaler.log_features_)}")
    X_train, X_test = _scaler.transform(X_train), _scaler.transform(X_test)

    # Initialize and fit the Gaussian Mixture Classifier
    # TODO - top-n=3!
    clf = TribbleClassifier()
    clf.fit(X_train, y_train)

    top_n_todo = clf.top_features_
    gaussian_memberships = clf.model_

    if augment_data:
        print("\nAugmenting training data to improve parity...")
        X_train_aug, y_train_aug = generate_synthetic_data(
            X_train, y_train, gaussian_memberships
        )
        print(f"New training set size: {len(X_train_aug)}")

        # Fit new model on augmented data
        clf.fit(X_train_aug, y_train_aug)
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

    # print("1-pass Total Model Stats:")
    # print("=" * 80)
    # print(f"N_rules={gaussian_memberships.n_rules}")
    # print(f"N_memberships={gaussian_memberships.n_membership_functions}")
    # print(f"Possible rules={gaussian_memberships.possible_rules}")

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

    print("Pre-Clean Total Model Stats:")
    print("=" * 80)
    print(f"N_rules={gaussian_memberships.n_rules}")
    print(f"N_memberships={gaussian_memberships.n_membership_functions}")
    print(f"Possible rules={gaussian_memberships.possible_rules}")

    simple_model = gaussian_memberships.to_simple_model()

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

    # Plot membership functions of the simple model
    plot_membership_functions(simple_model)


if __name__ == "__main__":
    main()
