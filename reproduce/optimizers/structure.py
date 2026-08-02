"""Searching the model's *shape* — how many rules, how many membership functions.

The antecedent study takes the structure as given and tunes the parameters
inside it. This module gives that up. Nothing here is told how many output
buckets to use (which is the rule count for a regression MoG-TSK), how many
Gaussians to put on each feature, how many features to keep, or what consequent
order to fit. Those are decision variables, and the optimizer picks them.

That makes it a mixed problem — four discrete choices and one continuous — which
is why it uses the `optimizers` package's `InputDiscreteVariable` alongside
`InputContinuousVariable` rather than the continuous-only setup next door.

**The objective is unchanged in kind and different in cost.** Still k-fold
held-out MSE, still with the consequents solved in closed form. But an
evaluation here rebuilds the whole model — re-partitions the output, re-ranks
the features, re-fits every mixture — where an evaluation in the antecedent
study only re-solved consequents against an existing structure. Evaluations are
therefore not comparable between the two studies, and the two must never share
a budget axis.

**Rule count is reported, never optimized.** The objective is pure accuracy. A
complexity penalty would bake in an exchange rate between rules and error that
nobody has justified, and the whole interpretability argument of this thesis
turns on that exchange rate being the reader's to set. So the search minimizes
error alone and the table prints what each answer cost in rules, which makes the
trade visible instead of deciding it.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "reproduce"))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))


# The search space. Discrete choices are listed rather than given as ranges,
# because `n_gaussians = -1` means "let the fitter choose per feature" and is a
# genuine option that no interval contains.
SPACE = {
    "n_buckets": [2, 3, 4, 5, 6, 8],          # == the rule count
    "n_gaussians": [-1, 1, 2, 3, 4],          # -1 = automatic, per feature
    "top_n": [2, 3, 4, 5, 6, 7, 8],           # features retained
    "order": ["0th", "1st", "2nd"],           # consequent order
    "log10_l2": (-4.0, 0.0),                  # ridge strength, continuous
}

# What the pipeline uses today when nobody searches: Chapter 6's configuration.
DEFAULT = {"n_buckets": 3, "n_gaussians": -1, "top_n": 8, "order": "2nd",
           "log10_l2": -2.0}


def decode(vec):
    """A raw optimizer vector -> a structure dict.

    The discrete axes arrive as floats regardless of variable type (the deck is
    one float array), so each is snapped to its nearest legal choice. Snapping
    rather than rejecting keeps the objective total: an optimizer that proposes
    3.4 buckets gets a real score for 3, instead of a penalty that teaches it
    nothing about the landscape.
    """
    v = np.asarray(vec, dtype=float).ravel()
    out = {}
    for i, key in enumerate(("n_buckets", "n_gaussians", "top_n", "order")):
        choices = SPACE[key]
        idx = int(np.clip(round(v[i]), 0, len(choices) - 1))
        out[key] = choices[idx]
    lo, hi = SPACE["log10_l2"]
    out["log10_l2"] = float(np.clip(v[4], lo, hi))
    return out


def encode(structure):
    """A structure dict -> the vector the optimizer sees. Inverse of `decode`."""
    return np.array(
        [SPACE["n_buckets"].index(structure["n_buckets"]),
         SPACE["n_gaussians"].index(structure["n_gaussians"]),
         SPACE["top_n"].index(structure["top_n"]),
         SPACE["order"].index(structure["order"]),
         structure["log10_l2"]], dtype=float)


def variables():
    """Bounded variables for the `optimizers` package, in `decode` order."""
    from optimizers.continuous.variables import InputContinuousVariable
    lo, hi = SPACE["log10_l2"]
    return [InputContinuousVariable("n_buckets", 0, len(SPACE["n_buckets"]) - 1),
            InputContinuousVariable("n_gaussians", 0, len(SPACE["n_gaussians"]) - 1),
            InputContinuousVariable("top_n", 0, len(SPACE["top_n"]) - 1),
            InputContinuousVariable("order", 0, len(SPACE["order"]) - 1),
            InputContinuousVariable("log10_l2", lo, hi)]


def bounds():
    lo, hi = SPACE["log10_l2"]
    return [(0, len(SPACE["n_buckets"]) - 1),
            (0, len(SPACE["n_gaussians"]) - 1),
            (0, len(SPACE["top_n"]) - 1),
            (0, len(SPACE["order"]) - 1),
            (lo, hi)]


class StructureProblem:
    """Build-and-score a MoG-TSK of any shape, on one train/test split."""

    def __init__(self, seed=0, n_folds=3, test_size=0.2):
        from sklearn.model_selection import train_test_split
        from tribblefis.gauss_math import (detect_and_apply_log_transform,
                                           standard_transform)
        import _fuzzy_models as FM

        data = FM.load_concrete()
        if data is None:
            raise RuntimeError("Concrete unavailable (no CSV, no UCI mirror).")
        X, y_raw = data

        # Same preprocessing as `table_concrete_reconciliation.prepare`, but the
        # output partition is NOT done here: its bucket count is a decision
        # variable, so it moves inside the objective.
        self.y_t = standard_transform(y_raw)
        Xt, self.logged = detect_and_apply_log_transform(X.copy(),
                                                         min_dynamic_range=2)
        self.X_t = standard_transform(Xt, column=Xt.columns)
        yr = np.asarray(y_raw, dtype=float)
        self.span = float(yr.max() - yr.min())

        idx = np.arange(len(self.X_t))
        self.tr_idx, self.te_idx = train_test_split(idx, test_size=test_size,
                                                    random_state=seed)
        self.seed, self.n_folds = seed, n_folds

    # -- the two things the study needs ------------------------------------- #
    def cv_mse(self, structure):
        """k-fold held-out MSE on the training split, for one structure."""
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        total, n = 0.0, 0
        for tr, val in kf.split(self.tr_idx):
            got = self._fit_predict(self.tr_idx[tr], self.tr_idx[val], structure)
            if got is None:
                return 1e6
            truth, pred = got
            total += float(np.mean((truth - pred) ** 2))
            n += 1
        return total / max(n, 1)

    def test_score(self, structure):
        """(R^2, RMSE in MPa, rule count, membership-function count)."""
        from sklearn.metrics import r2_score
        got = self._fit_predict(self.tr_idx, self.te_idx, structure, count=True)
        if got is None:
            return float("nan"), float("nan"), 0, 0
        truth, pred, n_rules, n_mfs = got
        rmse = float(np.sqrt(np.mean((truth - pred) ** 2))) * self.span
        return float(r2_score(truth, pred)), rmse, n_rules, n_mfs

    # -- one build ----------------------------------------------------------- #
    def _fit_predict(self, tr, te, structure, count=False):
        from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                           create_gaussian_membership_dict,
                                           take_top_features)
        from tribblefis.regression import (partition_output, predict_tsk,
                                           solve_tsk_consequents)

        X_tr, X_te = self.X_t.iloc[tr], self.X_t.iloc[te]
        try:
            # The output partition depends on the bucket count, so it is redone
            # per candidate -- and on the TRAINING rows only. Partitioning over
            # the whole set would leak the test target distribution into the
            # rule centres, which is the classic way this kind of search
            # flatters itself.
            y_tr_all, y_bucket_mean = partition_output(structure["n_buckets"],
                                                       self.y_t.iloc[tr])
            diffs = calculate_gaussian_correlation(X_tr, y_tr_all["y_bucket"])
            _, top_vars = take_top_features(diffs, top_n=structure["top_n"])
            model = create_gaussian_membership_dict(
                X_tr, y_tr_all["y_bucket"], top_n_var_names=top_vars,
                n_gaussians=structure["n_gaussians"])
            l2 = 10.0 ** structure["log10_l2"]
            corr, ybm = solve_tsk_consequents(
                X_tr, model, top_vars, y_bucket_mean, y_tr_all,
                n_output_buckets=structure["n_buckets"], order=structure["order"],
                l2_reg=l2, basis="raw", cross_pairs=None, verbose=False)
            pred = predict_tsk(X_te, model, top_vars, ybm, corr,
                               order=structure["order"], basis="raw",
                               cross_pairs=None)
        except Exception:  # noqa: BLE001 -- an infeasible shape scores as bad
            return None

        truth = np.asarray(self.y_t.iloc[te], dtype=float).ravel()
        pred = np.asarray(pred, dtype=float).ravel()
        keep = ~np.isnan(pred)
        if not np.any(keep):
            return None
        if not count:
            return truth[keep], pred[keep]
        n_mfs = sum(len(lm.memberships)
                    for fm in model.feature_models.values()
                    for lm in fm.label_models.values())
        return truth[keep], pred[keep], structure["n_buckets"], n_mfs

    def objective(self):
        """The callable the optimizers minimize: raw vector -> CV MSE."""
        def fn(vec):
            return self.cv_mse(decode(vec))
        return fn
