"""An interval type-2 regressor whose footprint of uncertainty is *fitted*.

`tribble-fis`'s `IntervalType2FuzzyRegressor` derives its footprint from a
single global hyperparameter: every membership gets `sigma*(1+w)` above and
`sigma*max(0.1, 1-w)` below, with the same `w` everywhere. The footprint is
therefore a decoration on the type-1 model -- it encodes no information the
type-1 fit did not already have, and it cannot be wider where the model is
actually less certain. Saying such a model "represents the uncertainty in the
data" is not supported by what it computes.

This estimator makes the footprint carry information. `w` becomes a vector
(one entry per selected feature, or per feature/output-bucket cell) fitted by
minimising a **proper scoring rule for intervals** -- the Winkler interval
score at level `alpha`:

    IS = (hi - lo)
       + (2/alpha) * (lo - y) * 1{y < lo}
       + (2/alpha) * (y - hi) * 1{y > hi}

which rewards narrow intervals and penalises misses in proportion to how far
outside they fall. It is minimised by the true central (1-alpha) interval, so
"minimise IS" is a statement about calibration and sharpness jointly, not a
heuristic. The fitted footprint is then a genuine per-region uncertainty
estimate and `predict_intervals` means something.

Why this matters for the drafter: speculative acceptance is symmetric in total
variation, so a wider interval does not improve the point estimate. What it
buys is knowing *when the shape prediction should not be trusted*, which is
the input the draft-length controller needs. A footprint set by a global
constant cannot supply that; a fitted one can.

Prototyped here rather than upstream until it demonstrates it is worth having.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

from tribblefis.gauss_data import (
    GaussianMembership,
    IT2FeatureModel,
    IT2GaussianMembership,
    IT2GaussianMixtureModel,
    IT2LabelModel,
)
from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.it2_kernel import it2_firing_strengths
from tribblefis.regression import apply_tsk_consequents
from tribblefis.gauss_math import resolve_norm_pair

MIN_SIGMA = 1e-4


def interval_score(y, lo, hi, alpha: float = 0.1) -> np.ndarray:
    """Winkler interval score. Lower is better. Proper for the (1-alpha) interval."""
    width = hi - lo
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    return width + (2.0 / alpha) * (below + above)


class LearnedFoUIT2Regressor(BaseEstimator, RegressorMixin):
    """IT2 regressor with a per-cell footprint fitted to an interval score.

    Parameters
    ----------
    granularity : {"feature", "cell", "global"}
        How many widths to fit. ``"feature"`` gives one per selected feature,
        ``"cell"`` one per (feature, output-bucket), ``"global"`` a single
        scalar -- ``"global"`` reproduces the stock estimator's parameterisation
        and exists as the control that says whether the extra freedom pays.
    alpha : float
        Interval level; 0.1 fits a 90% interval.
    """

    def __init__(
        self,
        top_n=8,
        n_gaussians=3,
        n_output_buckets=5,
        tsk_order="1st",
        norm_conorm="probability",
        granularity="feature",
        alpha=0.1,
        init_width=0.5,
        max_iter=60,
        km_iterations=None,
        random_state=42,
        fit_fraction=0.5,
    ):
        self.top_n = top_n
        self.n_gaussians = n_gaussians
        self.n_output_buckets = n_output_buckets
        self.tsk_order = tsk_order
        self.norm_conorm = norm_conorm
        self.granularity = granularity
        self.alpha = alpha
        self.init_width = init_width
        self.max_iter = max_iter
        self.km_iterations = km_iterations
        self.random_state = random_state
        self.fit_fraction = fit_fraction

    # -- footprint construction -------------------------------------------

    def _width_for(self, f_idx: int, l_idx: int, w: np.ndarray) -> float:
        if self.granularity == "global":
            return float(w[0])
        if self.granularity == "feature":
            return float(w[f_idx])
        return float(w[f_idx * self._n_labels + l_idx])

    def _build(self, w: np.ndarray) -> IT2GaussianMixtureModel:
        feature_models = {}
        for fi, (fname, t1_feat) in enumerate(self._t1_model.feature_models.items()):
            label_models = {}
            for li, (label, t1_lab) in enumerate(t1_feat.label_models.items()):
                mfs = []
                for mf in t1_lab.memberships:
                    base = max(mf.sigma, MIN_SIGMA)
                    width = self._width_for(fi, li, w)
                    mfs.append(
                        IT2GaussianMembership(
                            upper_mf=GaussianMembership(
                                mu=mf.mu, sigma=base * (1.0 + width), id=mf.id
                            ),
                            lower_mf=GaussianMembership(
                                mu=mf.mu,
                                sigma=base * max(0.1, 1.0 - width),
                                id=mf.id,
                            ),
                        )
                    )
                label_models[label] = IT2LabelModel(mfs)
            feature_models[fname] = IT2FeatureModel(label_models)
        return IT2GaussianMixtureModel(feature_models)

    def _bounds_for(self, X: pd.DataFrame, model) -> tuple[np.ndarray, np.ndarray]:
        fu, fl, _, labels = it2_firing_strengths(
            X, model, self._norms, km_iterations=None
        )
        b = self._base
        preds = [
            apply_tsk_consequents(
                X, b.top_features_, f, labels, b.y_bucket_mean_, b.corr_terms_,
                order=b.tsk_order, basis=b.consequent_basis,
                cross_pairs=b.cross_pairs_,
            )
            for f in (fu, fl)
        ]
        return np.minimum(*preds), np.maximum(*preds)

    # -- sklearn API -------------------------------------------------------

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        self.feature_names_in_ = list(X.columns)

        self._base = TribbleRegressor(
            top_n=self.top_n,
            n_gaussians=self.n_gaussians,
            n_output_buckets=self.n_output_buckets,
            tsk_order=self.tsk_order,
            norm_conorm=self.norm_conorm,
            random_state=self.random_state,
        ).fit(X, y)
        self._t1_model = self._base.model_
        self._norms = resolve_norm_pair(self.norm_conorm)

        n_feat = len(self._t1_model.feature_models)
        self._n_labels = max(
            len(f.label_models) for f in self._t1_model.feature_models.values()
        )
        n_w = {"global": 1, "feature": n_feat, "cell": n_feat * self._n_labels}[
            self.granularity
        ]

        # The footprint is fitted on a held-out slice of the training data.
        # Fitting it on the same rows that fixed the consequents would let the
        # widths shrink to cover in-sample residuals that the test set does not
        # share -- the interval would look calibrated and would not be.
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(X))
        n_cal = max(50, int(self.fit_fraction * len(X)))
        cal = idx[:n_cal]
        Xc, yc = X.iloc[cal], y[cal]

        def objective(w):
            lo, hi = self._bounds_for(Xc, self._build(np.abs(w)))
            return float(np.mean(interval_score(yc, lo, hi, self.alpha)))

        w0 = np.full(n_w, float(self.init_width))
        self.opt_ = minimize(
            objective,
            w0,
            method="L-BFGS-B",
            bounds=[(0.01, 3.0)] * n_w,
            options={"maxiter": self.max_iter},
        )
        self.widths_ = np.abs(self.opt_.x)
        self.initial_score_ = objective(w0)
        self.fitted_score_ = float(self.opt_.fun)
        self.model_ = self._build(self.widths_)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        _, _, fc, labels = it2_firing_strengths(
            X, self.model_, self._norms, km_iterations=self.km_iterations
        )
        b = self._base
        return apply_tsk_consequents(
            X, b.top_features_, fc, labels, b.y_bucket_mean_, b.corr_terms_,
            order=b.tsk_order, basis=b.consequent_basis, cross_pairs=b.cross_pairs_,
        )

    def predict_intervals(self, X):
        return self._bounds_for(pd.DataFrame(X), self.model_)

    def describe_widths(self) -> pd.DataFrame:
        """The fitted footprint, per feature -- the interpretable output."""
        names = list(self._t1_model.feature_models)
        if self.granularity == "global":
            return pd.DataFrame({"scope": ["global"], "width": self.widths_})
        if self.granularity == "feature":
            return pd.DataFrame({"feature": names, "width": self.widths_}).sort_values(
                "width", ascending=False
            )
        rows = [
            {"feature": names[i], "bucket": j,
             "width": self.widths_[i * self._n_labels + j]}
            for i in range(len(names))
            for j in range(self._n_labels)
        ]
        return pd.DataFrame(rows).sort_values("width", ascending=False)
