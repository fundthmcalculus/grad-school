"""Uniformity-preserving feature transforms for FIS membership-function placement.

Gaussian and triangular MFs are placed based on data statistics (means,
variances, or cluster centres).  When the input distribution is skewed or
heavy-tailed, MFs bunch up in high-density regions and leave gaps in the tails.
A **uniformity transform** re-maps each feature so the transformed values are
approximately uniformly distributed, which means MFs placed by data statistics
naturally cover the domain evenly.

``tribblefis.scaling`` offers ``log1p`` as its only nonlinear pre-step — it
tames right-skewed, multi-decade features but does nothing for other
distribution shapes.  This module prototypes three additional transforms, each
mapping to approximate uniformity and then affine-scaling to a target range:

1. **QuantileUniformScaler** — wraps ``sklearn.preprocessing.QuantileTransformer``
   with ``output_distribution='uniform'``, then affine-maps to ``feature_range``.
   The gold standard for marginal uniformity.

2. **EmpiricalCDFScaler** — evaluates the empirical CDF of the training data
   (equivalent to a rank transform normalised to [0, 1]).  A simpler, faster
   alternative to QuantileTransformer that does not interpolate between
   quantiles.  Affine-maps to ``feature_range``.

3. **PiecewiseLinearCDFScaler** — approximates each feature's empirical CDF
   with ``n_pieces`` equal-probability segments, each mapped by a single
   affine function.  This is the *"affine maps"* idiom: a piecewise-linear,
   monotone, invertible transform where each segment is a pure affine map, and
   the composition pushes the marginal toward uniform.  ``n_pieces`` trades
   fidelity for smoothness; at the limit of one piece per training point it
   degenerates to the full empirical CDF.

All three follow the sklearn transformer API (``fit`` / ``transform`` /
``inverse_transform`` / ``get_feature_names_out``), so they compose into
``sklearn.pipeline.Pipeline`` the same way the tribblefis scalers do.

Usage::

    from uniformity_transforms import QuantileUniformScaler
    scaler = QuantileUniformScaler(feature_range=(0.0, 1.0), n_quantiles=200)
    Xt = scaler.fit_transform(X_train)
    Xte_t = scaler.transform(X_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class QuantileUniformScaler(TransformerMixin, BaseEstimator):
    """Quantile transform to uniform, then affine-map to ``feature_range``.

    Wraps ``sklearn.preprocessing.QuantileTransformer`` for the hard part
    (interpolated quantile mapping), then applies the affine step so the
    output lands in a caller-specified range rather than always ``[0, 1]``.

    Args:
        feature_range: Desired ``(min, max)`` of transformed output.
        n_quantiles: Number of quantiles to compute (passed to sklearn).
        subsample: Maximum training-set size sklearn uses for fitting
            the quantile function (default 100_000).
    """

    def __init__(
        self,
        feature_range=(0.0, 1.0),
        n_quantiles=1000,
        subsample=100_000,
    ):
        self.feature_range = feature_range
        self.n_quantiles = n_quantiles
        self.subsample = subsample

    def fit(self, X, y=None):
        from sklearn.preprocessing import QuantileTransformer

        X_arr = np.asarray(X, dtype=float)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X_arr.shape[1])]
        self.n_features_in_ = X_arr.shape[1]

        self._qt = QuantileTransformer(
            n_quantiles=min(self.n_quantiles, X_arr.shape[0]),
            output_distribution="uniform",
            subsample=self.subsample,
        )
        self._qt.fit(X_arr)
        return self

    def transform(self, X):
        check_is_fitted(self, "_qt")
        X_arr = np.asarray(X, dtype=float)
        Xu = self._qt.transform(X_arr)
        lo, hi = self.feature_range
        return Xu * (hi - lo) + lo

    def inverse_transform(self, X):
        check_is_fitted(self, "_qt")
        X_arr = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        Xu = (X_arr - lo) / (hi - lo)
        return self._qt.inverse_transform(Xu)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "_qt")
        return np.asarray(self.feature_names_in_, dtype=object)


class EmpiricalCDFScaler(TransformerMixin, BaseEstimator):
    """Rank-based empirical CDF transform, then affine-map to ``feature_range``.

    For each feature, the transform is::

        F(x) = (rank of x among training values) / n_train

    which maps training values to ``[1/n, 1]``.  Values outside the training
    range are clipped to ``[0, 1]``.  The result is then affine-mapped to
    ``feature_range``.

    Simpler and faster than ``QuantileUniformScaler`` — no interpolation, no
    sklearn dependency beyond ``BaseEstimator`` — but coarser on small datasets.

    Args:
        feature_range: Desired ``(min, max)`` of transformed output.
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X_df = self._to_df(X)
        self.feature_names_in_ = list(X_df.columns)
        self.n_features_in_ = X_df.shape[1]
        self.sorted_values_ = {}
        for col in X_df.columns:
            vals = np.sort(X_df[col].dropna().to_numpy(dtype=float))
            self.sorted_values_[col] = vals
        return self

    def transform(self, X):
        check_is_fitted(self, "sorted_values_")
        X_df = self._to_df(X)
        out = np.empty_like(X_df.to_numpy(dtype=float))
        lo, hi = self.feature_range
        for i, col in enumerate(self.feature_names_in_):
            vals = self.sorted_values_[col]
            n = len(vals)
            ranks = np.searchsorted(vals, X_df[col].to_numpy(dtype=float), side="right")
            cdf = np.clip(ranks / n, 0.0, 1.0)
            out[:, i] = cdf * (hi - lo) + lo
        return out

    def inverse_transform(self, X):
        check_is_fitted(self, "sorted_values_")
        X_arr = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        cdf = np.clip((X_arr - lo) / (hi - lo), 0.0, 1.0)
        out = np.empty_like(X_arr)
        for i, col in enumerate(self.feature_names_in_):
            vals = self.sorted_values_[col]
            n = len(vals)
            indices = np.clip((cdf[:, i] * n).astype(int), 0, n - 1)
            out[:, i] = vals[indices]
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "sorted_values_")
        return np.asarray(self.feature_names_in_, dtype=object)

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            names = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
        return pd.DataFrame(np.asarray(X, dtype=float), columns=names)


class PiecewiseLinearCDFScaler(TransformerMixin, BaseEstimator):
    """Piecewise-linear CDF approximation — ``n_pieces`` affine maps per feature.

    Each feature's training distribution is partitioned into ``n_pieces``
    equal-probability segments (at the quantile boundaries), and each segment
    is mapped by a single affine function to a corresponding equal-width slice
    of ``feature_range``.  The result is:

    * **Piecewise affine** — each segment is ``y = a*x + b``, the simplest
      possible nonlinear-overall transform built from affine pieces.
    * **Monotone and invertible** — the slopes are all non-negative (strictly
      positive except on constant segments).
    * **Approximately uniformity-preserving** — with enough pieces, the
      marginal distribution of the output is close to uniform on
      ``feature_range``.

    ``n_pieces`` controls the fidelity/smoothness trade-off:

    * Large ``n_pieces`` ≈ full empirical CDF (many short affine segments).
    * Small ``n_pieces`` ≈ coarse histogram equalisation (few long segments).
    * ``n_pieces=1`` degenerates to plain min-max scaling.

    This is the *"affine maps which aid in FIS MF placement"* idiom: the
    transform is interpretable (plot the breakpoints), invertible (read off
    the original-space value), and the membership functions placed on the
    transformed space cover equal probability mass by construction.

    Args:
        feature_range: Desired ``(min, max)`` of transformed output.
        n_pieces: Number of affine segments per feature.
    """

    def __init__(self, feature_range=(0.0, 1.0), n_pieces=10):
        self.feature_range = feature_range
        self.n_pieces = n_pieces

    def fit(self, X, y=None):
        X_df = self._to_df(X)
        self.feature_names_in_ = list(X_df.columns)
        self.n_features_in_ = X_df.shape[1]

        self.breakpoints_ = {}
        for col in X_df.columns:
            vals = X_df[col].dropna().to_numpy(dtype=float)
            quantiles = np.linspace(0, 100, self.n_pieces + 1)
            bp = np.percentile(vals, quantiles)
            # Deduplicate breakpoints from constant segments while keeping
            # the first and last.
            bp = np.unique(bp)
            self.breakpoints_[col] = bp
        return self

    def transform(self, X):
        check_is_fitted(self, "breakpoints_")
        X_df = self._to_df(X)
        lo, hi = self.feature_range
        out = np.empty((len(X_df), self.n_features_in_), dtype=float)

        for i, col in enumerate(self.feature_names_in_):
            bp = self.breakpoints_[col]
            x = X_df[col].to_numpy(dtype=float)
            n_seg = len(bp) - 1
            if n_seg <= 0:
                out[:, i] = (lo + hi) / 2.0
                continue

            # Target breakpoints: equal-width slices of [lo, hi]
            tbp = np.linspace(lo, hi, n_seg + 1)

            # For each value, find its segment and apply the affine map.
            seg = np.clip(np.searchsorted(bp, x, side="right") - 1, 0, n_seg - 1)
            x_lo = bp[seg]
            x_hi = bp[seg + 1]
            t_lo = tbp[seg]
            t_hi = tbp[seg + 1]

            span = x_hi - x_lo
            safe_span = np.where(span > 0, span, 1.0)
            frac = np.clip((x - x_lo) / safe_span, 0.0, 1.0)
            out[:, i] = t_lo + frac * (t_hi - t_lo)

        return out

    def inverse_transform(self, X):
        check_is_fitted(self, "breakpoints_")
        X_arr = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        out = np.empty_like(X_arr)

        for i, col in enumerate(self.feature_names_in_):
            bp = self.breakpoints_[col]
            n_seg = len(bp) - 1
            if n_seg <= 0:
                out[:, i] = bp[0] if len(bp) > 0 else 0.0
                continue

            tbp = np.linspace(lo, hi, n_seg + 1)
            t = X_arr[:, i]
            seg = np.clip(np.searchsorted(tbp, t, side="right") - 1, 0, n_seg - 1)
            t_lo = tbp[seg]
            t_hi = tbp[seg + 1]
            x_lo = bp[seg]
            x_hi = bp[seg + 1]

            t_span = t_hi - t_lo
            safe_t_span = np.where(t_span > 0, t_span, 1.0)
            frac = np.clip((t - t_lo) / safe_t_span, 0.0, 1.0)
            out[:, i] = x_lo + frac * (x_hi - x_lo)

        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "breakpoints_")
        return np.asarray(self.feature_names_in_, dtype=object)

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            names = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
        return pd.DataFrame(np.asarray(X, dtype=float), columns=names)
