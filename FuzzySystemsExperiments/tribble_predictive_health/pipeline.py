"""`TribblePredictiveHealth` -- the whole RUL processing engine as one estimator.

A sibling to `TribbleRegressor` in spirit: a scikit-learn estimator you
construct with hyperparameters and then `fit` / `predict`. The difference is
what it consumes. `TribbleRegressor` takes a ready feature matrix; this takes a
raw run-to-failure sensor stream and owns everything between it and a RUL curve:

    condition correction  ->  memory / whole-cycle features  ->  RUL cap
                          ->  a TribbleRegressor  ->  monotone clamp

So the pipeline that used to be a page of loose functions is now one object you
can import, configure, fit on one dataset, reuse on another, or drop into a
service. `fit(X, y)` learns from a training stream (X carries the engine, cycle
and health columns; y is the per-sample RUL); `predict(X)` returns the RUL
trajectory for a new stream; `score(X, y)` reports both scoring conventions.

Nothing here is N-CMAPSS-specific except the default column names -- point the
`*_col` parameters (or `condition_cols` / `sensor_cols`) at any dataset shaped
the same way.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from tribblefis.gaussian_regressor import TribbleRegressor

from . import metrics
from .preprocessing import (
    apply_condition_correction,
    build_memory_features,
    build_whole_cycle_features,
    cap_rul,
    clamp_monotone,
    fit_condition_correction,
    onset_caps,
    per_cycle,
)


class TribblePredictiveHealth(BaseEstimator, RegressorMixin):
    """Remaining-useful-life from a run-to-failure sensor stream, end to end.

    Parameters mirror the pipeline's five steps plus the fuzzy system's own
    knobs. The defaults are the configuration that won the N-CMAPSS DS02
    design-of-experiments; `aggregation="whole_cycle"` with `tsk_order="1st"`
    is the variant that wins the per-engine metric instead.
    """

    def __init__(
        self,
        *,
        aggregation="raw_memory",
        monotone=True,
        condition_correction=True,
        condition_cols=None,
        sensor_cols=None,
        unit_col="unit",
        cycle_col="cycle",
        health_col="health",
        rul_col="rul",
        baseline_cycles=15,
        stride=200,
        window_size=5,
        memory_size=2,
        rul_ceiling=None,
        max_train_rows=None,
        tsk_order="full-2nd",
        n_gaussians=0,
        top_p=0.95,
        norm_conorm="hamacher",
        l2_reg=0.01,
        detect_interactions=False,
        select_interactions=False,
        interaction_top_p=0.95,
        n_output_buckets=2,
        member_function="gaussian",
        trapz_method="fast",
        trapz_width_reg=0.0,
        firing_exponent=1.0,
        max_samples=2000,
        random_state=42,
    ):
        self.aggregation = aggregation
        self.monotone = monotone
        self.condition_correction = condition_correction
        self.condition_cols = condition_cols
        self.sensor_cols = sensor_cols
        self.unit_col = unit_col
        self.cycle_col = cycle_col
        self.health_col = health_col
        self.rul_col = rul_col
        self.baseline_cycles = baseline_cycles
        self.stride = stride
        self.window_size = window_size
        self.memory_size = memory_size
        self.rul_ceiling = rul_ceiling
        self.max_train_rows = max_train_rows
        self.tsk_order = tsk_order
        self.n_gaussians = n_gaussians
        self.top_p = top_p
        self.norm_conorm = norm_conorm
        self.l2_reg = l2_reg
        self.detect_interactions = detect_interactions
        self.select_interactions = select_interactions
        self.interaction_top_p = interaction_top_p
        self.n_output_buckets = n_output_buckets
        self.member_function = member_function
        self.trapz_method = trapz_method
        self.trapz_width_reg = trapz_width_reg
        self.firing_exponent = firing_exponent
        self.max_samples = max_samples
        self.random_state = random_state

    # -- internals ----------------------------------------------------------
    def _resolve_columns(self, X):
        cond = self.condition_cols or [c for c in X.columns if c.startswith("W_")]
        sens = self.sensor_cols or [c for c in X.columns if c.startswith("Xs_")]
        if not cond or not sens:
            raise ValueError(
                "Could not find condition/sensor columns. Pass condition_cols "
                "and sensor_cols, or name them W_* and Xs_*."
            )
        return cond, sens

    def _build_features(self, df):
        common = dict(
            unit_col=self.unit_col,
            cycle_col=self.cycle_col,
            health_col=self.health_col,
            rul_col=self.rul_col,
        )
        if self.aggregation == "raw_memory":
            return build_memory_features(
                df,
                self.sensor_cols_,
                stride=self.stride,
                window_size=self.window_size,
                memory_size=self.memory_size,
                **common,
            )
        if self.aggregation == "whole_cycle":
            return build_whole_cycle_features(df, self.sensor_cols_, **common)
        raise ValueError(f"unknown aggregation {self.aggregation!r}")

    def _predict_matrix(self, table):
        Xs = self.scaler_.transform(table[self.feature_cols_].to_numpy(float))
        return self.regressor_.predict(Xs)

    def _featurize_test(self, X):
        df = X
        if self.condition_models_:
            df = apply_condition_correction(
                X, self.sensor_cols_, self.condition_cols_, self.condition_models_
            )
        table, _ = self._build_features(df)
        return table, self._predict_matrix(table)

    def _fit_table(self, table, feature_cols):
        """Steps 3(cap)-4 on an already-featurised training table: cap the RUL
        target, subsample if pooling, scale, and fit the fuzzy system. Shared by
        `fit` and `fit_featurized` so the two entry points can never diverge."""
        self.feature_cols_ = list(feature_cols)
        caps = onset_caps(
            table,
            unit_col=self.unit_col,
            cycle_col=self.cycle_col,
            health_col=self.health_col,
            rul_col=self.rul_col,
        )
        if self.max_train_rows and len(table) > self.max_train_rows:
            table = table.sample(self.max_train_rows, random_state=self.random_state)

        self.scaler_ = StandardScaler().fit(table[self.feature_cols_].to_numpy(float))
        X_train = self.scaler_.transform(table[self.feature_cols_].to_numpy(float))
        y_train = cap_rul(table, caps, unit_col=self.unit_col, rul_col=self.rul_col)
        if self.rul_ceiling is not None:
            # A constant RUL ceiling on top of the per-engine health-onset cap
            # (the classic C-MAPSS piecewise-linear-RUL trick): degradation is
            # only detectable in roughly the last `rul_ceiling` cycles, so
            # flattening the long healthy plateau removes samples the model
            # cannot fit anyway. Applied to the training target only; the test
            # target is scored uncapped.
            y_train = np.minimum(y_train, float(self.rul_ceiling))

        self.regressor_ = TribbleRegressor(
            random_state=self.random_state,
            max_samples=self.max_samples,
            tsk_order=self.tsk_order,
            n_gaussians=self.n_gaussians,
            top_p=self.top_p,
            norm_conorm=self.norm_conorm,
            l2_reg=self.l2_reg,
            detect_interactions=self.detect_interactions,
            select_interactions=self.select_interactions,
            interaction_top_p=self.interaction_top_p,
            n_output_buckets=self.n_output_buckets,
            member_function=self.member_function,
            trapz_method=self.trapz_method,
            trapz_width_reg=self.trapz_width_reg,
            firing_exponent=self.firing_exponent,
        )
        with contextlib.redirect_stdout(io.StringIO()):  # TRIBBLE is chatty
            self.regressor_.fit(X_train, y_train)
        self.n_rules_ = int(self.regressor_.model_.n_rules)
        return self

    # -- scikit-learn API ---------------------------------------------------
    def fit(self, X, y=None):
        self.condition_cols_, self.sensor_cols_ = self._resolve_columns(X)
        df = X.copy()
        if y is not None:
            df[self.rul_col] = np.asarray(y, dtype=float)

        # 1-2. condition correction, learned on the training engines' healthy
        # early cycles and applied to the whole stream. Skippable (`condition_
        # correction=False`) when the caller has already corrected the sensors --
        # e.g. pooling several datasets, each corrected against its own baseline.
        if self.condition_correction:
            self.condition_models_ = fit_condition_correction(
                df,
                self.sensor_cols_,
                self.condition_cols_,
                unit_col=self.unit_col,
                baseline_cycles=self.baseline_cycles,
            )
            df = apply_condition_correction(
                df, self.sensor_cols_, self.condition_cols_, self.condition_models_
            )
        else:
            self.condition_models_ = {}

        # 3(features)-4.
        table, feature_cols = self._build_features(df)
        return self._fit_table(table, feature_cols)

    def fit_featurized(self, table, feature_cols):
        """Fit on a table the caller has already condition-corrected and turned
        into features (pipeline steps 1-3) -- the entry point for pooling many
        datasets, each streamed, corrected against its own baseline, and
        featurised one at a time so peak memory stays near a single dataset.

        `table` carries the `unit_col`/`cycle_col`/`health_col`/`rul_col` columns
        (`unit_col` typically a globally-unique engine id); `feature_cols` names
        the model inputs. Pair with `predict_samples_featurized` / `score_
        featurized`, which likewise take an already-featurised table."""
        self.condition_models_ = {}
        return self._fit_table(table, feature_cols)

    def predict_frame(self, X, include_true=False):
        """The RUL trajectory for a stream: a DataFrame `[unit, cycle, rul]`,
        one row per flight cycle, monotone-clamped if `monotone` is set (step 5).
        This is the deployable output -- a curve that only ever falls. With
        `include_true=True` the cycle's actual RUL is carried in a `true` column
        (needs the target present in `X`)."""
        table, pred = self._featurize_test(X)
        true = None
        if include_true and self.rul_col in table.columns:
            true = table[self.rul_col].to_numpy(float)
        cyc = per_cycle(
            table[self.unit_col].to_numpy(),
            table[self.cycle_col].to_numpy(),
            pred,
            true=true,
        )
        if self.monotone:
            cyc = clamp_monotone(cyc)
        cyc = cyc.rename(columns={"pred": "rul"})
        keep = ["unit", "cycle", "rul"] + (["true"] if true is not None else [])
        return cyc[keep]

    def predict_samples(self, X):
        """The featurised table with a `predicted_rul` column -- one raw
        prediction per model-native row (per cycle for `whole_cycle`, per
        subsampled sample for `raw_memory`), before any per-cycle collapse or
        monotone clamp. For breaking a pooled score down by group."""
        table, pred = self._featurize_test(X)
        out = table.copy()
        out["predicted_rul"] = pred
        return out

    def predict(self, X):
        """Predicted RUL, one value per flight cycle (see :meth:`predict_frame`
        for the labelled version). Aggregating, so the output is per cycle, not
        per input row."""
        return self.predict_frame(X)["rul"].to_numpy()

    def score(self, X, y=None):
        """Every scoring convention on a held-out stream, as a flat dict (see
        :func:`metrics.score`): per-sample RMSE, raw and monotone per-cycle RMSE,
        and the canonical per-engine RMSE / NASA score. `y` (per-sample true RUL)
        may be passed or carried in `X`."""
        table, pred = self._featurize_test(X)
        return self._score_table(table, pred, y)

    def _score_table(self, table, pred, y=None):
        true = (
            np.asarray(y, float)
            if y is not None
            else table[self.rul_col].to_numpy(float)
        )
        return metrics.score(
            table[self.unit_col].to_numpy(),
            table[self.cycle_col].to_numpy(),
            true,
            pred,
        )

    # -- featurised-table entry points (pooling; see fit_featurized) ---------
    def predict_samples_featurized(self, table):
        """`predict_samples` for a table already condition-corrected and
        featurised: returns it with a raw `predicted_rul` column."""
        out = table.copy()
        out["predicted_rul"] = self._predict_matrix(table)
        return out

    def score_featurized(self, table, y=None):
        """`score` for a table already condition-corrected and featurised."""
        return self._score_table(table, self._predict_matrix(table), y)
