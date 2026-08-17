"""The arms under comparison, and the metrics they are scored on.

One FIS (`TribbleRegressor`, the DOE's own estimator), one ReLU network
(`experiments/fis-to-neural-net/fis2nn.py`, reused unmodified), and the four
initializations that network can be given: He-random, per-feature quantile
knots, random features, and the FIS-derived analytic seed.

The network is the *same architecture* in every arm -- one hidden ReLU layer
plus a linear skip, trained by the same NumPy Adam loop -- so the only variable
between arms is where layer 1's knots came from and what the read-out was
initialized to. That is the comparison the fis-to-neural-net study defines, run
here on a dataset it never saw.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "experiments", "fis-to-neural-net"))

import fis2nn  # noqa: E402

from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def nasa_score(y_true, y_pred) -> float:
    """The PHM08 asymmetric penalty, summed. Under-prediction (late warning is
    the dangerous direction) is penalized at exp(|d|/10), over-prediction at
    exp(|d|/13). It is a *sum*, so it scales with the number of scored rows --
    only compare it between arms scored on identical rows."""
    delta = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    alpha = np.where(delta > 0, 1.0 / 13.0, 1.0 / 10.0)
    return float(np.sum(np.exp(alpha * np.abs(delta))))


def endpoint_rows(unit: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """Index of each engine's last recorded cycle -- the canonical C-MAPSS
    protocol scores one RUL per test engine, at the end of its trajectory."""
    df = pd.DataFrame({"unit": unit, "cycle": cycle}).reset_index()
    return df.sort_values(["unit", "cycle"]).groupby("unit")["index"].last().to_numpy()


def evaluate(split, y_pred: np.ndarray) -> dict:
    """Both scoring conventions the DOE reports, on one prediction vector."""
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true = split.y_true
    err = y_true - y_pred
    idx = endpoint_rows(split.unit, split.cycle)
    e_err = y_true[idx] - y_pred[idx]
    return dict(
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        nasa=nasa_score(y_true, y_pred),
        rmse_endpoint=float(np.sqrt(np.mean(e_err**2))),
        mae_endpoint=float(np.mean(np.abs(e_err))),
        nasa_endpoint=nasa_score(y_true[idx], y_pred[idx]),
        n=int(len(y_true)),
        n_engines=int(len(idx)),
    )


# ---------------------------------------------------------------------------
# The FIS
# ---------------------------------------------------------------------------
# The DOE's two published DS02 configurations, quoted from
# `cmapss_rul_best.py`. NOTE: both were selected on `rmse_test_true` -- the
# official held-out test engines -- so they are test-selected configurations.
# They are carried here as the reference the network is compared against, and
# `sweep_fis.py` re-selects a config on the validation engines instead so the
# comparison has an arm that was chosen honestly on both sides.
FIS_CONFIGS = {
    "honest": dict(
        tsk_order="1st",
        n_gaussians=0,
        top_p=0.9,
        detect_interactions=False,
        norm_conorm="hamacher",
        l2_reg=0.01,
    ),
    "best": dict(
        tsk_order="full-2nd",
        n_gaussians=0,
        top_p=0.95,
        detect_interactions=False,
        norm_conorm="hamacher",
        l2_reg=0.01,
    ),
    # `memory18` is `best`'s configuration on the strict 18-sensor memory
    # pipeline (see cmapss_data.BUNDLES) -- the FIS-quality recommendation.
    "memory18": dict(
        tsk_order="full-2nd",
        n_gaussians=0,
        top_p=0.95,
        detect_interactions=False,
        norm_conorm="hamacher",
        l2_reg=0.01,
    ),
}


def as_frame(X: np.ndarray, feature_names) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(X, dtype=float), columns=list(feature_names))


def fit_fis(X, y, feature_names, seed: int = 42, **kwargs):
    """Fit a TribbleRegressor on a named frame and return it with its fit time.

    Passing a DataFrame rather than an array is what lets the conversion work
    later: `fis_knots` is keyed by feature name, and `analytic_seed_from_fis`
    needs to hand the regressor frames with those same columns.
    """
    Xdf = as_frame(X, feature_names)
    ydf = pd.Series(np.asarray(y, dtype=float))
    model = TribbleRegressor(random_state=seed, max_samples=2000, **kwargs)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(Xdf, ydf)
    return model, time.perf_counter() - t0


def fis_predict(model, X, feature_names) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        return np.asarray(
            model.predict(as_frame(X, feature_names)), dtype=float
        ).ravel()


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------
class Conversion:
    """A fitted FIS, its ReLU knots, and the seed network backed out of it.

    Holds the feature-subspace bookkeeping in one place: the FIS selects a
    subset of columns (`top_features_`), the seed network is built over exactly
    that subset, and every hot arm has to be fed `X[:, feature_index]` rather
    than the full matrix. Getting that wrong silently trains a network on the
    wrong columns, which is why it lives here rather than at each call site.
    """

    def __init__(
        self,
        model,
        feature_names,
        X_train,
        y_center,
        y_scale,
        seed=0,
        background: int = 256,
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.features = list(model.top_features_)
        self.index = np.array(
            [self.feature_names.index(f) for f in self.features], dtype=int
        )
        self.y_center = float(y_center)
        self.y_scale = float(y_scale)

        Xdf = as_frame(X_train, feature_names)
        # `fis_knots` reads membership functions, so it wants the mixture model
        # the regressor wraps -- not the regressor.
        self.knots = fis2nn.fis_knots(model.model_, self.features)
        self.n_hidden = int(sum(self.knots[f].size for f in self.features))

        def scaled_fis(frame):
            with contextlib.redirect_stdout(io.StringIO()):
                raw = np.asarray(model.predict(frame), dtype=float).ravel()
            return (raw - self.y_center) / self.y_scale

        t0 = time.perf_counter()
        self.net = fis2nn.analytic_seed_from_fis(
            scaled_fis,
            Xdf,
            self.features,
            self.knots,
            background_size=background,
            seed=seed,
        )
        self.analytic_seconds = time.perf_counter() - t0

    def subspace(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, self.index]


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------
def make_starts(
    conv, X_fit, y_fit_scaled, n_hidden=None, l2=1e-6, seed=0, arms=None
) -> tuple[dict, dict, dict]:
    """Build every initialization on the *training* rows only.

    Returns (nets, setup_seconds, feature_space): the last says whether an arm's
    network reads the FIS's selected columns ("fis") or all of them ("all"), so
    the caller feeds it the right matrix at train and predict time.

    `setup_seconds` charges each arm what it actually cost to get to its
    starting point. The hot arms carry the FIS fit and the conversion; the
    closed-form arms carry their own ridge solve; `he` carries nothing. This is
    the accounting that decides whether a warm start can ever repay itself.
    """
    all_arms = (
        "he",
        "quantile",
        "elm",
        "hot-analytic",
        "hot",
        "he-all",
        "quantile-all",
    )
    arms = tuple(arms) if arms is not None else all_arms
    n_hidden = int(n_hidden or conv.n_hidden)
    Xs = conv.subspace(X_fit)
    Xa = np.asarray(X_fit, dtype=float)
    y = np.asarray(y_fit_scaled, dtype=float).ravel()
    rng = np.random.default_rng(1000 + seed)

    nets, secs, space = {}, {}, {}

    def timed(name, build, where):
        t0 = time.perf_counter()
        nets[name] = build()
        secs[name] = time.perf_counter() - t0
        space[name] = where

    if "he" in arms:
        timed("he", lambda: fis2nn.he_start(rng, Xs.shape[1], n_hidden), "fis")
    if "quantile" in arms:
        timed("quantile", lambda: fis2nn.quantile_start(Xs, n_hidden, y, l2=l2), "fis")
    if "elm" in arms:
        timed(
            "elm",
            lambda: fis2nn.random_feature_start(rng, Xs, y, n_hidden, l2=l2),
            "fis",
        )
    if "hot-analytic" in arms:
        # The seed exactly as the conversion produced it: no labels, ever.
        nets["hot-analytic"] = conv.net.copy()
        secs["hot-analytic"] = conv.analytic_seconds
        space["hot-analytic"] = "fis"
    if "hot" in arms:
        # ...and the same seed given one anchored ridge solve against the labels.
        t0 = time.perf_counter()
        nets["hot"] = fis2nn.solve_readout(conv.net, Xs, y, l2=l2)
        secs["hot"] = conv.analytic_seconds + (time.perf_counter() - t0)
        space["hot"] = "fis"
    if "he-all" in arms:
        timed("he-all", lambda: fis2nn.he_start(rng, Xa.shape[1], n_hidden), "all")
    if "quantile-all" in arms:
        timed(
            "quantile-all", lambda: fis2nn.quantile_start(Xa, n_hidden, y, l2=l2), "all"
        )

    return nets, secs, space


def matrices(conv, space: str, *Xs):
    """Project each matrix into an arm's feature space."""
    if space == "fis":
        return [conv.subspace(X) for X in Xs]
    return [np.asarray(X, dtype=float) for X in Xs]
