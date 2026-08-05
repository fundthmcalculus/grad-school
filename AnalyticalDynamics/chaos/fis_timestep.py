"""Fuzzy-inference-system reproduction of the time-step operator in arXiv:2504.13453.

The paper's "time-step based approach" is not autoregressive. It fits a direct
operator

    (theta_1(0), ..., theta_n(0), t)  ->  (theta_1(t), ..., theta_n(t))

trained on a grid of initial conditions and evaluated on an "in-between" one.
That framing is what makes a Takagi-Sugeno FIS a plausible drop-in for their ten
neural / ML models: there is no rollout, so there is no error accumulation, and
the target is a plain (if violently oscillatory) 2-input regression surface.

Two preprocessing choices in the reference notebooks are load-bearing and are
reproduced here rather than silently improved on:

  * The target is min-max scaled **per trajectory**, independently per angle
    column, before pooling. Every trajectory therefore spans exactly [0, 1] in
    every output. A reported RMSE of 0.027 is 2.7% of that trajectory's own
    angular range, not 0.027 degrees.
  * Inputs are min-max scaled globally over the pooled training rows.

Both scalers are **unclipped**. The input scaler maps the training window's t to
[0, 1] and is then asked for t up to 20 s, which it maps to 2.0 rather than
saturating at 1.0 -- so a prediction past the training window diverges instead of
freezing at its t = 10 s value, and the failure is visible rather than disguised as
a plateau. The target scaler is fitted over each trajectory's full extent, the
holdout's included, so scaled truth stays in [0, 1] across all 20 s.

Because scaled RMSE is only interpretable relative to a trajectory's range, every
metric below is also reported in degrees.

Metric families reported (see `evaluate`):

  pooled     -- the paper's own split: rows pooled across all training initial
                conditions, random 80/20. Interleaves 5 ms-apart neighbours
                between train and test, so it flatters any smooth model.
  trained_ic -- all 2000 rows of a trajectory that was in training (the paper's
                Fig. 11 / 12 / 18B / 18C setting).
  holdout_ic -- all 2000 rows of the never-trained [120, ..., 2.05] trajectory
                (the paper's Fig. 13 / 18D setting). This is the only number
                that tests generalisation to a new initial condition.
"""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "tribble-fis" / "src"))

from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor  # noqa: E402

import pendulum_data as pdata  # noqa: E402

DATA_DIR = HERE / "data"
RESULT_DIR = HERE / "results"
FIG_DIR = HERE / "figures"

TRAINED_IC_THETA_DEG = 0.0  # the paper's "already trained on" case: [120, 0(, 0)]


@contextlib.contextmanager
def _quiet():
    """Swallow library stdout chatter without touching the submodule."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Split:
    """One (n_links, friction) problem in the paper's time-step layout."""

    n_links: int
    friction: bool
    t: np.ndarray  # (n_steps,)
    ic_deg: np.ndarray  # (n_ics, n_links)
    theta_deg: np.ndarray  # (n_ics, n_steps, n_links)   raw degrees
    theta_scaled: np.ndarray  # (n_ics, n_steps, n_links)   per-trajectory [0, 1]
    ranges: np.ndarray  # (n_ics, n_links, 2)  the (min, max) used per trajectory
    holdout_ic_deg: np.ndarray  # (n_links,)
    holdout_theta_deg: np.ndarray  # (n_test_steps, n_links)
    holdout_theta_scaled: np.ndarray
    holdout_range: np.ndarray  # (n_links, 2)
    holdout_t: np.ndarray  # (n_test_steps,)  runs to 20 s, twice the training window
    train_t_end: float = pdata.T_END  # where the training window stops

    @property
    def label(self):
        return pdata.dataset_label(self.n_links, self.friction)

    @property
    def in_window(self):
        """Boolean mask over holdout_t: samples inside the training time window."""
        return self.holdout_t < self.train_t_end

    @property
    def swept_index(self):
        """Which initial-angle column the sweep actually varies."""
        return int(np.argmax(np.ptp(self.ic_deg, axis=0)))


def _scale_per_trajectory(theta_deg):
    """Per-trajectory, per-column min-max to [0, 1]. Returns (scaled, ranges)."""
    lo = theta_deg.min(axis=-2, keepdims=True)
    hi = theta_deg.max(axis=-2, keepdims=True)
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    scaled = (theta_deg - lo) / span
    ranges = np.stack([lo.squeeze(-2), hi.squeeze(-2)], axis=-1)
    return scaled, ranges


def load(n_links, friction):
    label = pdata.dataset_label(n_links, friction)
    train = np.load(DATA_DIR / f"{label}.npz")
    hold = np.load(DATA_DIR / f"{label}_holdout.npz")

    theta_deg = train["theta_deg"]
    scaled, ranges = _scale_per_trajectory(theta_deg)

    h_deg = hold["theta_deg"][0]
    h_t = hold["t"]
    train_t_end = float(hold["train_t_end"]) if "train_t_end" in hold else pdata.T_END

    # The holdout target scaler is fitted on the *training window* only -- the
    # holdout's first 10 s -- and then applied to all 20 s.
    #
    # Per-trajectory min-max only makes training and test commensurable when both
    # are normalised over the same duration. Training targets span exactly [0, 1]
    # by construction, so fitting the holdout's scaler over its full 20 s would
    # leave its first 10 s spanning only [0, 0.678] on the frictionless double
    # pendulum, and a model trained to emit over [0, 1] would overshoot by ~1.5x
    # for reasons having nothing to do with its dynamics. Fitting on the window
    # keeps the two commensurable and keeps every in-window number comparable to
    # the 10 s protocol.
    #
    # It also leaks nothing new: the protocol already hands the model the test
    # trajectory's own min and max (METHOD_AND_PARAMETERS.md section 3), whereas
    # fitting over 20 s would additionally leak the range of the region being
    # extrapolated into. Scaled truth beyond 10 s may therefore fall outside
    # [0, 1], which is correct -- the chain genuinely leaves the window it was
    # normalised against.
    #
    # Friction datasets are indifferent to the choice (their 20 s range equals
    # their 10 s range bitwise); only the frictionless ones move.
    window = h_t < train_t_end
    _, h_range = _scale_per_trajectory(h_deg[window])
    lo, hi = h_range[:, 0][None, :], h_range[:, 1][None, :]
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    h_scaled = (h_deg - lo) / span

    return Split(
        n_links=n_links,
        friction=friction,
        t=train["t"],
        ic_deg=train["ic_deg"],
        theta_deg=theta_deg,
        theta_scaled=scaled,
        ranges=ranges,
        holdout_ic_deg=hold["ic_deg"][0],
        holdout_theta_deg=h_deg,
        holdout_theta_scaled=h_scaled,
        holdout_range=h_range,
        holdout_t=h_t,
        train_t_end=train_t_end,
    )


# ---------------------------------------------------------------------------
# Feature encodings
# ---------------------------------------------------------------------------
def encode(ic_deg_rows, t, encoding="raw", n_harmonics=0):
    """Build the input matrix for one or more trajectories.

    ic_deg_rows : (n_ics, n_links) initial angles in degrees
    t           : (n_steps,) sample times

    Returns (X, names). Rows are ordered trajectory-major, matching
    ``theta_scaled.reshape(-1, n_links)``.

    ``raw`` is the paper's own encoding: initial angles in radians plus t. The
    harmonic encodings append sin/cos of k*omega_0*t. They exist because the
    target oscillates ~20 times over the 10 s window while a TSK consequent is
    affine in its inputs: without a periodic basis, a rule can only ever draw a
    straight line through a full cycle. omega_0 is the small-angle frequency
    sqrt(g/l), which is the natural scale for this chain and involves no
    knowledge of the trajectories being fit.

    Measured: the idea does not work. At 40 rules per output on the frictionless
    double pendulum, trained-IC R^2 goes 0.864 (raw, 2 inputs) -> 0.668 (K=8, 18
    inputs) -> 0.489 (K=16, 34) -> -2.600 (K=24, 50). Firing strength is a t-norm
    product over one membership per feature, so widening the input vector drives
    every product toward zero and the normalised weights become numerical noise.
    Antecedent dimensionality is the binding constraint, not consequent order.
    Kept because the negative result is worth more than the idea was.
    """
    ic = np.asarray(ic_deg_rows, dtype=float).reshape(-1, np.shape(ic_deg_rows)[-1])
    n_ics, n_links = ic.shape
    n_steps = t.size

    ic_rad = np.deg2rad(ic)
    blocks = [np.repeat(ic_rad, n_steps, axis=0)]
    names = [f"theta{j + 1}_init" for j in range(n_links)]

    tt = np.tile(t, n_ics)[:, None]
    blocks.append(tt)
    names.append("t")

    if encoding == "raw":
        pass
    elif encoding == "harmonic":
        w0 = np.sqrt(pdata.G / pdata.L1)
        for k in range(1, n_harmonics + 1):
            blocks.append(np.sin(k * w0 * tt))
            blocks.append(np.cos(k * w0 * tt))
            names += [f"sin{k}w0t", f"cos{k}w0t"]
    else:
        raise ValueError(f"unknown encoding {encoding!r}")

    X = np.hstack(blocks)
    return X, names


def _drop_constant(X, names):
    """Drop zero-variance columns.

    theta_1(0) is fixed at 120 deg for every trajectory in the sweep (and
    theta_2(0) is fixed at 0 for the triple pendulum), so those columns carry no
    information. Gaussian membership fitting on a zero-variance feature yields
    sigma = 0 and a degenerate firing strength, so they must go. The reference
    notebooks feed them in regardless; a neural net can absorb a dead input, a
    fuzzy partition cannot.
    """
    keep = np.ptp(X, axis=0) > 0
    return X[:, keep], [n for n, k in zip(names, keep) if k], keep


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class FisConfig:
    """Hyperparameters for one collection-of-FIS model (one FIS per output)."""

    n_output_buckets: int = 40
    tsk_order: str = "1st"
    n_gaussians: int = 0
    output_partition: str = "uniform"
    consequent_basis: str = "raw"
    l2_reg: float = 1e-6
    norm_conorm: str = "probability"
    encoding: str = "raw"
    n_harmonics: int = 0
    #: 1.0 keeps every engineered feature. The library default of 0.95 drops any
    #: feature whose normalized differentiation score falls below 0.05, which
    #: silently discards high-order harmonics we deliberately added.
    top_p: float = 1.0
    random_state: int = 42

    def key(self):
        base = (
            f"nb{self.n_output_buckets}_{self.tsk_order}_g{self.n_gaussians}"
            f"_{self.output_partition}_{self.consequent_basis}_{self.norm_conorm}"
        )
        if self.encoding != "raw":
            base += f"_{self.encoding}{self.n_harmonics}"
        if self.l2_reg != 1e-6:
            base += f"_l2{self.l2_reg:g}"
        return base


class FisOperator:
    """A collection of TSK fuzzy inference systems, one per output angle.

    Wraps `MixtureOfGaussiansFuzzyRegressor` per output rather than using
    `MimoGaussianPredictor`, because that wrapper does not forward
    output_partition / l2_reg / consequent_basis / norm arguments and would
    silently substitute library defaults for anything swept here.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.models_ = []
        self.x_scaler_ = None
        self.keep_ = None
        self.names_ = None

    def _make(self):
        c = self.cfg
        return MixtureOfGaussiansFuzzyRegressor(
            n_output_buckets=c.n_output_buckets,
            tsk_order=c.tsk_order,
            n_gaussians=c.n_gaussians,
            output_partition=c.output_partition,
            consequent_basis=c.consequent_basis,
            l2_reg=c.l2_reg,
            norm_conorm=c.norm_conorm,
            top_p=c.top_p,
            random_state=c.random_state,
        )

    def fit(self, X_raw, names, Y):
        X, self.names_, self.keep_ = _drop_constant(X_raw, names)
        # clip=False is sklearn's default and is stated here because it is a load-
        # bearing choice, not an oversight: at t = 20 s this returns 2.0, putting
        # the query outside every Gaussian membership's support. With clip=True the
        # model would silently return its t = 10 s answer forever and the
        # extrapolation failure would look like a stable plateau.
        self.x_scaler_ = MinMaxScaler(clip=False).fit(X)
        Xs = self.x_scaler_.transform(X)
        self.models_ = []
        # gauss_math.rank_feature_differentiators prints its ranking table
        # unconditionally and the library is a submodule we do not edit, so the
        # ranking goes to the fit log instead of the console.
        with _quiet():
            for j in range(Y.shape[1]):
                m = self._make()
                m.fit(Xs, Y[:, j])
                self.models_.append(m)
        return self

    def predict(self, X_raw):
        Xs = self.x_scaler_.transform(X_raw[:, self.keep_])
        return np.column_stack([m.predict(Xs) for m in self.models_])

    @property
    def n_rules(self):
        return self.cfg.n_output_buckets * len(self.models_)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(y_true_scaled, y_pred_scaled, rng):
    """RMSE / R^2 in both scaled units and degrees.

    rng is (n_links, 2) of the (min, max) degrees used to scale this trajectory;
    multiplying a scaled residual by (max - min) puts it back in degrees.
    """
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))),
        "r2": float(r2_score(y_true_scaled, y_pred_scaled, multioutput="uniform_average")),
    }
    span = (rng[:, 1] - rng[:, 0])[None, :]
    resid_deg = (y_pred_scaled - y_true_scaled) * span
    out["rmse_deg"] = float(np.sqrt(np.mean(resid_deg**2)))
    per = np.sqrt(np.mean(resid_deg**2, axis=0))
    for j, v in enumerate(per):
        out[f"rmse_deg_theta{j + 1}"] = float(v)
    r2_each = r2_score(y_true_scaled, y_pred_scaled, multioutput="raw_values")
    for j, v in enumerate(np.atleast_1d(r2_each)):
        out[f"r2_theta{j + 1}"] = float(v)
    return out


def horizon_metrics(split, y_true_scaled, y_pred_scaled):
    """Split a 20 s holdout score into the in-window and extrapolated halves.

    Returns {"holdout": ..., "in_window": ..., "extrap": ..., "t_break": ...}.

    `holdout` covers 0-10 s and is the number every earlier table reports, so it
    stays directly comparable. `extrap` covers 10-20 s, where both the target time
    and the target range lie outside anything the model saw. `t_break` is the first
    time the absolute error exceeds 10% of the training-window range and stays
    above it, i.e. how far the model actually generalises, in seconds -- inf if it
    never does.
    """
    mask = split.in_window
    rng = split.holdout_range
    out = {
        "holdout": _metrics(y_true_scaled[mask], y_pred_scaled[mask], rng),
        "in_window": _metrics(y_true_scaled[mask], y_pred_scaled[mask], rng),
        "extrap": _metrics(y_true_scaled[~mask], y_pred_scaled[~mask], rng),
        "full": _metrics(y_true_scaled, y_pred_scaled, rng),
    }
    err = np.max(np.abs(y_pred_scaled - y_true_scaled), axis=1)
    over = err > 0.10
    # First index from which it never recovers below the threshold.
    never_back = np.flatnonzero(over & (np.cumsum(~over[::-1])[::-1] == 0))
    if never_back.size:
        out["t_break"] = float(split.holdout_t[never_back[0]])
    else:
        first = np.flatnonzero(over)
        out["t_break"] = float(split.holdout_t[first[0]]) if first.size else float("inf")
    return out


@dataclass
class Result:
    label: str
    config: str
    pooled: dict = field(default_factory=dict)
    trained_ic: dict = field(default_factory=dict)
    holdout_ic: dict = field(default_factory=dict)
    extrap_ic: dict = field(default_factory=dict)
    t_break: float = float("nan")
    n_rules: int = 0
    fit_seconds: float = 0.0

    def flat(self):
        row = {"dataset": self.label, "config": self.config, "n_rules": self.n_rules,
               "fit_seconds": round(self.fit_seconds, 2), "t_break": self.t_break}
        for fam, d in (("pooled", self.pooled), ("trained", self.trained_ic),
                       ("holdout", self.holdout_ic), ("extrap", self.extrap_ic)):
            for k, v in d.items():
                row[f"{fam}_{k}"] = v
        return row


def build_pooled(split, cfg):
    """Full pooled design matrix and target for a split."""
    X, names = encode(split.ic_deg, split.t, cfg.encoding, cfg.n_harmonics)
    Y = split.theta_scaled.reshape(-1, split.n_links)
    return X, names, Y


def run(split, cfg, test_fraction=0.2, seed=42, verbose=False):
    """Fit one FIS collection and evaluate all three metric families."""
    import time

    X, names, Y = build_pooled(split, cfg)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_train = int((1.0 - test_fraction) * X.shape[0])
    tr, te = perm[:n_train], perm[n_train:]

    model = FisOperator(cfg)
    t0 = time.perf_counter()
    model.fit(X[tr], names, Y[tr])
    fit_s = time.perf_counter() - t0

    # Pooled rows come from 31 trajectories with different angular ranges, so
    # there is no single correct degree conversion for them. The mean range makes
    # `pooled_rmse_deg` an order-of-magnitude figure only; `pooled_rmse` (scaled)
    # is the exact quantity. The trained_ic and holdout_ic degree figures below
    # each use their own trajectory's range and are exact.
    mean_range = split.ranges.mean(axis=0)

    res = Result(split.label, cfg.key(), n_rules=model.n_rules, fit_seconds=fit_s)
    res.pooled = _metrics(Y[te], model.predict(X[te]), mean_range)

    k = int(np.argmin(np.abs(split.ic_deg[:, split.swept_index] - TRAINED_IC_THETA_DEG)))
    Xk, _ = encode(split.ic_deg[k: k + 1], split.t, cfg.encoding, cfg.n_harmonics)
    res.trained_ic = _metrics(split.theta_scaled[k], model.predict(Xk), split.ranges[k])

    # The holdout is evaluated over the full 20 s, then split at the training
    # window edge: `holdout_ic` stays the 0-10 s number every earlier table quotes,
    # `extrap_ic` is the 10-20 s continuation.
    Xh, _ = encode(split.holdout_ic_deg[None, :], split.holdout_t,
                   cfg.encoding, cfg.n_harmonics)
    seg = horizon_metrics(split, split.holdout_theta_scaled, model.predict(Xh))
    res.holdout_ic = seg["holdout"]
    res.extrap_ic = seg["extrap"]
    res.t_break = seg["t_break"]

    if verbose:
        print(
            f"  {split.label:28s} {cfg.key():44s} "
            f"pooled R2={res.pooled['r2']:.4f} "
            f"trained R2={res.trained_ic['r2']:.4f} RMSE={res.trained_ic['rmse']:.4e} "
            f"holdout R2={res.holdout_ic['r2']:.4f} RMSE={res.holdout_ic['rmse']:.4e} "
            f"extrap R2={res.extrap_ic['r2']:.4f} t_break={res.t_break:.2f}s "
            f"({fit_s:.1f}s)"
        )
    return res, model


def baseline_bracket_midpoint(split, lower_deg=2.0, upper_deg=2.1):
    """No-learning reference: average the two bracketing trained trajectories.

    The holdout IC sits exactly between two ICs that *are* in the training grid,
    so the cheapest possible predictor is the mean of their two scaled
    trajectories. It fits nothing and has no parameters. Scored in exactly the
    metric the paper reports, so it is directly comparable to every row of the
    paper's tables -- and on the friction problems it beats all of them.

    Reported because a learned model that cannot beat this baseline has not been
    shown to have learned the dynamics, only to have interpolated the grid.

    Scored over the training window only. Both baselines are built out of training
    trajectories, which stop at 10 s, so neither has any value to offer beyond it --
    a fact worth stating rather than papering over: on the 10-20 s extrapolation
    the no-learning baselines do not merely score badly, they do not exist.
    """
    swept = split.ic_deg[:, split.swept_index]
    i_lo = int(np.argmin(np.abs(swept - lower_deg)))
    i_hi = int(np.argmin(np.abs(swept - upper_deg)))
    assert abs(swept[i_lo] - lower_deg) < 1e-6 and abs(swept[i_hi] - upper_deg) < 1e-6, (
        f"{split.label}: bracketing ICs {lower_deg}/{upper_deg} are not in the training grid"
    )
    pred = 0.5 * (split.theta_scaled[i_lo] + split.theta_scaled[i_hi])
    return _metrics(split.holdout_theta_scaled[split.in_window], pred, split.holdout_range)


def baseline_nearest(split):
    """No-learning reference: copy the single nearest trained trajectory.

    Training-window only, for the reason in `baseline_bracket_midpoint`.
    """
    swept = split.ic_deg[:, split.swept_index]
    target = split.holdout_ic_deg[split.swept_index]
    k = int(np.argmin(np.abs(swept - target)))
    return _metrics(split.holdout_theta_scaled[split.in_window],
                    split.theta_scaled[k], split.holdout_range)


def predictions_for(split, model, cfg, which="holdout"):
    """Scaled predictions plus ground truth for plotting.

    The holdout runs the full 20 s; the trained IC stays at the 10 s it was fitted
    on. `train_t_end` tells the plotting code where to draw the window edge, and is
    None when the whole series is inside the training window.
    """
    if which == "holdout":
        ic = split.holdout_ic_deg[None, :]
        truth, rng, t = split.holdout_theta_scaled, split.holdout_range, split.holdout_t
        t_end = split.train_t_end
    else:
        k = int(np.argmin(np.abs(split.ic_deg[:, split.swept_index] - TRAINED_IC_THETA_DEG)))
        ic = split.ic_deg[k: k + 1]
        truth, rng, t = split.theta_scaled[k], split.ranges[k], split.t
        t_end = None
    X, _ = encode(ic, t, cfg.encoding, cfg.n_harmonics)
    pred = model.predict(X)
    span = (rng[:, 1] - rng[:, 0])[None, :]
    lo = rng[:, 0][None, :]
    return {
        "t": t,
        "truth_scaled": truth,
        "pred_scaled": pred,
        "truth_deg": truth * span + lo,
        "pred_deg": pred * span + lo,
        "ic_deg": ic[0],
        "train_t_end": t_end,
    }


if __name__ == "__main__":
    split = load(2, friction=False)
    print(f"{split.label}: {split.theta_deg.shape}, swept column {split.swept_index}")
    print(f"  holdout IC {split.holdout_ic_deg}, range {split.holdout_range.tolist()}")
    for nb in (20, 40):
        run(split, FisConfig(n_output_buckets=nb), verbose=True)
