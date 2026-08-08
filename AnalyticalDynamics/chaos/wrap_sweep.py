"""Does bounding the angle representation help? A sweep over wrap limits.

The frictionless chains spin: theta_2 reaches 1501 deg on the double pendulum and
1589 deg on the quintuple. The benchmark's per-trajectory min-max scaling then
normalises against a range dominated by monotone accumulation rather than by
oscillation, and the target the FIS has to fit is a drifting quantity with no
bound. Wrapping the angle removes the drift.

Why wrapping is legitimate and clipping was not
-----------------------------------------------
The equations of motion depend on the angles only through sin and cos, so theta
and theta + 360 deg are the *same physical configuration*. Wrapping is therefore
an exact symmetry of the dynamics, not an approximation. That is the difference
from the input clipping tried in stable_extrapolation.py, which distorted the
dynamics at the boundary and made the rollout 25x worse: clipping changes the
physics, wrapping changes only its representation.

What "wrap to [-L, L]" means here
---------------------------------
A true modulo has period 360, so a window of width 480 ([-240, 240]) is not a
modulo and the phrase needs defining. This module uses *wrap-when-exceeded*:
subtract 360*sign(theta) repeatedly until |theta| <= L. It is stateless and
single-valued, and for L = 180 it reduces to the ordinary modulo.

For L > 180 there is an overlap band of width L - 180 on each side, inside which
a configuration may be represented either wrapped or unwrapped. That band is the
point of the exercise: a trajectory oscillating near the boundary crosses it
repeatedly under L = 180, injecting a 360 deg discontinuity on every crossing,
whereas a wider window lets the oscillation stay on one branch. The cost is a
wider target range. `discontinuities()` measures the first effect and the range
column measures the second, so the trade is visible rather than assumed.

Metrics
-------
Two are reported, because wrapping changes what an error means.

  rmse_scaled   the benchmark's own units: per-trajectory min-max on the wrapped
                target. Comparable between wrap settings, since each is scaled
                against its own representation.
  rmse_circ_deg circular angular error, ((pred - true + 180) mod 360) - 180, in
                degrees. This is the physically meaningful error and the only one
                comparable across every setting *including* no-wrap. It is more
                forgiving than the paper's metric -- a prediction of 430 deg
                against a truth of 790 deg scores zero, because they are the same
                configuration -- so it is reported alongside, never instead.

Run: python wrap_sweep.py
"""

from __future__ import annotations

import dataclasses
import json
import time

import numpy as np

import fis_timestep as fts

RESULT_DIR = fts.RESULT_DIR

#: Wrap limits in degrees; None is the no-wrap control and must be present.
WRAP_LIMITS = [None, 180.0, 240.0, 300.0, 360.0, 420.0]

#: Held fixed across wrap settings so the wrap is the only variable. Mid capacity
#: rather than the per-dataset winner, which differs between datasets and would
#: confound the comparison.
PROBE = fts.FisConfig(n_output_buckets=120, tsk_order="full-2nd", l2_reg=1e-9)

#: Same, but quantile-partitioned, for the representation comparison.
#: `output_partition="uniform"` crashes on sin/cos targets -- sin and cos of a
#: slowly-moving angle pile up near +-1, so an equal-width partition leaves empty
#: buckets and the library indexes past its own consequent array
#: (tribble-fis issue #82; reproduced here at 120 rules, having first been seen at
#: 300). Quantile partitioning is the documented workaround, and it is applied to
#: every representation rather than only to sin/cos so that the partition is not a
#: second thing changing between rows.
PROBE_REPR = fts.FisConfig(
    n_output_buckets=120, tsk_order="full-2nd", l2_reg=1e-9, output_partition="quantile"
)

DATASETS = [(2, False), (3, False), (5, False), (5, True), (2, True), (3, True)]


# ---------------------------------------------------------------------------
def wrap_angles(theta_deg, limit_deg):
    """Wrap-when-exceeded into [-limit, limit]. None passes through unchanged.

    Vectorised and stateless: the number of 360 deg turns to remove is computed
    directly rather than by looping, so a value at 1589 deg costs the same as one
    at 181 deg. For |theta| <= limit the value is returned untouched, which is
    what creates the overlap band when limit > 180.
    """
    if limit_deg is None:
        return np.asarray(theta_deg, dtype=float)
    x = np.asarray(theta_deg, dtype=float)
    over = np.abs(x) > limit_deg
    if not np.any(over):
        return x.copy()
    # Turns needed so that |x - 360k| <= limit. For |x| > limit the smallest such
    # k is round(x / 360) when limit >= 180, which always lands inside [-180, 180]
    # and hence inside [-limit, limit].
    out = x.copy()
    out[over] = x[over] - 360.0 * np.round(x[over] / 360.0)
    return out


def hysteresis_wrap(theta_deg, limit_deg):
    """Bounded wrap with branch memory: re-branch only on crossing +-limit.

    This is the correct implementation of the overlap-band idea. `wrap_angles`
    chooses a branch *pointwise*, so a trajectory oscillating inside the overlap
    band flips branch on every crossing and a wider band means more flipping --
    which is why the pointwise sweep found discontinuities *rising* with the
    limit. Here the branch is state: once chosen it is kept until the value leaves
    [-limit, limit], and a re-branch from +limit lands at limit - 360, i.e. deep
    inside the window when limit is large. Flip-flopping becomes impossible.

    `np.unwrap` cannot do this job. It is measured to be a no-op on this data
    (it changes the raw angles by 1e-13 deg, because RK4 output at 5 ms sampling
    never moves an angle more than 4.6 deg per step, far below the 180 deg
    discontinuity threshold), and wrap-then-unwrap round-trips to the original to
    2e-13 deg. np.unwrap *recovers* continuity after wrapping; it cannot impose a
    bound, because a bounded continuous scalar representation of a monotonically
    drifting angle does not exist. Escaping that needs two outputs -- see
    `sincos_targets`.

    Time is assumed to be the second-to-last axis, matching theta_deg layouts of
    (n_ics, n_steps, n_links) and (n_steps, n_links).
    """
    x = np.asarray(theta_deg, dtype=float)
    if limit_deg is None:
        return x.copy()
    xt = np.moveaxis(x, -2, 0)  # time first, so the scan is over axis 0
    k = np.zeros(xt.shape[1:], dtype=float)
    out = np.empty_like(xt)
    for i in range(xt.shape[0]):
        v = xt[i] - 360.0 * k
        hi, lo = v > limit_deg, v < -limit_deg
        if hi.any():
            k = np.where(hi, k + np.ceil((v - limit_deg) / 360.0), k)
        if lo.any():
            k = np.where(lo, k - np.ceil((-limit_deg - v) / 360.0), k)
        out[i] = xt[i] - 360.0 * k
    return np.moveaxis(out, 0, -2)


def sincos_targets(theta_deg):
    """(sin, cos) per angle: bounded in [-1, 1] *and* continuous everywhere.

    The only representation that has both properties. It works by spending a
    second output per angle rather than fighting the topology: a circle cannot be
    embedded in a bounded interval without a cut, but it embeds in the plane
    without one. The winding number is discarded, which costs nothing here since
    the dynamics depend on the angles only through sin and cos.

    Returns (..., 2n) with all sines first, then all cosines.
    """
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    return np.concatenate([np.sin(th), np.cos(th)], axis=-1)


def discontinuities(theta_deg, jump_deg=180.0):
    """Count sample-to-sample jumps exceeding `jump_deg`, per angle column.

    A wrap event shows up as a ~360 deg step between consecutive samples. This is
    the quantity the overlap band is supposed to reduce; at 5 ms sampling the true
    dynamics never move an angle by 180 deg in one step, so every such jump is an
    artefact of the representation.
    """
    d = np.abs(np.diff(np.asarray(theta_deg, dtype=float), axis=-2))
    return int(np.sum(d > jump_deg))


def circular_rmse_deg(pred_deg, true_deg):
    """RMSE of the shortest angular difference, in degrees."""
    d = (np.asarray(pred_deg) - np.asarray(true_deg) + 180.0) % 360.0 - 180.0
    return float(np.sqrt(np.mean(d**2)))


# ---------------------------------------------------------------------------
def wrapped_split(split, limit_deg):
    """A copy of `split` with angles wrapped and per-trajectory scaling redone.

    Scaling has to be recomputed after wrapping, not before: the whole point is
    that the wrapped target has a smaller range, and min-max scaling against the
    unwrapped range would discard exactly that gain.
    """
    theta = wrap_angles(split.theta_deg, limit_deg)
    scaled, ranges = fts._scale_per_trajectory(theta)

    # Fitted over the holdout's full 20 s, matching fis_timestep.load's unclipped
    # convention. Using the in-window range for the control and the full range for
    # the wrapped variants would make the wrap the *second* thing that changed
    # between rows and the comparison would mean nothing.
    h_deg = wrap_angles(split.holdout_theta_deg, limit_deg)
    _, h_range = fts._scale_per_trajectory(h_deg)
    lo, hi = h_range[:, 0][None, :], h_range[:, 1][None, :]
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    return dataclasses.replace(
        split,
        theta_deg=theta,
        theta_scaled=scaled,
        ranges=ranges,
        holdout_theta_deg=h_deg,
        holdout_theta_scaled=(h_deg - lo) / span,
        holdout_range=h_range,
    )


def evaluate(split, limit_deg, cfg=PROBE):
    """Fit at one wrap setting and score it, in both metrics."""
    ws = wrapped_split(split, limit_deg)
    t0 = time.perf_counter()
    res, model = fts.run(ws, cfg)
    fit_s = time.perf_counter() - t0

    pred = fts.predictions_for(ws, model, cfg, which="holdout")
    inw = ws.in_window
    # Circular error is computed against the *unwrapped* truth on purpose: a
    # wrapped prediction and an unwrapped truth describe the same configuration,
    # so the circular difference is the honest error either way, and this keeps
    # the column comparable to the no-wrap row.
    truth_unwrapped = split.holdout_theta_deg
    return {
        "dataset": split.label,
        "wrap_limit_deg": "none" if limit_deg is None else f"{limit_deg:.0f}",
        "config": cfg.key(),
        "train_seconds": round(fit_s, 1),
        "target_range_deg": round(
            float(np.max(ws.ranges[:, :, 1] - ws.ranges[:, :, 0])), 1
        ),
        "wrap_events_train": discontinuities(ws.theta_deg),
        "wrap_events_holdout": discontinuities(ws.holdout_theta_deg),
        "inwindow_rmse_scaled": round(res.holdout_ic["rmse"], 5),
        "inwindow_r2": round(res.holdout_ic["r2"], 5),
        "inwindow_rmse_circ_deg": round(
            circular_rmse_deg(pred["pred_deg"][inw], truth_unwrapped[inw]), 3
        ),
        "extrap_rmse_circ_deg": round(
            circular_rmse_deg(pred["pred_deg"][~inw], truth_unwrapped[~inw]), 3
        ),
    }


def wrapped_split_hyst(split, limit_deg):
    """Like `wrapped_split` but using the branch-remembering wrap."""
    theta = hysteresis_wrap(split.theta_deg, limit_deg)
    scaled, ranges = fts._scale_per_trajectory(theta)
    h_deg = hysteresis_wrap(split.holdout_theta_deg, limit_deg)
    _, h_range = fts._scale_per_trajectory(h_deg)
    lo, hi = h_range[:, 0][None, :], h_range[:, 1][None, :]
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    return dataclasses.replace(
        split,
        theta_deg=theta,
        theta_scaled=scaled,
        ranges=ranges,
        holdout_theta_deg=h_deg,
        holdout_theta_scaled=(h_deg - lo) / span,
        holdout_range=h_range,
    )


def evaluate_hysteresis(split, limit_deg, cfg=PROBE_REPR):
    ws = wrapped_split_hyst(split, limit_deg)
    t0 = time.perf_counter()
    res, model = fts.run(ws, cfg)
    fit_s = time.perf_counter() - t0
    pred = fts.predictions_for(ws, model, cfg, which="holdout")
    inw = ws.in_window
    truth = split.holdout_theta_deg
    return {
        "dataset": split.label,
        "representation": "hysteresis",
        "wrap_limit_deg": "none" if limit_deg is None else f"{limit_deg:.0f}",
        "train_seconds": round(fit_s, 1),
        "target_range_deg": round(
            float(np.max(ws.ranges[:, :, 1] - ws.ranges[:, :, 0])), 1
        ),
        "wrap_events_train": discontinuities(ws.theta_deg),
        "wrap_events_holdout": discontinuities(ws.holdout_theta_deg),
        "inwindow_rmse_scaled": round(res.holdout_ic["rmse"], 5),
        "inwindow_r2": round(res.holdout_ic["r2"], 5),
        "inwindow_rmse_circ_deg": round(
            circular_rmse_deg(pred["pred_deg"][inw], truth[inw]), 3
        ),
        "extrap_rmse_circ_deg": round(
            circular_rmse_deg(pred["pred_deg"][~inw], truth[~inw]), 3
        ),
    }


def evaluate_sincos(split, cfg=PROBE_REPR, seed=42):
    """Fit (sin, cos) per angle and recover the angle with atan2.

    No per-trajectory min-max here: sin and cos are already bounded and mutually
    commensurable, and scaling them independently would distort the circle into an
    ellipse and put the atan2 reconstruction on the wrong axis ratio. That means
    there is no scaled-RMSE column for this row -- only the circular error, which
    is the metric that was always the physically meaningful one.
    """
    X, names = fts.encode(split.ic_deg, split.t, cfg.encoding, cfg.n_harmonics)
    n = split.n_links
    Y = sincos_targets(split.theta_deg).reshape(-1, 2 * n)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    tr = perm[: int(0.8 * X.shape[0])]

    model = fts.FisOperator(cfg)
    t0 = time.perf_counter()
    model.fit(X[tr], names, Y[tr])
    fit_s = time.perf_counter() - t0

    Xh, _ = fts.encode(
        split.holdout_ic_deg[None, :], split.holdout_t, cfg.encoding, cfg.n_harmonics
    )
    P = model.predict(Xh)
    pred_deg = np.rad2deg(np.arctan2(P[:, :n], P[:, n:]))
    inw = split.in_window
    truth = split.holdout_theta_deg
    # How far the predicted (sin, cos) drifts off the unit circle is a free
    # diagnostic: the FIS is not constrained to produce a valid angle, and the
    # radius tells you how much of its output is inadmissible.
    radius = np.sqrt(P[:, :n] ** 2 + P[:, n:] ** 2)

    # Per-sample, per-link: does a shrunken radius flag the wrong predictions?
    # Correlated in-window only, matching every other in-window figure in this
    # module. Not calibrated uncertainty -- Table 8's note on this -- just
    # whether low radius and high error co-occur.
    abs_err_inw = np.abs((pred_deg[inw] - truth[inw] + 180.0) % 360.0 - 180.0)
    radius_shortfall_inw = 1.0 - radius[inw]
    radius_error_corr = float(
        np.corrcoef(radius_shortfall_inw.ravel(), abs_err_inw.ravel())[0, 1]
    )

    return {
        "dataset": split.label,
        "representation": "sin/cos",
        "wrap_limit_deg": "n/a",
        "n_output_buckets": cfg.n_output_buckets,
        "train_seconds": round(fit_s, 1),
        "target_range_deg": 2.0,
        "wrap_events_train": 0,
        "wrap_events_holdout": 0,
        "inwindow_rmse_scaled": "",
        "inwindow_r2": "",
        "inwindow_rmse_circ_deg": round(
            circular_rmse_deg(pred_deg[inw], truth[inw]), 3
        ),
        "extrap_rmse_circ_deg": round(
            circular_rmse_deg(pred_deg[~inw], truth[~inw]), 3
        ),
        "mean_unit_radius": round(float(np.mean(radius[inw])), 4),
        "radius_error_corr": round(radius_error_corr, 3),
    }


def main_sincos_capacity(
    n_output_buckets=(40, 120, 300), n_links=2, friction=False, log=print
):
    """Table 8: sin/cos accuracy against capacity, frictionless double pendulum.

    `evaluate_sincos` above is normally called once at a fixed probe capacity
    (matching the other representations for comparability); this is the sweep
    that actually varies n_output_buckets for the sin/cos representation, which
    is what shows it saturating rather than improving with capacity.
    """
    split = fts.load(n_links, friction)
    rows = []
    for nb in n_output_buckets:
        cfg = fts.FisConfig(
            n_output_buckets=nb,
            tsk_order="full-2nd",
            l2_reg=1e-9,
            output_partition="quantile",
        )
        r = evaluate_sincos(split, cfg)
        rows.append(r)
        log(
            f"  {nb:4d} rules  in-window {r['inwindow_rmse_circ_deg']:6.1f} deg  "
            f"past-window {r['extrap_rmse_circ_deg']:6.1f} deg  "
            f"mean radius {r['mean_unit_radius']:.3f}  "
            f"corr(1-r,|e|) {r['radius_error_corr']:+.3f}"
        )
    return rows


def build_representation_rows():
    """Table 7: pointwise wrap, hysteresis wrap, and sin/cos on one footing."""
    rows = []
    for n_links, friction in [(2, False), (3, False), (5, False), (5, True)]:
        split = fts.load(n_links, friction)
        print(
            f"\n{split.label}  (|theta| max = {np.abs(split.theta_deg).max():.0f} deg)"
        )

        r = evaluate(split, None, PROBE_REPR)
        r["representation"] = "pointwise"
        rows.append(r)
        print(
            f"  {'no wrap':22s} jumps={r['wrap_events_train']:5d}  "
            f"circErr in/out={r['inwindow_rmse_circ_deg']:7.2f}/"
            f"{r['extrap_rmse_circ_deg']:7.2f} deg"
        )

        for limit in [180.0, 360.0]:
            p = evaluate(split, limit, PROBE_REPR)
            p["representation"] = "pointwise"
            rows.append(p)
            h = evaluate_hysteresis(split, limit)
            rows.append(h)
            print(
                f"  {'pointwise +-' + str(int(limit)):22s} "
                f"jumps={p['wrap_events_train']:5d}  "
                f"circErr in/out={p['inwindow_rmse_circ_deg']:7.2f}/"
                f"{p['extrap_rmse_circ_deg']:7.2f} deg"
            )
            print(
                f"  {'hysteresis +-' + str(int(limit)):22s} "
                f"jumps={h['wrap_events_train']:5d}  "
                f"circErr in/out={h['inwindow_rmse_circ_deg']:7.2f}/"
                f"{h['extrap_rmse_circ_deg']:7.2f} deg"
            )

        s = evaluate_sincos(split)
        rows.append(s)
        print(
            f"  {'sin/cos':22s} jumps={s['wrap_events_train']:5d}  "
            f"circErr in/out={s['inwindow_rmse_circ_deg']:7.2f}/"
            f"{s['extrap_rmse_circ_deg']:7.2f} deg   "
            f"mean|(sin,cos)|={s['mean_unit_radius']:.3f}"
        )

    return rows


def build_wrap_sweep_rows(datasets=DATASETS, limits=WRAP_LIMITS, log=print):
    """Table 6-adjacent raw grid: every (dataset, wrap limit), pointwise wrap only."""
    rows = []
    for n_links, friction in datasets:
        split = fts.load(n_links, friction)
        raw_max = float(np.abs(split.theta_deg).max())
        log(f"\n{split.label}  (|theta| max = {raw_max:.0f} deg)")
        for limit in limits:
            r = evaluate(split, limit)
            rows.append(r)
            tag = "no wrap" if limit is None else f"+-{limit:.0f} deg"
            log(
                f"  {tag:12s} range={r['target_range_deg']:7.1f}  "
                f"jumps(train/hold)={r['wrap_events_train']:5d}/{r['wrap_events_holdout']:3d}  "
                f"scaledRMSE={r['inwindow_rmse_scaled']:.4f} R2={r['inwindow_r2']:+.4f}  "
                f"circErr in/out={r['inwindow_rmse_circ_deg']:7.2f}/"
                f"{r['extrap_rmse_circ_deg']:8.2f} deg"
            )
    return rows


def main_representations():
    """Standalone CLI: Table 7's comparison, written to results/representation.json."""
    rows = build_representation_rows()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "representation.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"representations": rows}, fh, indent=2)
    print(f"\nwrote {path} ({len(rows)} rows)")
    return rows


def main():
    """Standalone CLI: Table 6's raw wrap grid, written to results/wrap_sweep.json."""
    rows = build_wrap_sweep_rows()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "wrap_sweep.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2)
    print(f"\nwrote {path} ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    main()
    main_representations()
    sincos_rows = main_sincos_capacity()
    # Fold Table 8 into the file main_representations() just wrote, so a
    # standalone run leaves one complete results/representation.json behind
    # (same three keys run_all.py's `representation` stage writes, just without
    # its cache envelope).
    path = RESULT_DIR / "representation.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["sincos_capacity"] = sincos_rows
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"added sincos_capacity to {path}")
