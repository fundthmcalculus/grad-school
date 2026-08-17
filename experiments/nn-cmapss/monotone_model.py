"""A RUL model that is monotone-decreasing by construction, not by clamping.

`monotone.py` showed a causal post-hoc clamp (`mean5 -> cummin`) gets hard
monotonicity for ~0.2 RMSE, and that an *offline* monotone projection would
reach 8.35 on `honest` -- the ceiling, because it may look into the future.
This asks whether baking the constraint into the model, so it never has to be
clamped, can close some of the gap between the causal 10.23 and that 8.35
*causally*.

The model
---------
RUL falls by a per-cycle amount that cannot be negative:

    delta(t) = softplus(w . phi(x_t) + b)              >= 0   (a damage rate)
    RUL(t)   = A - sum_{s <= t} delta(s)                       (accumulated)

so ``RUL(t) - RUL(t-1) = -delta(t) <= 0`` -- monotone non-increasing, exactly,
for any weights. `A` is a learned intercept ("predicted RUL before any observed
degradation"), fit on data so the level is absolute rather than relative. The
cumulative sum is per engine and runs forward in cycle order, so a prediction
at cycle t uses only cycles <= t: deployable, unlike the offline oracle.

`phi` is the FIS's own selected, scaled features, optionally with the FIS's raw
per-cycle RUL prediction appended -- so this distils what TRIBBLE learned into a
monotone form rather than discarding it. The whole thing is a d+2 parameter
model trained by the same Adam the rest of the experiment uses; monotonicity is
structural, so training never fights the constraint.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

import cmapss_data
import models
import monotone as M
import report

OUT = report.OUT


def _softplus(z):
    # log(1 + e^z), stable for large |z|.
    return np.logaddexp(0.0, z)


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class MonotoneDamageRUL:
    """RUL as a starting level minus accumulated non-negative damage.

    Fit and predict both take (X, unit, cycle): the per-engine forward pass has
    to see each trajectory in cycle order to accumulate, so the model owns that
    bookkeeping rather than assuming the rows arrive sorted.
    """

    def __init__(
        self,
        l2=1e-4,
        lr=0.05,
        epochs=400,
        seed=0,
        anchor="full",
        floor=0.0,
        end_weight=0.0,
    ):
        # end_weight: extra loss on each *training* engine's last cycle, whose
        # true RUL is 0 (CMAPSS runs training units to failure). That pins the
        # total accumulated damage to the start level A, which is the one thing
        # per-cycle MSE under-weights -- one endpoint against a long plateau --
        # and it is exactly the slope drift that ruins the longest test engine.
        # A pure inference-time floor cannot fix a slope that is systematically
        # too steep; this fixes it at training time. Only cycles whose true RUL
        # is ~0 are anchored, so truncated test engines are never assumed to
        # have failed.
        self.end_weight = end_weight
        # floor: RUL cannot be negative, so predictions are clamped up to
        # `floor`. max(., 0) of a non-increasing sequence is still
        # non-increasing, so monotonicity survives -- and the clamp's gradient
        # is masked where it bites, so training stops driving a trajectory
        # further past zero instead of paying loss for an impossible value.
        # This is what rescues the longest engine, where a small per-cycle
        # damage-rate bias integrates into a large end-of-life overshoot.
        self.floor = floor
        # anchor="full": RUL(t) = A - sum_{s<=t} delta(s); the first prediction
        # already carries one increment, so engines differ from cycle one.
        # anchor="pre": RUL(t) = A - sum_{s<t} delta(s); the first prediction is
        # exactly A, common to all engines. "full" fits DS02's varied starts
        # better; both are offered because which is right is dataset-dependent.
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.anchor = anchor

    def _engine_order(self, unit, cycle):
        df = pd.DataFrame({"u": unit, "c": cycle, "i": np.arange(len(unit))})
        groups = []
        for _, sub in df.sort_values(["u", "c"]).groupby("u", sort=True):
            groups.append(sub["i"].to_numpy())
        return groups

    def _forward(self, X, groups):
        z = X @ self.w_ + self.b_
        delta = _softplus(z)
        raw = np.empty(X.shape[0], dtype=float)  # pre-floor, for gradient mask
        for idx in groups:
            d = delta[idx]
            csum = np.cumsum(d)
            if self.anchor == "pre":
                csum = csum - d  # exclude the current cycle's increment
            raw[idx] = self.A_ - csum
        pred = np.maximum(raw, self.floor)
        return z, delta, pred, raw

    def fit(self, X, y, unit, cycle):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        groups = self._engine_order(unit, cycle)

        # A starts at the mean first-cycle RUL; b so the initial damage rate is
        # ~1 cycle/cycle (softplus(0.54) ~ 1), the rate a healthy engine loses
        # RUL at; w at zero so training adds sensor dependence from a sane base.
        firsts = [idx[0] for idx in groups]
        self.w_ = np.zeros(d)
        self.b_ = 0.541
        self.A_ = float(np.mean(y[firsts])) if len(firsts) else float(np.mean(y))

        # Endpoints to anchor: an engine's last cycle, but only when its target
        # there is ~0 (a unit that actually ran to failure). n_end normalizes
        # the penalty so end_weight is comparable across datasets.
        ends = np.array([idx[-1] for idx in groups])
        ends = ends[y[ends] <= 1.0]
        n_end = max(len(ends), 1)

        params = ["w_", "b_", "A_"]
        m = {p: np.zeros_like(np.atleast_1d(getattr(self, p)), float) for p in params}
        v = {p: np.zeros_like(np.atleast_1d(getattr(self, p)), float) for p in params}
        b1, b2, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.epochs + 1):
            z, delta, pred, raw = self._forward(X, groups)
            g_pred = (2.0 / n) * (pred - y)
            # Where the floor is active the prediction no longer depends on the
            # parameters, so its loss must not flow back -- mask it out.
            g_pred = np.where(raw > self.floor, g_pred, 0.0)
            # Endpoint anchor: pull raw(T_e) toward 0 for units that failed.
            # Uses raw (not the floored pred) so it can still push a trajectory
            # that has bottomed out, and rides the same backward path since
            # d raw / d(params) == d pred / d(params) above the floor.
            if self.end_weight and len(ends):
                g_pred[ends] += (2.0 * self.end_weight / n_end) * raw[ends]

            g_w = np.zeros(d)
            g_b = 0.0
            g_A = float(g_pred.sum())  # dpred/dA = 1 everywhere it is unclamped
            sig = _sigmoid(z)
            for idx in groups:
                gp = g_pred[idx]
                # dL/ddelta(s) = sum_{t>=s} g_pred(t) * dpred(t)/ddelta(s).
                # "full": dpred(t)/ddelta(s) = -1 for t>=s -> -reverse-cumsum.
                # "pre":  dpred(t)/ddelta(s) = -1 for t>s   -> shift by one.
                rev = np.cumsum(gp[::-1])[::-1]
                if self.anchor == "pre":
                    rev = rev - gp  # drop the t==s term
                g_delta = -rev
                g_z = g_delta * sig[idx]
                g_w += X[idx].T @ g_z
                g_b += float(g_z.sum())
            g_w += self.l2 * self.w_

            grads = {"w_": g_w, "b_": np.atleast_1d(g_b), "A_": np.atleast_1d(g_A)}
            for p in params:
                g = grads[p]
                m[p] = b1 * m[p] + (1 - b1) * g
                v[p] = b2 * v[p] + (1 - b2) * (g * g)
                step = (
                    self.lr * (m[p] / (1 - b1**t)) / (np.sqrt(v[p] / (1 - b2**t)) + eps)
                )
                cur = getattr(self, p)
                setattr(self, p, cur - (step if p == "w_" else float(step[0])))
        return self

    def predict(self, X, unit, cycle):
        X = np.asarray(X, dtype=float)
        groups = self._engine_order(unit, cycle)
        _, _, pred, _ = self._forward(X, groups)
        return pred


# ---------------------------------------------------------------------------
# Per-cycle collapse (the model works on one row per (unit, cycle))
# ---------------------------------------------------------------------------
def to_per_cycle(bundle_split, feature_names):
    """Collapse a split to one row per (unit, cycle): features averaged, RUL
    taken as the cycle's value. The raw-memory pipeline carries several rows per
    cycle; a damage increment is per cycle, so it must be aggregated first."""
    X = np.asarray(bundle_split.X, dtype=float)
    df = pd.DataFrame(X, columns=list(feature_names))
    df["unit"] = bundle_split.unit
    df["cycle"] = bundle_split.cycle
    df["y"] = bundle_split.y
    df["y_true"] = bundle_split.y_true
    g = (
        df.groupby(["unit", "cycle"], as_index=False)
        .mean()
        .sort_values(["unit", "cycle"])
    )
    return (
        g[list(feature_names)].to_numpy(),
        g["y"].to_numpy(),
        g["y_true"].to_numpy(),
        g["unit"].to_numpy(),
        g["cycle"].to_numpy(),
    )


def run(which: str, use_fis_feature=True, anchor="full") -> dict:
    warnings.simplefilter("ignore")
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which], verbose=False)
    names = b.feature_names
    kw = models.FIS_CONFIGS[which]

    fis, _ = models.fit_fis(b.train.X, b.train.y, names, **kw)
    idx = np.array([names.index(f) for f in fis.top_features_], dtype=int)

    Xtr, ytr, ytr_true, utr, ctr = to_per_cycle(b.train, names)
    Xte, yte, yte_true, ute, cte = to_per_cycle(b.test, names)
    Ftr, Fte = Xtr[:, idx], Xte[:, idx]

    if use_fis_feature:
        # Distil the FIS: its own per-cycle RUL prediction, standardized on
        # train, becomes one more feature the damage rate can lean on.
        ptr = models.fis_predict(fis, Xtr, names)
        pte = models.fis_predict(fis, Xte, names)
        mu, sd = float(ptr.mean()), float(ptr.std()) or 1.0
        Ftr = np.column_stack([Ftr, (ptr - mu) / sd])
        Fte = np.column_stack([Fte, (pte - mu) / sd])

    model = MonotoneDamageRUL(anchor=anchor).fit(Ftr, ytr, utr, ctr)
    pred = model.predict(Fte, ute, cte)

    g = pd.DataFrame({"unit": ute, "cycle": cte, "true": yte_true, "pred": pred})
    per_engine = [
        M.score_engine(s["true"].to_numpy(), s["pred"].to_numpy())
        for _, s in g.groupby("unit")
    ]
    return dict(
        bundle=which,
        use_fis_feature=use_fis_feature,
        anchor=anchor,
        **M.aggregate(per_engine),
    )


def damage_predictions(which: str, end_weight=0.0):
    """Fit the damage model on a bundle; return the per-cycle test frame."""
    warnings.simplefilter("ignore")
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which], verbose=False)
    names = b.feature_names
    fis, _ = models.fit_fis(b.train.X, b.train.y, names, **models.FIS_CONFIGS[which])
    idx = np.array([names.index(f) for f in fis.top_features_], dtype=int)
    Xtr, ytr, _, utr, ctr = to_per_cycle(b.train, names)
    Xte, _, yte_true, ute, cte = to_per_cycle(b.test, names)
    model = MonotoneDamageRUL(anchor="full", end_weight=end_weight).fit(
        Xtr[:, idx], ytr, utr, ctr
    )
    pred = model.predict(Xte[:, idx], ute, cte)
    # The FIS reference is computed exactly as monotone.py does it -- on the
    # FULL test rows, then predictions averaged to per-cycle -- not on the
    # per-cycle-averaged features the damage model consumes. Averaging features
    # first and averaging predictions after are not the same on the raw-memory
    # pipeline, and the reference must be the one the rest of the study quotes.
    raw_ref = M.per_cycle(
        b.test.unit,
        b.test.cycle,
        b.test.y_true,
        models.fis_predict(fis, b.test.X, names),
    )
    return (
        pd.DataFrame({"unit": ute, "cycle": cte, "true": yte_true, "pred": pred}),
        raw_ref,
    )


def main(bundles=("honest", "best")) -> None:
    payload = {}
    for which in bundles:
        dmg, raw = damage_predictions(which, end_weight=0.0)

        # References straight from monotone.py, on the same raw FIS.
        variants = {
            "raw FIS": M.aggregate(M.apply_output(raw, M.out_raw)),
            "cummin (clamp)": M.aggregate(M.apply_output(raw, M.out_cummin)),
            "mean5->cummin (clamp)": M.aggregate(
                M.apply_output(raw, lambda p: M.out_mean_cummin(p, 5))
            ),
            "offline oracle": M.aggregate(M.apply_output(raw, M.out_iso_offline)),
        }

        # The monotone-by-construction model, and its endpoint-anchored variant.
        def _agg(frame):
            return M.aggregate(
                [
                    M.score_engine(s["true"].to_numpy(), s["pred"].to_numpy())
                    for _, s in frame.groupby("unit")
                ]
            )

        variants["monotone-damage"] = _agg(dmg)
        variants["monotone-damage+endpoint"] = _agg(
            damage_predictions(which, end_weight=1.0)[0]
        )
        payload[which] = variants

        print(f"\n=== {which} ===")
        print(f"  {'method':30s} {'up%':>5s} {'pos_tv':>7s} {'rmse':>6s} {'mae':>6s}")
        for tag, m in variants.items():
            print(
                f"  {tag:30s} {m['up_frac']*100:5.0f} {m['pos_tv']:7.1f} "
                f"{m['rmse']:6.2f} {m['mae']:6.2f}"
            )

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "monotone_model.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {os.path.relpath(path, cmapss_data.REPO)}")
    plot(bundles)


def plot(bundles=("honest", "best")) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings.simplefilter("ignore")
    runs = [(w, *damage_predictions(w, end_weight=0.0)) for w in bundles]
    units = sorted(runs[0][1]["unit"].unique().tolist())

    fig, axes = plt.subplots(
        len(runs),
        len(units),
        figsize=(4.6 * len(units), 3.9 * len(runs)),
        facecolor=report.SURFACE,
        squeeze=False,
        sharex="col",
    )
    C_RAW, C_CLAMP, C_DMG = report.C[0], report.C[3], report.C[1]
    for r, (tag, dmg, raw) in enumerate(runs):
        for c, u in enumerate(units):
            ax = axes[r][c]
            sub = raw[raw.unit == u]
            cyc, truth, praw = (sub[k].to_numpy() for k in ("cycle", "true", "pred"))
            pclamp = M.out_mean_cummin(praw, 5)
            dsub = dmg[dmg.unit == u]
            pdmg = dsub["pred"].to_numpy()
            ax.plot(
                cyc,
                truth,
                color=report.INK,
                linewidth=2.4,
                zorder=5,
                label="true RUL",
                solid_capstyle="round",
            )
            ax.plot(
                cyc,
                praw,
                color=C_RAW,
                linewidth=1.0,
                alpha=0.5,
                label="raw FIS",
                zorder=2,
            )
            ax.plot(
                cyc,
                pclamp,
                color=C_CLAMP,
                linewidth=1.6,
                alpha=0.9,
                label="clamp (mean5->cummin)",
                zorder=3,
            )
            ax.plot(
                dsub["cycle"].to_numpy(),
                pdmg,
                color=C_DMG,
                linewidth=2.2,
                label="monotone-damage (model)",
                zorder=4,
            )
            ax.axhline(0.0, color=report.GRID, linewidth=1.2, zorder=1)
            report._style(
                ax,
                "flight cycle" if r == len(runs) - 1 else "",
                "RUL (cycles)" if c == 0 else "",
                f"unit {u} — `{tag}`",
            )
            e_c = float(np.sqrt(np.mean((pclamp - truth) ** 2)))
            e_d = float(np.sqrt(np.mean((pdmg - dsub["true"].to_numpy()) ** 2)))
            ax.text(
                0.03,
                0.06,
                f"clamp {e_c:.1f}   model {e_d:.1f}",
                transform=ax.transAxes,
                fontsize=8.5,
                color=report.INK2,
                family="monospace",
            )
            if r == 0 and c == 0:
                ax.legend(
                    frameon=False,
                    fontsize=8.5,
                    labelcolor=report.INK2,
                    loc="upper right",
                )
    fig.suptitle(
        "Monotone by construction: a damage-accumulation model vs. clamping the FIS",
        color=report.INK,
        fontsize=13,
        x=0.006,
        ha="left",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    os.makedirs(report.FIG, exist_ok=True)
    path = os.path.join(report.FIG, "monotone_model.png")
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, cmapss_data.REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*", default=["honest", "best"])
    a = ap.parse_args()
    main(a.bundles)
