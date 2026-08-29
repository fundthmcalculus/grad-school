#!/usr/bin/env python3
"""Figure 4.1 -- per-feature Gaussian mixtures, and the rule they combine into.

Drawn from a real fit. The model is built by the same three calls the harness
uses in `table_4_4_openset.py` -- `calculate_gaussian_correlation`,
`take_top_features`, `create_gaussian_membership_dict` -- on the in-repo Glass
data, so the mixtures shown are the mixtures the method produces, degenerate
components and all.

**A correction the figure forces.** §4.3.1 said the per-feature memberships for
a class are combined "with a fuzzy OR -- a t-conorm -- and that combination *is*
the rule". The shipped `simple_gaussian_predict` does something different, and
the difference matters: within a feature the mixture components are combined by
the **t-conorm** (a class is recognised if this Gaussian fires *or* that one
does), and across features the results are combined by the **t-norm**. The rule
is a conjunction of disjunctions, not a disjunction. Drawing it as the prose
described would have drawn a model nobody has trained, so the figure follows the
code and the sentence in §4.3.1 has been corrected to match.

The aggregation here is not reimplemented, either: the same `t_norm`/`t_conorm`
functions and the same `resolve_norm_pair()` default the predictor uses are
called directly, and the resolved family is printed on the figure. If the
library's default norm changes, this figure changes with it.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "04-mog-classification"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET_CLASS = 1  # building windows, float processed -- the largest class
N_FEATURES = 3  # "two or three features stacked", per the caption


def _fit():
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        resolve_norm_pair,
        take_top_features,
    )

    sys.path.insert(0, REPO)  # repo root -> `import repro_data`
    from repro_data import load_glass

    # Glass now comes from the shared loader (reads data/glass.csv); the old
    # inline read + repo-root fallback lived here and in _mf_dedup.
    X, y = load_glass()

    diffs = calculate_gaussian_correlation(X, y)
    _, features = take_top_features(diffs, top_n=N_FEATURES)
    model = create_gaussian_membership_dict(X, y, top_n_var_names=features)
    return X, y, list(features), model, resolve_norm_pair()


def _feature_membership(components, values, conorm):
    """OR the mixture components of one feature -- what the predictor does first."""
    from tribblefis.gauss_math import t_conorm

    out = np.zeros_like(values, dtype=float)
    for mf in components:
        out = t_conorm(out, mf.evaluate(values), conorm)
    return out


def build():
    from tribblefis.gauss_math import t_norm

    X, y, features, model, norms = _fit()
    in_class = y.values == TARGET_CLASS

    fig = F._pyplot().figure(figsize=(F.W_WIDE, 5.0), dpi=F.DPI)
    fig.patch.set_facecolor(F.SURFACE)
    gs = fig.add_gridspec(
        2, N_FEATURES, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.28
    )

    clauses = []
    for col, feature in enumerate(features):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(F.SURFACE)
        components = (
            model.feature_models[feature].label_models[TARGET_CLASS].memberships
        )

        lo, hi = float(X[feature].min()), float(X[feature].max())
        pad = 0.06 * (hi - lo or 1.0)
        grid = np.linspace(lo - pad, hi + pad, 600)

        # The class's own data, as a rug. A histogram would compete with the
        # membership curves for the same vertical space and win.
        ax.plot(
            X[feature].values[in_class],
            np.full(in_class.sum(), -0.06),
            marker="|",
            ls="none",
            ms=4,
            mew=0.8,
            color=F.BLUE,
            zorder=3,
        )
        ax.plot(
            X[feature].values[~in_class],
            np.full((~in_class).sum(), -0.13),
            marker="|",
            ls="none",
            ms=4,
            mew=0.8,
            color=F.FAINT,
            zorder=2,
        )

        for mf in components:
            ax.plot(
                grid,
                mf.evaluate(grid),
                lw=0.9,
                ls=(0, (3, 2)),
                color=F.tint(F.BLUE, 0.45),
                zorder=3,
            )
        ax.plot(
            grid,
            _feature_membership(components, grid, norms.t_conorm),
            lw=2.0,
            color=F.BLUE,
            zorder=4,
        )

        plural = "" if len(components) == 1 else "s"
        F.style_axes(
            ax,
            title=f"{feature}   ({len(components)} component{plural})",
            xlabel=None,
            ylabel="membership" if col == 0 else None,
            grid=True,
            grid_axis="y",
        )
        ax.set_ylim(-0.2, 1.08)
        ax.set_yticks([0, 0.5, 1.0])

        centres = ", ".join(f"{mf.mu:.2f}" for mf in components)
        clauses.append(f"{feature} near {{{centres}}}")

    # -- the rule: t-norm across features ------------------------------------
    firing = np.ones(len(X))
    for feature in features:
        components = (
            model.feature_models[feature].label_models[TARGET_CLASS].memberships
        )
        firing = t_norm(
            _feature_membership(components, X[feature].values, norms.t_conorm),
            firing,
            norms.t_norm,
        )

    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor(F.SURFACE)
    bins = np.linspace(0, max(firing.max(), 1e-6), 36)
    ax.hist(
        firing[~in_class],
        bins=bins,
        color=F.tint(F.FAINT, 0.35),
        edgecolor=F.SURFACE,
        linewidth=0.4,
        label=f"other classes  ($n$ = {(~in_class).sum()})",
        zorder=2,
    )
    ax.hist(
        firing[in_class],
        bins=bins,
        color=F.BLUE,
        edgecolor=F.SURFACE,
        linewidth=0.4,
        label=f"Type {TARGET_CLASS}  ($n$ = {in_class.sum()})",
        zorder=3,
    )
    F.style_axes(
        ax,
        title=f"Firing strength of the Type-{TARGET_CLASS} rule, "
        f"over every sample in the dataset",
        xlabel="rule firing strength",
        ylabel="samples",
        grid=True,
        grid_axis="y",
    )
    F.legend(ax, loc="upper right")

    # Training-set accuracy of the whole rule set, for context under the panel:
    # a single rule's firing overlapping the other classes is not a failure, it
    # is what argmax-over-K-rules is for, and the figure should not imply
    # otherwise.
    from tribblefis.gauss_math import simple_gaussian_predict

    pred = simple_gaussian_predict(X, model.to_simple_model())
    accuracy = float((np.asarray(pred).astype(int) == y.values).mean())

    rule = f"IF  ({')  AND  ('.join(clauses)})   THEN   " f"Type = {TARGET_CLASS}"
    fig.text(
        0.5,
        0.015,
        rule,
        ha="center",
        va="center",
        fontsize=F.FS_ANNOT,
        color=F.INK,
        family="monospace",
    )
    fig.text(
        0.5,
        -0.035,
        f"Components within a feature are combined by the t-conorm (bold curve, "
        f"top row); the three results are combined across features by the "
        f"t-norm.\nBoth from the library default, resolved here as "
        f"'{norms.t_norm}' / '{norms.t_conorm}'. One rule is not a classifier — "
        f"prediction is the argmax over all six class rules, so the\noverlap in "
        f"the lower panel is expected; the argmax recovers {accuracy:.3f} on this "
        f"deliberately three-feature model. Fit on all 214 Glass samples — an "
        f"illustration of the\nconstruction, not a held-out measurement.",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
