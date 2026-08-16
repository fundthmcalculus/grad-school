#!/usr/bin/env python3
"""Figure 6.1 -- trained fuzzy trees on Concrete and PhiUSIIL, as rules.

§6.3.2's readability claim is that the tree "renders as a short list of IF-THEN
rules, one per root-to-leaf path, each mentioning only the variables on that
path". The way to show that is to print the rules, so the figure is the
`fuzzytree.render.render_tree_text` output of a real fit, typeset.

**The configuration is the demo's, and the choice is load-bearing.** These are
the settings from `tribble-tree/demo_concrete.py` and `demo_phishing.py` --
`max_depth=3, n_terms=2, top_n=4/5` -- not the library defaults. That matters
for what §6.3.2 claims: under the demo configuration the Concrete tree splits on
Cement and then on **Age at exactly 28**, recovering the standard curing mark
as the chapter says. Under the library defaults (`n_terms=3, top_n=-1`) the
second split is Superplasticizer and no Age boundary lands near 28. The claim is
true of the tuned tree, and the caption says so rather than leaving a reader to
find the difference.

Leaf consequents are truncated to their constant term plus a note. A first-order
leaf carries a four-variable linear model, and printing all of them turns a
figure about readability into a wall of coefficients.

PhiUSIIL is not vendored (57 MB; see `data/.gitignore` for the one-line recovery
from `tribble-fis` history). If it is absent the figure draws Concrete and says
in the empty panel what is missing and how to get it, rather than failing.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "06-fuzzy-tree"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "reproduce", "tables"))

# The variables §6.3.2 says each tree recovers, highlighted where they appear.
HIGHLIGHT = {
    "concrete": ("Cement", "Age"),
    "phiusiil": ("HasSocialNet", "HasCopyrightInfo", "URLSimilarityIndex"),
}

_LEAF = re.compile(r"=>\s*y\s*≈\s*([-\d.]+)(.*?)\(soft n=([\d.]+)\)")


def _shorten(line):
    """Collapse a first-order leaf's linear model to its constant plus a note."""
    m = _LEAF.search(line)
    if not m:
        return line.replace("(soft n=", "(n≈").rstrip()
    const, linear, mass = m.groups()
    tail = " + linear" if linear.strip() else ""
    indent = line[: len(line) - len(line.lstrip())]
    return f"{indent}=> y ≈ {float(const):.1f}{tail}   (n≈{float(mass):.0f})"


def _rules(est, limit=26):
    """(header, rule lines). The header is rebuilt rather than reused: the
    library's own first line lists every retained feature on one unwrapped line,
    which in a two-panel figure runs straight into the other panel."""
    import textwrap
    from fuzzytree.render import render_tree_text

    lines = render_tree_text(est).split("\n")
    body = [_shorten(ln) for ln in lines[1:] if ln.strip()]
    if len(body) > limit:
        body = body[:limit] + [f"… {len(body) - limit} further lines"]
    header = textwrap.fill(
        f"{est.n_leaves_} leaves · candidate splits: "
        f"{', '.join(map(str, est.top_features_))}",
        width=46,
    )
    return header, body


def _concrete():
    import _fuzzy_models as FM
    import fuzzytree

    data = FM.load_concrete()
    if data is None:
        return None
    X, y = data
    est = fuzzytree.FuzzyRegressionTree(
        tsk_order="1st",
        criterion="variance",
        max_depth=3,
        n_terms=2,
        top_n=4,
        min_soft_count=20,
        random_state=42,
    ).fit(X, y.values)
    return _rules(est)


def _phiusiil():
    import _fuzzy_models as FM
    import fuzzytree

    data = FM.load_phiusiil(sample_size=20000)
    if data is None:
        return None
    X, y = data
    est = fuzzytree.FuzzyClassificationTree(
        criterion="ambiguity",
        max_depth=3,
        n_terms=2,
        top_n=5,
        min_soft_count=50,
        random_state=42,
    ).fit(X, y)
    return _rules(est)


def _draw(ax, title, rendered, highlight, missing_note=None):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0,
        1.0,
        title,
        ha="left",
        va="top",
        fontsize=F.FS_LABEL,
        color=F.INK,
        fontweight="bold",
    )

    if rendered is None:
        ax.text(
            0.5,
            0.5,
            missing_note,
            ha="center",
            va="center",
            fontsize=F.FS_SMALL,
            color=F.MUTED,
            linespacing=1.6,
            style="italic",
        )
        return

    header, body = rendered
    ax.text(
        0,
        0.935,
        header,
        ha="left",
        va="top",
        fontsize=F.FS_SMALL - 0.5,
        color=F.MUTED,
        family="monospace",
    )

    step = 0.86 / max(len(body), 1)
    for i, line in enumerate(body):
        named = next((v for v in highlight if v in line), None)
        is_leaf = "=>" in line
        color = F.shade(F.BLUE, 0.25) if named else F.INK_2 if is_leaf else F.MUTED
        weight = "bold" if named else "normal"
        ax.text(
            0,
            0.875 - i * step,
            line,
            ha="left",
            va="top",
            fontsize=F.FS_SMALL - 0.5,
            color=color,
            family="monospace",
            fontweight=weight,
        )


def build():
    fig, (left, right) = F.grid_figure(1, 2, width=F.W_WIDE, height=4.6)

    concrete = _concrete()
    phiusiil = _phiusiil()

    _draw(
        left,
        "Concrete — FuzzyRegressionTree",
        concrete,
        HIGHLIGHT["concrete"],
        missing_note="Concrete unavailable: no data/Concrete_Data.csv\nand no "
        "network for the UCI fetch.",
    )
    _draw(
        right,
        "PhiUSIIL — FuzzyClassificationTree",
        phiusiil,
        HIGHLIGHT["phiusiil"],
        missing_note="PhiUSIIL is not vendored (57 MB).\nRecover it with the "
        "one-liner in data/.gitignore\nand re-run this generator.",
    )

    fig.text(
        0.5,
        0.015,
        "Bold: the variables §6.3.2 names. Both trees are at the "
        "`tribble-tree/demo_*.py` configuration (max_depth 3, n_terms 2), not "
        "the library defaults —\nunder the demo settings the Concrete tree "
        "splits Cement then Age at exactly 28, the standard curing mark; under "
        "the defaults the second split is Superplasticizer.\nFirst-order leaf "
        "consequents are shown as their constant plus a note, because printing "
        "four coefficients per leaf defeats the point of a figure about "
        "readability.",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.6,
    )
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
