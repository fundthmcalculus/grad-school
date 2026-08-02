"""The sixteen proposal figures, what produces each one, and what it needs.

One row per figure called out in the prose. This is the file `make_figures.py`
walks, the file `build_pdf.py` reads to know what to copy into the document, and
the file to edit when a figure is added, renamed, or finally produced.

Three things are worth saying about the shape of it.

**`skip` is a first-class outcome.** A figure whose experiment has not been run
is not a bug in this harness and must not look like one. Figure 4.3 wants a
before/after confusion matrix for a correction pass whose effect Chapter 4 says
in as many words it has not yet isolated; drawing it would mean inventing the
result. Those rows carry a `skip` reason, keep their placeholder PNG, and are
reported as skipped rather than failed -- the same discipline the table harness
applies when it prints `N/A` instead of guessing.

**`project` is the uv environment, not an import.** Chapter 3's and Chapter 5's
figures need the pinned submodules; the schematics need nothing but matplotlib.
Each generator runs in the smallest environment that can produce it, which is
also what keeps a schematic from silently depending on a library.

**Figure 3.2 is owned elsewhere.** It is generated as a by-product of the Table
3.1 three-arm sweep, because the exponents in that table and the slopes in that
figure have to come from the same measurements or they can disagree. It is
listed here so the inventory is complete, with `owner` naming the generator that
actually produces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Figure:
    name: str                       # prose/fig/<name>.png
    number: str                     # as cited in the text, e.g. "5.2"
    title: str                      # one line, for the driver's output
    module: Optional[str] = None    # reproduce/figures/<module>.py
    project: Optional[str] = None   # uv --project, or None for a bare env
    extras: tuple = ()              # uv --with packages
    skip: Optional[str] = None      # why this one is not drawn, if it is not
    owner: Optional[str] = None     # generator, when it is not `module`


FIGURES = [
    # -- Chapter 1 ---------------------------------------------------------- #
    Figure("01-structure-before-search", "1.1",
           "Conventional search-first route against the structure-first route",
           module="fig_01_structure_before_search", extras=("matplotlib",)),
    Figure("01-pipeline-roadmap", "1.2",
           "The end-to-end pipeline, annotated with chapters and claims",
           module="fig_01_pipeline_roadmap", extras=("matplotlib",)),

    # -- Chapter 2 ---------------------------------------------------------- #
    Figure("02-fis-components", "2.1",
           "FIS anatomy: what this work generates against what is fixed by design",
           module="fig_02_fis_components", extras=("matplotlib",)),
    Figure("02-vat-rdi", "2.2",
           "A dissimilarity matrix before and after VAT reordering",
           module="fig_02_vat_rdi", project="tribble-cluster",
           extras=("matplotlib", "scipy")),
    Figure("02-persistence", "2.3",
           "Single-linkage dendrogram and its persistence diagram",
           module="fig_02_persistence", extras=("matplotlib", "scipy", "numpy")),

    # -- Chapter 3 ---------------------------------------------------------- #
    Figure("03-pvat-reorder", "3.1",
           "The mergeVAT reorder against the classical linear-scan argmin",
           module="fig_03_pvat_reorder", extras=("matplotlib",)),
    Figure("03-complexity-fit", "3.2",
           "Measured reorder growth against the reference complexity curves",
           owner="reproduce/tables/table_3_1_reorder_three_arm.py"),

    # -- Chapter 4 ---------------------------------------------------------- #
    Figure("04-mog-classification", "4.1",
           "Per-feature Gaussian mixtures and the fuzzy-OR that forms a class rule",
           module="fig_04_mog_classification", project="tribble-fis",
           extras=("matplotlib",)),
    Figure("04-anomaly-sweep", "4.2",
           "The open-set operating curve of Table 4.6",
           module="fig_04_anomaly_sweep", extras=("matplotlib",)),
    Figure("04-rtiot-confusion", "4.3",
           "RT-IOT2022 confusion, before and after the correction-rule pass",
           skip="The experiment does not exist yet. §4.3.1 states plainly that the "
                "accuracy contribution of the correction pass 'has not yet been "
                "isolated' and calls the before/after comparison an experiment the "
                "author owes; RT-IOT2022 is also not among the datasets the harness "
                "can load. Drawing this would mean inventing both halves of the "
                "comparison it is supposed to report."),

    # -- Chapter 5 ---------------------------------------------------------- #
    Figure("05-minimax-transform", "5.1",
           "Concentric rings: raw dissimilarity against the minimax transform",
           module="fig_05_minimax_transform",
           extras=("matplotlib", "scipy", "numpy", "scikit-learn")),
    Figure("05-band-discovery", "5.2",
           "Band discovery on the log-birth spectrum, and the partitions it yields",
           module="fig_05_band_discovery",
           extras=("matplotlib", "scipy", "numpy", "scikit-learn")),
    Figure("05-persistence-ramp", "5.3",
           "A block's persistence ramp, read off the hierarchy",
           module="fig_05_persistence_ramp", extras=("matplotlib", "scipy", "numpy")),

    # -- Chapter 6 ---------------------------------------------------------- #
    Figure("06-fuzzy-tree", "6.1",
           "A trained fuzzy tree on Concrete and on PhiUSIIL, as rules",
           module="fig_06_fuzzy_tree", project="tribble-fis",
           extras=("matplotlib",)),
    Figure("06-hme-structure", "6.2",
           "Hierarchical mixture: gates over named inputs routing to TSK experts",
           module="fig_06_hme_structure", extras=("matplotlib",)),
    Figure("06-mimo-rollout", "6.3",
           "Double-pendulum rollout: truth, memoryless FIS, memory-augmented FIS",
           module="fig_06_mimo_rollout", project="tribble-fis",
           extras=("matplotlib", "scipy", "pandas")),
]

BY_NAME = {f.name: f for f in FIGURES}


def harness_name(prose_name):
    """`03-complexity-fit` -> `fig_03_complexity_fit`. Mirrors figstyle.harness_name."""
    return "fig_" + prose_name.replace("-", "_")


def figure_copies():
    """{harness basename: prose basename} for every figure that has a generator.

    `build_pdf.py` consumes this so that the copy step and the generator list
    cannot drift apart -- adding a figure here is the only edit needed to get it
    into the built document.
    """
    return {harness_name(f.name): f.name for f in FIGURES if f.skip is None}
