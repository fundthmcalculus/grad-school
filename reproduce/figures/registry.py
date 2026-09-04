"""Every figure generator in this directory, what produces each one, and what it needs.

One row per figure. This is the file `make_figures.py` walks, the file
`build_pdf.py` reads to know what to copy into the document, and the file to edit
when a figure is added, renamed, or finally produced.

**A generator absent from this list is a generator nothing runs.** Six of them
were: `fig_07` through `fig_12`, the §6.3.5 refinement and identification study
figures, were committed and referenced from
`reproduce/optimizers/RESULTS_2026-08-02.md` but never added here, so
`make_figures.py` did not draw them, `make_figures.py --list` did not report them
missing, and `build_pdf.py` did not copy them. They were hand-run, which is the
practice `PROVENANCE_MAP.md` note 12 spends a paragraph complaining about. The
docstring above this one said "the sixteen proposal figures" while nineteen
generators sat on disk; the count is now derived rather than asserted.

Four things are worth saying about the shape of it.

**`archive` is the figure's provenance, and it is declared, not discovered.**
`harness_data.table()` resolves an unpinned read to the newest archive by its
`generated:` stamp, then falls back to the loose files in `reproduce/outputs/`.
For a figure drawn from `run_all_tables.sh` output that is the right default. For
the six study figures it is not: their tables are produced by a *different*
driver, so the newest archive never contains them and the loose files are
gitignored — meaning on a clean checkout all six raised `FileNotFoundError`, and
on a dirty one they drew from whatever happened to run last. Naming the archive
here fixes both, and puts the answer to "which run is this figure from" in the
inventory instead of in the environment.

**`skip` is a first-class outcome.** A figure whose experiment has not been run
is not a bug in this harness and must not look like one. A row with a `skip`
reason keeps its placeholder PNG and is reported as skipped rather than
failed -- the same discipline the table harness applies when it prints `N/A`
instead of guessing. Figure 4.8 (4.3 before the 2026-09 renumbering) was the one example of this for most of the
document's life: it wanted a before/after confusion matrix for a correction
pass whose effect Chapter 4 said, in as many words, it had not yet isolated.
That RT-IOT2022 confusion matrix still cannot be drawn -- the dataset is still
not one the harness can load -- but the claim it was standing in for has since
been measured on Glass instead (`table_4_9_correction_pass.py`), so the figure
was retargeted to that measurement rather than left waiting on a dataset that
was never going to arrive. No row currently carries a `skip` reason as a
result, which is a fact about today's inventory, not a promise that the
mechanism has nothing left to do.

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
    name: str  # prose/fig/<name>.png
    number: str  # as cited in the text, e.g. "5.2"
    title: str  # one line, for the driver's output
    module: Optional[str] = None  # reproduce/figures/<module>.py
    project: Optional[str] = None  # uv --project, or None for a bare env
    extras: tuple = ()  # uv --with packages
    skip: Optional[str] = None  # why this one is not drawn, if it is not
    owner: Optional[str] = None  # generator, when it is not `module`
    archive: Optional[str] = None  # REPRO_ARCHIVE label to read; None = newest
    document: Optional[str] = None  # where it appears, if not the numbered prose


FIGURES = [
    # -- Chapter 1 ---------------------------------------------------------- #
    Figure(
        "01-structure-before-search",
        "1.1",
        "Conventional search-first route against the structure-first route",
        module="fig_01_structure_before_search",
        extras=("matplotlib",),
    ),
    Figure(
        "01-pipeline-roadmap",
        "1.2",
        "The end-to-end pipeline, annotated with chapters and claims",
        module="fig_01_pipeline_roadmap",
        extras=("matplotlib",),
    ),
    # -- Chapter 2 ---------------------------------------------------------- #
    Figure(
        "02-fis-components",
        "2.1",
        "FIS anatomy: what this work generates against what is fixed by design",
        module="fig_02_fis_components",
        extras=("matplotlib",),
    ),
    Figure(
        "02-tsk-inference",
        "2.2",
        "One-input TSK inference, and the linearity in the consequents",
        module="fig_02_tsk_inference",
        extras=(
            "matplotlib",
            "numpy",
        ),
    ),
    Figure(
        "02-norm-surfaces",
        "2.3",
        "The five t-norm / t-conorm families the library ships, as surfaces",
        module="fig_02_norm_surfaces",
        extras=(
            "matplotlib",
            "numpy",
        ),
    ),
    Figure(
        "02-vat-rdi",
        "2.4",
        "A dissimilarity matrix before and after VAT reordering",
        module="fig_02_vat_rdi",
        project="tribble-cluster",
        extras=("matplotlib", "scipy"),
    ),
    Figure(
        "02-minimax-ultrametric",
        "2.5",
        "The minimax distance: MST path, ultrametric, dendrogram",
        module="fig_02_minimax_ultrametric",
        extras=(
            "matplotlib",
            "scipy",
            "numpy",
        ),
    ),
    Figure(
        "02-persistence",
        "2.6",
        "Single-linkage dendrogram and its persistence diagram",
        module="fig_02_persistence",
        extras=("matplotlib", "scipy", "numpy"),
    ),
    # -- Chapter 3 ---------------------------------------------------------- #
    Figure(
        "03-pvat-reorder",
        "3.1",
        "The mergeVAT reorder against the classical linear-scan argmin",
        module="fig_03_pvat_reorder",
        extras=("matplotlib",),
    ),
    Figure(
        "03-inplace-permutation",
        "3.2",
        "The in-place permutation as cycles, and the k s N^2 footprint it buys",
        module="fig_03_inplace_permutation",
        extras=(
            "matplotlib",
            "numpy",
        ),
    ),
    Figure(
        "03-stitch",
        "3.3",
        "The divide-and-conquer stitch: blocks, farthest-point representatives, cross edges",
        module="fig_03_stitch",
        extras=(
            "matplotlib",
            "numpy",
        ),
    ),
    Figure(
        "03-complexity-fit",
        "3.4",
        "Measured reorder growth against the reference complexity curves",
        owner="reproduce/tables/table_3_1_reorder_three_arm.py",
    ),
    Figure(
        "03-three-arm-seconds",
        "3.5",
        "The three reorder arms in absolute seconds, and stage two's margin",
        module="fig_03_three_arm_seconds",
        extras=("matplotlib",),
        archive="full-14900hx-r2",
    ),
    Figure(
        "03-memory-ceiling",
        "3.6",
        "Dense footprint against N per scheme, with Table 3.3's ceilings as markers",
        module="fig_03_memory_ceiling",
        extras=(
            "matplotlib",
            "numpy",
        ),
    ),
    # -- Chapter 4 ---------------------------------------------------------- #
    Figure(
        "04-mog-classification",
        "4.1",
        "Per-feature Gaussian mixtures and the fuzzy-OR that forms a class rule",
        module="fig_04_mog_classification",
        project="tribble-fis",
        extras=("matplotlib",),
    ),
    Figure(
        "04-output-partitioning",
        "4.2",
        "Output partitioning: the zeroth-order cliff and quantile's instability (Tables 4.2, 4.3)",
        module="fig_04_output_partitioning",
        extras=("matplotlib",),
        archive="uniform-2026-08-03",
    ),
    Figure(
        "04-rule-count",
        "4.3",
        "Rules in the base against features in the data, from the dataset specs",
        module="fig_04_rule_count",
        extras=(
            "matplotlib",
            "numpy",
            "pyyaml",
        ),
    ),
    Figure(
        "04-anomaly-geometry",
        "4.4",
        "The geometry of the anomaly rule: saturation, the winning region, the one-class threshold",
        module="fig_04_anomaly_geometry",
        extras=(
            "matplotlib",
            "numpy",
        ),
    ),
    Figure(
        "04-speedup",
        "4.5",
        "Training time by method and dataset (Table 4.1b)",
        module="fig_04_speedup",
        extras=(
            "matplotlib",
            "numpy",
        ),
        archive="phiusiil-leakfree-2026-08-30",
    ),
    Figure(
        "04-anomaly-sweep",
        "4.6",
        "The open-set operating curve of Table 4.6",
        module="fig_04_anomaly_sweep",
        extras=("matplotlib",),
    ),
    Figure(
        "04-mf-dedup-sweep",
        "4.7",
        "The membership-function deduplication sweep behind Table 4.8, per problem",
        module="fig_04_mf_dedup_sweep",
        extras=("matplotlib",),
        archive="mf-dedup-2026-08-05",
    ),
    Figure(
        "04-rtiot-confusion",
        "4.8",
        "The correction-rule pass, quantified on Glass (MF count and accuracy)",
        module="fig_04_correction_pass",
        extras=("matplotlib",),
        archive="mf-dedup-2026-08-05",
    ),
    # -- Chapter 5 ---------------------------------------------------------- #
    Figure(
        "05-minimax-transform",
        "5.1",
        "Concentric rings: raw dissimilarity against the minimax transform",
        module="fig_05_minimax_transform",
        extras=("matplotlib", "scipy", "numpy", "scikit-learn"),
    ),
    Figure(
        "05-band-discovery",
        "5.2",
        "Band discovery on the log-birth spectrum, and the partitions it yields",
        module="fig_05_band_discovery",
        extras=("matplotlib", "scipy", "numpy", "scikit-learn"),
    ),
    Figure(
        "05-persistence-ramp",
        "5.3",
        "A block's persistence ramp, read off the hierarchy",
        module="fig_05_persistence_ramp",
        extras=("matplotlib", "scipy", "numpy"),
    ),
    # -- Chapter 6 ---------------------------------------------------------- #
    Figure(
        "06-fuzzy-tree",
        "6.1",
        "A trained fuzzy tree on Concrete and on PhiUSIIL, as rules",
        module="fig_06_fuzzy_tree",
        project="tribble-fis",
        extras=("matplotlib",),
    ),
    Figure(
        "06-hme-structure",
        "6.2",
        "Hierarchical mixture: gates over named inputs routing to TSK experts",
        module="fig_06_hme_structure",
        extras=("matplotlib",),
    ),
    # -- The §6.3.5 studies -------------------------------------------------- #
    #
    # Not numbered prose figures. They illustrate `reproduce/optimizers/
    # RESULTS_2026-08-02.md`, which PROVENANCE_MAP.md lists as superseding the
    # two-optimizer evidence behind §6.3.5. Listed here anyway, for the reason at
    # the top of this file: a generator absent from this list is a generator
    # nothing runs, and all six were hand-run for exactly that reason.
    #
    # Every one is a pure plotter over archived CSVs -- numpy and matplotlib, no
    # submodule -- so each needs its archive named. Which archive is not a free
    # choice: the identification and hot-start studies were both re-run after the
    # two `fit_gaussians` defects were fixed (RESULTS Addendum 4; the fix is
    # tribble-fis PR #72, which the parent repo now pins), and the pre-fix
    # archives are superseded. The `kmbic` labels are the post-fix runs.
    Figure(
        "07-optimizer-hotstart",
        "-",
        "Held-out R2 against evaluation budget, hot start against cold",
        module="fig_07_optimizer_hotstart",
        extras=("matplotlib",),
        archive="opt-hotcold-kmbic-2026-08-03",
        document="reproduce/optimizers/RESULTS_2026-08-02.md, Addendum 5",
    ),
    Figure(
        "08-identification",
        "-",
        "Concrete: the classical identification route against the construction",
        module="fig_08_identification",
        extras=("matplotlib",),
        archive="opt-identification-kmbic-pinned-2026-08-03",
        document="reproduce/optimizers/RESULTS_2026-08-02.md, Addendum 4",
    ),
    Figure(
        "09-phishing-scaling",
        "-",
        "PhiUSIIL: how the identification routes scale, and where the cost is",
        module="fig_09_phishing_scaling",
        extras=("matplotlib",),
        archive="opt-phishing-kmbic-pinned-2026-08-03",
        document="reproduce/optimizers/RESULTS_2026-08-02.md, Addendum 4",
    ),
    Figure(
        "10-convergence",
        "-",
        "Full convergence traces, and how much of the objective gain converts",
        module="fig_10_convergence",
        extras=("matplotlib",),
        archive="opt-hotcold-kmbic-2026-08-03",
        document="reproduce/optimizers/RESULTS_2026-08-02.md, Addendum 6",
    ),
    # Addendum 8 re-ran the HOT arms at ten seeds (`opt-phishing-hot10-2026-08-03`)
    # and found the first real arm ordering. These two figures stay on the
    # three-seed hot/cold run because they draw a cold column beside every hot one
    # and the ten-seed run has no cold arms -- so the bands they show are three
    # seeds wide where a ten-seed measurement now exists for half the panel. Worth
    # a redraw once a ten-seed cold arm is run; not worth mixing two seed counts
    # into one panel before then.
    Figure(
        "11-phishing-optimizer",
        "-",
        "PhiUSIIL: objective and test error per arm, with seed spreads",
        module="fig_11_phishing_optimizer",
        extras=("matplotlib",),
        archive="opt-phishing-hotcold-2026-08-03",
        document="reproduce/optimizers/RESULTS_2026-08-02.md, Addendum 7",
    ),
    Figure(
        "12-phishing-timing",
        "-",
        "PhiUSIIL: what construction costs against what searching costs",
        module="fig_12_phishing_timing",
        extras=("matplotlib",),
        archive="opt-phishing-hotcold-2026-08-03",
        document="reproduce/optimizers/RESULTS_2026-08-02.md, Addendum 7",
    ),
]

BY_NAME = {f.name: f for f in FIGURES}


def harness_name(prose_name):
    """`03-complexity-fit` -> `fig_03_complexity_fit`. Mirrors figstyle.harness_name."""
    return "fig_" + prose_name.replace("-", "_")


def prose_figures():
    """The figures the proposal itself cites, in prose order.

    Distinguished from the study figures by `document`: a figure that names an
    external document is illustrating that document, not the proposal, and must
    not be copied into `prose/fig/` -- doing so would add tracked files the built
    PDF never references and make the prose directory a poor answer to "which
    figures does this document have".
    """
    return [f for f in FIGURES if f.document is None]


def figure_copies():
    """{harness basename: prose basename} for every prose figure that has a generator.

    `build_pdf.py` consumes this so that the copy step and the generator list
    cannot drift apart -- adding a figure here is the only edit needed to get it
    into the built document.
    """
    return {harness_name(f.name): f.name for f in prose_figures() if f.skip is None}
