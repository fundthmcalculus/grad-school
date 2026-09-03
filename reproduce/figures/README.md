# `reproduce/figures/` — the proposal's figures

Thirty-five figures are cited in the proposal text (sixteen before the 2026-09 expansion, CHECKLIST D9). This directory produces them.

```bash
reproduce/figures/make_figures.py --list       # the inventory and what is skipped
reproduce/figures/make_figures.py              # draw everything, install into prose/fig/
reproduce/figures/make_figures.py 05-band-discovery
```

Each figure is written into `reproduce/outputs/figures/fig_<name>.{png,eps}` and
then copied to `research/proposal-defense/prose/fig/<name>.{png,eps}`. PNG is
what the Markdown embeds; EPS is what the LaTeX build will want. Both come out
of one `savefig` pass so they cannot drift.

## Why the split between `figstyle.py` and the generators

`figstyle.py` owns everything about how a figure *looks* — the validated
categorical palette, the type scale, the two figure widths, the recessive grid,
the schematic primitives. A generator owns only what it *draws*. That division
is the reason fifteen figures written across several sittings look like one set
rather than fifteen.

Two rules in there are not stylistic and should not be relaxed:

- **No alpha anywhere.** EPS has no alpha channel, so a translucent fill
  flattens differently in the two formats and the PNG and the EPS stop
  agreeing. Soft fills come from `figstyle.tint()`, which blends against the
  surface and returns a solid colour.
- **Raster panels are `rasterized=True`.** A 300 × 300 `imshow` exported as
  vector EPS is 90,000 filled paths. The VAT and dissimilarity panels rasterise
  the image and leave the frame and the type vector.

## Each figure runs in the smallest environment that can draw it

`registry.py` records the uv project and the extra packages per figure, and
`make_figures.py` dispatches accordingly — a schematic gets matplotlib and
nothing else, Chapter 2's VAT figure gets `tribble-cluster`, Chapter 6's tree
gets `tribble-fis`. Same discipline as `run_all_tables.sh`: the environment is
part of the provenance, and a schematic that quietly grew a dependency on a
pinned submodule would be a real problem rather than a convenience.

## Skipped figures are recorded, not hidden

A figure whose underlying experiment has not been run keeps its placeholder and
carries a `skip` reason in `registry.py`, printed by `--list`. This is the same
rule the table harness follows when it emits `N/A`: the harness says what it
could not produce rather than substituting something that looks like an answer.
Figure 4.8 (4.3 before the 2026-09 renumbering) was the long-running case — Chapter 4
stated that the accuracy contribution of the correction-rule pass had not been
isolated, so there was no before/after to plot — until the measurement was taken
on Glass and the figure retargeted to it.

Figure 3.4 is also not drawn from here. It is produced by
`reproduce/tables/table_3_1_reorder_three_arm.py`, deliberately: the fitted
exponents in Table 3.2 and the slopes in Figure 3.4 come from the same sweep, so
that a table and a figure describing the same measurement cannot disagree.

## Data, not decoration

Where a figure reports a measured quantity, that quantity is computed at draw
time from the same source the corresponding table uses, or read from the table's
own CSV under `reproduce/outputs/`. No number is typed into a generator by hand.
When a figure needs an illustrative construction rather than a measurement — the
schematics of Chapters 1, 2, 3 and 6 — it says so in its own docstring.
