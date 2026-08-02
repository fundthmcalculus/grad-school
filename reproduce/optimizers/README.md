# `reproduce/optimizers/` — what is left to find after the hot start

Chapter 6 §6.3.5 makes a claim that is load-bearing for the whole thesis:

> once the model has been built from the data's own structure, a global search
> has very little left to find, and what it does find is substantially noise.

That is the *structure before search* argument stated as an empirical result, and
until now the evidence for it was one dataset, two optimizers (SciPy's
differential evolution and L-BFGS-B), and no controlled budget. This harness
tests it properly: **same start, same objective, same box, same number of
objective evaluations — swap the optimizer.**

It is also how the pending `GA-tuned FIS` column of Table 4.5 gets filled. A
GA-tuned FIS *is* this pipeline with the GA arm selected, so the baseline and
the study come out of one piece of machinery rather than two that can disagree.

```bash
uv run --project tribble-fis --with-editable tribble-opt --with scikit-learn \
    python reproduce/optimizers/run_study.py --smoke        # wiring check
uv run --project tribble-fis --with-editable tribble-opt --with scikit-learn \
    python reproduce/optimizers/run_study.py                # the real thing
```

Outputs land in `reproduce/outputs/` as `table_opt_hotstart.{md,csv}` plus
`table_opt_hotstart_traces.csv`, the per-evaluation convergence trace that
`reproduce/figures/fig_07_optimizer_hotstart.py` plots.

## The three things that make it a fair comparison

**One objective.** Every arm minimizes `tribblefis.refine._make_kfold_fitness` —
the k-fold held-out MSE the shipped refinement already uses, with the
consequents re-solved in closed form at each candidate. Not a reimplementation:
the study imports it.

**One start.** `extract_gaussian_params` on a heuristically-fitted model. That
vector *is* the tribble-fis result, so "improvement" always means improvement on
the shipped pipeline, and it is scored off-budget so no arm is charged for it
while another is not.

**One budget, enforced by the objective.** `BudgetedObjective` counts calls and
raises past the cap; each arm unwinds and the driver keeps the best point seen.
Leaving the budget to each optimizer's own configuration would mean matching
SciPy's `maxiter × popsize × D` against the package's `population_size ×
num_generations` by hand, which is arithmetic that goes stale the first time a
default moves. Everything runs single-threaded so the counter is a true global
count — which is a real limit on what the study can say, and is stated in the
table's note: **an optimizer that parallelises well gets no credit here.**

## The hot start does not reach every optimizer the obvious way

Worth knowing before reading any result, and the reason `arms.py` carries three
mechanisms instead of one.

`InputContinuousVariable` accepts an `initial_value`, which looks like the
warm-start hook. Only two optimizers read it: `GradientDescentOptimizer` and
`MultiTypeOptimizer`. **GA, PSO and ACO ignore it** — they fill the initial
solution deck from `initial_random_value()`, so a hot start passed that way is
discarded silently and the run begins from a uniform sample of the box.

Two other seams do work:

- **Deck injection.** `SolutionDeck.initialize_solution_deck` preserves the
  first `int(archive_size × preserve_percent)` rows, and `solve()` takes
  `preserve_percent`. Writing `x0` into row 0 and solving with
  `preserve_percent = 1/archive_size` puts the incumbent in the initial
  population of any deck-based optimizer. Verified against a quadratic whose
  optimum was injected — the run returns it immediately.
- **The trust region.** `problem.build(radius=…)` shrinks the parameter box
  around `x0`. Every arm samples inside its bounds, so this warm-starts all of
  them, and the radius is the study's main hyperparameter: `1.0` is the full box
  from `build_param_bounds`, smaller values confine the search near the
  incumbent.

Also note `InputContinuousVariable` offsets a supplied `initial_value` by
`perturbation × domain` *deterministically*, and `perturbation` defaults to 0.1
— so passing a hot start with the default places it a tenth of the domain away
from the incumbent in every coordinate at once. `arms.py` passes `0.0`.

## Reading a result

`vs start` is the fraction of the starting objective removed, and `test R²` is
what the chapters quote. They can disagree, and when they do that is the finding
rather than a bug: an arm that drives the cross-validated objective down while
test R² stays flat has overfit the folds used to select it, which is exactly
what §6.3.5 reports for the population methods. `beat start` counts the seeds
where an arm improved the objective at all — reported per arm rather than
averaged away, because "how often did it find anything" is the question.

## Walking up the problem sizes

`problem.DATASETS` is the ladder, smallest first, and Concrete is the first rung
deliberately: ~144 free antecedent parameters, ten seeds × seven arms in well
under an hour, and it is the dataset §6.3.5's existing claim is measured on, so
the new numbers are directly comparable to the old ones. Add a rung by adding an
entry and a branch in `build()`; nothing else in the harness is dataset-specific.
