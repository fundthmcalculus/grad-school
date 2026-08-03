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

## The identification studies — before any optimizer runs

Two more studies live here and neither runs an optimizer. They ask the prior
question: *where does a rule base come from, and what does that cost?* One
answer is the Gaussian construction, the other is the way it was usually done —
cluster, then read one rule off each cluster.

```bash
# Concrete: matched rule count, construction against joint-space clustering
uv run --project tribble-fis --with-editable tribble-cluster --with scikit-learn \
    python reproduce/optimizers/run_identification_study.py \
        --rules 2,3,4,6,8,12 --seeds 0,1,2 --pin-components 1

# PhiUSIIL: the same contest against sample size
uv run --project tribble-fis --with-editable tribble-cluster --with scikit-learn \
    python reproduce/optimizers/run_phishing_study.py \
        --sizes 5000,20000,50000,120000,235795 --seeds 0,1,2 \
        --pin-components --cap-classical 20000
```

`classical.py` does the regression case (cluster the joint input-output space,
project onto the input axes), `phishing.py` the classification case (cluster
within each class). Both hand their model to the same predictor the construction
uses, so what is compared is placement and nothing else.

### Three flags that decide what the comparison means

Getting these wrong produces a number that looks like a result and is not. All
three exist because the first pass got them wrong; see Addendum 3 of
`RESULTS_2026-08-02.md`.

**`--pin-components`.** Left off, the construction chooses its component count
per (feature, bucket) by BIC — `find_optimal_gaussians` fits four EM mixtures,
keeps the winning *count*, throws the mixtures away, and runs k-means. That
search is **82–91% of the construction's identification time** on both datasets,
and it leaves the construction with ~3× the classical parameter count. A
classical route is never asked to do model selection, so leaving it in compares
two different jobs. Pinned, the parameter counts match exactly.

**`--cap-classical`.** `fit_gaussians` truncates every (feature, class) column
to its first `max_samples=20_000` rows before fitting. Above that the
construction's fitting cost stops growing — which is most of what a flat cost
curve on a large dataset is showing. A clustering that reads every row is not
its control; one given the same cap is, and it flattens the same way.

**`--kmeans-n-init`.** Restarts are a quality-versus-time dial, not a property
of the algorithm. The default of 10 makes k-means look slow; quoting the timing
without saying so would rig the comparison, so the table's note states it.

### Two things about the library worth knowing at the call site

Neither is a bug, both are load-bearing, and neither is visible from the
signature:

1. **`fit_gaussians` silently caps at 20,000 samples.** Any scaling claim made
   without knowing this is measuring the cap.
2. **`find_optimal_gaussians` discards the mixtures it fits.** The BIC winner's
   own component means are already computed and would make the following k-means
   redundant — the cheapest available speedup in the identification path, by a
   wide margin.
